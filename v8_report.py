#!/usr/bin/env python3
"""
ATLAS V8 — Full-Spectrum Report Formatter
Transforms engine output + extended data into a 10-section Slack report.

Each section is delivered as a separate threaded Slack message (<4000 chars each).
Design: Dense data in tables, natural language in narratives, everything in context.
"""

from datetime import datetime


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def _fmt_mc(val):
    """Format large monetary values."""
    if not val or val == 0:
        return "N/A"
    sign = "-" if val < 0 else ""
    av = abs(val)
    if av >= 1e12:
        return f"{sign}${av / 1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}${av / 1e9:.1f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:.0f}M"
    return f"{sign}${av:,.0f}"


def _fmt_pct(val, plus=False):
    """Format percentage."""
    if val is None or val == 0:
        return "N/A"
    prefix = "+" if plus and val > 0 else ""
    return f"{prefix}{val:.1f}%"


def _fmt_x(val):
    """Format multiple."""
    if not val or val == 0:
        return "N/A"
    return f"{val:.1f}x"


def _score_bar(score, width=10):
    """Visual bar: filled + empty blocks."""
    filled = max(0, min(width, round(score / 100 * width)))
    return '\u2588' * filled + '\u2591' * (width - filled)


def _truncate_title(title, max_len=60):
    """Truncate title at word boundary."""
    if len(title) <= max_len:
        return title
    truncated = title[:max_len].rsplit(' ', 1)[0]
    return truncated + '...'


def _trend_arrow(current, prior):
    if current > prior:
        return 'Rising'
    if current < prior:
        return 'Declining'
    return 'Stable'


# ============================================================================
# V8 SCORING SYSTEM
# ============================================================================

def _compute_v8_scores(summary, v8_data):
    """Compute 0-100 scores for each V8 dimension."""
    fin = v8_data.get('financials', {})
    tech = v8_data.get('technicals', {})
    news = v8_data.get('news', [])
    inst = v8_data.get('institutional', {})
    earnings = v8_data.get('earnings', [])

    # 1. Signal Strength (from engine)
    c_raw = summary.get('composite_raw', 0)
    tq = summary.get('trade_quality', 0)
    gate = summary.get('gate_value', 1)
    signal = max(0, min(100, 50 + c_raw * 0.5))
    signal = signal * min(1.3, 0.7 + tq * 2)
    signal = max(0, min(100, signal))

    # 2. Fundamental Score
    fundamental = 50
    rg = fin.get('revenue_growth', 0)
    if rg > 15:
        fundamental += 15
    elif rg > 5:
        fundamental += 8
    elif rg > 0:
        fundamental += 3
    elif rg < -5:
        fundamental -= 15
    else:
        fundamental -= 5

    nm = fin.get('net_margin', 0)
    if nm > 25:
        fundamental += 12
    elif nm > 15:
        fundamental += 6
    elif nm > 5:
        fundamental += 2
    elif nm < 0:
        fundamental -= 15

    roe = fin.get('roe', 0)
    if roe > 25:
        fundamental += 10
    elif roe > 15:
        fundamental += 5
    elif roe < 5:
        fundamental -= 5

    if fin.get('free_cash_flow', 0) > 0:
        fundamental += 5
    else:
        fundamental -= 8

    if earnings:
        beat_rate = sum(1 for e in earnings if e.get('beat', False)) / len(earnings)
        if beat_rate > 0.75:
            fundamental += 8
        elif beat_rate > 0.50:
            fundamental += 4
        elif beat_rate < 0.25:
            fundamental -= 10
    fundamental = max(0, min(100, fundamental))

    # 3. Technical Score
    b = tech.get('bullish_count', 0)
    be = tech.get('bearish_count', 0)
    n = tech.get('neutral_count', 0)
    total = b + be + n
    technical = round(b / total * 100) if total > 0 else 50
    rsi = tech.get('rsi_14', 50)
    if rsi > 70:
        technical = min(technical, 65)
    elif rsi < 30:
        technical = max(technical, 35)
    technical = max(0, min(100, technical))

    # 4. Sentiment Score
    sentiment = 50
    rec = fin.get('recommendation', 'none')
    if rec in ('strong_buy', 'buy'):
        sentiment += 20
    elif rec == 'hold':
        sentiment += 5
    elif rec in ('sell', 'strong_sell'):
        sentiment -= 20

    if news:
        pos = sum(1 for a in news if a.get('sentiment') == 'POSITIVE')
        neg = sum(1 for a in news if a.get('sentiment') == 'NEGATIVE')
        total_n = len(news)
        if total_n > 0:
            sentiment += (pos - neg) / total_n * 20

    sp = inst.get('short_pct', 0)
    if sp < 2:
        sentiment += 10
    elif sp > 10:
        sentiment -= 15
    sentiment = max(0, min(100, sentiment))

    # 5. Macro Score
    vix = summary.get('vix', 20)
    regime = summary.get('regime_label', 'Unknown')
    macro = 50
    if vix < 15:
        macro += 20
    elif vix < 20:
        macro += 10
    elif vix > 28:
        macro -= 20
    elif vix > 22:
        macro -= 10

    if regime == 'Calm':
        macro += 15
    elif regime == 'Chop':
        macro -= 5
    elif regime in ('Crisis Trend', 'Credit Stress'):
        macro -= 20
    elif regime == 'Tightening Shock':
        macro -= 15
    macro = max(0, min(100, macro))

    # 6. Risk Score
    risk = 50
    risk += gate * 20
    rel = summary.get('regime_reliability', 0.5)
    risk += rel * 15
    risk -= summary.get('risk_structural', 0) * 20
    risk -= summary.get('risk_tactical', 0) * 15
    risk = max(0, min(100, risk))

    # Composite
    composite = round(
        signal * 0.30 + fundamental * 0.25 + technical * 0.15 +
        sentiment * 0.10 + macro * 0.10 + risk * 0.10
    )
    composite = max(0, min(100, composite))

    return {
        'signal': round(signal),
        'fundamental': round(fundamental),
        'technical': round(technical),
        'sentiment': round(sentiment),
        'macro': round(macro),
        'risk': round(risk),
        'composite': composite,
    }


def _v8_verdict_label(score):
    """Map 0-100 composite to verdict label."""
    if score >= 80:
        return "STRONG BUY"
    if score >= 60:
        return "BUY"
    if score >= 50:
        return "LEAN BUY"
    if score >= 40:
        return "HOLD"
    if score >= 30:
        return "LEAN SELL"
    if score >= 15:
        return "SELL"
    return "STRONG SELL"


# ============================================================================
# NARRATIVE GENERATORS
# ============================================================================

def _build_v8_narrative(summary, v8_data, scores):
    """One-paragraph summary for the verdict section."""
    co = v8_data.get('company', {})
    fin = v8_data.get('financials', {})
    tech = v8_data.get('technicals', {})
    symbol = co.get('symbol', '???')
    price = co.get('price', 0)

    sma50 = tech.get('sma50', 0)
    sma200 = tech.get('sma200', 0)
    vix = summary.get('vix', 20)

    # Price action
    above_mas = []
    if tech.get('sma20', 0) > 0 and price > tech['sma20']:
        above_mas.append('20d')
    if sma50 > 0 and price > sma50:
        above_mas.append('50d')
    if sma200 > 0 and price > sma200:
        above_mas.append('200d')

    if len(above_mas) == 3:
        s1 = f"{symbol} at ${price:.2f} is above all major moving averages"
    elif len(above_mas) >= 1:
        s1 = f"{symbol} at ${price:.2f} is above its {'/'.join(above_mas)} MA(s)"
    else:
        s1 = f"{symbol} at ${price:.2f} is below its key moving averages"

    vol_desc = "calm" if vix < 18 else "normal" if vix < 23 else "elevated"
    s1 += f" in a {vol_desc} vol environment."

    # Fundamentals
    rg = fin.get('revenue_growth', 0)
    nm = fin.get('net_margin', 0)
    fcf = fin.get('free_cash_flow', 0)
    grade = "strong" if scores['fundamental'] >= 70 else "solid" if scores['fundamental'] >= 55 else "mixed" if scores['fundamental'] >= 40 else "weak"
    fcf_str = f", generating {_fmt_mc(fcf)} in FCF" if fcf > 0 else ""
    s2 = f"Fundamentals are {grade} with {_fmt_pct(rg, plus=True)} revenue growth, {_fmt_pct(nm)} net margins{fcf_str}."

    # Main risk
    fwd_pe = fin.get('forward_pe', 0)
    if fwd_pe > 28:
        s3 = f"Key risk: valuation at {fwd_pe:.1f}x forward leaves little room for disappointment."
    elif fwd_pe > 0 and fwd_pe < 12:
        s3 = f"Valuation at {fwd_pe:.1f}x forward is undemanding — but may reflect fundamental concerns."
    else:
        s3 = ""

    # Action
    verdict = _v8_verdict_label(scores['composite'])
    if 'BUY' in verdict and sma50 > 0:
        s4 = f"Weight of evidence supports buying pullbacks to ${sma50:.0f} (50d MA)."
    elif 'SELL' in verdict:
        s4 = "Balance of evidence favors defensive positioning."
    else:
        s4 = "Evidence is mixed — wait for clearer direction."

    parts = [s1, s2]
    if s3:
        parts.append(s3)
    parts.append(s4)
    return " ".join(parts)


# ============================================================================
# SECTION 1: THE VERDICT
# ============================================================================

def _section_verdict(summary, v8_data):
    """Verdict header + score breakdown + narrative."""
    co = v8_data.get('company', {})
    scores = _compute_v8_scores(summary, v8_data)
    verdict = _v8_verdict_label(scores['composite'])
    symbol = co.get('symbol', '???')
    name = co.get('name', symbol)
    price = co.get('price', 0)
    now = datetime.now().strftime('%b %d, %Y %I:%M %p ET')

    header = (
        f"```\n"
        f"{'=' * 50}\n"
        f"  ATLAS VERDICT: {verdict}  {_score_bar(scores['composite'])} {scores['composite']}/100\n"
        f"  {symbol} -- {name} -- ${price:.2f}\n"
        f"  As of {now}\n"
        f"{'=' * 50}\n"
        f"```\n"
    )

    breakdown = (
        f"```\n"
        f"Signal Strength  {_score_bar(scores['signal'])}  {scores['signal']:>3}\n"
        f"Fundamental      {_score_bar(scores['fundamental'])}  {scores['fundamental']:>3}\n"
        f"Technical        {_score_bar(scores['technical'])}  {scores['technical']:>3}\n"
        f"Sentiment        {_score_bar(scores['sentiment'])}  {scores['sentiment']:>3}\n"
        f"Macro Backdrop   {_score_bar(scores['macro'])}  {scores['macro']:>3}\n"
        f"Risk Profile     {_score_bar(scores['risk'])}  {scores['risk']:>3}\n"
        f"                                ---\n"
        f"Composite                        {scores['composite']:>3}\n"
        f"```\n"
    )

    narrative = _build_v8_narrative(summary, v8_data, scores)

    return header + breakdown + f"> {narrative}"


# ============================================================================
# SECTION 2: FUNDAMENTALS (2A-2B)
# ============================================================================

def _section_fundamentals(v8_data):
    """Key metrics dashboard + earnings track record."""
    fin = v8_data.get('financials', {})
    co = v8_data.get('company', {})
    earnings = v8_data.get('earnings', [])
    symbol = co.get('symbol', '???')

    # Metrics table
    lines = [
        f"*FUNDAMENTAL SNAPSHOT — {symbol}*",
        "```",
        f"{'Metric':<20} {'Value':>12}  {'Context':<20}",
        "-" * 54,
        f"{'Market Cap':<20} {_fmt_mc(fin.get('market_cap', 0)):>12}",
        f"{'Revenue (TTM)':<20} {_fmt_mc(fin.get('revenue_ttm', 0)):>12}  {_fmt_pct(fin.get('revenue_growth', 0), plus=True) + ' YoY':<20}",
        f"{'Net Income (TTM)':<20} {_fmt_mc(fin.get('net_income_ttm', 0)):>12}  {_fmt_pct(fin.get('earnings_growth', 0), plus=True) + ' YoY':<20}",
        f"{'Free Cash Flow':<20} {_fmt_mc(fin.get('free_cash_flow', 0)):>12}  {'Yield: ' + _fmt_pct(fin.get('fcf_yield', 0)):<20}",
        f"{'Gross Margin':<20} {_fmt_pct(fin.get('gross_margin', 0)):>12}",
        f"{'Operating Margin':<20} {_fmt_pct(fin.get('operating_margin', 0)):>12}",
        f"{'Net Margin':<20} {_fmt_pct(fin.get('net_margin', 0)):>12}",
        f"{'ROE':<20} {_fmt_pct(fin.get('roe', 0)):>12}",
        f"{'ROA':<20} {_fmt_pct(fin.get('roa', 0)):>12}",
        f"{'Debt/Equity':<20} {_fmt_x(fin.get('debt_equity', 0)):>12}",
        f"{'Current Ratio':<20} {_fmt_x(fin.get('current_ratio', 0)):>12}",
        "```",
    ]

    # Earnings table
    if earnings:
        lines.append("")
        lines.append("*EARNINGS TRACK RECORD*")
        lines.append("```")
        lines.append(f"{'Quarter':<12} {'EPS Est':>8} {'Actual':>8} {'Surprise':>9}")
        lines.append("-" * 39)
        for e in earnings:
            beat = '+' if e.get('beat') else '-'
            lines.append(
                f"{e.get('date', '?'):<12} "
                f"${e.get('eps_estimate', 0):>6.2f} "
                f"${e.get('eps_actual', 0):>6.2f} "
                f"{'+' if e.get('surprise', 0) >= 0 else ''}{e.get('surprise', 0):>7.2f} {beat}"
            )

        beat_count = sum(1 for e in earnings if e.get('beat'))
        total = len(earnings)
        rate = beat_count / total * 100 if total > 0 else 0
        lines.append("-" * 39)
        lines.append(f"Beat Rate: {rate:.0f}% ({beat_count}/{total})")
        lines.append("```")

    return "\n".join(lines)


# ============================================================================
# SECTION 3: BALANCE SHEET + REVENUE (2C-2D)
# ============================================================================

def _section_balance_sheet(v8_data):
    """Balance sheet health + dividend/buyback."""
    fin = v8_data.get('financials', {})
    co = v8_data.get('company', {})
    symbol = co.get('symbol', '???')

    total_debt = fin.get('total_debt', 0)
    total_cash = fin.get('total_cash', 0)
    net_debt = fin.get('net_debt', 0)

    lines = [
        f"*BALANCE SHEET & CAPITAL RETURN — {symbol}*",
        "```",
        "BALANCE SHEET SUMMARY",
        "-" * 40,
        f"{'Total Cash & Equiv:':<22} {_fmt_mc(total_cash):>15}",
        f"{'Total Debt:':<22} {_fmt_mc(total_debt):>15}",
        f"{'Net Debt:':<22} {_fmt_mc(net_debt):>15}",
        f"{'Net Debt/EBITDA:':<22} {fin.get('net_debt_ebitda', 0):>14.2f}x",
        f"{'Interest Coverage:':<22} {fin.get('interest_coverage', 0):>14.1f}x",
        f"{'Debt/Equity:':<22} {fin.get('debt_equity', 0):>14.2f}x",
        f"{'Current Ratio:':<22} {fin.get('current_ratio', 0):>14.2f}x",
        "",
        "CAPITAL RETURN",
        "-" * 40,
        f"{'Dividend Yield:':<22} {_fmt_pct(fin.get('dividend_yield', 0)):>15}",
        f"{'Payout Ratio:':<22} {_fmt_pct(fin.get('payout_ratio', 0)):>15}",
        f"{'Buyback Yield:':<22} {_fmt_pct(fin.get('buyback_yield', 0)):>15}",
        "```",
    ]

    # Narrative
    nd_ebitda = fin.get('net_debt_ebitda', 0)
    ic = fin.get('interest_coverage', 0)

    if nd_ebitda < 1 and ic > 10:
        health = "Balance sheet is strong"
        detail = f"Net debt is only {nd_ebitda:.1f}x EBITDA with {ic:.0f}x interest coverage."
    elif nd_ebitda < 3:
        health = "Balance sheet is adequate"
        detail = f"Leverage at {nd_ebitda:.1f}x EBITDA is manageable."
    else:
        health = "Balance sheet warrants caution"
        detail = f"Leverage at {nd_ebitda:.1f}x EBITDA is elevated."

    div_y = fin.get('dividend_yield', 0)
    bb_y = fin.get('buyback_yield', 0)
    total_return = div_y + bb_y
    if total_return > 3:
        cap_return = f"Total shareholder return yield of {total_return:.1f}% ({_fmt_pct(div_y)} dividend + {_fmt_pct(bb_y)} buyback) provides a structural tailwind."
    elif total_return > 0:
        cap_return = f"Total return yield is {total_return:.1f}%."
    else:
        cap_return = "No meaningful capital return program."

    lines.append("")
    lines.append(f"> {health}. {detail} {cap_return}")

    return "\n".join(lines)


# ============================================================================
# SECTION 4: VALUATION + PROJECTIONS
# ============================================================================

def _section_valuation(summary, v8_data):
    """Valuation multiples + DCF + price targets."""
    fin = v8_data.get('financials', {})
    dcf = v8_data.get('dcf', {})
    co = v8_data.get('company', {})
    symbol = co.get('symbol', '???')
    price = co.get('price', 0)

    lines = [
        f"*VALUATION & FAIR VALUE — {symbol}*",
        "```",
        f"{'Metric':<16} {'Current':>10}",
        "-" * 28,
        f"{'Trailing P/E':<16} {_fmt_x(fin.get('trailing_pe', 0)):>10}",
        f"{'Forward P/E':<16} {_fmt_x(fin.get('forward_pe', 0)):>10}",
        f"{'PEG Ratio':<16} {_fmt_x(fin.get('peg_ratio', 0)):>10}",
        f"{'P/B Ratio':<16} {_fmt_x(fin.get('price_to_book', 0)):>10}",
        f"{'P/S Ratio':<16} {_fmt_x(fin.get('price_to_sales', 0)):>10}",
        f"{'EV/EBITDA':<16} {_fmt_x(fin.get('ev_ebitda', 0)):>10}",
        f"{'EV/Revenue':<16} {_fmt_x(fin.get('ev_revenue', 0)):>10}",
        f"{'FCF Yield':<16} {_fmt_pct(fin.get('fcf_yield', 0)):>10}",
        "```",
    ]

    # DCF
    if dcf.get('base', 0) > 0:
        bear_fv = dcf['bear']
        base_fv = dcf['base']
        bull_fv = dcf['bull']
        bear_upside = (bear_fv / price - 1) * 100 if price > 0 else 0
        base_upside = (base_fv / price - 1) * 100 if price > 0 else 0
        bull_upside = (bull_fv / price - 1) * 100 if price > 0 else 0

        lines.append("")
        lines.append("```")
        lines.append("DCF FAIR VALUE ESTIMATE")
        lines.append("-" * 42)
        assumptions = dcf.get('assumptions', {})
        if assumptions:
            lines.append(f"  Rev Growth Y1: {assumptions.get('revenue_growth_y1', 0):.0f}%  |  FCF Margin: {assumptions.get('fcf_margin', 0):.0f}%")
            lines.append(f"  WACC: {assumptions.get('discount_rate', 10):.0f}%  |  Terminal: {assumptions.get('terminal_growth', 3):.0f}%")
            lines.append("-" * 42)
        lines.append(f"{'Scenario':<12} {'Fair Value':>12} {'vs Current':>14}")
        lines.append("-" * 42)
        lines.append(f"{'Bear Case':<12} {'$' + f'{bear_fv:.2f}':>12} {_fmt_pct(bear_upside, plus=True):>14}")
        lines.append(f"{'Base Case':<12} {'$' + f'{base_fv:.2f}':>12} {_fmt_pct(base_upside, plus=True):>14}")
        lines.append(f"{'Bull Case':<12} {'$' + f'{bull_fv:.2f}':>12} {_fmt_pct(bull_upside, plus=True):>14}")
        lines.append("```")

    # Analyst targets
    target = fin.get('target_mean', 0)
    t_high = fin.get('target_high', 0)
    t_low = fin.get('target_low', 0)
    if target > 0:
        upside = (target / price - 1) * 100 if price > 0 else 0
        lines.append("")
        lines.append(
            f"_Analyst Consensus Target: ${target:.2f} ({_fmt_pct(upside, plus=True)}) "
            f"| Range: ${t_low:.2f} - ${t_high:.2f} "
            f"| {fin.get('num_analysts', 0)} analysts_"
        )

    # Narrative
    fwd_pe = fin.get('forward_pe', 0)
    if fwd_pe > 30:
        val_grade = "PREMIUM"
        val_note = "trading well above typical multiples — priced for perfection"
    elif fwd_pe > 20:
        val_grade = "FAIR"
        val_note = "in line with growth profile — limited mispricing"
    elif fwd_pe > 0:
        val_grade = "DISCOUNT"
        val_note = "below typical multiples — potential value opportunity"
    else:
        val_grade = "N/A"
        val_note = "insufficient data"

    lines.append(f"\n> *VALUATION GRADE: {val_grade}* — {val_note}.")

    return "\n".join(lines)


# ============================================================================
# SECTION 5: TECHNICAL ANALYSIS
# ============================================================================

def _section_technicals(summary, v8_data):
    """Technical indicators dashboard + support/resistance."""
    tech = v8_data.get('technicals', {})
    co = v8_data.get('company', {})
    symbol = co.get('symbol', '???')
    price = tech.get('price', 0) or co.get('price', 0)

    def _sig(val, bull_cond, bear_cond):
        if bull_cond:
            return 'Bullish'
        if bear_cond:
            return 'Bearish'
        return 'Neutral'

    sma20 = tech.get('sma20', 0)
    sma50 = tech.get('sma50', 0)
    sma200 = tech.get('sma200', 0)
    rsi = tech.get('rsi_14', 50)
    macd_l = tech.get('macd_line', 0)
    macd_s = tech.get('macd_signal', 0)
    macd_h = tech.get('macd_histogram', 0)
    stoch_k = tech.get('stochastic_k', 50)
    stoch_d = tech.get('stochastic_d', 50)
    adx = tech.get('adx', 25)
    bb_up = tech.get('bollinger_upper', 0)
    bb_lo = tech.get('bollinger_lower', 0)
    obv_t = tech.get('obv_trend', 'Flat')

    lines = [
        f"*TECHNICAL ANALYSIS — {symbol}*",
        "```",
        f"{'Indicator':<18} {'Value':>10} {'Signal':>10}",
        "-" * 40,
        f"{'Price vs SMA20':<18} {'$' + f'{sma20:.0f}' if sma20 else 'N/A':>10} {_sig(0, price > sma20 and sma20 > 0, price < sma20 and sma20 > 0):>10}",
        f"{'Price vs SMA50':<18} {'$' + f'{sma50:.0f}' if sma50 else 'N/A':>10} {_sig(0, price > sma50 and sma50 > 0, price < sma50 and sma50 > 0):>10}",
        f"{'Price vs SMA200':<18} {'$' + f'{sma200:.0f}' if sma200 else 'N/A':>10} {_sig(0, price > sma200 and sma200 > 0, price < sma200 and sma200 > 0):>10}",
        f"{'RSI (14)':<18} {rsi:>10.1f} {_sig(rsi, 40 < rsi < 60, rsi > 70 or rsi < 30):>10}",
        f"{'MACD':<18} {macd_l:>+10.2f} {_sig(0, macd_l > macd_s, macd_l < macd_s):>10}",
        f"{'MACD Histogram':<18} {macd_h:>+10.2f} {_sig(0, macd_h > 0, macd_h < 0):>10}",
        f"{'Stochastic':<18} {f'{stoch_k:.0f}/{stoch_d:.0f}':>10} {_sig(stoch_k, stoch_k < 20, stoch_k > 80):>10}",
        f"{'Bollinger Band':<18} {'Mid-Upper' if price > (bb_up + bb_lo) / 2 and bb_up > 0 else 'Mid-Lower' if bb_lo > 0 else 'N/A':>10} {'Neutral':>10}",
        f"{'OBV Trend':<18} {obv_t:>10} {_sig(0, obv_t == 'Rising', obv_t == 'Falling'):>10}",
        f"{'ADX (14)':<18} {adx:>10.1f} {'Strong' if adx > 25 else 'Weak':>10}",
        "```",
    ]

    b = tech.get('bullish_count', 0)
    be = tech.get('bearish_count', 0)
    total = b + be + tech.get('neutral_count', 0)
    if total > 0:
        grade = "BULLISH" if b / total > 0.6 else "BEARISH" if be / total > 0.6 else "MIXED"
        lines.append(f"_TECHNICAL GRADE: {grade} ({b} of {total} indicators positive)_")

    # Support / Resistance
    lines.append("")
    lines.append("```")
    lines.append("KEY LEVELS")
    lines.append("-" * 40)

    for r in tech.get('resistance_levels', [])[:3]:
        lines.append(f"  Resistance:  ${r:.2f}")

    lines.append(f"  --- CURRENT  ${price:.2f} ---")

    for s in tech.get('support_levels', [])[:3]:
        lines.append(f"  Support:     ${s:.2f}")

    hi52 = tech.get('fifty_two_week_high', 0)
    lo52 = tech.get('fifty_two_week_low', 0)
    pos52 = tech.get('fifty_two_week_position', 50)
    if hi52 > 0 and lo52 > 0:
        lines.append(f"\n  52-Week Range: ${lo52:.2f} - ${hi52:.2f}")
        lines.append(f"  Position: {pos52:.0f}th percentile")

    lines.append("```")

    return "\n".join(lines)


# ============================================================================
# SECTION 6: COMPETITIVE LANDSCAPE
# ============================================================================

def _section_competitive(v8_data):
    """Peer comparison table."""
    peers = v8_data.get('peers', [])
    co = v8_data.get('company', {})
    fin = v8_data.get('financials', {})
    symbol = co.get('symbol', '???')

    if not peers:
        return f"*COMPETITIVE LANDSCAPE — {symbol}*\n_No peer data available._"

    # Build peer table including the target company
    target_row = {
        'symbol': symbol,
        'name': co.get('name', symbol)[:12],
        'price': co.get('price', 0),
        'market_cap': fin.get('market_cap', 0),
        'revenue_growth': fin.get('revenue_growth', 0),
        'profit_margin': fin.get('net_margin', 0),
        'forward_pe': fin.get('forward_pe', 0),
        'roe': fin.get('roe', 0),
    }

    all_companies = [target_row] + peers

    lines = [
        f"*COMPETITIVE LANDSCAPE — {symbol} vs PEERS*",
        "```",
        f"{'Ticker':<7} {'Price':>8} {'MktCap':>8} {'RevGr':>7} {'Margin':>7} {'FwdPE':>7} {'ROE':>7}",
        "=" * 53,
    ]

    for c in all_companies:
        marker = " <" if c['symbol'] == symbol else ""
        p = c.get('price', 0)
        price_str = f"${p:.0f}"
        lines.append(
            f"{c['symbol']:<7} "
            f"{price_str:>8} "
            f"{_fmt_mc(c.get('market_cap', 0)):>8} "
            f"{_fmt_pct(c.get('revenue_growth', 0), plus=True):>7} "
            f"{_fmt_pct(c.get('profit_margin', 0)):>7} "
            f"{_fmt_x(c.get('forward_pe', 0)):>7} "
            f"{_fmt_pct(c.get('roe', 0)):>6}{marker}"
        )

    lines.append("=" * 53)
    lines.append("```")

    # Competitive position
    rev_rank = sorted(all_companies, key=lambda x: x.get('revenue_growth', 0), reverse=True)
    margin_rank = sorted(all_companies, key=lambda x: x.get('profit_margin', 0), reverse=True)
    pe_rank = sorted(all_companies, key=lambda x: x.get('forward_pe', 0) if x.get('forward_pe', 0) > 0 else 999)

    rev_pos = next((i + 1 for i, c in enumerate(rev_rank) if c['symbol'] == symbol), 0)
    margin_pos = next((i + 1 for i, c in enumerate(margin_rank) if c['symbol'] == symbol), 0)
    pe_pos = next((i + 1 for i, c in enumerate(pe_rank) if c['symbol'] == symbol), 0)
    n = len(all_companies)

    lines.append(f"_Rank among {n} peers: Revenue Growth #{rev_pos}, Profit Margin #{margin_pos}, Valuation #{pe_pos}_")

    return "\n".join(lines)


# ============================================================================
# SECTION 7: SENTIMENT & NEWS
# ============================================================================

def _section_sentiment(v8_data):
    """Analyst consensus + news sentiment.
    Returns a dict with 'blocks' and 'text' when news is available (Block Kit),
    or a plain string when there's no news.
    """
    fin = v8_data.get('financials', {})
    news = v8_data.get('news', [])
    inst = v8_data.get('institutional', {})
    co = v8_data.get('company', {})
    symbol = co.get('symbol', '???')
    price = co.get('price', 0)

    rec = fin.get('recommendation', 'none')
    n_analysts = fin.get('num_analysts', 0)
    target = fin.get('target_mean', 0)
    t_high = fin.get('target_high', 0)
    t_low = fin.get('target_low', 0)
    upside = (target / price - 1) * 100 if price > 0 and target > 0 else 0

    lines = [
        f"*SENTIMENT & NEWS — {symbol}*",
        "```",
        "WALL STREET CONSENSUS",
        "-" * 40,
        f"  Recommendation:  {rec.upper().replace('_', ' ') if rec else 'N/A'}",
        f"  Analysts:        {n_analysts}",
        f"  Price Target:    ${target:.2f} ({_fmt_pct(upside, plus=True)})" if target > 0 else "  Price Target:    N/A",
        f"  Target Range:    ${t_low:.2f} - ${t_high:.2f}" if t_low > 0 else "",
        "```",
        "```",
        "SENTIMENT INDICATORS",
        "-" * 40,
        f"  Short Interest:        {inst.get('short_pct', 0):.1f}% of float",
        f"  Short Ratio:           {inst.get('short_ratio', 0):.1f} days",
        f"  Institutional:         {inst.get('institutional_pct', 0):.1f}%",
        f"  Insider Ownership:     {inst.get('insider_pct', 0):.1f}%",
        "```",
    ]

    # If no news, return plain text
    valid_news = [a for a in news if a.get('title', '').strip()] if news else []
    if not valid_news:
        return "\n".join(lines)

    pos = sum(1 for a in valid_news if a.get('sentiment') == 'POSITIVE')
    neg = sum(1 for a in valid_news if a.get('sentiment') == 'NEGATIVE')
    neu = len(valid_news) - pos - neg

    # Add news header + legend to the mrkdwn portion
    lines.append("")
    lines.append("*RECENT NEWS*")
    lines.append("```Positive ( + )  |  Negative ( - )  |  Neutral ( = )```")
    top_mrkdwn = "\n".join(lines)

    # Build rich_text_quote elements for articles (Block Kit)
    quote_elements = []
    for i, a in enumerate(valid_news[:7]):
        s = a.get('sentiment', 'NEUTRAL')
        marker = '( + )' if s == 'POSITIVE' else '( - )' if s == 'NEGATIVE' else '( = )'
        title = _truncate_title(a.get('title', ''))
        link = a.get('link', '')
        pub = a.get('publisher', '')
        date = a.get('date', '')
        pub = pub.replace(' Video', '').replace('.com', '')

        quote_elements.append({"type": "text", "text": f"{marker}  {title}"})

        if link and pub:
            quote_elements.append({"type": "text", "text": " \u2014 "})
            quote_elements.append({"type": "link", "url": link, "text": pub})
        elif pub:
            quote_elements.append({"type": "text", "text": f" \u2014 {pub}"})

        if date:
            quote_elements.append({"type": "text", "text": f" ({date})"})

        # Newline between articles (not after last)
        if i < min(len(valid_news), 7) - 1:
            quote_elements.append({"type": "text", "text": "\n"})

    # Sentiment summary line
    score = (pos - neg) / len(valid_news) if valid_news else 0
    label = "POSITIVE" if score > 0.2 else "NEGATIVE" if score < -0.2 else "NEUTRAL"
    sentiment_summary = f"_Sentiment: {label}  |  ( + ) {pos}  |  ( - ) {neg}  |  ( = ) {neu}_"

    # Assemble Block Kit blocks
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": top_mrkdwn}
        },
        {
            "type": "rich_text",
            "elements": [
                {
                    "type": "rich_text_quote",
                    "elements": quote_elements
                }
            ]
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": sentiment_summary}
        }
    ]

    fallback = top_mrkdwn + "\n" + sentiment_summary
    return {"blocks": blocks, "text": fallback}


# ============================================================================
# SECTION 8: RISK FACTORS + GROWTH CATALYSTS
# ============================================================================

def _generate_risks(summary, v8_data):
    """Generate ranked risk factors from data."""
    fin = v8_data.get('financials', {})
    tech = v8_data.get('technicals', {})
    risks = []

    fwd_pe = fin.get('forward_pe', 0)
    if fwd_pe > 25:
        risks.append(('Valuation Compression', 'Medium' if fwd_pe < 35 else 'High', 'High',
                       f"At {fwd_pe:.0f}x forward, any growth miss triggers P/E contraction"))

    rg = fin.get('revenue_growth', 0)
    if rg < 3:
        risks.append(('Revenue Stagnation', 'Medium', 'High',
                       f"Growth at {rg:.1f}% is below market expectations"))

    de = fin.get('debt_equity', 0)
    if de > 2:
        risks.append(('Leverage Risk', 'Low', 'High',
                       f"Debt/equity of {de:.1f}x exposes the company in a downturn"))

    vix = summary.get('vix', 20)
    if vix > 22:
        risks.append(('Volatility Risk', 'Medium', 'Medium',
                       f"VIX at {vix:.1f} signals elevated uncertainty"))

    nm = fin.get('net_margin', 0)
    eg = fin.get('earnings_growth', 0)
    if eg < 0 and nm > 0:
        risks.append(('Margin Pressure', 'Medium', 'Medium',
                       f"Earnings declining {eg:.1f}% despite positive margins"))

    regime = summary.get('regime_label', '')
    if regime in ('Crisis Trend', 'Credit Stress', 'Tightening Shock'):
        risks.append(('Macro Regime Stress', 'High', 'High',
                       f"ATLAS detects {regime} conditions"))

    sp = v8_data.get('institutional', {}).get('short_pct', 0)
    if sp > 5:
        risks.append(('High Short Interest', 'Medium', 'Medium',
                       f"Short interest at {sp:.1f}% — bears are positioned"))

    # Ensure at least one
    if not risks:
        risks.append(('Market Risk', 'Low', 'Medium', 'General market downturn'))

    return risks[:5]


def _generate_catalysts(summary, v8_data):
    """Generate ranked growth catalysts from data."""
    fin = v8_data.get('financials', {})
    earnings = v8_data.get('earnings', [])
    catalysts = []

    rg = fin.get('revenue_growth', 0)
    if rg > 10:
        catalysts.append(('Revenue Acceleration', 'Medium', 'HIGH',
                          f"Growing at {rg:.1f}% — above-market growth drives re-rating"))

    if earnings:
        beat_count = sum(1 for e in earnings if e.get('beat'))
        if beat_count >= len(earnings) * 0.75:
            catalysts.append(('Earnings Consistency', 'High', 'MEDIUM',
                              f"Beat rate of {beat_count}/{len(earnings)} — next print likely positive"))

    rec = fin.get('recommendation', '')
    if rec in ('strong_buy', 'buy'):
        catalysts.append(('Analyst Momentum', 'High', 'MEDIUM',
                          f"Consensus is {rec.replace('_', ' ')} — positive revision cycle"))

    bb = fin.get('buyback_yield', 0)
    dy = fin.get('dividend_yield', 0)
    if bb + dy > 2:
        catalysts.append(('Capital Return', 'Very High', 'MEDIUM',
                          f"{_fmt_pct(bb + dy)} total yield — structural EPS support"))

    nm = fin.get('net_margin', 0)
    om = fin.get('operating_margin', 0)
    if om > 20 and nm > 15:
        catalysts.append(('Margin Strength', 'High', 'MEDIUM',
                          f"Operating at {om:.1f}% margin — pricing power intact"))

    fcf = fin.get('free_cash_flow', 0)
    if fcf > 0:
        catalysts.append(('Cash Generation', 'High', 'MEDIUM',
                          f"Generating {_fmt_mc(fcf)} in FCF — funds growth and returns"))

    if not catalysts:
        catalysts.append(('Valuation Support', 'Medium', 'MEDIUM', 'Trading at reasonable valuation'))

    return catalysts[:5]


def _section_risk_catalysts(summary, v8_data):
    """Risk factors and growth catalysts."""
    co = v8_data.get('company', {})
    symbol = co.get('symbol', '???')

    risks = _generate_risks(summary, v8_data)
    catalysts = _generate_catalysts(summary, v8_data)

    lines = [
        f"*RISK FACTORS & GROWTH CATALYSTS — {symbol}*",
        "",
        "```",
        "TOP RISK FACTORS",
        "=" * 50,
    ]

    for i, (name, prob, impact, desc) in enumerate(risks, 1):
        lines.append(f" {i}. {name:<28} P:{prob:<7} I:{impact}")
        lines.append(f"    {desc}")

    lines.append("=" * 50)
    lines.append("```")

    lines.append("")
    lines.append("```")
    lines.append("GROWTH CATALYSTS")
    lines.append("=" * 50)

    for i, (name, prob, impact, desc) in enumerate(catalysts, 1):
        lines.append(f" {i}. {name:<28} P:{prob:<10} I:{impact}")
        lines.append(f"    {desc}")

    lines.append("=" * 50)
    lines.append("```")

    return "\n".join(lines)


# ============================================================================
# SECTION 9: MACRO & MARKET CONTEXT
# ============================================================================

def _section_macro(v8_data):
    """Market dashboard + sector performance + economic indicators."""
    mkt = v8_data.get('market', {})
    sector = v8_data.get('sector', {})
    econ = v8_data.get('economic', {})

    lines = ["*MACRO & MARKET CONTEXT*"]

    # Market indices
    indices = mkt.get('indices', [])
    if indices:
        lines.append("```")
        lines.append(f"{'Index':<16} {'Level':>10} {'1D':>8} {'1M':>8} {'YTD':>8}")
        lines.append("-" * 52)
        for idx in indices:
            lines.append(
                f"{idx['name']:<16} "
                f"{idx['price']:>10,.0f} "
                f"{_fmt_pct(idx['1d'], plus=True):>8} "
                f"{_fmt_pct(idx['1m'], plus=True):>8} "
                f"{_fmt_pct(idx['ytd'], plus=True):>8}"
            )

        # Bonds
        for b in mkt.get('bonds', []):
            lines.append(f"{b['name']:<16} {b['value']:>9.3f}% {b['1d_change']:>+7.3f}")

        # Commodities
        for c in mkt.get('commodities', []):
            cprice = f"${c.get('price', 0):,.0f}"
            lines.append(
                f"{c['name']:<16} "
                f"{cprice:>10} "
                f"{_fmt_pct(c.get('1d', 0), plus=True):>8} "
                f"{_fmt_pct(c.get('1m', 0), plus=True):>8}"
            )

        # Crypto
        for cr in mkt.get('crypto', []):
            crprice = f"${cr.get('price', 0):,.0f}"
            lines.append(
                f"{cr['name']:<16} "
                f"{crprice:>10} "
                f"{_fmt_pct(cr.get('1d', 0), plus=True):>8} "
                f"{_fmt_pct(cr.get('1m', 0), plus=True):>8}"
            )

        lines.append("```")

    # Sector performance
    all_sectors = sector.get('all_sectors', [])
    if all_sectors:
        target_sector = sector.get('name', '')
        lines.append("")
        lines.append("```")
        lines.append(f"{'Sector':<18} {'1-Week':>8} {'1-Month':>8} {'YTD':>8}")
        lines.append("-" * 44)
        for s in all_sectors:
            marker = "  <" if s.get('sector_key') == target_sector else ""
            lines.append(
                f"{s['name']:<18} "
                f"{_fmt_pct(s['1w'], plus=True):>8} "
                f"{_fmt_pct(s['1m'], plus=True):>8} "
                f"{_fmt_pct(s['ytd'], plus=True):>8}"
                f"{marker}"
            )
        lines.append("```")

    # Economic indicators
    indicators = econ.get('indicators', [])
    if indicators:
        lines.append("")
        lines.append("```")
        lines.append(f"{'Indicator':<22} {'Latest':>8} {'Prior':>8} {'Trend':>10}")
        lines.append("-" * 50)
        for ind in indicators:
            lines.append(
                f"{ind['name']:<22} "
                f"{ind['value']:>8.2f} "
                f"{ind['prior']:>8.2f} "
                f"{ind['trend']:>10}"
            )
        lines.append("```")

    return "\n".join(lines)


# ============================================================================
# SECTION 10: ATLAS ENGINE + FINAL WORD
# ============================================================================

def _section_engine_final(summary, v8_data):
    """ATLAS engine output + The Final Word."""
    co = v8_data.get('company', {})
    fin = v8_data.get('financials', {})
    tech = v8_data.get('technicals', {})
    symbol = co.get('symbol', '???')
    price = co.get('price', 0)

    scores_dict = summary.get('scores', {})
    contributions = summary.get('contributions', {})
    w_dyn = summary.get('w_dynamic', {})
    c_raw = summary.get('composite_raw', 0)
    c_adj = summary.get('composite_adjusted', 0)
    tq = summary.get('trade_quality', 0)
    gate = summary.get('gate_value', 0)
    dc = summary.get('data_confidence', 0)
    regime = summary.get('regime_label', 'Unknown')
    rel = summary.get('regime_reliability', 0)
    exec_mode = summary.get('execution_mode', 'N/A')

    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']

    lines = [
        f"*ATLAS ENGINE SIGNAL — {symbol}*",
        "```",
        f"Composite: {c_raw:+.1f}/100  |  Adjusted: {c_adj:+.1f}/100",
        f"Trade Quality: {tq:.3f}  |  Gate: {gate:.2f}  |  DC: {dc:.0f}%",
        f"Regime: {regime}  |  Reliability: {rel:.2f}  |  Mode: {exec_mode}",
        "",
        f"{'Engine':<13} {'Score':>6} {'Wt':>6} {'Contrib':>8}",
        "-" * 35,
    ]

    c_sum = 0
    for eng in engines:
        sc = float(scores_dict.get(eng, 0))
        w = float(w_dyn.get(eng, 1 / 8))
        c = float(contributions.get(eng, 0))
        c_sum += c
        lines.append(f"{eng:<13} {sc:>+6.1f} {w:>6.3f} {c:>+8.2f}")

    lines.append(f"{'':>13} {'':>6} {'-----':>6} {'------':>8}")
    lines.append(f"{'TOTAL':<13} {'':>6} {'1.000':>6} {c_sum:>+8.2f}")
    lines.append("```")

    # Trade levels from engine
    stop = summary.get('stop_loss', 0)
    tp = summary.get('take_profit', (0, 0))
    buy_zone = summary.get('buy_zone', (0, 0))
    pos_size = summary.get('position_size', 0)
    pos_pct = summary.get('position_pct', 0)

    if stop > 0:
        lines.append("```")
        lines.append("ATLAS TRADE PLAN")
        lines.append("-" * 35)
        if buy_zone and buy_zone[0] > 0:
            lines.append(f"Buy Zone:       ${buy_zone[0]:.2f} - ${buy_zone[1]:.2f}")
        lines.append(f"Stop Loss:      ${stop:.2f}")
        if tp and tp[0] > 0:
            lines.append(f"Target:         ${tp[0]:.2f} - ${tp[1]:.2f}")
        lines.append(f"Position:       ${pos_size:,.0f} ({pos_pct:.1f}%)")
        lines.append(f"Mode:           {exec_mode}")
        lines.append("```")

    # THE FINAL WORD
    scores = _compute_v8_scores(summary, v8_data)
    verdict = _v8_verdict_label(scores['composite'])

    sma50 = tech.get('sma50', 0) or summary.get('sma50', 0)
    sma200 = tech.get('sma200', 0) or summary.get('sma200', 0)

    lines.append("")
    lines.append(f"```")
    lines.append(f"{'=' * 50}")
    lines.append(f"  THE FINAL WORD ON {symbol}")
    lines.append(f"{'=' * 50}")
    lines.append("")

    # Verdict line
    if 'BUY' in verdict:
        if sma50 > 0 and price > sma50 * 1.03:
            lines.append(f"VERDICT: {verdict} on pullbacks to ${sma50:.0f}")
        else:
            lines.append(f"VERDICT: {verdict} at current levels")
    elif 'SELL' in verdict:
        lines.append(f"VERDICT: {verdict} — reduce exposure")
    elif verdict == 'HOLD':
        lines.append(f"VERDICT: {verdict} — maintain, no new positions")
    else:
        lines.append(f"VERDICT: {verdict}")

    lines.append("")

    # WHY section
    lines.append("WHY:")
    if scores['fundamental'] >= 60:
        rg = fin.get('revenue_growth', 0)
        fcf = fin.get('free_cash_flow', 0)
        lines.append(f"  + Fundamentals: {_fmt_pct(rg, plus=True)} rev growth, {_fmt_mc(fcf)} FCF")
    else:
        lines.append(f"  - Fundamentals: score {scores['fundamental']}/100")

    if scores['technical'] >= 55:
        lines.append(f"  + Technicals: {tech.get('bullish_count', 0)} of {tech.get('bullish_count', 0) + tech.get('bearish_count', 0) + tech.get('neutral_count', 0)} indicators bullish")
    else:
        lines.append(f"  - Technicals: score {scores['technical']}/100")

    if scores['sentiment'] >= 55:
        lines.append(f"  + Sentiment: analyst consensus {fin.get('recommendation', 'N/A')}")
    else:
        lines.append(f"  - Sentiment: score {scores['sentiment']}/100")

    if scores['macro'] >= 50:
        lines.append(f"  + Macro: {regime} regime, VIX {summary.get('vix', 0):.1f}")
    else:
        lines.append(f"  - Macro: {regime} regime, VIX {summary.get('vix', 0):.1f}")

    # ACTION PLAN
    lines.append("")
    lines.append("ACTION:")
    if 'BUY' in verdict:
        if sma50 > 0:
            lines.append(f"  Entry:  ${sma50:.0f} area (50d MA pullback)")
        if stop > 0:
            lines.append(f"  Stop:   ${stop:.2f}")
        if tp and tp[0] > 0:
            lines.append(f"  Target: ${tp[0]:.2f} - ${tp[1]:.2f}")
    elif 'SELL' in verdict:
        lines.append(f"  Reduce or hedge existing positions")
        if stop > 0:
            lines.append(f"  Hard stop at ${stop:.2f}")
    else:
        lines.append(f"  Wait for clearer direction")
        if sma50 > 0:
            lines.append(f"  Watch ${sma50:.0f} (50d MA) for signal")

    lines.append("")
    lines.append(f"{'=' * 50}")
    lines.append(f"ATLAS Confidence: {scores['composite']}/100")
    now = datetime.now().strftime('%b %d, %Y %I:%M %p ET')
    lines.append(f"Generated: {now}")
    lines.append("This is model output, NOT financial advice.")
    lines.append(f"{'=' * 50}")
    lines.append("```")

    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def format_v8_report(summary, v8_data):
    """
    Generate complete V8 report as list of Slack messages.
    10 messages covering all sections of the full-spectrum analysis.

    Args:
        summary: dict from run_atlas() — engine output
        v8_data: dict from fetch_v8_data() — extended data

    Returns:
        list of strings, each under 4000 chars
    """
    messages = [
        _section_verdict(summary, v8_data),
        _section_fundamentals(v8_data),
        _section_balance_sheet(v8_data),
        _section_valuation(summary, v8_data),
        _section_technicals(summary, v8_data),
        _section_competitive(v8_data),
        _section_sentiment(v8_data),
        _section_risk_catalysts(summary, v8_data),
        _section_macro(v8_data),
        _section_engine_final(summary, v8_data),
    ]

    # Ensure no message exceeds Slack's 4000 char limit
    result = []
    for msg in messages:
        if msg:
            if isinstance(msg, dict):
                # Block Kit message — pass through as-is
                result.append(msg)
            elif len(msg) > 3900:
                # Truncate with warning
                msg = msg[:3850] + "\n```\n_[Truncated — section too large]_"
                result.append(msg)
            else:
                result.append(msg)

    return result
