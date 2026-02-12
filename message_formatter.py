"""
ATLAS — Trader Abstraction Layer (Layer 11)
Transforms quantitative engine output into trader-ready Slack messages.

Design principles:
  - Write like a senior trader's morning desk note
  - Every section uses human language, not data labels
  - Drivers are explained in context with actual price levels
  - Suppression rules prevent showing non-tradable artifacts
  - One risk gate value throughout — zero contradictions
  - Reads in under 30 seconds
"""

import numpy as np


# ============================================================================
# DECISION RULES
# ============================================================================

def _trader_verdict(s):
    tq = s.get('trade_quality', 0)
    rel = s.get('regime_reliability', 0)
    gate = s.get('gate_value', 0)
    c_raw = s.get('composite_raw', 0)
    if tq < 0.12 or rel < 0.25 or gate < 0.35:
        return 'CASH'
    if tq < 0.20:
        return 'WAIT'
    if c_raw > 0:
        return 'BUY'
    elif c_raw < 0:
        return 'SELL'
    return 'WAIT'


def _bias(c_raw):
    if c_raw > 5:
        return 'Bullish'
    elif c_raw < -5:
        return 'Bearish'
    return 'Neutral'


def _risk_env(regime_label):
    return {
        'Calm': 'Calm', 'Chop': 'Normal',
        'Tightening Shock': 'Elevated',
        'Crisis Trend': 'Stress', 'Credit Stress': 'Stress',
    }.get(regime_label, 'Normal')


def _readability_label(rel):
    if rel < 0.25:
        return 'Low'
    if rel < 0.50:
        return 'Mixed'
    if rel < 0.75:
        return 'Readable'
    return 'Clean'


def _conviction(tq):
    if tq < 0.12:
        return 'No Edge'
    if tq < 0.20:
        return 'Low Conviction'
    if tq < 0.35:
        return 'Tradable'
    return 'High Conviction'


def _verdict_emoji(verdict):
    return {
        'BUY': ':chart_with_upwards_trend:',
        'SELL': ':chart_with_downwards_trend:',
        'WAIT': ':eyes:',
        'CASH': ':no_entry_sign:',
    }.get(verdict, ':bar_chart:')


# ============================================================================
# NARRATIVE BUILDER
# ============================================================================

def _build_narrative(s):
    """Build a 3-4 sentence plain English summary of the situation."""
    symbol = s.get('symbol', '???')
    price = s.get('price', 0)
    sma20 = s.get('sma20', 0)
    sma50 = s.get('sma50', 0)
    sma200 = s.get('sma200', 0)
    verdict = _trader_verdict(s)
    regime = s.get('regime_label', 'Unknown')
    rel = s.get('regime_reliability', 0)
    tq = s.get('trade_quality', 0)
    c_raw = s.get('composite_raw', 0)
    scores = s.get('scores', {})
    vix = s.get('vix', 20)

    # Sentence 1: Price action context with MA levels
    above = []
    below = []
    if sma20 > 0:
        (above if price > sma20 else below).append(f"20-day (${sma20:.0f})")
    if sma50 > 0:
        (above if price > sma50 else below).append(f"50-day (${sma50:.0f})")
    if sma200 > 0:
        (above if price > sma200 else below).append(f"200-day (${sma200:.0f})")

    if above and below:
        s1 = f"{symbol} is at ${price:.2f}, holding above its {' and '.join(above)} but trading below the {' and '.join(below)}."
    elif below:
        s1 = f"{symbol} is at ${price:.2f}, trading below its {' and '.join(below)} moving averages."
    elif above:
        s1 = f"{symbol} is at ${price:.2f}, holding above its {' and '.join(above)} moving averages."
    else:
        s1 = f"{symbol} is at ${price:.2f}."

    # Sentence 2: Regime / environment
    regime_desc = {
        'Calm': "The overall environment is calm with stable volatility, which typically favors trending moves.",
        'Chop': "The market is in a choppy, range-bound regime — signals are noisy and price keeps reversing before trends develop.",
        'Tightening Shock': "Rates are moving sharply, creating stress across rate-sensitive sectors and compressing multiples.",
        'Crisis Trend': "The model detects crisis-level conditions with risk signals firing across multiple inputs.",
        'Credit Stress': "Credit spreads are widening, a warning sign that often precedes broader equity weakness.",
    }
    s2 = regime_desc.get(regime, "Market conditions are mixed with no dominant regime.")

    # Sentence 3: Verdict justification
    if verdict == 'CASH':
        if rel < 0.25:
            s3 = f"Market readability is near zero — the model can't distinguish signal from noise right now. No trade until the picture clears."
        elif tq < 0.05:
            s3 = f"Signal strength is extremely weak (TQ {tq:.3f}). There's no edge in either direction. Sit on hands."
        else:
            s3 = "Conditions don't meet the threshold for deploying capital. Stay flat and wait for a cleaner setup."
    elif verdict == 'WAIT':
        bias = _bias(c_raw)
        s3 = f"There's a {bias.lower()} lean forming but conviction isn't strong enough to size a position yet. Watch for confirmation."
    elif verdict == 'BUY':
        s3 = f"Signals are aligned to the upside with trade quality at {tq:.2f}. The model supports going long with defined risk."
    elif verdict == 'SELL':
        s3 = f"Downside signals are dominant with trade quality at {tq:.2f}. Short or defensive positioning is warranted."
    else:
        s3 = ""

    # Sentence 4: Earnings / fundamental context
    consensus_score = scores.get('consensus', 0)
    val_score = scores.get('valuation', 0)

    if consensus_score > 30:
        s4 = "Analyst estimate revisions are strongly positive — the street sees improving fundamentals, which could fuel the next move higher on an earnings catalyst."
    elif consensus_score > 10:
        s4 = "The street is modestly constructive on the outlook, with more analysts revising estimates up than down."
    elif consensus_score < -30:
        s4 = "Analysts are actively cutting estimates — fundamental deterioration may be underway. Watch for guidance downgrades on the next earnings print."
    elif consensus_score < -10:
        s4 = "Analyst momentum is slightly negative, with more estimate cuts than raises recently."
    else:
        # No strong consensus signal — comment on valuation instead
        if val_score < -15:
            s4 = "Valuation is stretched relative to history — the stock needs strong earnings execution to justify current multiples."
        elif val_score > 15:
            s4 = "Valuations look reasonable, providing a margin of safety if fundamentals hold."
        else:
            s4 = ""

    parts = [s1, s2, s3]
    if s4:
        parts.append(s4)
    return " ".join(parts)


# ============================================================================
# CONTEXT-AWARE DRIVER DESCRIPTIONS
# ============================================================================

def _expand_driver(engine, is_positive, s):
    """Generate a rich, context-aware driver description with actual data."""
    price = s.get('price', 0)
    sma20 = s.get('sma20', 0)
    sma50 = s.get('sma50', 0)
    sma200 = s.get('sma200', 0)
    vix = s.get('vix', 20)
    scores = s.get('scores', {})
    rv = s.get('regime_vector', {})

    if engine == 'trend':
        if is_positive:
            parts = []
            if sma20 > 0 and price > sma20:
                parts.append(f"20-day (${sma20:.2f})")
            if sma50 > 0 and price > sma50:
                parts.append(f"50-day (${sma50:.2f})")
            if sma200 > 0 and price > sma200:
                parts.append(f"200-day (${sma200:.2f})")
            if parts:
                return f"*Trend is constructive.* Price is holding above its {' and '.join(parts)}. Momentum is positive and the trend structure supports directional bets."
            return "*Trend is positive.* Price momentum is building with the stock holding above key support levels."
        else:
            parts = []
            if sma20 > 0 and price < sma20:
                parts.append(f"20-day (${sma20:.2f})")
            if sma50 > 0 and price < sma50:
                parts.append(f"50-day (${sma50:.2f})")
            below_str = f"below its {' and '.join(parts)}" if parts else "below key moving averages"
            return f"*Price action is a problem.* Trading {below_str}. Short-term trend structure is broken and momentum is negative."

    elif engine == 'valuation':
        if is_positive:
            return "*Valuations are attractive.* Current multiples sit below historical averages, providing a margin of safety. When valuation is cheap AND trend confirms, the probability of a sustained move is higher."
        else:
            return "*Valuations are stretched.* The stock is trading above its historical average multiples — you're paying a premium that needs to be backed by strong earnings delivery."

    elif engine == 'consensus':
        if is_positive:
            return "*Analyst momentum is positive.* Estimate revisions are trending higher over the past month, suggesting the street sees improving fundamentals. This often leads price — but needs price action confirmation to be tradable."
        else:
            return "*Analysts are getting cautious.* Estimate revisions are trending lower. When the street starts cutting numbers, it tends to be a leading indicator of further price weakness."

    elif engine == 'volatility':
        if is_positive:
            return f"*Volatility is favorable.* VIX at {vix:.1f} signals a calm options market. Low-vol environments historically favor directional strategies — trends tend to be cleaner and more persistent."
        else:
            return f"*Elevated volatility is a headwind.* VIX at {vix:.1f} means higher uncertainty and wider daily ranges. This favors smaller positions, wider stops, or sitting out entirely."

    elif engine == 'macro':
        if is_positive:
            return "*Macro backdrop is supportive.* Rates are stable and credit conditions are benign — the kind of environment where risk assets tend to grind higher without macro surprises disrupting the tape."
        else:
            return "*Macro headwinds present.* Rate movements or credit spread widening are creating pressure. This is particularly negative for high-multiple growth names and rate-sensitive sectors."

    elif engine == 'liquidity':
        if is_positive:
            return "*Market breadth is healthy.* A broad range of stocks are participating in the move, which makes it more sustainable. Good breadth means this isn't a narrow rally driven by a handful of names."
        else:
            return "*Breadth is thin.* Fewer stocks are participating, which is a classic warning sign. Narrow leadership makes moves fragile and more prone to sharp reversals."

    elif engine == 'global':
        if is_positive:
            return "*Global flows are supportive.* Overnight sessions in Asia and Europe were positive, suggesting healthy global risk appetite that typically carries into the US session."
        else:
            return "*Global markets flagged risk-off.* Negative overnight action from international markets signals a cautious tone that often spills into US trading."

    elif engine == 'correlation':
        if is_positive:
            return "*Cross-asset signals are stable.* Bonds and equities are behaving normally relative to each other. When correlations are predictable, all other model signals become more reliable."
        else:
            return "*Cross-asset correlation breakdown.* Unusual behavior between stocks and bonds (e.g., both falling together) signals systemic stress and makes every directional signal less trustworthy."

    return f"*{'Supportive' if is_positive else 'Negative'} signal* from underlying analysis."


def _get_top_drivers(s):
    """Get sorted drivers by contribution magnitude."""
    contributions = s.get('contributions', {})
    if not contributions:
        return [], []

    positive = sorted(
        [(k, v) for k, v in contributions.items() if v > 0.5],
        key=lambda x: x[1], reverse=True
    )[:2]

    negative = sorted(
        [(k, v) for k, v in contributions.items() if v < -0.5],
        key=lambda x: x[1]
    )[:2]

    return positive, negative


# ============================================================================
# TRIGGER GENERATION
# ============================================================================

def _generate_triggers(s, verdict):
    """Generate 3-5 specific, measurable triggers with context."""
    triggers = []
    rv = s.get('regime_vector', {})
    tq = s.get('trade_quality', 0)
    gate = s.get('gate_value', 0)
    c_raw = s.get('composite_raw', 0)
    price = s.get('price', 0)
    sma50 = s.get('sma50', 0)
    sma200 = s.get('sma200', 0)
    vix = s.get('vix', 20)
    rel = s.get('regime_reliability', 0)
    scores = s.get('scores', {})

    if verdict in ('CASH', 'WAIT'):
        # Price-based triggers
        if sma50 > 0 and price < sma50:
            triggers.append(f"Price reclaims the 50-day MA at ${sma50:.2f} with above-average volume — that would flip the short-term trend")
        if sma200 > 0 and price < sma200:
            triggers.append(f"Price holds above the 200-day MA at ${sma200:.2f} — a break below is a bigger structural concern")
        # Model-based triggers
        if tq < 0.20:
            triggers.append(f"Trade quality improves past 0.20 (currently {tq:.3f}) — right now there's simply no edge")
        if rel < 0.25:
            triggers.append(f"Market readability rises above 0.25 (currently {rel:.2f}) — the chop needs to subside before signals are trustworthy")
        if rv.get('CH', 0) > 0.60:
            triggers.append(f"Choppiness drops below 0.60 (currently {rv['CH']:.2f}) — the range-bound action is killing signal quality")
        if vix > 22:
            triggers.append(f"VIX drops below 22 (currently {vix:.1f}) — a calmer vol regime would improve the setup")
        if abs(c_raw) < 15:
            triggers.append(f"Composite signal strength rises above 15 in either direction (currently {c_raw:+.1f}) — a stronger lean would help")
        # Earnings trigger
        if scores.get('consensus', 0) > 20:
            triggers.append("Upcoming earnings could be the catalyst — positive estimate momentum suggests a beat is possible")
        elif scores.get('consensus', 0) < -20:
            triggers.append("Watch next earnings print closely — downward estimate revisions suggest risk of a miss")
    elif verdict == 'BUY':
        if sma50 > 0:
            triggers.append(f"Price breaks below the 50-day MA at ${sma50:.2f} — that would invalidate the current trend setup")
        triggers.append(f"VIX spikes above 25 (currently {vix:.1f}) — a vol shock would warrant reducing exposure")
        triggers.append(f"Trend score drops below -0.30 (currently {rv.get('TS', 0):+.2f}) — that means momentum has fully reversed")
        if rv.get('CS', 0) < 0.50:
            triggers.append(f"Credit spreads widen sharply (stress currently {rv['CS']:.2f}) — a credit event would change everything")
        triggers.append(f"Risk gate closes below 0.40 (currently {gate:.2f}) — if systemic risk rises, the model will pull the position")
    elif verdict == 'SELL':
        if sma50 > 0:
            triggers.append(f"Price reclaims the 50-day MA at ${sma50:.2f} — would flip the short-term trend to bullish")
        triggers.append(f"Trend score rises above +0.30 (currently {rv.get('TS', 0):+.2f}) — momentum reversal would invalidate the short")
        triggers.append(f"VIX declines below 18 (currently {vix:.1f}) — calming vol reduces the urgency of defensive positioning")
        triggers.append(f"Risk gate closes below 0.40 (currently {gate:.2f}) — paradoxically, too much risk can invalidate shorts too")

    return triggers[:5]


# ============================================================================
# MESSAGE 1: VERDICT + THE BOTTOM LINE
# ============================================================================

def format_trader_snapshot(s):
    """Compact verdict card + narrative summary."""
    verdict = _trader_verdict(s)
    tq = s.get('trade_quality', 0)
    rel = s.get('regime_reliability', 0)
    gate = s.get('gate_value', 0)
    c_raw = s.get('composite_raw', 0)
    price = s.get('price', 0)
    symbol = s.get('symbol', '???')
    pos_size = s.get('position_size', 0)
    pos_pct = s.get('position_pct', 0)

    emoji = _verdict_emoji(verdict)
    risk_env = _risk_env(s.get('regime_label', 'Unknown'))
    readability = _readability_label(rel)
    conv = _conviction(tq)

    lines = [
        f"{emoji} *ATLAS — {symbol}* | ${price:.2f}",
        "",
    ]

    # Verdict card
    if verdict in ('CASH', 'WAIT'):
        lines.append(f"*Verdict: {verdict}* | Bias: {_bias(c_raw)} | Risk: {risk_env}")
        lines.append(f"Tradable: NO | Readability: {readability} | Conviction: {conv}")
        lines.append("Position: $0")
    else:
        lines.append(f"*Verdict: {verdict}* | Risk: {risk_env}")
        lines.append(f"Tradable: YES | Readability: {readability} | Conviction: {conv}")
        lines.append(f"Position: ${pos_size:,.0f} ({pos_pct:.1f}%)")

    lines.append("")
    lines.append("*THE BOTTOM LINE*")
    lines.append(_build_narrative(s))

    return '\n'.join(lines)


# ============================================================================
# MESSAGE 2: WHAT'S DRIVING THIS
# ============================================================================

def format_driver_analysis(s):
    """Detailed, context-aware explanation of the key forces."""
    positive_drivers, negative_drivers = _get_top_drivers(s)

    lines = ["*WHAT'S DRIVING THIS*", ""]

    if positive_drivers:
        for engine, contrib in positive_drivers:
            desc = _expand_driver(engine, True, s)
            lines.append(f":small_blue_diamond: {desc}")
            lines.append("")

    if negative_drivers:
        for engine, contrib in negative_drivers:
            desc = _expand_driver(engine, False, s)
            lines.append(f":small_orange_diamond: {desc}")
            lines.append("")

    if not positive_drivers and not negative_drivers:
        lines.append("No dominant driver in either direction. All inputs are generating weak, offsetting signals — which is exactly why the model says no trade.")
        lines.append("")

    return '\n'.join(lines)


# ============================================================================
# MESSAGE 3: LEVELS + RISK + TRIGGERS
# ============================================================================

def format_levels_and_triggers(s):
    """Levels, risk check, and what would change the call."""
    verdict = _trader_verdict(s)
    tq = s.get('trade_quality', 0)
    rel = s.get('regime_reliability', 0)
    gate = s.get('gate_value', 0)
    price = s.get('price', 0)
    rv = s.get('regime_vector', {})
    run_count = s.get('run_count', 0)

    lines = []

    # ── LEVELS ──
    if verdict in ('BUY', 'SELL'):
        atr_val = s.get('atr', 1.0)
        stop = s.get('stop_loss', 0)
        tp_lo, tp_hi = s.get('take_profit', (0, 0))

        if verdict == 'BUY':
            entry_lo = round(price - atr_val * 0.75, 2)
            entry_hi = round(price, 2)
        else:
            entry_lo = round(price, 2)
            entry_hi = round(price + atr_val * 0.75, 2)

        entry_mid = (entry_lo + entry_hi) / 2
        tp_mid = (tp_lo + tp_hi) / 2
        risk_amt = abs(entry_mid - stop)
        reward_amt = abs(tp_mid - entry_mid)
        rr = reward_amt / risk_amt if risk_amt > 0 else 0

        lines.append("*TRADE LEVELS*")
        lines.append(f"Entry Zone: ${entry_lo:.2f} - ${entry_hi:.2f}")
        lines.append(f"Stop Loss: ${stop:.2f}")
        lines.append(f"Target 1: ${tp_lo:.2f} | Stretch: ${tp_hi:.2f}")
        if rr > 0:
            lines.append(f"Risk/Reward: {rr:.1f}:1")
        lines.append("")

    elif verdict == 'WAIT':
        sma50 = s.get('sma50', 0)
        sma200 = s.get('sma200', 0)
        stop = s.get('stop_loss', 0)

        lines.append("*WATCH LEVELS* _(observation only — not trade entries)_")
        if sma50 > 0:
            if price < sma50:
                lines.append(f"50-day MA at ${sma50:.2f} — price needs to reclaim this to flip the short-term trend")
            else:
                lines.append(f"50-day MA at ${sma50:.2f} — support to hold for the bullish case to stay intact")
        if sma200 > 0:
            lines.append(f"200-day MA at ${sma200:.2f} — the structural line in the sand")
        if stop > 0:
            lines.append(f"Support floor at ${stop:.2f}")
        lines.append("")

    else:
        # CASH: Key levels to watch
        sma50 = s.get('sma50', 0)
        sma200 = s.get('sma200', 0)
        atr_val = s.get('atr', 1.0)

        lines.append("*KEY LEVELS TO WATCH*")
        if sma50 > 0:
            if price < sma50:
                lines.append(f"50-day MA: ${sma50:.2f} — reclaiming this would be the first sign of trend repair")
            else:
                lines.append(f"50-day MA: ${sma50:.2f} — holding above keeps the structure intact")
        if sma200 > 0:
            if price < sma200:
                lines.append(f"200-day MA: ${sma200:.2f} — below this level, the bigger picture is bearish")
            else:
                lines.append(f"200-day MA: ${sma200:.2f} — long-term support, a break below would escalate concerns")
        if atr_val > 0:
            lines.append(f"Daily range: ~${atr_val:.2f} (14-day ATR) — gives you a sense of normal daily movement")
        lines.append("")

    # ── RISK CHECK ──
    conv = _conviction(tq)
    lines.append("*RISK CHECK*")
    lines.append(f"Risk Gate: {gate:.2f} | Readability: {rel:.2f} | TQ: {tq:.3f} | Conviction: {conv}")

    # Warnings in human language
    saturated = [k for k, v in rv.items() if v >= 0.95 or v <= -0.95]
    if saturated:
        lines.append(f"_Note: Some regime inputs are maxed out ({', '.join(saturated)}) — this usually means the model is operating in an unusual environment and signals should be taken with extra caution._")
    if run_count < 10:
        lines.append(f"_Model is still calibrating (run {run_count} of 10). Adaptive weights will begin adjusting after 10 runs to learn which signals work best in the current environment._")
    lines.append("")

    # ── WHAT WOULD CHANGE MY MIND ──
    lines.append("*WHAT WOULD CHANGE MY MIND*")
    triggers = _generate_triggers(s, verdict)
    for i, t in enumerate(triggers, 1):
        lines.append(f"{i}. {t}")

    return '\n'.join(lines)


# ============================================================================
# MESSAGE 4: MODEL INTERNALS
# ============================================================================

def format_model_internals(s):
    """Compact model internals in code block. Numbers only, no formulas."""
    c_raw = s.get('composite_raw', 0)
    c_adj = s.get('composite_adjusted', 0)
    dc = s.get('data_confidence', 0)
    run_count = s.get('run_count', 0)
    rel = s.get('regime_reliability', 0)
    regime = s.get('regime_label', 'Unknown')

    learning = "Cold Start" if run_count < 10 else "Learning" if run_count < 50 else "Converged"

    lines = [
        "--- MODEL INTERNALS (ADVANCED) ---",
        f"Composite: Raw {c_raw:+.1f} -> Adjusted {c_adj:+.1f}",
        f"DC: {dc:.0f}% | Learning: {learning} ({run_count})",
        "",
        f"{'Engine':<14} {'Score':>6} {'Weight':>7} {'Contrib':>8}",
        "-" * 38,
    ]

    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']
    scores = s.get('scores', {})
    w_dyn = s.get('w_dynamic', {})
    contributions = s.get('contributions', {})

    contrib_sum = 0.0
    for engine in engines:
        sc = scores.get(engine, 0)
        w = w_dyn.get(engine, 1/8)
        c = contributions.get(engine, 0)
        contrib_sum += c
        lines.append(f"{engine:<14} {float(sc):>+6.1f} {float(w):>7.3f} {float(c):>+8.2f}")

    lines.append(f"{'':14} {'':>6} {'':>7} {'------':>8}")
    lines.append(f"{'Total':14} {'':>6} {'':>7} {contrib_sum:>+8.2f}")

    lines.append("")
    lines.append(f"Regime: {regime} | Reliability: {rel:.2f}")

    rv = s.get('regime_vector', {})
    feature_order = ['TS', 'CH', 'VL', 'VS', 'CI', 'RS', 'CS', 'GR', 'BM_f', 'BEI']
    rv_line1 = "  ".join([f"{f}={float(rv.get(f, 0)):+.2f}" if f == 'TS' else f"{f}={float(rv.get(f, 0)):.2f}" for f in feature_order[:5]])
    rv_line2 = "  ".join([f"{f}={float(rv.get(f, 0)):.2f}" for f in feature_order[5:]])
    lines.append(rv_line1)
    lines.append(rv_line2)

    return "```\n" + '\n'.join(lines) + "\n```"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def format_full_trader_report(summary):
    """
    Generate complete trader report as list of Slack messages.
    4 messages: Snapshot, Drivers, Levels+Risk+Triggers, Internals
    """
    messages = []
    messages.append(format_trader_snapshot(summary))
    messages.append(format_driver_analysis(summary))
    messages.append(format_levels_and_triggers(summary))
    messages.append(format_model_internals(summary))
    return messages


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def split_report_for_slack(report_text):
    """Legacy: Split raw pyramid report for Slack (kept for CLI mode)."""
    lines = report_text.split('\n')
    messages = []
    current_chunk = []
    current_size = 0

    def is_boundary(line):
        stripped = line.strip()
        return stripped.startswith('==') or stripped.startswith('--')

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > 3500 and current_chunk:
            split_idx = None
            for i in range(len(current_chunk) - 1, -1, -1):
                if is_boundary(current_chunk[i]):
                    split_idx = i
                    break
            if split_idx and split_idx > 0:
                messages.append('\n'.join(current_chunk[:split_idx]))
                current_chunk = current_chunk[split_idx:]
                current_size = sum(len(l) + 1 for l in current_chunk)
            else:
                messages.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        messages.append('\n'.join(current_chunk))

    return [f"```\n{msg}\n```" for msg in messages]


def format_summary_message(summary):
    """Legacy: backward compatible."""
    return format_trader_snapshot(summary)
