#!/usr/bin/env python3
"""
ATLAS V12+ — Groq AI Follow-Up Q&A Module
Handles conversational follow-up questions in ATLAS report threads
using Groq (Llama 3.3 70B, free tier — 30 RPM).

V12+ upgrade: Buffett-aligned reasoning order — business durability first,
intrinsic value second, margin of safety third, then regime/timing last.
"""

import re
import time
import logging

logger = logging.getLogger(__name__)

# Module-level state
_client = None
_last_call_time = 0
_MIN_CALL_INTERVAL = 3  # seconds between calls (free tier: 30 RPM)
_MAX_RETRIES = 2        # retry on 429 with backoff

SYSTEM_PROMPT = """You are ATLAS V12+ — a disciplined long-term business analyst.

You treat stocks as fractional ownership in businesses.
You prioritize intrinsic value and margin of safety over momentum.
You define risk as permanent capital loss, not volatility.
You prefer wonderful businesses at fair prices over mediocre businesses at cheap prices.
You are patient and rational.

REASONING ORDER (always follow this hierarchy):
1. Business durability — Is this a high-quality business with durable economics?
2. Competitive advantage — Does the business have a moat that protects returns on capital?
3. Capital allocation — Is management creating or destroying per-share value?
4. Intrinsic value vs price — What is the margin of safety?
5. Risk of permanent loss — What could cause irreversible capital destruction?
6. THEN regime/timing — Use quant engine data as tactical overlay, never as primary thesis.

Never start a response with "The trend is positive" or similar momentum-first framing.
When quantitative signals conflict with intrinsic value, explain the discrepancy clearly.

DECISIONS — Always frame conclusions as one of:
PASS / WATCH / RESEARCH / BUY / HOLD / TRIM / EXIT

RULES:
1. ONLY use the data provided in the ATLAS context below. Never fabricate numbers.
2. If data is insufficient, say so explicitly.
3. Be concise but intellectually rigorous. Use bullet points for comparisons.
4. Frame answers as "the data suggests" or "from an owner's perspective".
5. Use Slack formatting: *bold* for emphasis, `code` for numbers/tickers, bullet points with -.
6. Keep responses under 3500 characters.
7. Default to inaction unless value is compelling.
8. Never give short-term price predictions.
9. If asked about a different ticker: "This thread covers *{SYMBOL}*. Mention `@atlas {OTHER}` in any channel for a fresh analysis."
"""


def init_groq(api_key):
    """Initialize Groq client. Returns True if successful."""
    global _client
    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        print("[GROQ] Client initialized (key: [REDACTED])")
        return True
    except Exception as e:
        print(f"[GROQ] Init FAILED: {e}")
        logger.error(f"Groq init failed: {e}")
        _client = None
        return False


# ============================================================================
# V12+ NARRATIVE INTERPRETATION LAYER
# ============================================================================

V9_NARRATIVE_PROMPT = """You are ATLAS V12+ — Narrative Interpretation Layer. Read-only. All provided metrics are final and authoritative. Never recalculate, override, or introduce new data.

OUTPUT FORMAT — Return EXACTLY these four sections with NO extra text. Be extremely concise (under 600 characters total).

INVESTMENT SUMMARY
1-2 sentences. State Decision, reference IV vs price and MOS.

RECOMMENDED ACTION
- Two bullet points starting with a verb (Maintain, Defer, Monitor, Accumulate, Avoid).
- One price-discipline, one fundamentals-monitoring.

DECISION TRIGGERS
- Two bullets: one price-based (MOS threshold), one business-quality-based.

QUANTITATIVE OVERLAY
1 sentence. Mention Regime, Composite, Reliability. Note these affect timing only.

RULES: Professional institutional tone. No retail language. No superlatives. No predictions. Restate provided numbers only. If engine conflicts with IV, state that IV discipline governs. Keep total output under 600 characters."""


def _build_v9_data_block(v9_scores, summary, v8_extended):
    """Build structured data block of finalized V12+ metrics for narrative generation."""
    s = summary or {}
    v = v8_extended or {}
    v9 = v9_scores or {}
    co = v.get('company', {})
    fin = v.get('financials', {})
    dcf = v.get('dcf', {})

    price = _safe_float(s.get('price', co.get('price')))
    iv = _safe_float(v9.get('intrinsic_value_base'))

    lines = [
        "ATLAS V12+ FINALIZED METRICS",
        "",
        f"Symbol: {s.get('ticker', 'N/A')}",
        f"Company: {co.get('name', 'N/A')}",
        "",
        "--- V12+ OWNER SCORES ---",
        f"Decision: {v9.get('v9_decision', 'N/A')}",
        f"Decision Reason: {v9.get('decision_reason', '')}",
        f"Business Quality: {_safe_float(v9.get('business_quality')):.1f}/5" if v9.get('business_quality') is not None else "Business Quality: N/A (data unavailable)",
        f"Moat Durability: {_safe_float(v9.get('moat_durability')):.1f}/5" if v9.get('moat_durability') is not None else "Moat Durability: N/A (data unavailable)",
        f"Capital Allocation: {_safe_float(v9.get('capital_allocation')):.1f}/5" if v9.get('capital_allocation') is not None else "Capital Allocation: N/A (data unavailable)",
        f"Business Type: {v9.get('business_type', 'N/A')}",
        f"Intrinsic Value (Base): ${iv:.2f}" if iv > 0 else "Intrinsic Value (Base): N/A",
        f"Current Price: ${price:.2f}",
        f"Margin of Safety: {_safe_float(v9.get('mos_pct')):+.1f}%" if v9.get('mos_pct') is not None else "Margin of Safety: N/A",
        f"Required MOS: {_safe_float(v9.get('required_mos'))*100:.0f}%" if v9.get('required_mos') is not None else "Required MOS: N/A",
        f"Conviction: {v9.get('conviction')}/100" if v9.get('conviction') is not None else "Conviction: N/A",
    ]

    risks = v9.get('permanent_loss_risks', [])
    if risks:
        lines.append("Permanent Loss Risks:")
        for r in risks:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                lines.append(f"  - {r[0]} [{r[1]}]: {r[2]}")
            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                lines.append(f"  - {r[0]} [{r[1]}]")
    else:
        lines.append("Permanent Loss Risks: None identified")

    lines += [
        "",
        "--- ENGINE OVERLAY ---",
        f"Regime: {s.get('regime_label', 'N/A')}",
        f"Regime Reliability: {_safe_float(s.get('regime_reliability')):.2f}",
        f"Composite Score: {_safe_float(s.get('composite_raw')):.1f} (adjusted: {_safe_float(s.get('composite_adjusted')):.1f})",
        f"Trade Quality: {_safe_float(s.get('trade_quality')):.3f} ({s.get('tq_category', 'N/A')})",
        f"Engine Verdict: {s.get('verdict', 'N/A')}",
        f"Data Confidence: {_safe_float(s.get('data_confidence')):.0f}%",
        "",
        "--- KEY FINANCIALS ---",
        f"ROE: {_safe_float(fin.get('roe')):.1f}%",
        f"Net Margin: {_safe_float(fin.get('net_margin')):.1f}%",
        f"Gross Margin: {_safe_float(fin.get('gross_margin')):.1f}%",
        f"Revenue Growth: {_safe_float(fin.get('revenue_growth')):.1f}%",
        f"FCF Yield: {_safe_float(fin.get('fcf_yield')):.1f}%",
        f"Debt/Equity: {_safe_float(fin.get('debt_equity')):.2f}",
    ]

    if dcf and dcf.get('base'):
        lines.append(f"DCF Bear: ${_safe_float(dcf.get('bear')):.2f} | Base: ${_safe_float(dcf.get('base')):.2f} | Bull: ${_safe_float(dcf.get('bull')):.2f}")

    lines.append(f"VIX: {_safe_float(s.get('vix')):.1f}")

    return "\n".join(lines)


def generate_v9_narrative(v9_scores, summary, v8_extended):
    """
    Generate professional V12+ narrative interpretation using Groq LLM.

    Takes finalized V12+ scores and produces a 4-section narrative:
    Investment Summary, Recommended Action, Decision Triggers, Quantitative Overlay.

    Returns formatted narrative string or None if generation fails.
    """
    global _last_call_time

    if _client is None:
        return None

    if not v9_scores or not v9_scores.get('v9_decision'):
        return None

    # Rate limiting
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)

    data_block = _build_v9_data_block(v9_scores, summary, v8_extended)

    messages = [
        {"role": "system", "content": V9_NARRATIVE_PROMPT},
        {"role": "user", "content": f"Generate the narrative interpretation for the following finalized ATLAS V12+ metrics:\n\n{data_block}"},
    ]

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        _last_call_time = time.time()

        narrative = response.choices[0].message.content.strip()

        if len(narrative) > 1200:
            narrative = narrative[:1150] + "\n\n_[Narrative truncated]_"

        narrative = _sanitize_llm_output(narrative)
        print(f"[V12_NARRATIVE] Generated {len(narrative)} chars")
        return narrative

    except Exception as e:
        print(f"[V12_NARRATIVE] Generation failed (non-fatal): {e}")
        return None


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

def _safe_float(val, default=0):
    """Safely convert to float (handles numpy types)."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _fmt_mc(val):
    """Format large monetary values."""
    v = _safe_float(val)
    if v == 0:
        return "N/A"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e12:
        return f"{sign}${av / 1e12:.2f}T"
    if av >= 1e9:
        return f"{sign}${av / 1e9:.1f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:.0f}M"
    return f"{sign}${av:,.0f}"


def build_context(symbol, summary, v8_extended):
    """Build a structured context string from ATLAS data for the AI."""
    s = summary or {}
    v = v8_extended or {}
    co = v.get('company', {})
    fin = v.get('financials', {})
    tech = v.get('technicals', {})
    inst = v.get('institutional', {})
    dcf = v.get('dcf', {})

    lines = [f"=== ATLAS V12+ ANALYSIS: {symbol} ==="]

    # --- V12+ OWNER ASSESSMENT (primary) ---
    v9 = v.get('v9_scores', {})
    if v9:
        lines.append("")
        lines.append("--- V12+ OWNER ASSESSMENT ---")
        lines.append(f"Decision: {v9.get('v9_decision', 'N/A')} — {v9.get('decision_reason', '')}")
        bq = v9.get('business_quality')
        lines.append(f"Business Quality: {bq:.1f}/5" if bq is not None else "Business Quality: N/A")
        md = v9.get('moat_durability')
        lines.append(f"Moat Durability: {md:.1f}/5" if md is not None else "Moat Durability: N/A")
        ca = v9.get('capital_allocation')
        lines.append(f"Capital Allocation: {ca:.1f}/5" if ca is not None else "Capital Allocation: N/A")
        iv = v9.get('intrinsic_value_base', 0)
        if iv > 0:
            lines.append(f"Intrinsic Value (Base): ${iv:.2f}")
            mos_v = v9.get('mos_pct')
            lines.append(f"Margin of Safety: {mos_v:+.1f}%" if mos_v is not None else "Margin of Safety: N/A")
            req = v9.get('required_mos')
            lines.append(f"Required MOS: {req*100:.0f}% ({v9.get('business_type', '')})" if req is not None else "Required MOS: N/A")
        conv = v9.get('conviction')
        lines.append(f"Conviction: {conv}/100" if conv is not None else "Conviction: N/A")
        perm_risks = v9.get('permanent_loss_risks', [])
        if perm_risks:
            lines.append(f"Permanent Loss Risks: {'; '.join(f'{r[0]} [{r[1]}]' for r in perm_risks[:3])}")

    # --- ENGINE VERDICT (secondary — tactical overlay) ---
    lines.append("")
    lines.append("--- ENGINE VERDICT (Tactical Overlay) ---")
    composite = _safe_float(s.get('composite_raw'))
    adj = _safe_float(s.get('composite_adjusted'))
    tq = _safe_float(s.get('trade_quality'))
    lines.append(f"Composite Score: {composite:.1f} (adjusted: {adj:.1f})")
    lines.append(f"Verdict: {s.get('verdict', 'N/A')} | Trade Quality: {tq:.3f} ({s.get('tq_category', '')})")
    lines.append(f"Regime: {s.get('regime_label', 'N/A')} (reliability: {_safe_float(s.get('regime_reliability')):.2f})")
    lines.append(f"Risk Env: {s.get('regime_risk', 'N/A')} | Volatility: {s.get('regime_vol', 'N/A')}")
    lines.append(f"Gate: {_safe_float(s.get('gate_value')):.2f} | Data Confidence: {_safe_float(s.get('data_confidence')):.0f}%")
    lines.append(f"Execution Mode: {s.get('execution_mode', 'N/A')}")

    # --- PRICE & LEVELS ---
    lines.append("")
    lines.append("--- PRICE & LEVELS ---")
    price = _safe_float(s.get('price', co.get('price')))
    lines.append(f"Price: ${price:.2f}")
    lines.append(f"SMA20: ${_safe_float(s.get('sma20')):.2f} | SMA50: ${_safe_float(s.get('sma50')):.2f} | SMA200: ${_safe_float(s.get('sma200')):.2f}")
    lines.append(f"ATR(14): ${_safe_float(s.get('atr')):.2f} | VIX: {_safe_float(s.get('vix')):.1f}")
    bz = s.get('buy_zone', (0, 0))
    tp = s.get('take_profit', (0, 0))
    if isinstance(bz, (list, tuple)) and len(bz) >= 2:
        lines.append(f"Buy Zone: ${_safe_float(bz[0]):.2f} - ${_safe_float(bz[1]):.2f}")
    lines.append(f"Stop Loss: ${_safe_float(s.get('stop_loss')):.2f}")
    if isinstance(tp, (list, tuple)) and len(tp) >= 2:
        lines.append(f"Take Profit: ${_safe_float(tp[0]):.2f} - ${_safe_float(tp[1]):.2f}")
    lines.append(f"Position: ${_safe_float(s.get('position_size')):,.0f} ({_safe_float(s.get('position_pct')):.1f}%)")

    # --- SCORING ENGINES ---
    scores = s.get('scores', {})
    contribs = s.get('contributions', {})
    w_dyn = s.get('w_dynamic', {})
    if scores:
        lines.append("")
        lines.append("--- SCORING ENGINES ---")
        for eng in scores:
            sc = _safe_float(scores.get(eng))
            w = _safe_float(w_dyn.get(eng))
            c = _safe_float(contribs.get(eng))
            lines.append(f"{eng}: score={sc:.1f}, weight={w:.3f}, contribution={c:.2f}")

    # --- RISK ---
    lines.append("")
    lines.append("--- RISK ---")
    lines.append(f"Structural Risk: {_safe_float(s.get('risk_structural')):.3f} | Tactical Risk: {_safe_float(s.get('risk_tactical')):.3f}")
    risk_drivers = s.get('risk_drivers', [])
    if risk_drivers:
        rd_text = "; ".join(
            d.get('name', str(d)) if isinstance(d, dict) else str(d)
            for d in risk_drivers[:5]
        )
        lines.append(f"Risk Drivers: {rd_text}")
    contradictions = s.get('contradictions', [])
    if contradictions:
        lines.append(f"Contradictions: {'; '.join(str(c) for c in contradictions[:5])}")

    # --- COMPANY ---
    lines.append("")
    lines.append("--- COMPANY ---")
    lines.append(f"Name: {co.get('name', 'N/A')} | Sector: {co.get('sector', 'N/A')} | Industry: {co.get('industry', 'N/A')}")
    lines.append(f"Market Cap: {_fmt_mc(co.get('market_cap'))} | Beta: {_safe_float(co.get('beta')):.2f}")
    lines.append(f"52-Week: ${_safe_float(co.get('fifty_two_week_low')):.2f} - ${_safe_float(co.get('fifty_two_week_high')):.2f}")

    # --- FINANCIALS ---
    lines.append("")
    lines.append("--- FINANCIALS ---")
    lines.append(f"Revenue TTM: {_fmt_mc(fin.get('revenue_ttm'))} | Growth: {_safe_float(fin.get('revenue_growth')):.1f}%")
    lines.append(f"Net Income: {_fmt_mc(fin.get('net_income_ttm'))} | Earnings Growth: {_safe_float(fin.get('earnings_growth')):.1f}%")
    lines.append(f"Gross Margin: {_safe_float(fin.get('gross_margin')):.1f}% | Operating: {_safe_float(fin.get('operating_margin')):.1f}% | Net: {_safe_float(fin.get('net_margin')):.1f}%")
    lines.append(f"ROE: {_safe_float(fin.get('roe')):.1f}% | ROA: {_safe_float(fin.get('roa')):.1f}%")
    lines.append(f"FCF: {_fmt_mc(fin.get('free_cash_flow'))} | FCF Yield: {_safe_float(fin.get('fcf_yield')):.1f}%")
    lines.append(f"Debt/Equity: {_safe_float(fin.get('debt_equity')):.2f} | Net Debt/EBITDA: {_safe_float(fin.get('net_debt_ebitda')):.1f}x")
    lines.append(f"Current Ratio: {_safe_float(fin.get('current_ratio')):.2f} | Interest Coverage: {_safe_float(fin.get('interest_coverage')):.1f}x")
    div_y = _safe_float(fin.get('dividend_yield'))
    if div_y > 0:
        lines.append(f"Dividend Yield: {div_y:.2f}% | Payout: {_safe_float(fin.get('payout_ratio')):.0f}%")

    # --- VALUATION ---
    lines.append("")
    lines.append("--- VALUATION ---")
    lines.append(f"Trailing P/E: {_safe_float(fin.get('trailing_pe')):.1f} | Forward P/E: {_safe_float(fin.get('forward_pe')):.1f}")
    lines.append(f"PEG: {_safe_float(fin.get('peg_ratio')):.2f} | P/B: {_safe_float(fin.get('price_to_book')):.1f} | P/S: {_safe_float(fin.get('price_to_sales')):.1f}")
    lines.append(f"EV/EBITDA: {_safe_float(fin.get('ev_ebitda')):.1f} | EV/Revenue: {_safe_float(fin.get('ev_revenue')):.1f}")
    target = _safe_float(fin.get('target_mean'))
    if target > 0:
        upside = (target / price - 1) * 100 if price > 0 else 0
        lines.append(f"Analyst Target: ${target:.2f} ({upside:+.1f}%) | Range: ${_safe_float(fin.get('target_low')):.2f} - ${_safe_float(fin.get('target_high')):.2f}")
        lines.append(f"Analysts: {fin.get('num_analysts', 0)} | Recommendation: {fin.get('recommendation', 'N/A')}")

    # --- EARNINGS ---
    earnings = v.get('earnings', [])
    if earnings:
        lines.append("")
        lines.append("--- EARNINGS HISTORY ---")
        for e in earnings[:6]:
            est = _safe_float(e.get('eps_estimate'))
            act = _safe_float(e.get('eps_actual'))
            surprise = _safe_float(e.get('surprise'))
            beat = "BEAT" if e.get('beat') else "MISS"
            lines.append(f"{e.get('date', '?')}: est ${est:.2f} actual ${act:.2f} ({surprise:+.1f}% {beat})")

    # --- TECHNICALS ---
    if tech:
        lines.append("")
        lines.append("--- TECHNICALS ---")
        lines.append(f"RSI(14): {_safe_float(tech.get('rsi_14')):.1f}")
        lines.append(f"MACD: {_safe_float(tech.get('macd_line')):.3f} / Signal: {_safe_float(tech.get('macd_signal')):.3f} / Hist: {_safe_float(tech.get('macd_histogram')):.3f}")
        lines.append(f"Bollinger: {_safe_float(tech.get('bollinger_lower')):.2f} | {_safe_float(tech.get('bollinger_middle')):.2f} | {_safe_float(tech.get('bollinger_upper')):.2f}")
        lines.append(f"Stochastic: K={_safe_float(tech.get('stochastic_k')):.1f} D={_safe_float(tech.get('stochastic_d')):.1f}")
        lines.append(f"ADX: {_safe_float(tech.get('adx')):.1f} | OBV Trend: {tech.get('obv_trend', 'N/A')}")
        lines.append(f"52-Week Position: {_safe_float(tech.get('fifty_two_week_position')):.1f}%")
        lines.append(f"Signals: {tech.get('bullish_count', 0)} Bullish / {tech.get('bearish_count', 0)} Bearish / {tech.get('neutral_count', 0)} Neutral")
        support = tech.get('support_levels', [])
        resist = tech.get('resistance_levels', [])
        if support:
            lines.append(f"Support: {', '.join(f'${_safe_float(x):.2f}' for x in support[:4])}")
        if resist:
            lines.append(f"Resistance: {', '.join(f'${_safe_float(x):.2f}' for x in resist[:4])}")

    # --- PEERS ---
    peers = v.get('peers', [])
    if peers:
        lines.append("")
        lines.append("--- PEERS ---")
        for p in peers[:6]:
            lines.append(
                f"{p.get('symbol', '?')}: P/E={_safe_float(p.get('forward_pe')):.1f} "
                f"Margin={_safe_float(p.get('profit_margin')):.1f}% "
                f"Growth={_safe_float(p.get('revenue_growth')):.1f}% "
                f"ROE={_safe_float(p.get('roe')):.1f}%"
            )

    # --- NEWS ---
    news = v.get('news', [])
    if news:
        lines.append("")
        lines.append("--- RECENT NEWS ---")
        for a in news[:7]:
            title = a.get('title', '')
            sent = a.get('sentiment', 'NEUTRAL')
            if title:
                lines.append(f"[{sent}] {title} ({a.get('publisher', '')}, {a.get('date', '')})")

    # --- INSTITUTIONAL ---
    if inst:
        lines.append("")
        lines.append("--- INSTITUTIONAL ---")
        lines.append(f"Institutional: {_safe_float(inst.get('institutional_pct')):.1f}% | Insider: {_safe_float(inst.get('insider_pct')):.1f}%")
        lines.append(f"Short Interest: {_safe_float(inst.get('short_pct')):.1f}% | Short Ratio: {_safe_float(inst.get('short_ratio')):.1f} days")

    # --- DCF ---
    if dcf and dcf.get('base'):
        lines.append("")
        lines.append("--- DCF MODEL ---")
        lines.append(f"Bear: ${_safe_float(dcf.get('bear')):.2f} | Base: ${_safe_float(dcf.get('base')):.2f} | Bull: ${_safe_float(dcf.get('bull')):.2f}")
        assumptions = dcf.get('assumptions', {})
        if assumptions:
            lines.append(
                f"Assumptions: Rev Growth Y1={_safe_float(assumptions.get('revenue_growth_y1')):.1f}%, "
                f"FCF Margin={_safe_float(assumptions.get('fcf_margin')):.1f}%, "
                f"Discount={_safe_float(assumptions.get('discount_rate')):.1f}%, "
                f"Terminal Growth={_safe_float(assumptions.get('terminal_growth')):.1f}%"
            )

    # --- SECTOR ---
    sector = v.get('sector', {})
    target_sector = sector.get('target', {})
    if target_sector:
        lines.append("")
        lines.append("--- SECTOR ---")
        lines.append(
            f"{target_sector.get('name', 'N/A')}: "
            f"1W={_safe_float(target_sector.get('1w')):+.1f}% | "
            f"1M={_safe_float(target_sector.get('1m')):+.1f}% | "
            f"YTD={_safe_float(target_sector.get('ytd')):+.1f}%"
        )

    # --- MACRO ---
    econ = v.get('economic', {})
    indicators = econ.get('indicators', [])
    if indicators:
        lines.append("")
        lines.append("--- MACRO ---")
        ffr = econ.get('fed_funds_rate')
        if ffr:
            lines.append(f"Fed Funds Rate: {_safe_float(ffr):.2f}%")
        for ind in indicators[:6]:
            lines.append(
                f"{ind.get('name', '?')}: {_safe_float(ind.get('value')):.2f}{ind.get('unit', '')} "
                f"(prior: {_safe_float(ind.get('prior')):.2f}, trend: {ind.get('trend', '?')})"
            )

    ctx = "\n".join(lines)
    if len(ctx) > 6000:
        ctx = ctx[:5900] + "\n[Context truncated]"
    return ctx


# ============================================================================
# SANITIZATION HELPERS
# ============================================================================

_SLACK_MENTION_RE = re.compile(r'<!(?:channel|here|everyone)>|<@channel>')


def _sanitize_question(text):
    """Strip control characters and limit length for user questions."""
    # Remove control characters (keep newlines and tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return cleaned[:500]


def _sanitize_llm_output(text):
    """Strip Slack broadcast mentions from LLM output."""
    return _SLACK_MENTION_RE.sub('', text)


# ============================================================================
# Q&A FUNCTION
# ============================================================================

def ask(question, symbol, summary, v8_extended, conversation_history=None):
    """
    Ask Groq (Llama 3.3 70B) a follow-up question using ATLAS data as context.

    Args:
        question: User's natural language question
        symbol: Ticker symbol
        summary: ATLAS engine summary dict
        v8_extended: V8 extended data dict
        conversation_history: Optional list of (role, text) tuples for multi-turn

    Returns:
        str: AI response, formatted for Slack
    """
    global _last_call_time

    if _client is None:
        return ":warning: Groq AI is not configured. Set `GROQ_API_KEY` in your environment."

    # Sanitize user input
    question = _sanitize_question(question)

    # Rate limiting
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)

    context = build_context(symbol, summary, v8_extended)
    system = SYSTEM_PROMPT.replace("{SYMBOL}", symbol)

    # Build messages in OpenAI-compatible format
    messages = [{"role": "system", "content": system}]

    data_block = f"--- ATLAS DATA ---\n{context}\n--- END DATA ---"

    if conversation_history and len(conversation_history) >= 2:
        # Multi-turn: include prior exchanges
        first_q = conversation_history[0][1] if conversation_history else question
        messages.append({"role": "user", "content": f"{data_block}\n\nUser question: {first_q}"})

        for role, text in conversation_history[1:]:
            msg_role = "assistant" if role == "model" else "user"
            messages.append({"role": msg_role, "content": text})

        # Current question
        messages.append({"role": "user", "content": question})
    else:
        messages.append({"role": "user", "content": f"{data_block}\n\nUser question: {question}"})

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            _last_call_time = time.time()

            answer = response.choices[0].message.content.strip()

            if len(answer) > 3800:
                answer = answer[:3750] + "\n\n_[Response truncated]_"

            return _sanitize_llm_output(answer)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            print(f"[GROQ] Attempt {attempt + 1}/{_MAX_RETRIES + 1} error: {e}")

            is_rate_limit = '429' in error_str or 'quota' in error_str or 'rate_limit' in error_str

            # Retry on rate limit with exponential backoff
            if is_rate_limit and attempt < _MAX_RETRIES:
                wait = 10 * (attempt + 1)  # 10s, 20s
                print(f"[GROQ] Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            # Final failure
            if is_rate_limit:
                return ":hourglass: AI is rate-limited right now. Wait ~30 seconds and try again."
            else:
                logger.error(f"Groq error: {e}")
                return f":x: AI error: {str(e)[:300]}"

    return f":x: AI error after retries: {str(last_error)[:300]}"
