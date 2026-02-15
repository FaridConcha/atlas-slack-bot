#!/usr/bin/env python3
"""
ATLAS V12 — Full-Spectrum Report Formatter
Transforms engine output + extended data into an 11-section Slack report.

V12adds a Buffett-aligned Owner Assessment layer on top of V8 quant signals.
Interpretation shifts from "signal says BUY" to "would a rational capital
allocator buy this entire business at this price?"

Each section is delivered as a separate threaded Slack message (<4000 chars each).
Design: Dense data in tables, natural language in narratives, everything in context.
"""

from datetime import datetime
from valuation_config import (
    CONFIG, MOAT_PROTECTED_INDUSTRIES, THIN_MARGIN_NET_THRESHOLD,
    THIN_MARGIN_OP_THRESHOLD, FRAGILITY_LOW_WACC, FRAGILITY_HIGH_TERMINAL_DEP,
    FRAGILITY_LOW_DATA_CONF, FRAGILITY_FLAT_MARGINS,
    BetaPath, SectorProvenance,
    INDUSTRY_PRIORS, PRIOR_CAP,
)


def _n(val, default=0):
    """None-safe numeric: return val if not None, else default. For scoring math."""
    return val if val is not None else default


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
    data_status = fin.get('_data_status', 'OK')

    # 1. Signal Strength (from engine)
    c_raw = summary.get('composite_raw', 0)
    tq = summary.get('trade_quality', 0)
    gate = summary.get('gate_value', 1)
    signal = max(0, min(100, 50 + c_raw * 0.5))
    signal = signal * min(1.3, 0.7 + tq * 2)
    signal = max(0, min(100, signal))

    # 2. Fundamental Score — gated on data availability
    if data_status == 'INVALID':
        fundamental = None  # Cannot compute
    else:
        fundamental = 50
        rg = _n(fin.get('revenue_growth'))
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

        nm = _n(fin.get('net_margin'))
        if nm > 25:
            fundamental += 12
        elif nm > 15:
            fundamental += 6
        elif nm > 5:
            fundamental += 2
        elif nm < 0:
            fundamental -= 15

        roe = _n(fin.get('roe'))
        if roe > 25:
            fundamental += 10
        elif roe > 15:
            fundamental += 5
        elif roe < 5:
            fundamental -= 5

        fcf_val = _n(fin.get('free_cash_flow'))
        if fcf_val > 0:
            fundamental += 5
        elif fin.get('free_cash_flow') is not None:
            fundamental -= 8
        # If FCF is None (missing), don't penalize — data absent

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

    # Composite — if fundamental is None (INVALID data), exclude it and reweight
    if fundamental is not None:
        composite = round(
            signal * 0.30 + fundamental * 0.25 + technical * 0.15 +
            sentiment * 0.10 + macro * 0.10 + risk * 0.10
        )
    else:
        # Reweight without fundamental (0.25 redistributed)
        composite = round(
            signal * 0.40 + technical * 0.20 +
            sentiment * 0.13 + macro * 0.13 + risk * 0.14
        )
    composite = max(0, min(100, composite))

    return {
        'signal': round(signal),
        'fundamental': round(fundamental) if fundamental is not None else None,
        'technical': round(technical),
        'sentiment': round(sentiment),
        'macro': round(macro),
        'risk': round(risk),
        'composite': composite,
        '_data_status': data_status,
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
# STAGE 5: FRAGILITY, DYNAMIC MOS, RECONCILIATION
# ============================================================================

def _compute_fragility(dcf, financials, summary):
    """
    Identify valuation fragility contributors.

    Returns:
        list[str] of contributor codes (e.g. LOW_WACC, HIGH_TERMINAL_DEP, etc.)
    """
    if not CONFIG.flags.fragility_scoring:
        return []

    contributors = []
    cfg = CONFIG.fragility

    # LOW_WACC: discount rate too low to be credible
    assumptions = dcf.get('assumptions', {}) if dcf else {}
    discount_rate = assumptions.get('discount_rate', 0) or 0
    if discount_rate > 0 and discount_rate < cfg.low_wacc_threshold:
        contributors.append(FRAGILITY_LOW_WACC)

    # HIGH_TERMINAL_DEP: terminal value dominates fair value
    terminal_pct = assumptions.get('terminal_value_pct', 0) or 0
    if terminal_pct >= cfg.high_terminal_dep_threshold:
        contributors.append(FRAGILITY_HIGH_TERMINAL_DEP)

    # LOW_DATA_CONF: engine data confidence is low
    dc = summary.get('data_confidence', 100) or 100
    if dc < cfg.low_data_confidence_threshold:
        contributors.append(FRAGILITY_LOW_DATA_CONF)

    # FLAT_MARGINS_ASSUMPTION: if DCF uses constant margin over forecast
    # (always true in current model — flag if margin is negative or very thin)
    fcf_margin = assumptions.get('fcf_margin', 0) or 0
    if 0 < fcf_margin < 5:
        contributors.append(FRAGILITY_FLAT_MARGINS)

    return contributors


def _compute_dynamic_mos(business_type, dcf, financials, data_confidence, fragility_contributors):
    """
    Additive MOS model: base + uplift components.

    Returns:
        (required_mos: float (decimal, e.g. 0.86), mos_build: list[dict])
    """
    if not CONFIG.flags.dynamic_mos:
        # Legacy static MOS
        if business_type == 'Very Stable':
            return (0.20, [{'component': 'base', 'adjustment': 20, 'reason': 'Very Stable (legacy)'}])
        elif business_type == 'Normal':
            return (0.30, [{'component': 'base', 'adjustment': 30, 'reason': 'Normal (legacy)'}])
        else:
            return (0.45, [{'component': 'base', 'adjustment': 45, 'reason': 'Cyclical (legacy)'}])

    cfg = CONFIG.mos
    build = []
    total_pp = 0.0

    # 1. Base — always present
    base_pct = cfg.base_for_type(business_type) * 100
    build.append({'component': 'base', 'adjustment': base_pct, 'reason': f'{business_type} business'})
    total_pp += base_pct

    # P4: Show ALL components for audit transparency, even those at 0
    # 2. Data confidence uplift
    dc = data_confidence if data_confidence is not None else 100
    if dc < cfg.data_confidence_threshold:
        build.append({'component': 'data_confidence', 'adjustment': cfg.uplift_low_data_confidence,
                      'reason': f'Data confidence {dc:.0f}% < {cfg.data_confidence_threshold:.0f}%'})
        total_pp += cfg.uplift_low_data_confidence
    else:
        build.append({'component': 'data_confidence', 'adjustment': 0,
                      'reason': f'Data confidence {dc:.0f}% >= {cfg.data_confidence_threshold:.0f}% (no uplift)'})

    # 3. Terminal dependence uplift
    assumptions = dcf.get('assumptions', {}) if dcf else {}
    terminal_pct = assumptions.get('terminal_value_pct', 0) or 0
    if terminal_pct >= cfg.terminal_dep_threshold:
        build.append({'component': 'terminal_dependence', 'adjustment': cfg.uplift_high_terminal_dep,
                      'reason': f'Terminal dependence {terminal_pct:.0f}% >= {cfg.terminal_dep_threshold:.0f}%'})
        total_pp += cfg.uplift_high_terminal_dep
    else:
        build.append({'component': 'terminal_dependence', 'adjustment': 0,
                      'reason': f'Terminal dependence {terminal_pct:.0f}% < {cfg.terminal_dep_threshold:.0f}% (no uplift)'})

    # 4. WACC clamp uplift
    wacc_clamp_codes = assumptions.get('wacc_clamp_codes', []) or []
    if wacc_clamp_codes:
        build.append({'component': 'wacc_clamp', 'adjustment': cfg.uplift_wacc_clamp,
                      'reason': f'WACC clamp applied: {", ".join(wacc_clamp_codes)}'})
        total_pp += cfg.uplift_wacc_clamp
    else:
        build.append({'component': 'wacc_clamp', 'adjustment': 0,
                      'reason': 'No WACC clamp applied'})

    # 5. Leverage uplift
    de = _n(financials.get('debt_equity'))
    if de > cfg.leverage_extreme_threshold:
        build.append({'component': 'leverage', 'adjustment': cfg.uplift_extreme_leverage,
                      'reason': f'D/E {de:.1f}x > {cfg.leverage_extreme_threshold:.0f}x (extreme)'})
        total_pp += cfg.uplift_extreme_leverage
    elif de > cfg.leverage_high_threshold:
        build.append({'component': 'leverage', 'adjustment': cfg.uplift_high_leverage,
                      'reason': f'D/E {de:.1f}x > {cfg.leverage_high_threshold:.0f}x (high)'})
        total_pp += cfg.uplift_high_leverage
    else:
        build.append({'component': 'leverage', 'adjustment': 0,
                      'reason': f'D/E {de:.1f}x <= {cfg.leverage_high_threshold:.0f}x (no uplift)'})

    # 6. Value creation shortfall (ROE < WACC proxy)
    roe = _n(financials.get('roe'))
    wacc_proxy = assumptions.get('discount_rate', 8) or 8
    if roe > 0 and roe < wacc_proxy:
        build.append({'component': 'value_creation', 'adjustment': cfg.uplift_value_destruction,
                      'reason': f'ROE {roe:.1f}% < WACC {wacc_proxy:.1f}% (value destruction)'})
        total_pp += cfg.uplift_value_destruction
    elif roe > 0:
        build.append({'component': 'value_creation', 'adjustment': 0,
                      'reason': f'ROE {roe:.1f}% >= WACC {wacc_proxy:.1f}% (value creator)'})
    else:
        build.append({'component': 'value_creation', 'adjustment': 0,
                      'reason': 'ROE not available'})

    # 7. Fragility uplift (per extra contributor beyond 1)
    n_frag = len(fragility_contributors)
    if n_frag > 1:
        extra = n_frag - 1
        frag_uplift = extra * cfg.uplift_fragility_per
        build.append({'component': 'fragility', 'adjustment': frag_uplift,
                      'reason': f'{n_frag} fragility contributors ({", ".join(fragility_contributors)})'})
        total_pp += frag_uplift
    elif n_frag == 1:
        build.append({'component': 'fragility', 'adjustment': 0,
                      'reason': f'1 fragility contributor ({fragility_contributors[0]})'})
    else:
        build.append({'component': 'fragility', 'adjustment': 0,
                      'reason': 'No fragility contributors'})

    required_mos = total_pp / 100.0
    return (round(required_mos, 4), build)


def _reconciliation_checks(summary, v8_data, v9_scores):
    """
    Cross-module consistency checks.

    Returns:
        list[dict] with keys: check, status ('OK'|'WARN'|'ERROR'), detail
    """
    if not CONFIG.flags.reconciliation_checks:
        return []

    errors = []

    # 1. MOS sign invariant: if price > IV, MOS must be negative
    price = v8_data.get('company', {}).get('price', 0) or 0
    iv = v9_scores.get('intrinsic_value_base', 0) or 0
    mos = v9_scores.get('mos_pct', 0) or 0
    if price > 0 and iv > 0 and price > iv and mos > 0:
        errors.append({
            'check': 'MOS_SIGN_INVARIANT',
            'status': 'ERROR',
            'detail': f'Price ${price:.2f} > IV ${iv:.2f} but MOS is +{mos:.1f}% (should be negative)'
        })

    # 2. DCF bridge: PV sum + TV = EV
    dcf = v8_data.get('dcf', {})
    assumptions = dcf.get('assumptions', {}) if dcf else {}
    pv_sum = assumptions.get('pv_fcf_sum', 0) or 0
    tv_pv = assumptions.get('terminal_value_pv', 0) or 0
    ev = assumptions.get('enterprise_value', 0) or 0
    if ev > 0 and abs((pv_sum + tv_pv) - ev) > 2:
        errors.append({
            'check': 'DCF_BRIDGE',
            'status': 'ERROR',
            'detail': f'PV sum ({pv_sum:.0f}) + TV PV ({tv_pv:.0f}) != EV ({ev:.0f})'
        })

    # 3. Weight sum (from engine)
    w_dynamic = summary.get('w_dynamic', {})
    if w_dynamic:
        wsum = sum(w_dynamic.values())
        if abs(wsum - 1.0) > 0.01:
            errors.append({
                'check': 'WEIGHT_SUM',
                'status': 'WARN',
                'detail': f'Engine weight sum = {wsum:.4f} (expected 1.0)'
            })

    # 4. Composite sum verification
    contributions = summary.get('contributions', {})
    composite_raw = summary.get('composite_raw')
    if contributions and composite_raw is not None:
        c_sum = sum(contributions.values())
        if abs(c_sum - composite_raw) > 0.5:
            errors.append({
                'check': 'COMPOSITE_SUM',
                'status': 'WARN',
                'detail': f'Contribution sum {c_sum:.2f} != composite_raw {composite_raw:.2f}'
            })

    # G9: MOS build sum reconciliation
    mos_build = v9_scores.get('mos_build', [])
    required_mos = v9_scores.get('required_mos')
    if mos_build and required_mos is not None:
        build_sum = sum(item.get('adjustment', 0) for item in mos_build) / 100.0
        if abs(build_sum - required_mos) > 0.01:
            errors.append({
                'check': 'MOS_BUILD_SUM',
                'status': 'ERROR',
                'detail': f'MOS build sum {build_sum:.4f} != required_mos {required_mos:.4f}'
            })

    return errors


# ============================================================================
# V12OWNER INTELLIGENCE LAYER
# ============================================================================

def _compute_v9_owner_scores(summary, v8_data):
    """
    Compute Buffett-aligned owner intelligence scores.

    Returns dict with:
        business_quality (0-5), moat_durability (0-5),
        capital_allocation (0-5), mos_pct (float),
        intrinsic_value_base (float), v9_decision (str),
        required_mos (float), business_type (str),
        conviction (0-100), permanent_loss_risks (list)
    """
    fin = v8_data.get('financials', {})
    co = v8_data.get('company', {})
    dcf = v8_data.get('dcf', {})
    inst = v8_data.get('institutional', {})
    earnings = v8_data.get('earnings', [])
    price = co.get('price', 0)
    data_status = fin.get('_data_status', 'OK')

    # --- GATE: If fundamentals INVALID, return degraded assessment ---
    # P3: Return None (not 0) for BQ/Moat/CA — "not scoreable" not "scored zero"
    if data_status == 'INVALID':
        data_reasons = fin.get('_data_reasons', [])
        return {
            'business_quality': None,
            'moat_durability': None,
            'capital_allocation': None,
            'mos_pct': None,
            'mos_price_based': None,
            'mos_iv_basis': None,
            'premium_to_iv': None,
            'intrinsic_value_base': 0,
            'intrinsic_value_bear': 0,
            'intrinsic_value_bull': 0,
            'v9_decision': 'RESEARCH',
            'decision_reason': f"Fundamental data unavailable ({', '.join(data_reasons[:3])}). Cannot assess.",
            'required_mos': None,
            'required_price': 0,
            'business_type': 'Unknown',
            'conviction': None,
            'permanent_loss_risks': [("Data integrity failure", "High",
                "Core financial data missing or invalid — analysis suppressed")],
            '_data_status': data_status,
            'mos_build': [],
            'fragility_score': 0,
            'fragility_contributors': [],
            'ca_evidence': {},
            'iv_confidence': None,
            'terminal_penalty': 0,
            'terminal_pct': 0,
            'terminal_flags': [],
            '_dcf_disabled': True,
        }

    # --- Business Quality Score (0-5) ---
    bq = 0.0
    roe = _n(fin.get('roe'))
    if roe > 25:
        bq += 1.2
    elif roe > 15:
        bq += 0.8
    elif roe > 10:
        bq += 0.4

    nm = _n(fin.get('net_margin'))
    if nm > 20:
        bq += 1.0
    elif nm > 12:
        bq += 0.6
    elif nm > 5:
        bq += 0.3

    rg = _n(fin.get('revenue_growth'))
    if rg > 10:
        bq += 0.8
    elif rg > 3:
        bq += 0.5
    elif rg > 0:
        bq += 0.2
    elif rg < -5:
        bq -= 0.5

    fcf = _n(fin.get('free_cash_flow'))
    if fcf > 0:
        bq += 0.5
        fcf_yield = _n(fin.get('fcf_yield'))
        if fcf_yield > 5:
            bq += 0.3
    elif fin.get('free_cash_flow') is not None:
        bq -= 0.5
    # If FCF is None (missing), don't penalize

    de = _n(fin.get('debt_equity'))
    if de < 0.5:
        bq += 0.5
    elif de < 1.0:
        bq += 0.2
    elif de > 2.0:
        bq -= 0.5

    if earnings:
        beat_rate = sum(1 for e in earnings if e.get('beat', False)) / len(earnings)
        if beat_rate > 0.75:
            bq += 0.5
        elif beat_rate > 0.5:
            bq += 0.2
        elif beat_rate < 0.25:
            bq -= 0.3

    business_quality = max(0, min(5, round(bq, 1)))

    # --- Moat Durability Score (0-5) ---
    moat = 0.0
    gm = _n(fin.get('gross_margin'))
    if gm > 50:
        moat += 1.5  # Strong pricing power
    elif gm > 35:
        moat += 1.0
    elif gm > 20:
        moat += 0.4

    om = _n(fin.get('operating_margin'))
    if om > 25:
        moat += 1.0
    elif om > 15:
        moat += 0.6
    elif om > 8:
        moat += 0.3

    mc = _n(fin.get('market_cap')) or _n(co.get('market_cap')) or 0
    if mc > 200e9:
        moat += 0.8  # Scale advantage
    elif mc > 50e9:
        moat += 0.5
    elif mc > 10e9:
        moat += 0.2

    if roe > 20 and nm > 15:
        moat += 0.7  # Sustainable competitive advantage indicator

    ic = _n(fin.get('interest_coverage'))
    if ic > 15:
        moat += 0.5
    elif ic > 8:
        moat += 0.3

    moat_durability = max(0, min(5, round(moat, 1)))

    # --- G1: Industry Prior Application ---
    industry = co.get('industry', '')
    prior = INDUSTRY_PRIORS.get(industry) if CONFIG.flags.industry_priors else None
    prior_audit = {'prior_applied': False, 'prior_value': 0, 'data_missing_fields': [], 'cap_applied': False}

    # Count missing metric fields to determine prior weight
    missing_fields = []
    if fin.get('roe') is None: missing_fields.append('roe')
    if fin.get('net_margin') is None: missing_fields.append('net_margin')
    if fin.get('revenue_growth') is None: missing_fields.append('revenue_growth')
    if fin.get('gross_margin') is None: missing_fields.append('gross_margin')
    if fin.get('operating_margin') is None: missing_fields.append('operating_margin')
    if fin.get('free_cash_flow') is None: missing_fields.append('free_cash_flow')

    if prior and len(missing_fields) >= 2:
        # Scale prior by missing data proportion (more missing = more prior weight)
        prior_weight = min(1.0, len(missing_fields) / 6)

        bq_adj = min(PRIOR_CAP, prior.bq_prior * prior_weight)
        business_quality = max(0, min(5, round(bq + bq_adj, 1)))

        moat_adj = min(PRIOR_CAP, prior.moat_prior * prior_weight)
        moat_durability = max(0, min(5, round(moat + moat_adj, 1)))

        ca_adj = min(PRIOR_CAP, prior.ca_prior * prior_weight)
        # ca_adj applied after CA score computed below

        prior_audit = {
            'prior_applied': True,
            'industry': industry,
            'prior_value': {'bq': round(bq_adj, 2), 'moat': round(moat_adj, 2), 'ca': round(ca_adj, 2)},
            'data_missing_fields': missing_fields,
            'cap_applied': bq_adj >= PRIOR_CAP or moat_adj >= PRIOR_CAP,
            'rationale': prior.rationale,
        }
    else:
        ca_adj = 0

    # --- Capital Allocation Score (0-5) ---
    ca = 0.0

    # G2: ROIC proxy vs WACC — handle missing ROE as "uncertain"
    roe_available = fin.get('roe') is not None
    roic_proxy = roe * (1 - de / (1 + de)) if roe_available and de >= 0 and (1 + de) > 0 else (roe if roe_available else None)
    wacc_proxy = 8.0  # Rough estimate

    if roic_proxy is not None:
        if roic_proxy > wacc_proxy * 2:
            ca += 1.5
        elif roic_proxy > wacc_proxy:
            ca += 1.0
        elif roic_proxy > 0:
            ca += 0.3
        # else: roic_proxy <= 0 — no credit, no penalty
    else:
        ca += 0.0  # Uncertain — no penalty for missing data
        # roic_proxy remains None → "value creation uncertain"

    # G3: Buyback discipline — includes IV-price check
    bb = _n(fin.get('buyback_yield'))
    fwd_pe = _n(fin.get('forward_pe'))
    dcf_disabled = dcf.get('_dcf_disabled', False)
    base_iv = dcf.get('base', 0) if not dcf_disabled else 0
    is_value_creator = roic_proxy is not None and roic_proxy > wacc_proxy
    if bb > 0:
        if not is_value_creator:
            pass  # buyback credit handled via ca_evidence, not raw score
        elif base_iv > 0 and price > base_iv:
            ca += 0.3  # POTENTIALLY_DESTRUCTIVE: buying back above IV
        elif fwd_pe > 0 and fwd_pe < 20:
            ca += 1.0  # Buying back at reasonable valuation
        elif fwd_pe > 30:
            ca += 0.2  # Buying back at high valuation — questionable
        else:
            ca += 0.6

    # Dividend discipline
    dy = _n(fin.get('dividend_yield'))
    pr = _n(fin.get('payout_ratio'))
    if dy > 0 and pr > 0 and pr < 60:
        ca += 0.5  # Sustainable dividend
    elif pr > 90:
        ca -= 0.3  # Unsustainable payout

    # Debt discipline
    nd_ebitda = _n(fin.get('net_debt_ebitda'))
    if nd_ebitda < 1:
        ca += 0.8
    elif nd_ebitda < 2:
        ca += 0.4
    elif nd_ebitda > 4:
        ca -= 0.5

    # Conservative if large cash position
    total_cash = _n(fin.get('total_cash'))
    total_debt = _n(fin.get('total_debt'))
    if total_cash > total_debt:
        ca += 0.5

    # Apply industry prior CA adjustment (G1)
    if ca_adj > 0:
        ca += ca_adj
    capital_allocation = max(0, min(5, round(ca, 1)))

    # --- Business Type Classification ---
    if business_quality >= 3.5 and moat_durability >= 3.5:
        business_type = "Very Stable"
    elif business_quality >= 2.0 and moat_durability >= 2.0:
        business_type = "Normal"
    else:
        business_type = "Cyclical"

    # --- Fragility + Dynamic MOS (Stage 5) ---
    data_confidence = summary.get('data_confidence', 100) or 100
    fragility_contributors = _compute_fragility(dcf, fin, summary)
    fragility_score = len(fragility_contributors)

    required_mos, mos_build = _compute_dynamic_mos(
        business_type, dcf, fin, data_confidence, fragility_contributors
    )

    # --- Margin of Safety ---
    dcf_disabled = dcf.get('_dcf_disabled', False)
    base_iv = dcf.get('base', 0) if not dcf_disabled else 0
    if base_iv > 0 and price > 0:
        mos_pct = round((base_iv - price) / base_iv * 100, 1)
    else:
        mos_pct = 0

    # Invariant: If price > IV, MOS must be negative
    if price > base_iv and base_iv > 0 and mos_pct > 0:
        mos_pct = -abs(mos_pct)

    # --- Permanent Loss Risks ---
    perm_risks = []
    if de > 3:
        perm_risks.append(("Excessive leverage", "High", "Debt covenant breach in downturn"))
    if rg < -10:
        perm_risks.append(("Secular decline", "High", "Revenue in structural free-fall"))
    if nm < 0:
        perm_risks.append(("Unprofitable operations", "High", "Cash burn without path to profitability"))
    if fwd_pe > 50 and rg < 20:
        perm_risks.append(("Valuation collapse risk", "Medium", "Growth doesn't justify extreme multiple"))
    sp = inst.get('short_pct', 0)
    if sp > 10:
        perm_risks.append(("Heavy short positioning", "Medium", f"Short interest {sp:.1f}% signals structural concern"))
    regime = summary.get('regime_label', '')
    if regime in ('Crisis Trend', 'Credit Stress'):
        perm_risks.append(("Macro regime stress", "Medium", f"Operating in {regime} environment"))
    if ic > 0 and ic < 3:
        perm_risks.append(("Debt service risk", "Medium", f"Interest coverage only {ic:.1f}x"))
    if base_iv > 0 and price > base_iv * 1.5:
        perm_risks.append(("Severe overvaluation", "Medium",
            f"Price {price/base_iv:.1f}x intrinsic value — overpayment risk material"))
    # G6: Missing data → "Uncertain" risk entries (not high risk, not absent)
    if fin.get('debt_equity') is None:
        perm_risks.append(("Leverage position", "Uncertain", "Debt/equity data unavailable — cannot assess leverage risk"))
    if fin.get('interest_coverage') is None and fin.get('debt_equity') is not None and de > 1:
        perm_risks.append(("Debt service capacity", "Uncertain", "Interest coverage unavailable — debt service risk unknown"))
    if fin.get('revenue_growth') is None:
        perm_risks.append(("Revenue trajectory", "Uncertain", "Revenue growth data unavailable — secular trends unknown"))
    if fin.get('free_cash_flow') is None:
        perm_risks.append(("Cash flow generation", "Uncertain", "FCF data unavailable — cash generation uncertain"))

    if not perm_risks:
        perm_risks.append(("General market risk", "Low", "Systemic drawdown exposure"))

    # --- V12Decision Logic ---
    # Step 1: Business Quality Gate
    if business_quality < 1.5:
        v9_decision = "PASS"
        decision_reason = "Business quality too low for capital allocation"
    # Step 2: MOS Check
    elif base_iv <= 0:
        v9_decision = "RESEARCH"
        decision_reason = "Insufficient data for intrinsic value estimate"
    elif mos_pct < -20:
        v9_decision = "PASS"
        decision_reason = "Deeply overvalued relative to intrinsic value"
    elif mos_pct < 0:
        v9_decision = "WATCH"
        decision_reason = "Trading above intrinsic value — no margin of safety"
    elif mos_pct < required_mos * 100:
        v9_decision = "WATCH"
        decision_reason = f"MOS {mos_pct:.0f}% below required {required_mos*100:.0f}% for {business_type.lower()} business"
    # Step 3: Capital allocation quality
    elif capital_allocation < 1.5:
        v9_decision = "RESEARCH"
        decision_reason = "Adequate value but capital allocation concerns"
    # Step 4: Permanent loss risk asymmetry
    elif any(r[1] == "High" for r in perm_risks):
        v9_decision = "RESEARCH"
        decision_reason = "Material permanent loss risks require deeper analysis"
    # Step 5: Full pass — good business, good price
    else:
        if mos_pct > required_mos * 100 * 1.5 and business_quality >= 3.5:
            v9_decision = "BUY"
            decision_reason = "Strong business with compelling margin of safety"
        elif mos_pct > required_mos * 100:
            v9_decision = "BUY"
            decision_reason = "Adequate business quality and sufficient margin of safety"
        else:
            v9_decision = "HOLD"
            decision_reason = "Fair value — maintain position, no new capital"

    # --- Conviction Score (0-100) ---
    conviction = 0
    conviction += min(25, business_quality * 5)  # 0-25
    conviction += min(20, moat_durability * 4)   # 0-20
    conviction += min(20, capital_allocation * 4) # 0-20
    conviction += min(20, max(0, mos_pct) * 0.5) # 0-20 (capped)
    conviction += min(15, (5 - len([r for r in perm_risks if r[1] == "High"])) * 5)  # 0-15

    # Terminal-dependence penalty: if >70% of IV from terminal year,
    # reduce conviction (model is fragile to r-g assumptions)
    # Stage 5: Enhanced penalty cap 20 (was 15)
    terminal_pct = dcf.get('assumptions', {}).get('terminal_value_pct', 0) or 0
    terminal_penalty = 0
    terminal_flags = dcf.get('assumptions', {}).get('terminal_flags', []) or []
    if terminal_pct > CONFIG.terminal.penalty_threshold:
        terminal_penalty = min(CONFIG.terminal.penalty_cap,
                               round((terminal_pct - CONFIG.terminal.penalty_threshold) * CONFIG.terminal.penalty_per_pct))
        conviction -= terminal_penalty

    conviction = max(0, min(100, round(conviction)))

    # IV confidence: LOW when terminal_pct >= extreme threshold
    # G10: Use severe_threshold (80%) not extreme_threshold (90%) — matches spec Section IV.3
    iv_confidence = 'LOW' if terminal_pct >= CONFIG.terminal.severe_threshold else 'NORMAL'

    # P0: MOS semantics — explicit IV-basis and premium-to-IV fields
    # mos_iv_basis: (IV - Price) / IV — "discount from IV" (positive = undervalued)
    # premium_to_iv: (Price / IV) - 1 — "how much over IV" (positive = overvalued)
    mos_iv_basis = round((base_iv - price) / base_iv * 100, 1) if base_iv > 0 else None
    premium_to_iv = round((price / base_iv - 1) * 100, 1) if base_iv > 0 else None

    # Price-based MOS for display (legacy: relative to price)
    mos_price_based = round((base_iv - price) / price * 100, 1) if price > 0 else 0

    # --- Capital Allocation Evidence (Stage 5 + P5 fix + G3 IV check + G4 completeness) ---
    ca_evidence = {}
    if CONFIG.flags.ca_evidence:
        wacc_proxy_used = dcf.get('assumptions', {}).get('discount_rate', 8) or 8
        # G2: Use the already-computed roic_proxy (handles None for missing ROE)
        ca_roic = roic_proxy  # May be None if ROE missing
        ca_is_value_creator = ca_roic is not None and ca_roic > wacc_proxy_used

        # G3: Buyback discipline must account for value creation AND IV-price check
        if bb > 0:
            if ca_roic is None or not ca_is_value_creator:
                buyback_discipline = 'DESTROYS_VALUE'  # P5: buybacks at any PE destroy value
            elif base_iv > 0 and price > base_iv:
                buyback_discipline = 'POTENTIALLY_DESTRUCTIVE'  # G3: IV < price → overpaying
            elif fwd_pe > 0 and fwd_pe < 20:
                buyback_discipline = 'GOOD'
            elif fwd_pe > 30:
                buyback_discipline = 'QUESTIONABLE'
            else:
                buyback_discipline = 'ADEQUATE'
        elif bb == 0 and fin.get('buyback_yield') is not None:
            buyback_discipline = 'NONE'
        else:
            buyback_discipline = 'INSUFFICIENT_EVIDENCE'

        ca_evidence = {
            'roic_proxy': round(ca_roic, 2) if ca_roic is not None else None,
            'wacc_proxy': round(wacc_proxy_used, 2),
            'roic_wacc_spread': round(ca_roic - wacc_proxy_used, 2) if ca_roic is not None else None,
            'buyback_yield': round(_n(fin.get('buyback_yield')), 2),
            'buyback_fwd_pe': round(fwd_pe, 1),
            'buyback_discipline': buyback_discipline,
            'reason_codes': [],
        }
        if ca_roic is not None:
            if ca_is_value_creator:
                ca_evidence['reason_codes'].append('VALUE_CREATOR')
            else:
                ca_evidence['reason_codes'].append('VALUE_DESTROYER')
        else:
            ca_evidence['reason_codes'].append('ROIC_UNAVAILABLE')
        if bb > 0 and ca_is_value_creator and fwd_pe > 0 and fwd_pe < 20:
            ca_evidence['reason_codes'].append('BUYBACK_DISCIPLINED')
        elif bb > 0 and ca_is_value_creator and fwd_pe > 30:
            ca_evidence['reason_codes'].append('BUYBACK_OVERPRICED')
        elif bb > 0 and ca_is_value_creator and base_iv > 0 and price > base_iv:
            ca_evidence['reason_codes'].append('BUYBACK_ABOVE_IV')
        elif bb > 0 and not ca_is_value_creator:
            ca_evidence['reason_codes'].append('BUYBACK_VALUE_DESTRUCTIVE')

        # G4: Data completeness field
        ca_evidence['data_completeness'] = {
            'roe_available': fin.get('roe') is not None,
            'roic_method': 'proxy_from_roe' if roe_available else 'unavailable',
            'buyback_available': fin.get('buyback_yield') is not None,
            'fwd_pe_available': fin.get('forward_pe') is not None,
            'debt_equity_available': fin.get('debt_equity') is not None,
        }

    return {
        'business_quality': business_quality,
        'moat_durability': moat_durability,
        'capital_allocation': capital_allocation,
        'mos_pct': mos_pct,
        'mos_price_based': mos_price_based,
        'mos_iv_basis': mos_iv_basis,
        'premium_to_iv': premium_to_iv,
        'intrinsic_value_base': base_iv,
        'intrinsic_value_bear': dcf.get('bear', 0) if not dcf_disabled else 0,
        'intrinsic_value_bull': dcf.get('bull', 0) if not dcf_disabled else 0,
        'v9_decision': v9_decision,
        'decision_reason': decision_reason,
        'required_mos': required_mos,
        'required_mos_used': round(required_mos, 4),
        'required_price': round(base_iv * (1 - required_mos), 2) if base_iv > 0 else 0,
        'business_type': business_type,
        'conviction': conviction,
        'terminal_penalty': terminal_penalty,
        'terminal_pct': terminal_pct,
        'terminal_flags': terminal_flags,
        'iv_confidence': iv_confidence,
        'permanent_loss_risks': perm_risks[:5],
        'mos_build': mos_build,
        'fragility_score': fragility_score,
        'fragility_contributors': fragility_contributors,
        'ca_evidence': ca_evidence,
        'prior_audit': prior_audit,
        '_data_status': data_status,
        '_dcf_disabled': dcf_disabled,
    }


def _stars(score, max_stars=5):
    """Render score as filled/empty stars."""
    filled = int(round(score))
    return '\u2605' * filled + '\u2606' * (max_stars - filled)


# ============================================================================
# SECTION 0: V12OWNER ASSESSMENT
# ============================================================================

def _section_owner_assessment(summary, v8_data):
    """V12Owner Assessment — Buffett-aligned business evaluation."""
    co = v8_data.get('company', {})
    fin = v8_data.get('financials', {})
    v9 = _compute_v9_owner_scores(summary, v8_data)
    symbol = co.get('symbol', '???')
    name = co.get('name', symbol)
    price = co.get('price', 0)
    now = datetime.now().strftime('%b %d, %Y %I:%M %p ET')

    decision = v9['v9_decision']
    data_status = v9.get('_data_status', 'OK')
    dec_emoji = {
        'BUY': ':white_check_mark:', 'HOLD': ':pause_button:',
        'WATCH': ':eyes:', 'RESEARCH': ':mag:', 'PASS': ':no_entry_sign:',
        'TRIM': ':scissors:', 'EXIT': ':door:',
    }.get(decision, ':grey_question:')

    header = (
        f"```\n"
        f"{'=' * 52}\n"
        f"  ATLAS V12 — OWNER ASSESSMENT\n"
        f"  {symbol} — {name} — ${price:.2f}\n"
        f"  As of {now}\n"
        f"{'=' * 52}\n"
        f"```\n"
    )

    # Data integrity warning
    data_warning = ""
    if data_status == 'INVALID':
        data_warning = (
            ":warning: *DATA INTEGRITY: INVALID* — Core financial data is missing or anomalous. "
            "Scores and valuations are suppressed. Treat this report as directional only.\n\n"
        )
    elif data_status == 'DEGRADED':
        reasons = fin.get('_data_reasons', [])
        data_warning = (
            ":warning: *DATA INTEGRITY: DEGRADED* — Some metrics are missing or estimated"
            + (f" ({', '.join(reasons[:2])})" if reasons else "")
            + ". Results may have reduced accuracy.\n\n"
        )

    # Scorecard — P3: Handle None scores
    iv = v9['intrinsic_value_base']
    mos = v9['mos_pct']
    bq = v9['business_quality']
    moat = v9['moat_durability']
    ca_score = v9['capital_allocation']
    conv = v9['conviction']
    req_mos = v9['required_mos']

    # P0: Use premium_to_iv for overvalued, mos_iv_basis for undervalued
    if iv > 0 and mos is not None:
        premium = v9.get('premium_to_iv')
        if premium is not None and premium > 0:
            mos_str = f"{mos:+.1f}% (trading at {premium:.0f}% premium to IV)"
        else:
            mos_str = f"{mos:+.1f}%"
    else:
        mos_str = "N/A"

    bq_str = f"{_stars(bq)}  {bq:.1f}/5" if bq is not None else "N/A  (data unavailable)"
    moat_str = f"{_stars(moat)}  {moat:.1f}/5" if moat is not None else "N/A  (data unavailable)"
    ca_str = f"{_stars(ca_score)}  {ca_score:.1f}/5" if ca_score is not None else "N/A  (data unavailable)"
    conv_str = f"{conv}/100" if conv is not None else "N/A"
    req_mos_str = f"{req_mos*100:.0f}%  ({v9['business_type']})" if req_mos is not None else "N/A"

    scorecard = (
        f"```\n"
        f"Business Quality:     {bq_str}\n"
        f"Moat Durability:      {moat_str}\n"
        f"Capital Allocation:   {ca_str}\n"
        f"\n"
        f"Intrinsic Value:      ${iv:.2f}  (base DCF)\n"
        f"Current Price:        ${price:.2f}\n"
        f"Margin of Safety:     {mos_str}\n"
        f"Required MOS:         {req_mos_str}\n"
        f"\n"
        f"Conviction:           {conv_str}\n"
        f"```\n"
    )

    decision_block = f"{dec_emoji} *Decision: {decision}*\n> {v9['decision_reason']}\n"

    # WHY (3 bullets) — P6: Evidence-gated assertions
    why_bullets = []
    if bq is not None:
        if bq >= 3.5:
            why_bullets.append(f"Business quality {bq:.1f}/5 — high-quality compounder with durable economics")
        elif bq >= 2.0:
            why_bullets.append(f"Business quality {bq:.1f}/5 — adequate but not exceptional")
        else:
            why_bullets.append(f"Business quality {bq:.1f}/5 — structural weaknesses in business model")
    else:
        why_bullets.append("Business quality: data insufficient to assess")

    if iv > 0 and mos is not None and mos > 0:
        why_bullets.append(f"Trading at {mos:.0f}% discount to intrinsic value (${iv:.2f}) — margin of safety present")
    elif iv > 0 and mos is not None:
        why_bullets.append(f"Trading at {abs(mos):.0f}% premium to intrinsic value — no margin of safety")
    else:
        why_bullets.append("Insufficient data for reliable intrinsic value estimate")

    if ca_score is not None:
        # P5/P6: Gate CA narrative on evidence — check ca_evidence for contradictions
        ca_ev = v9.get('ca_evidence', {})
        if ca_ev.get('reason_codes') and 'VALUE_DESTROYER' in ca_ev.get('reason_codes', []):
            why_bullets.append(f"Capital allocation {ca_score:.1f}/5 — ROIC below cost of capital (value destruction)")
        elif ca_score >= 3.5:
            why_bullets.append(f"Capital allocation {ca_score:.1f}/5 — management creating per-share value")
        elif ca_score >= 2.0:
            why_bullets.append(f"Capital allocation {ca_score:.1f}/5 — reasonable but room for improvement")
        else:
            why_bullets.append(f"Capital allocation {ca_score:.1f}/5 — potential value destruction")
    else:
        why_bullets.append("Capital allocation: data insufficient to assess")

    # G7: Moat narrative — evidence-gated
    _prior_audit = v9.get('prior_audit', {})
    if moat is not None:
        if moat >= 3.5:
            why_bullets.append(f"Moat durability {moat:.1f}/5 — structural competitive advantages present")
        elif moat >= 2.0:
            why_bullets.append(f"Moat durability {moat:.1f}/5 — moderate competitive position")
        else:
            if _prior_audit.get('prior_applied'):
                why_bullets.append(f"Moat durability {moat:.1f}/5 — moat strength uncertain due to limited data (industry prior applied)")
            else:
                why_bullets.append(f"Moat durability {moat:.1f}/5 — limited evidence of durable competitive advantage")
    else:
        why_bullets.append("Moat durability: data insufficient to assess")

    why_text = "\n".join(f"- {b}" for b in why_bullets[:4])

    # RISKS (3 bullets)
    risk_bullets = []
    for rname, severity, rdesc in v9['permanent_loss_risks'][:3]:
        risk_bullets.append(f"[{severity}] {rname} — {rdesc}")
    risk_text = "\n".join(f"- {b}" for b in risk_bullets)

    # WHAT WOULD CHANGE MY MIND — P6: guard None values
    change_bullets = []
    if decision in ('PASS', 'WATCH'):
        if mos is not None and req_mos is not None and mos < req_mos * 100:
            needed_price = iv * (1 - req_mos) if iv > 0 else 0
            if needed_price > 0:
                change_bullets.append(f"Price decline to ~${needed_price:.0f} would provide adequate margin of safety")
        change_bullets.append("Structural improvement in business quality or competitive position")
    elif decision == 'BUY':
        change_bullets.append(f"Erosion of margin of safety if price rises above ${iv:.0f}" if iv > 0 else "Loss of competitive advantage")
        change_bullets.append("Deterioration in management capital allocation discipline")
    else:
        change_bullets.append("Clearer data on business durability and intrinsic value")
        change_bullets.append("Resolution of identified permanent loss risks")

    change_text = "\n".join(f"- {b}" for b in change_bullets[:2])

    # Tactical overlay from engine
    regime = summary.get('regime_label', 'N/A')
    composite = summary.get('composite_raw', 0)
    tq = summary.get('trade_quality', 0)
    vix = summary.get('vix', 0)

    # Engine conflict protocol — P6: guard None mos
    conflict_note = ""
    if mos is not None and req_mos is not None:
        if composite > 30 and mos < 0:
            conflict_note = (
                "\n:warning: *Engine Conflict:* Quant engine favors momentum, but intrinsic value "
                "suggests full valuation. Under capital preservation framework, patience is warranted."
            )
        elif composite < -30 and mos > req_mos * 100:
            conflict_note = (
                "\n:bulb: *Engine Conflict:* Engine reacting to short-term trend weakness. However, "
                "intrinsic value provides significant margin of safety. Long-term case remains intact."
            )

    tactical = (
        f"\n_Tactical Overlay (engine): Regime={regime} | Composite={composite:+.1f} | "
        f"TQ={tq:.4f} | VIX={vix:.1f}_"
    )

    # Temperament note
    if vix > 28:
        temperament = "\n> :thermometer: *Market Sentiment:* Fear elevated. Historically, fear creates opportunity for patient capital."
    elif vix < 14:
        temperament = "\n> :thermometer: *Market Sentiment:* Extreme optimism. Exercise caution — complacency breeds risk."
    else:
        temperament = "\n> :thermometer: *Default action in absence of clear margin of safety is inaction.*"

    parts = [
        header,
        data_warning,
        scorecard,
        decision_block,
        "\n*Why:*",
        why_text,
        "\n*Permanent Loss Risks:*",
        risk_text,
        "\n*What Would Change My Mind:*",
        change_text,
        conflict_note,
        tactical,
        temperament,
    ]

    return "\n".join(parts)


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
    rg = _n(fin.get('revenue_growth'))
    nm = _n(fin.get('net_margin'))
    fcf = _n(fin.get('free_cash_flow'))
    fund_score = scores.get('fundamental')
    if fund_score is not None:
        grade = "strong" if fund_score >= 70 else "solid" if fund_score >= 55 else "mixed" if fund_score >= 40 else "weak"
        fcf_str = f", generating {_fmt_mc(fcf)} in FCF" if fcf > 0 else ""
        s2 = f"Fundamentals are {grade} with {_fmt_pct(rg, plus=True)} revenue growth, {_fmt_pct(nm)} net margins{fcf_str}."
    else:
        s2 = "Fundamental data is unavailable — quantitative assessment suppressed."

    # Main risk
    fwd_pe = _n(fin.get('forward_pe'))
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

    fund_val = scores['fundamental']
    fund_line = f"Fundamental      {_score_bar(fund_val)}  {fund_val:>3}" if fund_val is not None else "Fundamental      [DATA N/A]     N/A"
    breakdown = (
        f"```\n"
        f"Signal Strength  {_score_bar(scores['signal'])}  {scores['signal']:>3}\n"
        f"{fund_line}\n"
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

    total_debt = fin.get('total_debt')
    total_cash = fin.get('total_cash')
    net_debt = fin.get('net_debt', 0)

    lines = [
        f"*BALANCE SHEET & CAPITAL RETURN — {symbol}*",
        "```",
        "BALANCE SHEET SUMMARY",
        "-" * 40,
        f"{'Total Cash & Equiv:':<22} {_fmt_mc(total_cash):>15}",
        f"{'Total Debt:':<22} {_fmt_mc(total_debt):>15}",
        f"{'Net Debt:':<22} {_fmt_mc(net_debt):>15}",
        f"{'Net Debt/EBITDA:':<22} {_fmt_x(fin.get('net_debt_ebitda')):>15}",
        f"{'Interest Coverage:':<22} {_fmt_x(fin.get('interest_coverage')):>15}",
        f"{'Debt/Equity:':<22} {_fmt_x(fin.get('debt_equity')):>15}",
        f"{'Current Ratio:':<22} {_fmt_x(fin.get('current_ratio')):>15}",
        "",
        "CAPITAL RETURN",
        "-" * 40,
        f"{'Dividend Yield:':<22} {_fmt_pct(fin.get('dividend_yield')):>15}",
        f"{'Payout Ratio:':<22} {_fmt_pct(fin.get('payout_ratio')):>15}",
        f"{'Buyback Yield:':<22} {_fmt_pct(fin.get('buyback_yield')):>15}",
        "```",
    ]

    # Narrative
    nd_ebitda = _n(fin.get('net_debt_ebitda'))
    ic = _n(fin.get('interest_coverage'))

    if nd_ebitda < 1 and ic > 10:
        health = "Balance sheet is strong"
        detail = f"Net debt is only {nd_ebitda:.1f}x EBITDA with {ic:.0f}x interest coverage."
    elif nd_ebitda < 3:
        health = "Balance sheet is adequate"
        detail = f"Leverage at {nd_ebitda:.1f}x EBITDA is manageable."
    else:
        health = "Balance sheet warrants caution"
        detail = f"Leverage at {nd_ebitda:.1f}x EBITDA is elevated."

    div_y = fin.get('dividend_yield') or 0
    bb_y = fin.get('buyback_yield') or 0
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

    # DCF — skip if disabled
    if not dcf.get('_dcf_disabled') and dcf.get('base', 0) > 0:
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
    target = _n(fin.get('target_mean'))
    t_high = _n(fin.get('target_high'))
    t_low = _n(fin.get('target_low'))
    if target > 0:
        upside = (target / price - 1) * 100 if price > 0 else 0
        lines.append("")
        lines.append(
            f"_Analyst Consensus Target: ${target:.2f} ({_fmt_pct(upside, plus=True)}) "
            f"| Range: ${t_low:.2f} - ${t_high:.2f} "
            f"| {fin.get('num_analysts', 0)} analysts_"
        )

    # Narrative
    fwd_pe = _n(fin.get('forward_pe'))
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
        'market_cap': fin.get('market_cap'),
        'revenue_growth': fin.get('revenue_growth'),
        'profit_margin': fin.get('net_margin'),
        'forward_pe': fin.get('forward_pe'),
        'roe': fin.get('roe'),
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
    rev_rank = sorted(all_companies, key=lambda x: _n(x.get('revenue_growth')), reverse=True)
    margin_rank = sorted(all_companies, key=lambda x: _n(x.get('profit_margin')), reverse=True)
    pe_rank = sorted(all_companies, key=lambda x: _n(x.get('forward_pe')) if _n(x.get('forward_pe')) > 0 else 999)

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
    n_analysts = _n(fin.get('num_analysts'))
    target = _n(fin.get('target_mean'))
    t_high = _n(fin.get('target_high'))
    t_low = _n(fin.get('target_low'))
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

    fwd_pe = _n(fin.get('forward_pe'))
    if fwd_pe > 25:
        risks.append(('Valuation Compression', 'Medium' if fwd_pe < 35 else 'High', 'High',
                       f"At {fwd_pe:.0f}x forward, any growth miss triggers P/E contraction"))

    rg = _n(fin.get('revenue_growth'))
    if rg < 3 and fin.get('revenue_growth') is not None:
        risks.append(('Revenue Stagnation', 'Medium', 'High',
                       f"Growth at {rg:.1f}% is below market expectations"))

    de = _n(fin.get('debt_equity'))
    if de > 2:
        risks.append(('Leverage Risk', 'Low', 'High',
                       f"Debt/equity of {de:.1f}x exposes the company in a downturn"))

    vix = summary.get('vix', 20)
    if vix > 22:
        risks.append(('Volatility Risk', 'Medium', 'Medium',
                       f"VIX at {vix:.1f} signals elevated uncertainty"))

    nm = _n(fin.get('net_margin'))
    eg = _n(fin.get('earnings_growth'))
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

    rg = _n(fin.get('revenue_growth'))
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

    bb = _n(fin.get('buyback_yield'))
    dy = _n(fin.get('dividend_yield'))
    if bb + dy > 2:
        catalysts.append(('Capital Return', 'Very High', 'MEDIUM',
                          f"{_fmt_pct(bb + dy)} total yield — structural EPS support"))

    nm = _n(fin.get('net_margin'))
    om = _n(fin.get('operating_margin'))
    if om > 20 and nm > 15:
        catalysts.append(('Margin Strength', 'High', 'MEDIUM',
                          f"Operating at {om:.1f}% margin — pricing power intact"))

    fcf = _n(fin.get('free_cash_flow'))
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
        f"Trade Quality: {tq:.4f}  |  Gate: {gate:.2f}  |  DC: {dc:.0f}%",
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

    # THE FINAL WORD — V12Owner's Perspective
    scores = _compute_v8_scores(summary, v8_data)
    v9 = v8_data.get('v9_scores') or _compute_v9_owner_scores(summary, v8_data)
    verdict = v9['v9_decision']

    sma50 = tech.get('sma50', 0) or summary.get('sma50', 0)
    sma200 = tech.get('sma200', 0) or summary.get('sma200', 0)

    lines.append("")
    lines.append(f"```")
    lines.append(f"{'=' * 50}")
    lines.append(f"  THE FINAL WORD ON {symbol}")
    lines.append(f"  (V12Owner's Perspective)")
    lines.append(f"{'=' * 50}")
    lines.append("")

    # V12Decision
    lines.append(f"OWNER DECISION: {verdict}")
    lines.append(f"  {v9['decision_reason']}")
    lines.append("")

    # Business case — P3/P6: guard None scores
    lines.append("BUSINESS CASE:")
    bq = v9['business_quality']
    moat = v9['moat_durability']
    ca = v9['capital_allocation']
    mos = v9['mos_pct']
    iv = v9['intrinsic_value_base']
    bq_s = f"{bq:.1f}" if bq is not None else "N/A"
    moat_s = f"{moat:.1f}" if moat is not None else "N/A"
    ca_s = f"{ca:.1f}" if ca is not None else "N/A"
    lines.append(f"  Quality: {bq_s}/5  Moat: {moat_s}/5  CapAlloc: {ca_s}/5")
    if iv > 0 and mos is not None:
        lines.append(f"  Intrinsic Value: ${iv:.2f}  |  MOS: {mos:+.1f}%")
    elif iv > 0:
        lines.append(f"  Intrinsic Value: ${iv:.2f}  |  MOS: N/A")
    lines.append("")

    # Quant overlay
    quant_verdict = _v8_verdict_label(scores['composite'])
    lines.append("QUANT ENGINE OVERLAY:")
    lines.append(f"  Signal: {quant_verdict} ({scores['composite']}/100)")
    lines.append(f"  Regime: {regime} | VIX: {summary.get('vix', 0):.1f}")

    # Conflict note — P6: guard None mos
    c_raw = summary.get('composite_raw', 0)
    req_mos = v9.get('required_mos')
    if mos is not None and req_mos is not None:
        if c_raw > 30 and mos < 0:
            lines.append("")
            lines.append("  NOTE: Engine bullish but overvalued.")
            lines.append("  Patience over momentum.")
        elif c_raw < -30 and mos > req_mos * 100:
            lines.append("")
            lines.append("  NOTE: Engine bearish but undervalued.")
            lines.append("  Temporary weakness in durable business.")

    # Tactical levels (secondary)
    lines.append("")
    lines.append("TACTICAL LEVELS (if deploying capital):")
    if verdict == 'BUY':
        if sma50 > 0:
            lines.append(f"  Entry:  ${sma50:.0f} area (50d MA pullback)")
        if stop > 0:
            lines.append(f"  Stop:   ${stop:.2f}")
        if tp and tp[0] > 0:
            lines.append(f"  Target: ${tp[0]:.2f} - ${tp[1]:.2f}")
    elif verdict in ('WATCH', 'RESEARCH'):
        if iv > 0 and req_mos is not None:
            needed = iv * (1 - req_mos)
            lines.append(f"  Buy below: ~${needed:.0f} (MOS threshold)")
        if sma50 > 0:
            lines.append(f"  Watch: ${sma50:.0f} (50d MA)")
    elif verdict == 'PASS':
        lines.append(f"  No deployment recommended")
    else:
        if stop > 0:
            lines.append(f"  Protect capital: stop at ${stop:.2f}")

    # Position sizing by conviction — P3: guard None
    conv = v9['conviction']
    if conv is not None:
        if conv >= 80:
            size_note = "Candidate for top-10 holding"
        elif conv >= 60:
            size_note = "Meaningful but not core position"
        elif conv >= 40:
            size_note = "Opportunistic / small allocation"
        else:
            size_note = "Insufficient conviction for capital deployment"
        lines.append(f"  Sizing: {size_note} (conviction {conv}/100)")
    else:
        lines.append(f"  Sizing: Data insufficient for conviction assessment")

    lines.append("")
    lines.append(f"{'=' * 50}")
    now = datetime.now().strftime('%b %d, %Y %I:%M %p ET')
    lines.append(f"Generated: {now}")
    lines.append("This is model output, NOT financial advice.")
    lines.append("Stocks are fractional ownership in businesses.")
    lines.append(f"{'=' * 50}")
    lines.append("```")

    return "\n".join(lines)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def format_v8_report(summary, v8_data):
    """
    Generate complete V12report as list of Slack messages.
    11 messages: Owner Assessment + 10 full-spectrum analysis sections.

    V12leads with business owner intelligence (Buffett-aligned),
    then provides full quant detail for tactical overlay.

    Args:
        summary: dict from run_atlas() — engine output
        v8_data: dict from fetch_v8_data() — extended data

    Returns:
        list of strings, each under 4000 chars
    """
    # Compute and attach V12scores for downstream consumers (web dashboard, Q&A)
    v9_scores = _compute_v9_owner_scores(summary, v8_data)
    v8_data['v9_scores'] = v9_scores

    # Stage 5: Reconciliation checks
    recon_errors = _reconciliation_checks(summary, v8_data, v9_scores)
    v8_data['reconciliation_errors'] = recon_errors

    messages = [
        _section_owner_assessment(summary, v8_data),   # V12: Owner's view first
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
