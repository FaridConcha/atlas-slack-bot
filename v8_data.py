#!/usr/bin/env python3
"""
ATLAS V12+ Phase 3 — Extended Data Fetcher + Monte Carlo DCF
Fetches additional data beyond what the ATLAS engine needs.
Used by v8_report.py to generate the full 10-section report.
V12+: Added Monte Carlo DCF (N=1000), sensitivity analysis, DCF kernel extraction.
Phase 1: Fat-tailed growth (Student's t, sector-specific ν), WACC σ regime scaling.
Phase 2: prob_permanent_loss, cross-sensitivity ∂²IV/(∂WACC×∂Growth).

Data sources:
  - yfinance: Company info, financials, earnings, peers, news, technicals
  - yfinance batch: Market dashboard, sector ETFs
  - FRED (optional): Economic indicators
  - Computed: Technical indicators (RSI, MACD, Bollinger, etc.)
"""

import time
import zlib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from data_fetcher import resolve_price
from valuation_config import (
    CONFIG, SECTOR_BETA_BOUNDS, DEFAULT_BETA_BOUNDS, SECTOR_WACC_FLOORS,
    BetaPath, SectorProvenance,
    SECTOR_GROWTH_SIGMA, MC_CORRELATION_MATRIX, MonteCarloConfig,
    SECTOR_TAIL_DF, REGIME_VARIANCE_MULTIPLIER,
)



# ============================================================================
# SECTOR PEER MAPPING
# ============================================================================

SECTOR_PEERS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSM', 'AVGO'],
    'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'],
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
    'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'PM', 'MO'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
    'Industrials': ['CAT', 'BA', 'HON', 'UPS', 'RTX', 'DE', 'GE', 'LMT'],
    'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL'],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'PSA', 'DLR'],
    'Basic Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'NUE'],
}

SECTOR_ETFS = {
    'Technology': ('XLK', 'Technology'),
    'Financial Services': ('XLF', 'Financials'),
    'Healthcare': ('XLV', 'Healthcare'),
    'Consumer Cyclical': ('XLY', 'Consumer Disc.'),
    'Consumer Defensive': ('XLP', 'Consumer Staples'),
    'Energy': ('XLE', 'Energy'),
    'Industrials': ('XLI', 'Industrials'),
    'Communication Services': ('XLC', 'Comm. Services'),
    'Utilities': ('XLU', 'Utilities'),
    'Real Estate': ('XLRE', 'Real Estate'),
    'Basic Materials': ('XLB', 'Materials'),
}

MARKET_TICKERS = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Dow Jones': '^DJI',
    'Russell 2000': '^RUT',
}

BOND_TICKERS = {
    '10-Year Treasury': '^TNX',
    '2-Year Treasury': '^IRX',
}

COMMODITY_TICKERS = {
    'Gold': 'GC=F',
    'Crude Oil (WTI)': 'CL=F',
    'Natural Gas': 'NG=F',
}

FX_TICKERS = {
    'USD Index (DXY)': 'DX-Y.NYB',
    'EUR/USD': 'EURUSD=X',
    'USD/JPY': 'JPY=X',
}

CRYPTO_TICKERS = {
    'Bitcoin': 'BTC-USD',
}

# Simple sentiment keywords
POSITIVE_KEYWORDS = [
    'beat', 'surge', 'rally', 'upgrade', 'buy', 'bullish', 'record', 'growth',
    'strong', 'outperform', 'gain', 'profit', 'raise', 'positive', 'opportunity',
    'momentum', 'recovery', 'expand', 'accelerat', 'boost', 'upbeat', 'optimis',
]
NEGATIVE_KEYWORDS = [
    'miss', 'decline', 'downgrade', 'sell', 'bearish', 'weak', 'loss', 'risk',
    'concern', 'drop', 'fall', 'cut', 'warning', 'layoff', 'slowdown', 'fine',
    'lawsuit', 'investigation', 'recession', 'plunge', 'slump', 'disappoint',
]


# ============================================================================
# TECHNICAL INDICATOR COMPUTATIONS
# ============================================================================

def _ema(data, period):
    """Exponential moving average."""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    result[period - 1] = np.mean(data[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def _compute_rsi(closes, period=14):
    """RSI(14)."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _compute_macd(closes):
    """MACD(12,26,9). Returns (macd_line, signal, histogram)."""
    if len(closes) < 35:
        return 0, 0, 0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    # Signal line is EMA(9) of MACD line, starting from index 25
    valid_macd = macd_line[25:]
    if len(valid_macd) < 9:
        return round(float(macd_line[-1]), 3), 0, 0
    signal = _ema(valid_macd, 9)
    macd_val = float(macd_line[-1])
    sig_val = float(signal[-1]) if not np.isnan(signal[-1]) else 0
    hist_val = macd_val - sig_val
    return round(macd_val, 3), round(sig_val, 3), round(hist_val, 3)


def _compute_bollinger(closes, period=20, num_std=2):
    """Bollinger Bands(20,2). Returns (upper, middle, lower)."""
    if len(closes) < period:
        return 0, 0, 0
    middle = float(np.mean(closes[-period:]))
    std = float(np.std(closes[-period:]))
    upper = round(middle + num_std * std, 2)
    lower = round(middle - num_std * std, 2)
    return upper, round(middle, 2), lower


def _compute_stochastic(highs, lows, closes, period=14, smooth=3):
    """%K and %D Stochastic."""
    if len(closes) < period:
        return 50, 50
    high_period = np.max(highs[-period:])
    low_period = np.min(lows[-period:])
    denom = high_period - low_period
    if denom < 1e-10:
        return 50, 50
    k = 100.0 * (closes[-1] - low_period) / denom
    # Compute %D as average of last smooth %K values
    k_values = []
    for i in range(smooth):
        idx = len(closes) - 1 - i
        if idx < period:
            break
        h = np.max(highs[idx - period + 1:idx + 1])
        lo = np.min(lows[idx - period + 1:idx + 1])
        d = h - lo
        k_values.append(100.0 * (closes[idx] - lo) / d if d > 1e-10 else 50)
    d = np.mean(k_values) if k_values else k
    return round(k, 1), round(d, 1)


def _compute_adx(highs, lows, closes, period=14):
    """ADX(14). Simplified computation."""
    if len(closes) < period * 2:
        return 25.0
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)
        plus_dm_list.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm_list.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

    if len(tr_list) < period:
        return 25.0

    # Smoothed averages
    atr = np.mean(tr_list[-period:])
    plus_di = 100 * np.mean(plus_dm_list[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm_list[-period:]) / atr if atr > 0 else 0
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
    return round(dx, 1)


def _compute_obv(closes, volumes):
    """On-Balance Volume. Returns (current OBV, trend direction)."""
    if len(closes) < 20:
        return 0, 'Flat'
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])

    # Trend: compare last 5 OBV to 20-bar OBV SMA
    recent = np.mean(obv[-5:])
    older = np.mean(obv[-20:])
    if recent > older * 1.02:
        trend = 'Rising'
    elif recent < older * 0.98:
        trend = 'Falling'
    else:
        trend = 'Flat'
    return obv[-1], trend


def _find_swing_levels(highs, lows, closes, window=5, lookback=60):
    """Find recent swing highs and lows for support/resistance."""
    n = min(lookback, len(closes))
    recent_highs = highs[-n:]
    recent_lows = lows[-n:]
    price = closes[-1]

    resistances = []
    supports = []

    for i in range(window, n - window):
        # Swing high
        if recent_highs[i] == max(recent_highs[i - window:i + window + 1]):
            if recent_highs[i] > price:
                resistances.append(round(float(recent_highs[i]), 2))
        # Swing low
        if recent_lows[i] == min(recent_lows[i - window:i + window + 1]):
            if recent_lows[i] < price:
                supports.append(round(float(recent_lows[i]), 2))

    resistances = sorted(set(resistances))[:3]
    supports = sorted(set(supports), reverse=True)[:3]
    return supports, resistances


# ============================================================================
# HELPER: SAFE DATA ACCESS
# ============================================================================

def _safe_fast_info(ticker):
    """Get ticker.fast_info as dict with error handling.

    fast_info is more reliable than info — it uses a lighter API endpoint
    and returns market_cap, shares, yearHigh, yearLow, lastPrice, etc.
    """
    try:
        fi = ticker.fast_info
        if fi is None:
            return {}
        # Convert fast_info to a dict with info-compatible keys
        return {
            'marketCap': getattr(fi, 'market_cap', None),
            'sharesOutstanding': getattr(fi, 'shares', None),
            'currentPrice': getattr(fi, 'last_price', None),
            'previousClose': getattr(fi, 'previous_close', None),
            'fiftyTwoWeekHigh': getattr(fi, 'year_high', None),
            'fiftyTwoWeekLow': getattr(fi, 'year_low', None),
            'fiftyDayAverage': getattr(fi, 'fifty_day_average', None),
            'twoHundredDayAverage': getattr(fi, 'two_hundred_day_average', None),
            '_source': 'fast_info',
        }
    except Exception:
        return {}


def _safe_info(ticker):
    """Get ticker.info with fast_info fallback for critical fields.

    When ticker.info fails (transient API error, rate limiting), critical
    fundamentals like market_cap, shares, 52W range are lost, cascading into
    full analysis suppression. fast_info is a lighter, more reliable endpoint
    that provides these core fields.
    """
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    # If info is empty or missing critical fields, enrich from fast_info
    critical_keys = ('marketCap', 'sharesOutstanding', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow')
    missing_critical = not info or any(info.get(k) is None for k in critical_keys)

    if missing_critical:
        fast = _safe_fast_info(ticker)
        if fast:
            enriched = 0
            for key, val in fast.items():
                if key.startswith('_'):
                    continue
                if val is not None and (info.get(key) is None or info.get(key) == 0):
                    info[key] = val
                    enriched += 1
            if enriched > 0:
                info['_fast_info_enriched'] = True
                info['_fast_info_fields'] = enriched
                print(f"[V12]   fast_info enriched {enriched} missing fields")

    return info


def _score_sentiment(title):
    """Simple keyword-based sentiment scoring."""
    title_lower = title.lower()
    pos = sum(1 for w in POSITIVE_KEYWORDS if w in title_lower)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in title_lower)
    if pos > neg:
        return 'POSITIVE'
    elif neg > pos:
        return 'NEGATIVE'
    return 'NEUTRAL'


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def fetch_v8_data(symbol, fred_api_key=None, regime_variance_mult=1.0):
    """
    Fetch all extended data for V8 report.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL')
        fred_api_key: Optional FRED API key for economic indicators
        regime_variance_mult: WACC sigma scaling from engine regime (default 1.0)

    Returns:
        dict with keys: company, financials, earnings, technicals,
        peers, news, sector, market, economic, institutional, dcf
    """
    print(f"[V12] Fetching extended data for {symbol}...")
    ticker = yf.Ticker(symbol)
    info = _safe_info(ticker)

    # Fetch OHLCV for technicals (1 year)
    try:
        hist = ticker.history(period="1y", interval="1d", prepost=True, timeout=30)
    except Exception:
        hist = None

    print(f"[V12]   Building company info...")
    company = _build_company_info(info, symbol, hist=hist)

    print(f"[V12]   Building financials...")
    financials = _build_financials(ticker, info, hist=hist)

    # Sync fundamentals quality with financial integrity status and serialize
    fq = company.get('_fundamentals_quality')
    if fq:
        fq.data_status = financials.get('_data_status', 'OK')
        fq.data_reasons = financials.get('_data_reasons', [])
        fq.shares_available = financials.get('shares_outstanding') is not None and financials['shares_outstanding'] > 0
        fq.revenue_available = financials.get('revenue_ttm') is not None and financials['revenue_ttm'] > 0
        # Serialize to dict for JSON compatibility
        company['_fundamentals_quality'] = {
            'market_cap_available': fq.market_cap_available,
            'shares_available': fq.shares_available,
            'revenue_available': fq.revenue_available,
            'beta_available': fq.beta_available,
            'sector_available': fq.sector_available,
            'price_available': fq.price_available,
            'high_52w_available': fq.high_52w_available,
            'low_52w_available': fq.low_52w_available,
            'beta_defaulted': fq.beta_defaulted,
            'sector_defaulted': fq.sector_defaulted,
            'data_status': fq.data_status,
            'data_reasons': fq.data_reasons,
            'report_mode': fq.report_mode.value,
        }

    print(f"[V12]   Building earnings history...")
    earnings = _build_earnings(ticker)

    print(f"[V12]   Computing technicals...")
    technicals = _build_technicals(hist, info)

    print(f"[V12]   Fetching peer data...")
    peers = _build_peers(symbol, info)

    print(f"[V12]   Fetching news...")
    news = _build_news(ticker)

    print(f"[V12]   Fetching sector performance...")
    sector = _build_sector(info)

    print(f"[V12]   Building market dashboard...")
    market = _build_market_dashboard()

    print(f"[V12]   Fetching economic indicators...")
    economic = _build_economic(fred_api_key)

    print(f"[V12]   Building institutional data...")
    institutional = _build_institutional(info)

    print(f"[V12]   Computing DCF model...")
    _price_for_dcf = company.get('price')
    dcf = _build_dcf(info, financials, sector=info.get('sector'),
                     ticker=symbol, price=_price_for_dcf,
                     regime_variance_mult=regime_variance_mult)

    print(f"[V12] Extended data complete for {symbol}")

    return {
        'company': company,
        'financials': financials,
        'earnings': earnings,
        'technicals': technicals,
        'peers': peers,
        'news': news,
        'sector': sector,
        'market': market,
        'economic': economic,
        'institutional': institutional,
        'dcf': dcf,
    }


# ============================================================================
# COMPANY INFO
# ============================================================================

def _build_company_info(info, symbol, hist=None):
    """Extract company overview from yfinance info.

    Stage 5.1: Missing fields propagate as None, never coerced to 0/1.0/'Unknown'.
    The _fundamentals_quality object tracks which fields are real vs defaulted.
    """
    from valuation_config import FundamentalsQuality, CONFIG

    price = resolve_price(info, hist)

    # --- Raw extraction: None if truly missing ---
    # Beta/52w: treat 0 as missing (yfinance returns 0 for unavailable)
    raw_beta = _safe_num(info.get('beta'))
    if raw_beta is not None and raw_beta <= 0:
        raw_beta = None
    raw_sector = info.get('sector')
    raw_industry = info.get('industry')
    raw_mc = _safe_num(info.get('marketCap'), min_val=0)
    raw_employees = _safe_num(info.get('fullTimeEmployees'), min_val=0)
    raw_52h = _safe_num(info.get('fiftyTwoWeekHigh'), min_val=0)
    if raw_52h is not None and raw_52h <= 0:
        raw_52h = None
    raw_52l = _safe_num(info.get('fiftyTwoWeekLow'), min_val=0)
    if raw_52l is not None and raw_52l <= 0:
        raw_52l = None

    # Fallback: derive 52W range from price history if info doesn't have it
    if (raw_52h is None or raw_52l is None) and hist is not None and not getattr(hist, 'empty', True):
        try:
            if raw_52h is None:
                raw_52h = float(hist['High'].max())
            if raw_52l is None:
                raw_52l = float(hist['Low'].min())
        except Exception:
            pass

    # Sector: treat empty string or 'Unknown' from yfinance as missing
    sector_available = bool(raw_sector and raw_sector not in ('', 'Unknown', 'N/A'))
    industry_available = bool(raw_industry and raw_industry not in ('', 'Unknown', 'N/A'))

    # Build quality tracker
    fq = FundamentalsQuality(
        market_cap_available=raw_mc is not None and raw_mc > 0,
        shares_available=True,  # updated later by _build_financials
        revenue_available=True,  # updated later by _build_financials
        beta_available=raw_beta is not None and raw_beta > 0,
        sector_available=sector_available,
        price_available=price is not None and price > 0,
        high_52w_available=raw_52h is not None and raw_52h > 0,
        low_52w_available=raw_52l is not None and raw_52l > 0,
    )

    return {
        'symbol': symbol,
        'name': info.get('longName', '') or info.get('shortName', symbol),
        'sector': raw_sector if sector_available else None,
        'industry': raw_industry if industry_available else None,
        'market_cap': raw_mc,
        'employees': int(raw_employees) if raw_employees else None,
        'beta': raw_beta,
        'price': price,
        'fifty_two_week_high': raw_52h,
        'fifty_two_week_low': raw_52l,
        '_fundamentals_quality': fq,
    }


# ============================================================================
# NONE-SAFE DATA HELPERS
# ============================================================================

def _safe_num(val, min_val=None, max_val=None):
    """Return a numeric value or None. Never coerces missing data to 0."""
    if val is None:
        return None
    try:
        v = float(val)
        if v != v:  # NaN check
            return None
        if min_val is not None and v < min_val:
            return None
        if max_val is not None and v > max_val:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_pct(val):
    """Convert a decimal (0.15) to percentage (15.0), or return None."""
    v = _safe_num(val)
    if v is None:
        return None
    return v * 100


def _check_fundamental_integrity(mc, shares, revenue, price):
    """
    Validate core fundamentals. Returns (status, reasons).
    Status: 'OK', 'DEGRADED', 'INVALID'
    """
    reasons = []

    # Market cap must exist and be >100M for public equities
    if mc is None or mc <= 0:
        reasons.append('market_cap_missing')
    elif mc < 100e6:
        reasons.append('market_cap_implausible')

    # Shares must exist and be >1M
    if shares is None or shares <= 0:
        reasons.append('shares_missing')
    elif shares < 1e6:
        reasons.append('shares_implausible')

    # Revenue should exist and be >0 for operating companies
    if revenue is None or revenue <= 0:
        reasons.append('revenue_missing')

    # Cross-check: mc ≈ shares * price within 20%
    if mc and shares and price and mc > 0 and shares > 0 and price > 0:
        implied_mc = shares * price
        if abs(implied_mc - mc) / mc > 0.20:
            reasons.append('mc_shares_mismatch')

    if any(r in reasons for r in ('market_cap_missing', 'shares_missing')):
        return ('INVALID', reasons)
    elif any(r in reasons for r in ('market_cap_implausible', 'shares_implausible', 'revenue_missing')):
        return ('DEGRADED', reasons)
    elif reasons:
        return ('DEGRADED', reasons)
    else:
        return ('OK', [])


# ============================================================================
# DATA SANITY HELPERS
# ============================================================================

def _sanitize_yield(info, price):
    """
    Normalize dividend yield to canonical internal representation.

    Returns (display_yield_pct, anomaly_info_dict).
      display_yield_pct: float (e.g. 1.36 meaning 1.36%) or None if anomalous.
      anomaly_info: provenance dict with raw values, unit determination, and flags.

    Pipeline:
      1. Capture raw provider value exactly as received.
      2. Determine unit via cross-check with dividend_rate/price when available.
      3. Normalize to canonical decimal (e.g. 0.0136 for 1.36%).
      4. Out-of-range evaluated AFTER normalization (decimal > 0.25 → flag).
      5. Convert to display percent for return value.
    """
    raw_dy = info.get('dividendYield')
    div_rate = info.get('dividendRate')
    anomaly = {
        'raw_provider_value': raw_dy,
        'dividend_rate': div_rate,
        'provider': 'yfinance',
        'unit_hint': None,
        'normalized_decimal': None,
        'normalized_pct': None,
        'expected_yield_decimal': None,
        'expected_yield_pct': None,
        'reason_codes': [],
    }

    if raw_dy is None and div_rate is None:
        anomaly['reason_codes'].append('DIV_YIELD_MISSING')
        anomaly['reason_codes'].append('DIV_PER_SHARE_MISSING')
        return None, anomaly

    raw_dy = raw_dy or 0
    div_rate = div_rate or 0

    # ── Step 1: Compute expected yield from dividend_rate / price ──
    expected_decimal = None
    if price > 0 and div_rate > 0:
        expected_decimal = div_rate / price
        anomaly['expected_yield_decimal'] = round(expected_decimal, 6)
        anomaly['expected_yield_pct'] = round(expected_decimal * 100, 3)

    # ── Step 2: Determine raw_dy unit ──
    # Compare two hypotheses:
    #   H_decimal: raw_dy IS a decimal (e.g. 0.0136)  → normalized = raw_dy
    #   H_percent: raw_dy IS percent points (e.g. 1.36) → normalized = raw_dy / 100
    # Pick whichever is closer to expected_decimal (the ground truth).
    if expected_decimal is not None and raw_dy > 0:
        error_if_decimal = abs(raw_dy - expected_decimal)
        error_if_percent = abs(raw_dy / 100 - expected_decimal)

        if error_if_percent < error_if_decimal:
            # raw is percent points (e.g. 1.36 meaning 1.36%)
            normalized_decimal = raw_dy / 100
            anomaly['unit_hint'] = 'percent_points'
        else:
            # raw is decimal (e.g. 0.0136 meaning 1.36%)
            normalized_decimal = raw_dy
            anomaly['unit_hint'] = 'decimal'

        # Cross-check: flag if normalized still diverges from expected
        if expected_decimal > 0:
            rel_error = abs(normalized_decimal - expected_decimal) / expected_decimal
            if rel_error > 0.20:
                anomaly['reason_codes'].append('DIV_YIELD_INCONSISTENT_WITH_DIV_RATE')
    elif raw_dy > 0:
        # No div_rate for cross-check — use heuristic
        # yfinance typically returns decimal; values > 0.50 are implausible as decimal
        if raw_dy > 0.50:
            normalized_decimal = raw_dy / 100
            anomaly['unit_hint'] = 'percent_points_heuristic'
        else:
            normalized_decimal = raw_dy
            anomaly['unit_hint'] = 'decimal_assumed'
    else:
        if div_rate == 0:
            anomaly['normalized_decimal'] = 0.0
            anomaly['normalized_pct'] = 0.0
            return 0, anomaly  # Company doesn't pay dividends
        normalized_decimal = 0
        anomaly['reason_codes'].append('DIV_YIELD_ZERO_BUT_RATE_EXISTS')

    anomaly['normalized_decimal'] = round(normalized_decimal, 6)
    anomaly['normalized_pct'] = round(normalized_decimal * 100, 3)

    # ── Step 3: Out-of-range evaluated AFTER normalization ──
    if normalized_decimal > 0.25:
        anomaly['reason_codes'].append('DIV_YIELD_OUT_OF_RANGE')
        anomaly['final_value'] = None
        anomaly['rejected_value_pct'] = round(normalized_decimal * 100, 2)
        return None, anomaly

    display_pct = round(normalized_decimal * 100, 2)
    anomaly['final_value'] = display_pct
    return display_pct, anomaly


def _sanitize_payout(info):
    """Sanitize payout ratio — cap at 200%. yfinance returns decimal (0.35 = 35%)."""
    raw_pr = _safe_num(info.get('payoutRatio'))
    if raw_pr is None:
        return None
    # yfinance returns decimal: 0.35 = 35%, 1.5 = 150%, 3.5 = 350%
    payout = raw_pr * 100
    if payout > 200:
        return None
    if payout < 0:
        return None
    return round(payout, 1)


def _validate_financials(fin, info, price):
    """Post-hoc sanity checks. Nullify anomalous values."""
    mc = fin.get('market_cap') or 0
    shares = fin.get('shares_outstanding') or 0

    # 1. Yield sanity: dividend_yield > 25% -> None
    if fin.get('dividend_yield') is not None and fin['dividend_yield'] > 25:
        fin['dividend_yield'] = None

    # 2. Payout ratio sanity: > 200% -> None
    if fin.get('payout_ratio') is not None and fin['payout_ratio'] > 200:
        fin['payout_ratio'] = None

    # 3. Share count cross-check: shares * price ~ market_cap (+-10%)
    if shares > 0 and price > 0 and mc > 0:
        implied_mc = shares * price
        if abs(implied_mc - mc) / mc > 0.10:
            fin['_shares_warning'] = True

    # 4. EV consistency: EV ~ MC + net_debt (+-15%)
    reported_ev = _safe_num(info.get('enterpriseValue')) or 0
    computed_ev = mc + (fin.get('net_debt') or 0)
    if reported_ev > 0 and computed_ev > 0:
        if abs(reported_ev - computed_ev) / reported_ev > 0.15:
            fin['_ev_warning'] = True

    # 5. PEG: if 0, set to None (missing growth, not zero growth)
    if fin.get('peg_ratio') == 0:
        fin['peg_ratio'] = None

    return fin


# ============================================================================
# DETAILED FINANCIALS
# ============================================================================

def _build_financials(ticker, info, hist=None):
    """Comprehensive financial metrics with None-safe data propagation."""
    price = resolve_price(info, hist)

    # --- Core fundamentals: use None for missing, NEVER coerce to 0/1 ---
    mc = _safe_num(info.get('marketCap'), min_val=0)
    shares = _safe_num(info.get('sharesOutstanding'), min_val=0)
    revenue = _safe_num(info.get('totalRevenue'))
    net_income = _safe_num(info.get('netIncomeToCommon'))
    fcf = _safe_num(info.get('freeCashflow'))
    total_debt = _safe_num(info.get('totalDebt'), min_val=0)
    total_cash = _safe_num(info.get('totalCash'), min_val=0)
    ebitda = _safe_num(info.get('ebitda'))

    # --- Secondary fallbacks from financial statements when info is sparse ---
    if revenue is None or net_income is None or fcf is None:
        try:
            inc = ticker.income_stmt
            if inc is not None and not inc.empty and inc.shape[1] > 0:
                latest = inc.iloc[:, 0]
                if revenue is None:
                    revenue = _safe_num(latest.get('Total Revenue'))
                if net_income is None:
                    net_income = _safe_num(latest.get('Net Income'))
        except Exception:
            pass
    if fcf is None:
        try:
            cf = ticker.cash_flow
            if cf is not None and not cf.empty and cf.shape[1] > 0:
                latest_cf = cf.iloc[:, 0]
                op_cf = _safe_num(latest_cf.get('Operating Cash Flow'))
                capex = _safe_num(latest_cf.get('Capital Expenditure'))
                if op_cf is not None:
                    fcf = op_cf + (capex or 0)  # capex is negative
        except Exception:
            pass

    # Fallback: derive shares from mc/price if sharesOutstanding missing
    if shares is None and mc is not None and price > 0:
        shares = mc / price

    # Fallback: derive mc from shares*price if marketCap missing
    if mc is None and shares is not None and price > 0:
        mc = shares * price

    # --- Fundamental integrity check ---
    data_status, data_reasons = _check_fundamental_integrity(mc, shares, revenue, price)

    # --- Derived metrics: only compute when inputs are valid ---
    # Interest coverage
    om_raw = _safe_num(info.get('operatingMargins'))
    operating_income = (revenue * om_raw) if (revenue and om_raw) else None
    interest_expense = (total_debt * 0.04) if total_debt else None
    if operating_income and interest_expense and interest_expense > 0:
        interest_coverage = round(operating_income / interest_expense, 1)
    else:
        interest_coverage = None

    # FCF yield: requires valid mc
    if fcf is not None and mc and mc > 100e6:
        fcf_yield = round(fcf / mc * 100, 2)
    else:
        fcf_yield = None

    # Net debt
    td = total_debt or 0
    tc = total_cash or 0
    net_debt = td - tc

    # Net debt / EBITDA
    if ebitda and ebitda > 0:
        net_debt_ebitda = round(net_debt / ebitda, 2)
    else:
        net_debt_ebitda = None

    # Buyback yield: requires valid mc
    buyback_yield = None
    if mc and mc > 100e6:
        try:
            cf = ticker.cashflow
            if cf is not None and not cf.empty and 'Repurchase Of Capital Stock' in cf.index:
                buyback = abs(float(cf.loc['Repurchase Of Capital Stock'].iloc[0] or 0))
                raw_bb = buyback / mc * 100
                buyback_yield = round(raw_bb, 2) if raw_bb <= 25 else None
        except Exception:
            buyback_yield = None

    # Debt/equity — yfinance returns debtToEquity as percentage (150.0 = 1.5x)
    de_raw = _safe_num(info.get('debtToEquity'), min_val=0)
    debt_equity = round(de_raw / 100, 2) if de_raw is not None else None
    if debt_equity is not None and debt_equity > 10.0:
        debt_equity = None  # Anomalous: D/E > 10x — likely data error

    # Dividend yield with anomaly provenance
    div_yield_val, div_anomaly = _sanitize_yield(info, price)

    result = {
        'revenue_ttm': revenue,
        'revenue_growth': _safe_pct(info.get('revenueGrowth')),
        'earnings_growth': _safe_pct(info.get('earningsGrowth')),
        'net_income_ttm': net_income,
        'ebitda': ebitda,
        'gross_margin': _safe_pct(info.get('grossMargins')),
        'operating_margin': _safe_pct(info.get('operatingMargins')),
        'net_margin': _safe_pct(info.get('profitMargins')),
        'ebitda_margin': _safe_pct(info.get('ebitdaMargins')),
        'roe': _safe_pct(info.get('returnOnEquity')),
        'roa': _safe_pct(info.get('returnOnAssets')),
        'free_cash_flow': fcf,
        'fcf_yield': fcf_yield,
        'total_debt': total_debt,
        'total_cash': total_cash,
        'net_debt': net_debt,
        'net_debt_ebitda': net_debt_ebitda,
        'debt_equity': debt_equity,
        'current_ratio': _safe_num(info.get('currentRatio'), min_val=0),
        'interest_coverage': interest_coverage,
        'dividend_yield': div_yield_val,
        '_dividend_anomaly': div_anomaly,
        'payout_ratio': _sanitize_payout(info),
        'buyback_yield': buyback_yield,
        'trailing_pe': _safe_num(info.get('trailingPE'), min_val=0),
        'forward_pe': _safe_num(info.get('forwardPE'), min_val=0),
        'peg_ratio': _safe_num(info.get('pegRatio'), min_val=0),
        'price_to_book': _safe_num(info.get('priceToBook'), min_val=0),
        'price_to_sales': _safe_num(info.get('priceToSalesTrailing12Months'), min_val=0),
        'ev_ebitda': _safe_num(info.get('enterpriseToEbitda'), min_val=0),
        'ev_revenue': _safe_num(info.get('enterpriseToRevenue'), min_val=0),
        'trailing_eps': _safe_num(info.get('trailingEps')),
        'forward_eps': _safe_num(info.get('forwardEps')),
        'revenue_per_share': _safe_num(info.get('revenuePerShare'), min_val=0),
        'book_value': _safe_num(info.get('bookValue')),
        'shares_outstanding': shares,
        'market_cap': mc,
        'target_mean': _safe_num(info.get('targetMeanPrice'), min_val=0),
        'target_high': _safe_num(info.get('targetHighPrice'), min_val=0),
        'target_low': _safe_num(info.get('targetLowPrice'), min_val=0),
        'num_analysts': _safe_num(info.get('numberOfAnalystOpinions'), min_val=0) or 0,
        'recommendation': info.get('recommendationKey', 'none'),
        # Data integrity fields
        '_data_status': data_status,
        '_data_reasons': data_reasons,
    }

    return _validate_financials(result, info, price)


# ============================================================================
# EARNINGS HISTORY
# ============================================================================

def _build_earnings(ticker):
    """Earnings surprise history from yfinance."""
    try:
        eh = ticker.earnings_history
        if eh is None or eh.empty:
            return []

        records = []
        for _, row in eh.iterrows():
            eps_est = row.get('epsEstimate', 0) or 0
            eps_act = row.get('epsActual', 0) or 0
            surprise = eps_act - eps_est
            date_val = row.get('quarter', '')
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            records.append({
                'date': date_str,
                'eps_estimate': round(float(eps_est), 2),
                'eps_actual': round(float(eps_act), 2),
                'surprise': round(float(surprise), 2),
                'beat': eps_act > eps_est,
            })
        return records[-8:]  # Last 8 quarters
    except Exception:
        return []


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def _build_technicals(hist, info):
    """Compute all technical indicators from OHLCV data."""
    default = {
        'rsi_14': 50, 'macd_line': 0, 'macd_signal': 0, 'macd_histogram': 0,
        'bollinger_upper': 0, 'bollinger_middle': 0, 'bollinger_lower': 0,
        'stochastic_k': 50, 'stochastic_d': 50, 'adx': 25, 'obv': 0,
        'obv_trend': 'Flat', 'sma20': 0, 'sma50': 0, 'sma200': 0,
        'price': 0, 'atr': 0,
        'fifty_two_week_high': 0, 'fifty_two_week_low': 0,
        'fifty_two_week_position': 50, 'support_levels': [], 'resistance_levels': [],
        'bullish_count': 0, 'bearish_count': 0, 'neutral_count': 0,
    }

    if hist is None or len(hist) < 30:
        default['price'] = resolve_price(info)
        return default

    closes = hist['Close'].values.astype(float)
    highs = hist['High'].values.astype(float)
    lows = hist['Low'].values.astype(float)
    volumes = hist['Volume'].values.astype(float)
    price = float(closes[-1])

    # Moving averages
    sma20 = round(float(np.mean(closes[-20:])), 2) if len(closes) >= 20 else 0
    sma50 = round(float(np.mean(closes[-50:])), 2) if len(closes) >= 50 else 0
    sma200 = round(float(np.mean(closes[-200:])), 2) if len(closes) >= 200 else 0

    # ATR(14)
    if len(closes) >= 15:
        trs = []
        for i in range(1, min(15, len(closes))):
            tr = max(highs[-i] - lows[-i],
                     abs(highs[-i] - closes[-i - 1]),
                     abs(lows[-i] - closes[-i - 1]))
            trs.append(tr)
        atr = round(float(np.mean(trs)), 2)
    else:
        atr = 0

    # RSI
    rsi = _compute_rsi(closes)

    # MACD
    macd_line, macd_signal, macd_hist = _compute_macd(closes)

    # Bollinger
    bb_upper, bb_middle, bb_lower = _compute_bollinger(closes)

    # Stochastic
    stoch_k, stoch_d = _compute_stochastic(highs, lows, closes)

    # ADX
    adx = _compute_adx(highs, lows, closes)

    # OBV
    obv, obv_trend = _compute_obv(closes, volumes)

    # 52-week range
    hi52 = info.get('fiftyTwoWeekHigh', 0) or float(np.max(highs))
    lo52 = info.get('fiftyTwoWeekLow', 0) or float(np.min(lows))
    range_52 = hi52 - lo52
    position_52 = round((price - lo52) / range_52 * 100, 1) if range_52 > 0 else 50

    # Support / Resistance
    supports, resistances = _find_swing_levels(highs, lows, closes)
    # Add MAs as support/resistance
    for ma in [sma20, sma50, sma200]:
        if ma > 0:
            if ma < price:
                supports.append(round(ma, 2))
            else:
                resistances.append(round(ma, 2))
    if hi52 > price:
        resistances.append(round(hi52, 2))
    supports = sorted(set(supports), reverse=True)[:4]
    resistances = sorted(set(resistances))[:4]

    # Count bullish/bearish signals
    bullish = 0
    bearish = 0
    neutral = 0

    # Price vs MAs
    for ma in [sma20, sma50, sma200]:
        if ma > 0:
            if price > ma:
                bullish += 1
            else:
                bearish += 1

    # RSI
    if 40 < rsi < 60:
        neutral += 1
    elif rsi >= 60:
        bullish += 1
    else:
        bearish += 1

    # MACD
    if macd_line > macd_signal:
        bullish += 1
    else:
        bearish += 1

    # MACD Histogram
    if macd_hist > 0:
        bullish += 1
    elif macd_hist < 0:
        bearish += 1
    else:
        neutral += 1

    # Stochastic
    if stoch_k > 80:
        bearish += 1  # Overbought
    elif stoch_k < 20:
        bullish += 1  # Oversold
    else:
        neutral += 1

    # Bollinger position
    if bb_upper > 0 and bb_lower > 0:
        if price > bb_upper:
            bearish += 1  # Stretched
        elif price < bb_lower:
            bullish += 1  # Oversold
        else:
            neutral += 1

    # OBV
    if obv_trend == 'Rising':
        bullish += 1
    elif obv_trend == 'Falling':
        bearish += 1
    else:
        neutral += 1

    # ADX
    if adx > 25:
        neutral += 1  # Strong trend (direction-neutral)
    else:
        neutral += 1

    return {
        'rsi_14': rsi,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_histogram': macd_hist,
        'bollinger_upper': bb_upper,
        'bollinger_middle': bb_middle,
        'bollinger_lower': bb_lower,
        'stochastic_k': stoch_k,
        'stochastic_d': stoch_d,
        'adx': adx,
        'obv': obv,
        'obv_trend': obv_trend,
        'sma20': sma20,
        'sma50': sma50,
        'sma200': sma200,
        'price': price,
        'atr': atr,
        'fifty_two_week_high': round(hi52, 2),
        'fifty_two_week_low': round(lo52, 2),
        'fifty_two_week_position': position_52,
        'support_levels': supports,
        'resistance_levels': resistances,
        'bullish_count': bullish,
        'bearish_count': bearish,
        'neutral_count': neutral,
    }


# ============================================================================
# PEER DATA
# ============================================================================

def _fetch_single_peer(sym):
    """Fetch basic info for one peer ticker."""
    try:
        t = yf.Ticker(sym)
        i = t.info or {}
        return {
            'symbol': sym,
            'name': i.get('shortName', sym),
            'price': resolve_price(i),
            'market_cap': i.get('marketCap', 0) or 0,
            'revenue_growth': round((i.get('revenueGrowth', 0) or 0) * 100, 1),
            'profit_margin': round((i.get('profitMargins', 0) or 0) * 100, 1),
            'forward_pe': round(i.get('forwardPE', 0) or 0, 1),
            'roe': round((i.get('returnOnEquity', 0) or 0) * 100, 1),
        }
    except Exception:
        return None


def _build_peers(symbol, info):
    """Fetch fundamentals for 5-6 peer companies."""
    sector = info.get('sector', '')
    peer_list = SECTOR_PEERS.get(sector, [])

    # Remove the target symbol from peers
    peer_list = [p for p in peer_list if p != symbol.upper()][:6]

    if not peer_list:
        # Fallback: use mega-cap defaults
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        peer_list = [p for p in fallback if p != symbol.upper()][:5]

    # Fetch in parallel
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(_fetch_single_peer, peer_list))
        return [r for r in results if r is not None]
    except Exception:
        return []


# ============================================================================
# NEWS & SENTIMENT
# ============================================================================

def _build_news(ticker):
    """Fetch and score recent news articles.
    Handles both old and new yfinance news formats.
    """
    try:
        news_list = ticker.news
        if not news_list:
            return []

        articles = []
        for article in news_list[:10]:
            # New yfinance format: data nested under 'content'
            content = article.get('content', {}) or {}

            # Title: try content.title first, then top-level
            title = content.get('title', '') or article.get('title', '')
            if not title:
                continue  # Skip articles with no title

            # Publisher: try content.provider, then top-level
            provider = content.get('provider', {}) or {}
            publisher = provider.get('displayName', '') or article.get('publisher', '')

            # Link: try content.clickThroughUrl, content.canonicalUrl, then top-level
            click_url = content.get('clickThroughUrl', {}) or {}
            canon_url = content.get('canonicalUrl', {}) or {}
            link = (click_url.get('url', '') or canon_url.get('url', '')
                    or article.get('link', ''))

            # Date: try content.pubDate (ISO string), then top-level timestamp
            pub_date_str = content.get('pubDate', '')
            pub_time = article.get('providerPublishTime', 0)
            if pub_date_str:
                try:
                    # ISO format: "2024-02-12T14:30:00Z"
                    dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    date_str = dt.strftime('%b %d')
                except Exception:
                    date_str = ''
            elif pub_time:
                date_str = datetime.fromtimestamp(pub_time).strftime('%b %d')
            else:
                date_str = ''

            sentiment = _score_sentiment(title)
            articles.append({
                'title': title,
                'publisher': publisher or 'Unknown',
                'date': date_str,
                'link': link,
                'sentiment': sentiment,
            })
        return articles
    except Exception as e:
        print(f"[V12]   News fetch error: {e}")
        return []


# ============================================================================
# SECTOR PERFORMANCE
# ============================================================================

def _build_sector(info):
    """Sector ETF performance across multiple timeframes."""
    sector = info.get('sector', 'Unknown')
    etf_info = SECTOR_ETFS.get(sector, ('SPY', 'Market'))

    all_sectors = []
    etf_tickers = [v[0] for v in SECTOR_ETFS.values()]
    etf_names = {v[0]: v[1] for v in SECTOR_ETFS.values()}
    etf_sectors = {v[0]: k for k, v in SECTOR_ETFS.items()}

    try:
        data = yf.download(etf_tickers, period="6mo", interval="1d",
                           group_by='ticker', progress=False, timeout=30)

        for etf in etf_tickers:
            try:
                if len(etf_tickers) > 1:
                    etf_data = data[etf]
                else:
                    etf_data = data

                if etf_data is None or etf_data.empty:
                    continue

                closes = etf_data['Close'].dropna()
                if len(closes) < 5:
                    continue

                current = float(closes.iloc[-1])
                ret_1w = round((current / float(closes.iloc[-6]) - 1) * 100, 2) if len(closes) >= 6 else 0
                ret_1m = round((current / float(closes.iloc[-22]) - 1) * 100, 2) if len(closes) >= 22 else 0

                # YTD: find first trading day of current year
                year_start = closes.index[0]
                for idx in closes.index:
                    if hasattr(idx, 'year') and idx.year == datetime.now().year:
                        year_start = idx
                        break
                ytd_start = float(closes.loc[year_start])
                ret_ytd = round((current / ytd_start - 1) * 100, 2) if ytd_start > 0 else 0

                all_sectors.append({
                    'name': etf_names.get(etf, etf),
                    'etf': etf,
                    'sector_key': etf_sectors.get(etf, ''),
                    '1w': ret_1w,
                    '1m': ret_1m,
                    'ytd': ret_ytd,
                })
            except Exception:
                continue

    except Exception:
        pass

    # Sort by YTD performance
    all_sectors.sort(key=lambda x: x.get('ytd', 0), reverse=True)

    # Find the target sector
    target = next((s for s in all_sectors if s['sector_key'] == sector), None)

    return {
        'name': sector,
        'etf': etf_info[0],
        'display_name': etf_info[1],
        'target': target,
        'all_sectors': all_sectors,
    }


# ============================================================================
# MARKET DASHBOARD
# ============================================================================

def _build_market_dashboard():
    """Fetch major indices, bonds, commodities, FX, crypto."""
    result = {'indices': [], 'bonds': [], 'commodities': [], 'fx': [], 'crypto': []}

    # Batch all market tickers
    all_tickers = {}
    all_tickers.update(MARKET_TICKERS)
    all_tickers.update(BOND_TICKERS)
    all_tickers.update(COMMODITY_TICKERS)
    all_tickers.update(FX_TICKERS)
    all_tickers.update(CRYPTO_TICKERS)

    ticker_symbols = list(all_tickers.values())

    try:
        data = yf.download(ticker_symbols, period="6mo", interval="1d",
                           group_by='ticker', progress=False, timeout=30)

        def _get_returns(sym):
            try:
                if len(ticker_symbols) > 1:
                    d = data[sym]
                else:
                    d = data
                if d is None or d.empty:
                    return 0, 0, 0, 0
                closes = d['Close'].dropna()
                if len(closes) < 2:
                    return 0, 0, 0, 0
                current = float(closes.iloc[-1])
                r1d = round((current / float(closes.iloc[-2]) - 1) * 100, 2) if len(closes) >= 2 else 0
                r1m = round((current / float(closes.iloc[-22]) - 1) * 100, 2) if len(closes) >= 22 else 0
                # YTD
                year_start = closes.index[0]
                for idx in closes.index:
                    if hasattr(idx, 'year') and idx.year == datetime.now().year:
                        year_start = idx
                        break
                ytd_start = float(closes.loc[year_start])
                r_ytd = round((current / ytd_start - 1) * 100, 2) if ytd_start > 0 else 0
                return current, r1d, r1m, r_ytd
            except Exception:
                return 0, 0, 0, 0

        # Indices
        for name, sym in MARKET_TICKERS.items():
            price, r1d, r1m, r_ytd = _get_returns(sym)
            result['indices'].append({
                'name': name, 'ticker': sym, 'price': round(price, 2),
                '1d': r1d, '1m': r1m, 'ytd': r_ytd,
            })

        # Bonds (yields, not prices - return raw values)
        for name, sym in BOND_TICKERS.items():
            try:
                if len(ticker_symbols) > 1:
                    d = data[sym]
                else:
                    d = data
                closes = d['Close'].dropna()
                current = float(closes.iloc[-1])
                prev = float(closes.iloc[-2]) if len(closes) >= 2 else current
                change = round(current - prev, 3)
                result['bonds'].append({
                    'name': name, 'value': round(current, 3), '1d_change': change,
                })
            except Exception:
                pass

        # Commodities
        for name, sym in COMMODITY_TICKERS.items():
            price, r1d, r1m, r_ytd = _get_returns(sym)
            result['commodities'].append({
                'name': name, 'price': round(price, 2), '1d': r1d, '1m': r1m,
            })

        # FX
        for name, sym in FX_TICKERS.items():
            price, r1d, r1m, r_ytd = _get_returns(sym)
            result['fx'].append({
                'name': name, 'value': round(price, 4), '1d': r1d,
            })

        # Crypto
        for name, sym in CRYPTO_TICKERS.items():
            price, r1d, r1m, r_ytd = _get_returns(sym)
            result['crypto'].append({
                'name': name, 'price': round(price, 2), '1d': r1d, '1m': r1m,
            })

    except Exception as e:
        print(f"[V12]   Market dashboard error: {e}")

    return result


# ============================================================================
# ECONOMIC INDICATORS
# ============================================================================

def _build_economic(fred_api_key=None):
    """Fetch economic indicators from FRED if available."""
    default = {
        'fed_funds_rate': 4.33,
        'indicators': [],
        'source': 'unavailable',
    }

    if not fred_api_key:
        # Try to get basic data from yfinance proxies
        try:
            tnx = yf.Ticker("^TNX")
            tnx_hist = tnx.history(period="5d", timeout=15)
            if not tnx_hist.empty:
                rate_10y = float(tnx_hist['Close'].iloc[-1])
                default['indicators'].append({
                    'name': '10-Year Treasury', 'value': round(rate_10y, 2),
                    'prior': round(float(tnx_hist['Close'].iloc[-2]), 2) if len(tnx_hist) >= 2 else 0,
                    'trend': 'Stable', 'unit': '%',
                })
        except Exception:
            pass

        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d", timeout=15)
            if not vix_hist.empty:
                vix_val = float(vix_hist['Close'].iloc[-1])
                default['indicators'].append({
                    'name': 'VIX', 'value': round(vix_val, 2),
                    'prior': round(float(vix_hist['Close'].iloc[-2]), 2) if len(vix_hist) >= 2 else 0,
                    'trend': 'Low' if vix_val < 18 else 'Elevated' if vix_val > 25 else 'Normal',
                    'unit': '',
                })
        except Exception:
            pass

        default['source'] = 'yfinance_proxy'
        return default

    # FRED API available
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_api_key)

        indicators = []
        fred_series = {
            'CPI (YoY)': ('CPIAUCSL', '%', True),
            'Core CPI (YoY)': ('CPILFESL', '%', True),
            'Unemployment': ('UNRATE', '%', False),
            'Fed Funds Rate': ('FEDFUNDS', '%', False),
            'GDP Growth (QoQ)': ('A191RL1Q225SBEA', '%', False),
            'Consumer Confidence': ('UMCSENT', '', False),
            'ISM Manufacturing': ('MANEMP', 'K', False),
            'Retail Sales (MoM)': ('RSXFS', 'M$', True),
        }

        for name, (series_id, unit, yoy) in fred_series.items():
            try:
                data = fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=400))
                data = data.dropna()
                if len(data) < 2:
                    continue

                current = float(data.iloc[-1])
                prior = float(data.iloc[-2])

                if yoy and len(data) >= 13:
                    # Year-over-year change
                    year_ago = float(data.iloc[-13])
                    current_yoy = round((current / year_ago - 1) * 100, 1) if year_ago != 0 else 0
                    prior_idx = max(0, len(data) - 14)
                    prior_yago = float(data.iloc[prior_idx])
                    prior_yoy = round((prior / prior_yago - 1) * 100, 1) if prior_yago != 0 else 0
                    value = current_yoy
                    prior_val = prior_yoy
                else:
                    value = round(current, 1)
                    prior_val = round(prior, 1)

                trend = 'Rising' if value > prior_val else 'Declining' if value < prior_val else 'Stable'

                indicators.append({
                    'name': name, 'value': value, 'prior': prior_val,
                    'trend': trend, 'unit': unit,
                })

                if name == 'Fed Funds Rate':
                    default['fed_funds_rate'] = value

            except Exception:
                continue

        default['indicators'] = indicators
        default['source'] = 'fred'
        return default

    except ImportError:
        print("[V12]   fredapi not installed, using yfinance proxies")
        return default
    except Exception as e:
        print(f"[V12]   FRED error: {e}")
        return default


# ============================================================================
# INSTITUTIONAL DATA
# ============================================================================

def _build_institutional(info):
    """Institutional ownership, insider activity, short interest."""
    return {
        'institutional_pct': round((info.get('heldPercentInstitutions', 0) or 0) * 100, 1),
        'insider_pct': round((info.get('heldPercentInsiders', 0) or 0) * 100, 1),
        'short_pct': round((info.get('shortPercentOfFloat', 0) or 0) * 100, 1),
        'short_ratio': round(info.get('shortRatio', 0) or 0, 1),
    }


# ============================================================================
# BETA STABILIZATION + WACC GOVERNANCE (Stage 5)
# ============================================================================

def _stabilize_beta(raw_beta, sector, beta_path=None):
    """
    Clamp raw beta to sector-appropriate bounds.

    P2: When beta_path is PROXY, skip sector-specific stabilization
    and only apply general bounds (the proxy already IS the fallback).

    Returns:
        (beta_used, flags: list[str])
    """
    if not CONFIG.flags.beta_stabilization:
        return (raw_beta, [])

    flags = []

    # P2: Proxy betas skip sector stabilization — they're already conservative defaults
    if beta_path == BetaPath.PROXY:
        flags.append('BETA_PROXY_1_0')
        return (round(raw_beta, 2), flags)

    bounds = SECTOR_BETA_BOUNDS.get(sector, DEFAULT_BETA_BOUNDS)

    if raw_beta < 0.5:
        flags.append('LOW_BETA_WARNING')

    beta_used = raw_beta
    if raw_beta < bounds.floor:
        beta_used = bounds.floor
        flags.append('BETA_FLOOR_APPLIED')
    elif raw_beta > bounds.cap:
        beta_used = bounds.cap
        flags.append('BETA_CAP_APPLIED')

    return (round(beta_used, 2), flags)


def _compute_wacc_governed(beta, sector, de, sector_provenance=None):
    """
    Multi-layer WACC computation with sector and general floors.

    P1: When sector_provenance is DEFAULTED, only apply general floor
    (don't silently apply sector-specific floors for unknown sectors).

    Returns:
        (discount_rate, wacc_raw, reason_codes: list[str], audit: dict)
    """
    cfg = CONFIG.wacc
    reason_codes = []

    # CAPM: Ke = Rf + beta * ERP
    cost_of_equity = cfg.risk_free_rate + beta * cfg.equity_risk_premium

    # After-tax cost of debt
    after_tax_cost_of_debt = cfg.pre_tax_cost_of_debt * (1 - cfg.tax_rate)

    # Capital structure weights from D/E
    cap_structure_warning = None
    if de > 0:
        debt_weight = de / (1 + de)
        equity_weight = 1 - debt_weight
        wacc_raw = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
    else:
        debt_weight = 0.0
        equity_weight = 1.0
        wacc_raw = cost_of_equity

    # Guard: weights must be in [0,1] and sum to ~1.0
    if equity_weight < 0 or equity_weight > 1.0 or debt_weight < 0 or debt_weight > 1.0:
        cap_structure_warning = 'WEIGHT_OUT_OF_BOUNDS'
    if abs(equity_weight + debt_weight - 1.0) > 1e-3:
        cap_structure_warning = 'CAP_STRUCTURE_NOT_NORMALIZED'

    if not CONFIG.flags.wacc_governance:
        # Legacy behavior: simple floor/ceiling
        clamp_floor = 0.06
        clamp_ceiling = 0.15
        if wacc_raw < clamp_floor:
            discount_rate = clamp_floor
            reason_codes.append('FLOOR_ABSOLUTE')
        elif wacc_raw > clamp_ceiling:
            discount_rate = clamp_ceiling
            reason_codes.append('CEILING_ABSOLUTE')
        else:
            discount_rate = wacc_raw
    else:
        # Multi-layer governance clamp
        general_floor = cfg.wacc_general_floor
        ceiling = cfg.wacc_ceiling

        # P1: Only apply sector floor when sector is from PROVIDER (real data)
        if sector_provenance == SectorProvenance.DEFAULTED or not sector:
            sector_floor = general_floor  # Don't apply sector-specific floor
        else:
            sector_floor = SECTOR_WACC_FLOORS.get(sector, general_floor)

        discount_rate = wacc_raw

        # Apply floors (take the highest)
        if discount_rate < general_floor:
            reason_codes.append('FLOOR_GENERAL')
        if sector_provenance != SectorProvenance.DEFAULTED and sector and discount_rate < sector_floor:
            reason_codes.append(f'FLOOR_SECTOR_{sector.upper().replace(" ", "_")}')

        effective_floor = max(general_floor, sector_floor)
        if discount_rate < effective_floor:
            discount_rate = effective_floor

        # Apply ceiling
        if discount_rate > ceiling:
            discount_rate = ceiling
            reason_codes.append('CEILING_ABSOLUTE')

    audit = {
        'cost_of_equity': round(cost_of_equity, 6),
        'after_tax_cost_of_debt': round(after_tax_cost_of_debt, 6),
        'equity_weight': round(equity_weight, 4),
        'debt_weight': round(debt_weight, 4),
        'cap_structure_warning': cap_structure_warning,
        'wacc_general_floor': round(cfg.wacc_general_floor * 100, 2) if CONFIG.flags.wacc_governance else None,
        'wacc_sector_floor': round(sector_floor * 100, 2) if CONFIG.flags.wacc_governance else None,
        'sector_provenance': sector_provenance.value if sector_provenance else None,
    }

    return (round(discount_rate, 6), round(wacc_raw, 6), reason_codes, audit)


# ============================================================================
# DCF KERNEL + MONTE CARLO ENGINE
# ============================================================================

def _evaluate_dcf_single(revenue, fcf_margin, discount_rate, terminal_growth,
                          growth_rates_array, shares, net_debt, forecast_years=5):
    """
    Pure DCF kernel: deterministic IV per share for given parameters.
    Extracted for reuse by both deterministic and Monte Carlo paths.
    """
    r = revenue
    projected_fcf = []
    for g in growth_rates_array:
        r *= (1 + g)
        projected_fcf.append(r * fcf_margin)

    pv_fcf_sum = sum(
        f / (1 + discount_rate) ** (i + 1) for i, f in enumerate(projected_fcf)
    )

    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    denom = discount_rate - terminal_growth
    if denom <= 0:
        denom = 0.001  # prevent div-by-zero
    terminal_value = terminal_fcf / denom
    tv_pv = terminal_value / (1 + discount_rate) ** forecast_years

    enterprise_value = pv_fcf_sum + tv_pv
    equity_value = max(0, enterprise_value - net_debt)
    iv_per_share = equity_value / shares if shares > 0 else 0
    return iv_per_share


def _ticker_seed(ticker):
    """Deterministic, reproducible seed from ticker string."""
    return zlib.crc32(ticker.upper().encode()) & 0xFFFFFFFF


def _student_t_unit(rng, nu, size):
    """
    Generate unit-variance Student's t samples.
    For ν > 2, raw t has variance ν/(ν-2); we rescale to unit variance.
    Falls back to standard normal when ν >= 30 (negligible tail difference).
    """
    if nu >= 30:
        return rng.standard_normal(size)
    z = rng.standard_normal(size)
    chi2 = rng.chisquare(nu, size)
    t = z / np.sqrt(chi2 / nu)
    if nu > 2:
        t *= np.sqrt((nu - 2) / nu)  # rescale to unit variance
    return t


def _run_monte_carlo_dcf(revenue, fcf_margin, discount_rate, terminal_growth,
                          growth_rates, shares, net_debt, sector, ticker_seed,
                          cash_flow_source='fcf', price=None, n_simulations=1000,
                          regime_variance_mult=1.0):
    """
    Fully vectorized Monte Carlo DCF engine.
    Returns distribution statistics for intrinsic value.
    """
    t0 = time.perf_counter()
    mc_cfg = CONFIG.monte_carlo
    N = n_simulations
    forecast_years = len(growth_rates)

    rng = np.random.default_rng(ticker_seed)

    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(MC_CORRELATION_MATRIX)

    # Draw correlated standard normals: (N, 4) → [growth, margin, wacc, terminal_g]
    z = rng.standard_normal((N, 4))
    correlated = z @ L.T

    # Sigma multiplier for proxy cash flows
    proxy_mult = mc_cfg.proxy_sigma_multiplier if cash_flow_source != 'fcf' else 1.0

    # Growth shocks with AR(1) structure across years
    growth_sigma = SECTOR_GROWTH_SIGMA.get(sector, 0.06) * proxy_mult
    rho = mc_cfg.growth_ar1_rho
    growth_base = np.array(growth_rates)  # (forecast_years,)

    # Build growth paths: (N, forecast_years)
    growth_paths = np.tile(growth_base, (N, 1))  # start from base
    # Year 1: innovation from correlated draw — fat-tailed via Student's t
    tail_df = SECTOR_TAIL_DF.get(sector, 6)
    growth_innovations = _student_t_unit(rng, tail_df, (N, forecast_years)) * growth_sigma
    growth_innovations[:, 0] += correlated[:, 0] * growth_sigma  # add correlated component
    # AR(1) propagation
    for t in range(forecast_years):
        if t == 0:
            growth_paths[:, t] += growth_innovations[:, t]
        else:
            growth_paths[:, t] += rho * (growth_paths[:, t-1] - growth_base[t-1]) + (1 - rho) * growth_innovations[:, t]
    growth_paths = np.maximum(growth_paths, mc_cfg.growth_floor)

    # Margin shocks: (N,)
    margin_sigma = max(abs(fcf_margin) * mc_cfg.margin_cv * proxy_mult, mc_cfg.margin_sigma_floor)
    margins = fcf_margin + correlated[:, 1] * margin_sigma
    margins = np.maximum(margins, 0.01)  # floor at 1%

    # WACC shocks: (N,) — regime-scaled uncertainty
    wacc_sigma = mc_cfg.wacc_sigma * proxy_mult * regime_variance_mult
    waccs = discount_rate + correlated[:, 2] * wacc_sigma
    waccs = np.clip(waccs, 0.03, 0.20)

    # Terminal growth shocks: (N,)
    tg_sigma = mc_cfg.terminal_growth_sigma
    term_gs = terminal_growth + correlated[:, 3] * tg_sigma
    term_gs = np.clip(term_gs, mc_cfg.terminal_growth_floor, mc_cfg.terminal_growth_cap)
    # Enforce terminal_g < wacc (with buffer)
    term_gs = np.minimum(term_gs, waccs - 0.005)
    term_gs = np.maximum(term_gs, 0.005)

    # Vectorized revenue projections: (N, forecast_years)
    cum_growth = np.cumprod(1 + growth_paths, axis=1)
    revenues = revenue * cum_growth

    # FCFs = revenues × margins
    fcfs = revenues * margins[:, np.newaxis]

    # PV factors: 1/(1+wacc)^t for t=1..forecast_years
    t_arr = np.arange(1, forecast_years + 1)
    pv_factors = 1.0 / (1 + waccs[:, np.newaxis]) ** t_arr[np.newaxis, :]

    # PV of FCFs
    pv_sum = np.sum(fcfs * pv_factors, axis=1)

    # Terminal value
    fcf_last = fcfs[:, -1]
    terminal_fcf_mc = fcf_last * (1 + term_gs)
    tv_denom = waccs - term_gs
    tv_denom = np.maximum(tv_denom, 0.001)
    terminal_values = terminal_fcf_mc / tv_denom
    tv_pv_factors = 1.0 / (1 + waccs) ** forecast_years
    tv_pv = terminal_values * tv_pv_factors

    # Enterprise value → equity → IV per share
    ev = pv_sum + tv_pv
    equity = np.maximum(0, ev - net_debt)
    iv = equity / shares if shares > 0 else np.zeros(N)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Distribution shape metrics
    iv_skew = float(np.mean(((iv - np.mean(iv)) / max(np.std(iv), 1e-10)) ** 3)) if np.std(iv) > 0 else 0.0
    iv_kurtosis = float(np.mean(((iv - np.mean(iv)) / max(np.std(iv), 1e-10)) ** 4)) - 3.0 if np.std(iv) > 0 else 0.0

    # Percentile extraction
    result = {
        'n_simulations': N,
        'seed': int(ticker_seed),
        'elapsed_ms': round(elapsed_ms, 1),
        'tail_df': tail_df,
        'regime_variance_mult': round(regime_variance_mult, 2),
        'iv_mean': round(float(np.mean(iv)), 2),
        'iv_median': round(float(np.median(iv)), 2),
        'iv_std': round(float(np.std(iv)), 2),
        'iv_p5': round(float(np.percentile(iv, 5)), 2),
        'iv_p10': round(float(np.percentile(iv, 10)), 2),
        'iv_p25': round(float(np.percentile(iv, 25)), 2),
        'iv_p50': round(float(np.percentile(iv, 50)), 2),
        'iv_p75': round(float(np.percentile(iv, 75)), 2),
        'iv_p90': round(float(np.percentile(iv, 90)), 2),
        'iv_p95': round(float(np.percentile(iv, 95)), 2),
        'iv_skewness': round(iv_skew, 4),
        'iv_kurtosis': round(iv_kurtosis, 4),
    }

    if price is not None and price > 0:
        result['prob_undervalued'] = round(float(np.mean(iv > price)), 4)
        mos_vals = np.maximum(0, (iv - price) / np.where(iv > 0, iv, 1))
        result['expected_mos'] = round(float(np.mean(mos_vals)), 4)
        # Phase 2f: prob_permanent_loss = P(IV < 0.5 × Price)
        result['prob_permanent_loss'] = round(float(np.mean(iv < 0.5 * price)), 4)
    else:
        result['prob_undervalued'] = None
        result['expected_mos'] = None
        result['prob_permanent_loss'] = None

    return result


def _compute_sensitivity(revenue, fcf_margin, discount_rate, terminal_growth,
                          growth_rates, shares, net_debt):
    """
    Sensitivity analysis via central finite differences.
    Returns partial derivatives and identifies most sensitive parameter.
    """
    base_iv = _evaluate_dcf_single(
        revenue, fcf_margin, discount_rate, terminal_growth,
        growth_rates, shares, net_debt
    )

    # WACC sensitivity: ±50 bps
    dw = 0.005
    iv_up = _evaluate_dcf_single(revenue, fcf_margin, discount_rate + dw, terminal_growth,
                                  growth_rates, shares, net_debt)
    iv_dn = _evaluate_dcf_single(revenue, fcf_margin, discount_rate - dw, terminal_growth,
                                  growth_rates, shares, net_debt)
    dIV_dWACC = (iv_up - iv_dn) / (2 * dw)

    # Growth sensitivity: ±200 bps on all years
    dg = 0.02
    gr_up = [g + dg for g in growth_rates]
    gr_dn = [g - dg for g in growth_rates]
    iv_up = _evaluate_dcf_single(revenue, fcf_margin, discount_rate, terminal_growth,
                                  gr_up, shares, net_debt)
    iv_dn = _evaluate_dcf_single(revenue, fcf_margin, discount_rate, terminal_growth,
                                  gr_dn, shares, net_debt)
    dIV_dGrowth = (iv_up - iv_dn) / (2 * dg)

    # Margin sensitivity: ±200 bps
    dm = 0.02
    iv_up = _evaluate_dcf_single(revenue, fcf_margin + dm, discount_rate, terminal_growth,
                                  growth_rates, shares, net_debt)
    iv_dn = _evaluate_dcf_single(revenue, fcf_margin - dm, discount_rate, terminal_growth,
                                  growth_rates, shares, net_debt)
    dIV_dMargin = (iv_up - iv_dn) / (2 * dm)

    # Terminal growth sensitivity: ±50 bps
    dt = 0.005
    tg_up = min(terminal_growth + dt, discount_rate - 0.005)
    tg_dn = max(terminal_growth - dt, 0.005)
    iv_up = _evaluate_dcf_single(revenue, fcf_margin, discount_rate, tg_up,
                                  growth_rates, shares, net_debt)
    iv_dn = _evaluate_dcf_single(revenue, fcf_margin, discount_rate, tg_dn,
                                  growth_rates, shares, net_debt)
    dIV_dTerminalG = (iv_up - iv_dn) / (tg_up - tg_dn) if (tg_up - tg_dn) > 0 else 0

    # Phase 2f: Cross-sensitivity (WACC×Growth interaction)
    iv_wg_up = _evaluate_dcf_single(revenue, fcf_margin, discount_rate + dw, terminal_growth,
                                     gr_up, shares, net_debt)
    iv_wg_dn = _evaluate_dcf_single(revenue, fcf_margin, discount_rate - dw, terminal_growth,
                                     gr_dn, shares, net_debt)
    # Mixed partial: ∂²IV/(∂WACC ∂Growth)
    cross_wacc_growth = ((iv_wg_up - iv_up) - (iv_wg_dn - iv_dn)) / (4 * dw * dg) if base_iv > 0 else 0.0

    sensitivities = {
        'dIV_dWACC': round(dIV_dWACC, 2),
        'dIV_dGrowth': round(dIV_dGrowth, 2),
        'dIV_dMargin': round(dIV_dMargin, 2),
        'dIV_dTerminalG': round(dIV_dTerminalG, 2),
        'cross_wacc_growth': round(cross_wacc_growth, 2),
    }

    # Identify most sensitive parameter (by absolute magnitude per unit)
    # Normalize: WACC/terminal_g use 100bps, growth/margin use 100bps
    abs_sens = {
        'WACC': abs(dIV_dWACC * 0.01),
        'Growth': abs(dIV_dGrowth * 0.01),
        'Margin': abs(dIV_dMargin * 0.01),
        'TerminalG': abs(dIV_dTerminalG * 0.01),
    }
    sensitivities['most_sensitive_to'] = max(abs_sens, key=abs_sens.get)

    return sensitivities


# ============================================================================
# DCF MODEL
# ============================================================================

def _build_dcf(info, financials, sector=None, ticker=None, price=None, regime_variance_mult=1.0):
    """
    DCF fair value estimate with full audit trail.

    Discounts UNLEVERED free cash flow at WACC.
    Produces EV, then subtracts net debt to arrive at equity value per share.
    regime_variance_mult: WACC sigma scaling from engine regime (1.0 = no scaling).
    """
    _disabled = {'bear': 0, 'base': 0, 'bull': 0, '_dcf_disabled': True}
    try:
        # Gate: refuse to run DCF on INVALID fundamentals
        if financials.get('_data_status') == 'INVALID':
            _disabled['_dcf_reason'] = 'fundamentals_invalid'
            return _disabled

        revenue = financials.get('revenue_ttm')
        shares = financials.get('shares_outstanding')
        fcf = financials.get('free_cash_flow') or 0
        net_income = financials.get('net_income_ttm') or 0
        ebitda = financials.get('ebitda') or 0
        rg_pct = financials.get('revenue_growth')
        rev_growth = (rg_pct / 100) if rg_pct is not None else 0.05

        if not revenue or revenue <= 0 or not shares or shares <= 0:
            _disabled['_dcf_reason'] = 'missing_revenue_or_shares'
            return _disabled

        # Cash flow proxy: FCF → Net Income (75% haircut) → EBITDA (50% haircut)
        # NOTE: All proxies approximate UNLEVERED FCF (FCFF)
        cash_flow_source = 'fcf'
        cash_proxy = fcf
        if cash_proxy <= 0 and net_income > 0:
            cash_proxy = net_income * 0.75
            cash_flow_source = 'net_income'
        if cash_proxy <= 0 and ebitda > 0:
            cash_proxy = ebitda * 0.50
            cash_flow_source = 'ebitda'
        if cash_proxy <= 0:
            _disabled['_dcf_reason'] = 'no_positive_cash_flow'
            return _disabled

        fcf_margin = cash_proxy / revenue

        # ── DISCOUNT RATE DECOMPOSITION (Stage 5 Governance) ──
        cfg = CONFIG.wacc
        _raw_b = _safe_num(info.get('beta'))
        beta_defaulted = _raw_b is None or _raw_b <= 0
        beta_raw = round(_raw_b, 2) if _raw_b and _raw_b > 0 else 1.0
        _sector = sector or info.get('sector') or ''
        # Treat 'Unknown' sector as empty for governance purposes
        if _sector in ('Unknown', 'N/A', ''):
            _sector = ''

        # P1: Sector provenance — track whether sector is real or defaulted
        _sector_provenance = SectorProvenance.PROVIDER if _sector else SectorProvenance.DEFAULTED

        # P2: Beta path — separate measured vs proxy
        _beta_path = BetaPath.PROXY if beta_defaulted else BetaPath.MEASURED

        # Beta stabilization: clamp to sector bounds (P2: proxy skips sector stabilization)
        beta, beta_flags = _stabilize_beta(beta_raw, _sector, beta_path=_beta_path)

        risk_free = cfg.risk_free_rate
        erp = cfg.equity_risk_premium
        cost_of_equity = risk_free + beta * erp  # CAPM: Ke = Rf + β×ERP

        # Cost of debt
        pre_tax_cost_of_debt = cfg.pre_tax_cost_of_debt
        tax_rate = cfg.tax_rate
        after_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)

        # Capital structure weights from D/E — stored as decimals in [0,1]
        de = financials.get('debt_equity') or 0

        # WACC governance: multi-layer clamp (P1: pass sector provenance)
        discount_rate, wacc, wacc_clamp_codes, wacc_audit = _compute_wacc_governed(
            beta, _sector, de, sector_provenance=_sector_provenance
        )
        equity_weight = wacc_audit['equity_weight']
        debt_weight = wacc_audit['debt_weight']
        cap_structure_warning = wacc_audit['cap_structure_warning']

        discount_rate_type = 'WACC'
        clamp_rule = wacc_clamp_codes[0] if wacc_clamp_codes else None

        terminal_growth = 0.03
        forecast_years = 5

        # ── YEAR-BY-YEAR PROJECTIONS ──
        growth_rates = [
            max(0.02, rev_growth),
            max(0.02, rev_growth * 0.85),
            max(0.02, rev_growth * 0.70),
            0.05,
            0.05,
        ]

        projections = []  # Audit trail: year-by-year
        projected_fcf = []
        r = revenue
        for i, g in enumerate(growth_rates):
            r *= (1 + g)
            yr_fcf = r * fcf_margin
            projected_fcf.append(yr_fcf)
            projections.append({
                'year': i + 1,
                'revenue': round(r, 0),
                'growth_rate': round(g * 100, 2),
                'fcf': round(yr_fcf, 0),
                'pv_factor': round(1 / (1 + discount_rate) ** (i + 1), 4),
                'pv_fcf': round(yr_fcf / (1 + discount_rate) ** (i + 1), 0),
            })

        # Terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)

        # Present value
        pv_fcf_sum = sum(f / (1 + discount_rate) ** (i + 1) for i, f in enumerate(projected_fcf))
        tv_pv = terminal_value / (1 + discount_rate) ** forecast_years
        enterprise_value = pv_fcf_sum + tv_pv

        # Terminal value as % of total DCF
        terminal_pct = tv_pv / enterprise_value * 100 if enterprise_value > 0 else 0

        # ── EV → EQUITY BRIDGE ──
        net_debt = financials.get('net_debt') or 0
        equity_value = max(0, enterprise_value - net_debt)
        base_fv = equity_value / shares

        # ── Bridge reconciliation checks ──
        bridge_warnings = []
        if abs((pv_fcf_sum + tv_pv) - enterprise_value) > 1.0:
            bridge_warnings.append('PV_SUM_MISMATCH')
        if abs(equity_value - max(0, enterprise_value - net_debt)) > 1.0:
            bridge_warnings.append('EV_EQUITY_MISMATCH')
        if shares > 0 and abs(equity_value / shares - base_fv) > 0.01:
            bridge_warnings.append('PER_SHARE_MISMATCH')

        # Widen bear/bull range when using proxy cash flows
        bear_mult = 0.70 if cash_flow_source != 'fcf' else 0.80
        bull_mult = 1.30 if cash_flow_source != 'fcf' else 1.25

        # Terminal governance: haircut bear_mult if terminal dependence is severe
        terminal_flags = []
        if CONFIG.flags.terminal_governance and terminal_pct >= CONFIG.terminal.severe_threshold:
            bear_mult *= CONFIG.terminal.bear_mult_haircut
            terminal_flags.append('BEAR_MULT_HAIRCUT')
        if terminal_pct >= CONFIG.terminal.extreme_threshold:
            terminal_flags.append('IV_CONFIDENCE_LOW')

        # Effective clamp floors for audit
        _gen_floor = CONFIG.wacc.wacc_general_floor if CONFIG.flags.wacc_governance else 0.06
        _sec_floor = SECTOR_WACC_FLOORS.get(_sector, _gen_floor) if CONFIG.flags.wacc_governance else _gen_floor

        # ── MONTE CARLO DCF (V12+) ──
        mc_result = None
        sensitivity_result = None

        if CONFIG.flags.monte_carlo_dcf:
            _ticker = ticker or ''
            _price = price
            seed = _ticker_seed(_ticker) if _ticker else 42

            mc_result = _run_monte_carlo_dcf(
                revenue=revenue, fcf_margin=fcf_margin,
                discount_rate=discount_rate, terminal_growth=terminal_growth,
                growth_rates=growth_rates, shares=shares, net_debt=net_debt,
                sector=_sector, ticker_seed=seed,
                cash_flow_source=cash_flow_source,
                price=_price, n_simulations=CONFIG.monte_carlo.n_simulations,
                regime_variance_mult=regime_variance_mult,
            )

            sensitivity_result = _compute_sensitivity(
                revenue=revenue, fcf_margin=fcf_margin,
                discount_rate=discount_rate, terminal_growth=terminal_growth,
                growth_rates=growth_rates, shares=shares, net_debt=net_debt,
            )

        # Replace bear/base/bull with MC percentiles if available
        bear_fv = round(base_fv * bear_mult, 2)
        base_fv_final = round(base_fv, 2)
        bull_fv = round(base_fv * bull_mult, 2)

        if mc_result is not None:
            bear_fv = mc_result['iv_p5']
            base_fv_final = mc_result['iv_p50']
            bull_fv = mc_result['iv_p95']

        result = {
            'bear': bear_fv,
            'base': base_fv_final,
            'bull': bull_fv,
            'base_deterministic': round(base_fv, 2),
            'cash_flow_source': cash_flow_source,
            'cash_flow_definition': 'unlevered_fcf',
            'forecast_years': forecast_years,
            'projections': projections,
            'assumptions': {
                # Discount rate decomposition
                'discount_rate_type': discount_rate_type,
                'risk_free_rate': round(risk_free * 100, 2),
                'equity_risk_premium': round(erp * 100, 2),
                'beta': round(beta, 2),
                'beta_raw': round(beta_raw, 2),
                'beta_defaulted': beta_defaulted,
                'beta_path': _beta_path.value,
                'beta_source': 'yfinance_5y_monthly',  # G5: Document provider source
                'beta_flags': beta_flags,
                'cost_of_equity': round(cost_of_equity * 100, 2),
                'pre_tax_cost_of_debt': round(pre_tax_cost_of_debt * 100, 2),
                'tax_rate': round(tax_rate * 100, 1),
                'after_tax_cost_of_debt': round(after_tax_cost_of_debt * 100, 2),
                # Capital structure weights — decimals in [0,1]
                'equity_weight': round(equity_weight, 4),
                'debt_weight': round(debt_weight, 4),
                'cap_structure_warning': cap_structure_warning,
                # WACC
                'wacc_raw': round(wacc * 100, 2),
                'discount_rate': round(discount_rate * 100, 2),
                # Clamp details — fully explicit
                'clamp_rule': clamp_rule,
                'wacc_clamp_codes': wacc_clamp_codes,
                'clamp_floor': round(_gen_floor * 100, 1),
                'clamp_ceiling': round(CONFIG.wacc.wacc_ceiling * 100, 1),
                'wacc_general_floor': round(_gen_floor * 100, 2),
                'wacc_sector_floor': round(_sec_floor * 100, 2),
                'sector': _sector,
                'sector_provenance': _sector_provenance.value,
                # Growth & margin
                'revenue_growth_y1': round(growth_rates[0] * 100, 1),
                'fcf_margin': round(fcf_margin * 100, 1),
                'margin_assumption': 'constant',  # G8: constant | mean_reverting | scenario_adjusted
                'cash_flow_source': cash_flow_source,  # G8: fcf | net_income | ebitda
                'cash_flow_haircut': 0.0 if cash_flow_source == 'fcf' else (0.25 if cash_flow_source == 'net_income' else 0.50),  # G8
                'terminal_growth': round(terminal_growth * 100, 1),
                # Terminal
                'terminal_fcf': round(terminal_fcf, 0),
                'terminal_value': round(terminal_value, 0),
                'terminal_value_pv': round(tv_pv, 0),
                'terminal_value_pct': round(terminal_pct, 1),
                'terminal_flags': terminal_flags,
                # EV → Equity bridge
                'enterprise_value': round(enterprise_value, 0),
                'pv_fcf_sum': round(pv_fcf_sum, 0),
                'net_debt_subtracted': round(net_debt, 0),
                'equity_value': round(equity_value, 0),
                'shares': round(shares, 0),
                # Reconciliation
                'bridge_warnings': bridge_warnings,
            },
        }

        if mc_result is not None:
            result['monte_carlo'] = mc_result
        if sensitivity_result is not None:
            result['sensitivity'] = sensitivity_result

        return result
    except Exception as e:
        _disabled['_dcf_reason'] = f'exception: {str(e)[:80]}'
        return _disabled


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    import json
    symbol = sys.argv[1] if len(sys.argv) > 1 else "KO"
    fred_key = sys.argv[2] if len(sys.argv) > 2 else None
    data = fetch_v8_data(symbol, fred_key)
    print(f"\nV8 Data for {symbol}:")
    print(f"  Company: {data['company']['name']}")
    print(f"  Sector: {data['company']['sector']}")
    print(f"  Peers: {len(data['peers'])} found")
    print(f"  News: {len(data['news'])} articles")
    print(f"  Technicals: RSI={data['technicals']['rsi_14']}, MACD={data['technicals']['macd_line']}")
    print(f"  Sectors: {len(data['sector']['all_sectors'])} sectors")
    print(f"  Market: {len(data['market']['indices'])} indices")
    print(f"  Economic: {len(data['economic']['indicators'])} indicators")
    print(f"  DCF: Bear=${data['dcf'].get('bear',0)}, Base=${data['dcf'].get('base',0)}, Bull=${data['dcf'].get('bull',0)}")
