#!/usr/bin/env python3
"""
ATLAS V8 — Extended Data Fetcher
Fetches additional data beyond what the ATLAS engine needs.
Used by v8_report.py to generate the full 10-section report.

Data sources:
  - yfinance: Company info, financials, earnings, peers, news, technicals
  - yfinance batch: Market dashboard, sector ETFs
  - FRED (optional): Economic indicators
  - Computed: Technical indicators (RSI, MACD, Bollinger, etc.)
"""

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor


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

def _safe_info(ticker):
    """Get ticker.info with error handling."""
    try:
        return ticker.info or {}
    except Exception:
        return {}


def _pct_return(hist, days_back):
    """Compute return from days_back to current."""
    if hist is None or len(hist) < 2:
        return 0.0
    current = float(hist['Close'].iloc[-1])
    idx = min(days_back, len(hist) - 1)
    past = float(hist['Close'].iloc[-idx - 1]) if idx < len(hist) else float(hist['Close'].iloc[0])
    if past == 0:
        return 0.0
    return round((current / past - 1) * 100, 2)


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

def fetch_v8_data(symbol, fred_api_key=None):
    """
    Fetch all extended data for V8 report.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL')
        fred_api_key: Optional FRED API key for economic indicators

    Returns:
        dict with keys: company, financials, earnings, technicals,
        peers, news, sector, market, economic, institutional, dcf
    """
    print(f"[V8] Fetching extended data for {symbol}...")
    ticker = yf.Ticker(symbol)
    info = _safe_info(ticker)

    # Fetch OHLCV for technicals (1 year)
    try:
        hist = ticker.history(period="1y", interval="1d")
    except Exception:
        hist = None

    print(f"[V8]   Building company info...")
    company = _build_company_info(info, symbol)

    print(f"[V8]   Building financials...")
    financials = _build_financials(ticker, info)

    print(f"[V8]   Building earnings history...")
    earnings = _build_earnings(ticker)

    print(f"[V8]   Computing technicals...")
    technicals = _build_technicals(hist, info)

    print(f"[V8]   Fetching peer data...")
    peers = _build_peers(symbol, info)

    print(f"[V8]   Fetching news...")
    news = _build_news(ticker)

    print(f"[V8]   Fetching sector performance...")
    sector = _build_sector(info)

    print(f"[V8]   Building market dashboard...")
    market = _build_market_dashboard()

    print(f"[V8]   Fetching economic indicators...")
    economic = _build_economic(fred_api_key)

    print(f"[V8]   Building institutional data...")
    institutional = _build_institutional(info)

    print(f"[V8]   Computing DCF model...")
    dcf = _build_dcf(info, financials)

    print(f"[V8] Extended data complete for {symbol}")

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

def _build_company_info(info, symbol):
    """Extract company overview from yfinance info."""
    price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0
    return {
        'symbol': symbol,
        'name': info.get('longName', '') or info.get('shortName', symbol),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'market_cap': info.get('marketCap', 0) or 0,
        'employees': info.get('fullTimeEmployees', 0) or 0,
        'beta': info.get('beta', 1.0) or 1.0,
        'price': price,
        'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0) or 0,
        'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0) or 0,
    }


# ============================================================================
# DETAILED FINANCIALS
# ============================================================================

def _build_financials(ticker, info):
    """Comprehensive financial metrics."""
    mc = info.get('marketCap', 0) or 1
    revenue = info.get('totalRevenue', 0) or 0
    net_income = info.get('netIncomeToCommon', 0) or 0
    fcf = info.get('freeCashflow', 0) or 0
    total_debt = info.get('totalDebt', 0) or 0
    total_cash = info.get('totalCash', 0) or 0
    ebitda = info.get('ebitda', 0) or 0
    price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0
    shares = mc / price if price > 0 else 1

    # Interest coverage from operating income / interest expense
    operating_income = revenue * ((info.get('operatingMargins', 0) or 0))
    # Approximate interest expense from debt at ~4% rate
    interest_expense = total_debt * 0.04
    interest_coverage = operating_income / interest_expense if interest_expense > 0 else 0

    # Buyback yield approximation
    try:
        cf = ticker.cashflow
        if cf is not None and not cf.empty and 'Repurchase Of Capital Stock' in cf.index:
            buyback = abs(float(cf.loc['Repurchase Of Capital Stock'].iloc[0] or 0))
            buyback_yield = buyback / mc * 100 if mc > 0 else 0
        else:
            buyback_yield = 0
    except Exception:
        buyback_yield = 0

    return {
        'revenue_ttm': revenue,
        'revenue_growth': (info.get('revenueGrowth', 0) or 0) * 100,
        'earnings_growth': (info.get('earningsGrowth', 0) or 0) * 100,
        'net_income_ttm': net_income,
        'ebitda': ebitda,
        'gross_margin': (info.get('grossMargins', 0) or 0) * 100,
        'operating_margin': (info.get('operatingMargins', 0) or 0) * 100,
        'net_margin': (info.get('profitMargins', 0) or 0) * 100,
        'ebitda_margin': (info.get('ebitdaMargins', 0) or 0) * 100,
        'roe': (info.get('returnOnEquity', 0) or 0) * 100,
        'roa': (info.get('returnOnAssets', 0) or 0) * 100,
        'free_cash_flow': fcf,
        'fcf_yield': (fcf / mc * 100) if mc > 0 else 0,
        'total_debt': total_debt,
        'total_cash': total_cash,
        'net_debt': total_debt - total_cash,
        'net_debt_ebitda': (total_debt - total_cash) / ebitda if ebitda > 0 else 0,
        'debt_equity': (info.get('debtToEquity', 0) or 0) / 100,
        'current_ratio': info.get('currentRatio', 0) or 0,
        'interest_coverage': round(interest_coverage, 1),
        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
        'payout_ratio': (info.get('payoutRatio', 0) or 0) * 100,
        'buyback_yield': round(buyback_yield, 2),
        'trailing_pe': info.get('trailingPE', 0) or 0,
        'forward_pe': info.get('forwardPE', 0) or 0,
        'peg_ratio': info.get('pegRatio', 0) or 0,
        'price_to_book': info.get('priceToBook', 0) or 0,
        'price_to_sales': info.get('priceToSalesTrailing12Months', 0) or 0,
        'ev_ebitda': info.get('enterpriseToEbitda', 0) or 0,
        'ev_revenue': info.get('enterpriseToRevenue', 0) or 0,
        'trailing_eps': info.get('trailingEps', 0) or 0,
        'forward_eps': info.get('forwardEps', 0) or 0,
        'revenue_per_share': info.get('revenuePerShare', 0) or 0,
        'book_value': info.get('bookValue', 0) or 0,
        'shares_outstanding': shares,
        'market_cap': mc,
        'target_mean': info.get('targetMeanPrice', 0) or 0,
        'target_high': info.get('targetHighPrice', 0) or 0,
        'target_low': info.get('targetLowPrice', 0) or 0,
        'num_analysts': info.get('numberOfAnalystOpinions', 0) or 0,
        'recommendation': info.get('recommendationKey', 'none'),
    }


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
            'price': i.get('currentPrice', 0) or i.get('regularMarketPrice', 0) or 0,
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
                'title': title[:75],
                'publisher': publisher or 'Unknown',
                'date': date_str,
                'link': link,
                'sentiment': sentiment,
            })
        return articles
    except Exception as e:
        print(f"[V8]   News fetch error: {e}")
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
                           group_by='ticker', progress=False)

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
                           group_by='ticker', progress=False)

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
        print(f"[V8]   Market dashboard error: {e}")

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
            tnx_hist = tnx.history(period="5d")
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
            vix_hist = vix.history(period="5d")
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
        print("[V8]   fredapi not installed, using yfinance proxies")
        return default
    except Exception as e:
        print(f"[V8]   FRED error: {e}")
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
# DCF MODEL
# ============================================================================

def _build_dcf(info, financials):
    """Simplified DCF fair value estimate."""
    try:
        mc = info.get('marketCap', 0) or 0
        price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0
        revenue = financials.get('revenue_ttm', 0)
        fcf = financials.get('free_cash_flow', 0)
        rev_growth = financials.get('revenue_growth', 5) / 100  # As decimal
        shares = mc / price if price > 0 else 1

        if revenue <= 0 or fcf <= 0 or shares <= 0:
            return {'bear': 0, 'base': 0, 'bull': 0}

        fcf_margin = fcf / revenue
        discount_rate = 0.10
        terminal_growth = 0.03

        # 5-year FCF projections
        growth_rates = [
            max(0.02, rev_growth),
            max(0.02, rev_growth * 0.85),
            max(0.02, rev_growth * 0.70),
            0.05,
            0.05,
        ]

        projected_fcf = []
        r = revenue
        for g in growth_rates:
            r *= (1 + g)
            projected_fcf.append(r * fcf_margin)

        # Terminal value
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # Present value
        pv = sum(f / (1 + discount_rate) ** (i + 1) for i, f in enumerate(projected_fcf))
        pv += terminal_value / (1 + discount_rate) ** 5

        base_fv = pv / shares

        return {
            'bear': round(base_fv * 0.80, 2),
            'base': round(base_fv, 2),
            'bull': round(base_fv * 1.25, 2),
            'assumptions': {
                'revenue_growth_y1': round(growth_rates[0] * 100, 1),
                'fcf_margin': round(fcf_margin * 100, 1),
                'discount_rate': discount_rate * 100,
                'terminal_growth': terminal_growth * 100,
            },
        }
    except Exception:
        return {'bear': 0, 'base': 0, 'bull': 0}


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
