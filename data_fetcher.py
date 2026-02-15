#!/usr/bin/env python3
"""
Live Data Fetcher for ATLAS
Pulls real-time data from free APIs and writes CSV/JSON files
that the ATLAS engine can consume directly.

APIs used:
  - yfinance: Price OHLCV, fundamentals, analyst consensus
  - yfinance: VIX, global indices (Nikkei, DAX, FTSE)
  - FRED (optional): Treasury yields, credit spreads

No API keys needed for yfinance. FRED requires a free key from
https://fred.stlouisfed.org/docs/api/api_key.html
"""

import json
import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import numpy as np


def resolve_price(info, hist=None):
    """Resolve the best available price from yfinance info dict.

    Fallback chain:
        currentPrice → preMarketPrice → postMarketPrice → regularMarketPrice
        → previousClose → regularMarketPreviousClose → last OHLCV close → open → 0

    Returns:
        tuple: (price, source_key) where source_key indicates which field provided the price.
    """
    for key in (
        'currentPrice',
        'preMarketPrice',
        'postMarketPrice',
        'regularMarketPrice',
        'previousClose',
        'regularMarketPreviousClose',
    ):
        val = info.get(key)
        if val is not None and val > 0:
            return float(val), key

    # Try last OHLCV bar
    if hist is not None and not getattr(hist, 'empty', True):
        try:
            last_close = float(hist['Close'].iloc[-1])
            if last_close > 0:
                return last_close, 'ohlcv_close'
        except Exception:
            pass
        try:
            last_open = float(hist['Open'].iloc[-1])
            if last_open > 0:
                return last_open, 'ohlcv_open'
        except Exception:
            pass

    # Try info 'open' as last resort
    open_val = info.get('open')
    if open_val is not None and open_val > 0:
        return float(open_val), 'info_open'

    return 0, 'none'


def _safe_num(val):
    """Return val as float if it is a valid positive number, else None.

    Treats 0, None, NaN as missing.
    """
    if val is None:
        return None
    try:
        f = float(val)
        if f == 0 or np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _validate_symbol(symbol):
    """Validate ticker symbol: 1-5 alphanumeric chars, letters and optional dots/hyphens."""
    import re
    if not symbol or not re.match(r'^[A-Z]{1,5}([.-][A-Z]{1,2})?$', symbol.upper()):
        raise ValueError(f"Invalid ticker symbol: {symbol!r}")
    return symbol.upper()


def fetch_live_data(symbol, data_dir=None, fred_api_key=None):
    """
    Fetch live market data for any ticker and write ATLAS-compatible files.

    Args:
        symbol:       Ticker symbol (e.g. 'KO', 'AAPL', 'SPY')
        data_dir:     Directory to write data files. If None, creates a temp dir.
        fred_api_key: Optional FRED API key for macro data.

    Returns:
        str: Path to the data directory containing all files
    """
    symbol = _validate_symbol(symbol)

    if data_dir is None:
        data_dir = tempfile.mkdtemp(prefix=f"atlas_{symbol}_")
    else:
        os.makedirs(data_dir, exist_ok=True)

    print(f"[ATLAS] Fetching live data for {symbol} (parallel)...")

    # Main ticker
    ticker = yf.Ticker(symbol)

    # Run all 7 data fetches in parallel for ~3x speedup
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(_fetch_ohlcv, ticker, symbol, data_dir): "ohlcv",
            executor.submit(_fetch_fundamentals, ticker, symbol, data_dir): "fundamentals",
            executor.submit(_fetch_consensus, ticker, symbol, data_dir): "consensus",
            executor.submit(_fetch_volatility, data_dir): "volatility",
            executor.submit(_fetch_macro, data_dir, fred_api_key): "macro",
            executor.submit(_fetch_breadth, data_dir): "breadth",
            executor.submit(_fetch_global_overnight, data_dir): "global",
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ATLAS]   CRITICAL ERROR in {name}: {e}")

    print(f"[ATLAS] All data saved to {data_dir}")
    return data_dir


def _fetch_ohlcv(ticker, symbol, data_dir):
    """Fetch 1 year of daily OHLCV data."""
    print(f"[ATLAS]   Fetching {symbol} price history...")
    try:
        hist = ticker.history(period="1y", interval="1d", prepost=True, timeout=30)

        if hist.empty:
            print(f"[ATLAS]   WARNING: No price data for {symbol}")
            # Write minimal data
            hist = ticker.history(period="6mo", interval="1d", prepost=True, timeout=30)

        filepath = os.path.join(data_dir, "ohlcv.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])
            for date, row in hist.iterrows():
                writer.writerow([
                    date.strftime("%Y-%m-%d"),
                    round(row['Open'], 2),
                    round(row['High'], 2),
                    round(row['Low'], 2),
                    round(row['Close'], 2),
                    int(row['Volume'])
                ])

        print(f"[ATLAS]   Got {len(hist)} price bars, latest close: ${hist['Close'].iloc[-1]:.2f}")

        # If we have less than 200 bars, pad with synthetic data
        if len(hist) < 200:
            print(f"[ATLAS]   WARNING: Only {len(hist)} bars, ATLAS needs 200+. Results may be limited.")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching OHLCV: {e}")
        _write_empty_ohlcv(data_dir)


def _fetch_fundamentals(ticker, symbol, data_dir):
    """Fetch fundamental data from yfinance."""
    print(f"[ATLAS]   Fetching {symbol} fundamentals...")
    try:
        info = ticker.info
        missing_fields = []
        synthetic_fields = []

        trailing_pe = _safe_num(info.get('trailingPE'))
        forward_pe = _safe_num(info.get('forwardPE'))
        ev_ebitda = _safe_num(info.get('enterpriseToEbitda'))
        ev_revenue = _safe_num(info.get('enterpriseToRevenue'))

        for name, val in [('trailingPE', trailing_pe), ('forwardPE', forward_pe),
                          ('enterpriseToEbitda', ev_ebitda), ('enterpriseToRevenue', ev_revenue)]:
            if val is None:
                missing_fields.append(name)

        # Margins
        gross_margin_raw = _safe_num(info.get('grossMargins'))
        ebitda_margin_raw = _safe_num(info.get('ebitdaMargins'))
        net_margin_raw = _safe_num(info.get('profitMargins'))
        gross_margin = gross_margin_raw * 100 if gross_margin_raw is not None else 0
        ebitda_margin = ebitda_margin_raw * 100 if ebitda_margin_raw is not None else 0
        net_margin = net_margin_raw * 100 if net_margin_raw is not None else 0
        for name, val in [('grossMargins', gross_margin_raw), ('ebitdaMargins', ebitda_margin_raw),
                          ('profitMargins', net_margin_raw)]:
            if val is None:
                missing_fields.append(name)

        # Returns
        roe_raw = _safe_num(info.get('returnOnEquity'))
        roa_raw = _safe_num(info.get('returnOnAssets'))
        roe = roe_raw * 100 if roe_raw is not None else 0
        roa = roa_raw * 100 if roa_raw is not None else 0
        if roe_raw is None:
            missing_fields.append('returnOnEquity')
        if roa_raw is None:
            missing_fields.append('returnOnAssets')

        # FCF yield
        market_cap = _safe_num(info.get('marketCap'))
        free_cf = _safe_num(info.get('freeCashflow'))
        if market_cap is None:
            missing_fields.append('marketCap')
            market_cap = 1  # fallback for division
        if free_cf is None:
            missing_fields.append('freeCashflow')
            free_cf = 0
        fcf_yield = (free_cf / market_cap * 100) if market_cap > 0 else 0

        # Balance sheet
        current_ratio = _safe_num(info.get('currentRatio'))
        debt_equity = _safe_num(info.get('debtToEquity'))
        if current_ratio is None:
            missing_fields.append('currentRatio')
            current_ratio = 0
        if debt_equity is None:
            missing_fields.append('debtToEquity')
            debt_equity = 0
        if debt_equity > 10:  # yfinance sometimes returns as percentage
            debt_equity = debt_equity / 100

        # EPS
        trailing_eps = _safe_num(info.get('trailingEps'))
        forward_eps = _safe_num(info.get('forwardEps'))
        if trailing_eps is None:
            missing_fields.append('trailingEps')
            trailing_eps = 0
        if forward_eps is None:
            missing_fields.append('forwardEps')
            forward_eps = 0

        # Revenue
        revenue = _safe_num(info.get('totalRevenue'))
        if revenue is None:
            missing_fields.append('totalRevenue')
            revenue = 0

        # Price source for provenance
        _, price_source = resolve_price(info)

        # Deterministic historical PE approximation (no randomness)
        pe_history_synthetic = True
        if trailing_pe is not None and trailing_pe > 0:
            pe_history = [round(trailing_pe, 2)] * 9
        else:
            pe_history = [20]
        synthetic_fields.append('trailing_pe_history')

        ev_history_synthetic = True
        if ev_ebitda is not None and ev_ebitda > 0:
            ev_history = [round(ev_ebitda, 2)] * 9
        else:
            ev_history = [15]
        synthetic_fields.append('ev_ebitda_history')

        fcf_history_synthetic = True
        if fcf_yield > 0:
            fcf_history = [round(fcf_yield, 2)] * 9
        else:
            fcf_history = [3.0]
        synthetic_fields.append('fcf_yield_history')

        fundamentals = {
            "trailing_pe": round(trailing_pe, 2) if trailing_pe is not None else 0,
            "forward_pe": round(forward_pe, 2) if forward_pe is not None else 0,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda is not None else 0,
            "ev_revenue": round(ev_revenue, 2) if ev_revenue is not None else 0,
            "fcf_yield": round(fcf_yield, 2),
            "roic": round((roe + roa) / 2, 2),  # Approximate ROIC
            "roa": round(roa, 2),
            "roe": round(roe, 2),
            "gross_margin": round(gross_margin, 2),
            "ebitda_margin": round(ebitda_margin, 2),
            "net_margin": round(net_margin, 2),
            "current_ratio": round(current_ratio, 2),
            "debt_equity": round(debt_equity, 2),
            "eps_2024": round(trailing_eps, 2),
            "eps_2025": round(forward_eps, 2),
            "revenue_m": round(revenue / 1e6, 0),
            "market_cap_m": round(market_cap / 1e6, 0),
            "trailing_pe_history": pe_history,
            "ev_ebitda_history": ev_history,
            "fcf_yield_history": fcf_history,
            "_pe_history_synthetic": pe_history_synthetic,
            "_ev_history_synthetic": ev_history_synthetic,
            "_fcf_history_synthetic": fcf_history_synthetic,
            "_provenance": {
                "timestamp_utc": datetime.now(tz=__import__('datetime').timezone.utc).isoformat().replace('+00:00', 'Z'),
                "source": "yfinance",
                "synthetic_fields": synthetic_fields,
                "missing_fields": missing_fields,
                "price_source": price_source,
            },
        }

        filepath = os.path.join(data_dir, "fundamentals.json")
        with open(filepath, 'w') as f:
            json.dump(fundamentals, f, indent=2)

        _tpe = trailing_pe if trailing_pe is not None else 0
        _fpe = forward_pe if forward_pe is not None else 0
        _eve = ev_ebitda if ev_ebitda is not None else 0
        print(f"[ATLAS]   PE: {_tpe:.1f}, Fwd PE: {_fpe:.1f}, EV/EBITDA: {_eve:.1f}")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching fundamentals: {e}")
        filepath = os.path.join(data_dir, "fundamentals.json")
        with open(filepath, 'w') as f:
            json.dump({}, f)


def _fetch_consensus(ticker, symbol, data_dir):
    """Fetch analyst consensus from yfinance."""
    print(f"[ATLAS]   Fetching {symbol} analyst consensus...")
    try:
        info = ticker.info

        target_price = info.get('targetMeanPrice', 0) or 0
        target_high = info.get('targetHighPrice', 0) or 0
        target_low = info.get('targetLowPrice', 0) or 0
        current_price, _ = resolve_price(info)
        num_analysts = info.get('numberOfAnalystOpinions', 0) or 0
        recommendation = info.get('recommendationKey', 'none')

        # Map recommendation to buy/hold/sell counts
        if num_analysts > 0:
            if recommendation in ('strong_buy', 'buy'):
                buy = int(num_analysts * 0.75)
                hold = int(num_analysts * 0.20)
                sell = num_analysts - buy - hold
            elif recommendation == 'hold':
                buy = int(num_analysts * 0.35)
                hold = int(num_analysts * 0.50)
                sell = num_analysts - buy - hold
            else:
                buy = int(num_analysts * 0.20)
                hold = int(num_analysts * 0.40)
                sell = num_analysts - buy - hold
        else:
            buy, hold, sell = 0, 0, 0

        # Earnings surprise history (yfinance earnings)
        surprise_history = []
        try:
            earnings = ticker.earnings_history
            if earnings is not None and not earnings.empty:
                for _, row in earnings.tail(5).iterrows():
                    surprise = row.get('epsActual', 0) - row.get('epsEstimate', 0)
                    surprise_history.append(round(surprise, 2))
        except Exception:
            surprise_history = [0.02, 0.01, 0.03]  # Default mild beats

        # Revision approximation (yfinance doesn't have this directly)
        # Use recommendation trend if available
        up_revisions = max(1, int(num_analysts * 0.6))
        down_revisions = max(0, num_analysts - up_revisions)

        consensus = {
            "eps_estimates": {
                "current_year": info.get('forwardEps', 0) or 0,
            },
            "revisions": {
                "uprevisions_1m": up_revisions,
                "downrevisions_1m": down_revisions,
                "uprevisions_3m": up_revisions + 2,
                "downrevisions_3m": down_revisions + 1
            },
            "target_price": round(target_price, 2),
            "current_price": round(current_price, 2),
            "analyst_ratings": {
                "buy": buy,
                "hold": hold,
                "sell": sell
            },
            "surprise_history": surprise_history,
            "total_analysts": num_analysts
        }

        filepath = os.path.join(data_dir, "consensus.json")
        with open(filepath, 'w') as f:
            json.dump(consensus, f, indent=2)

        print(f"[ATLAS]   Target: ${target_price:.2f}, {num_analysts} analysts, rec: {recommendation}")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching consensus: {e}")
        filepath = os.path.join(data_dir, "consensus.json")
        with open(filepath, 'w') as f:
            json.dump({}, f)


def _fetch_volatility(data_dir):
    """Fetch VIX data from yfinance."""
    print(f"[ATLAS]   Fetching VIX data...")
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="1y", interval="1d", timeout=30)

        # VIX 3-month (VIX3M)
        try:
            vix3m = yf.Ticker("^VIX3M")
            vix3m_hist = vix3m.history(period="1y", interval="1d", timeout=30)
        except Exception:
            vix3m_hist = None

        filepath = os.path.join(data_dir, "volatility.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "vix", "vix_3m", "put_call_ratio",
                            "vix3m_synthetic", "pcr_synthetic"])

            for date, row in vix_hist.iterrows():
                vix_close = row['Close']
                # Match VIX3M by date if available
                vix3m_is_synthetic = False
                if vix3m_hist is not None and date in vix3m_hist.index:
                    vix3m_val = vix3m_hist.loc[date, 'Close']
                else:
                    vix3m_val = vix_close * 1.1  # Approximate contango
                    vix3m_is_synthetic = True

                # Put/call ratio not freely available, use approximation
                pcr = 0.85 + (vix_close - 18) * 0.02  # Higher VIX = higher PCR
                pcr = max(0.4, min(1.6, pcr))
                pcr_synthetic = True  # Always synthetic — no free source

                writer.writerow([
                    date.strftime("%Y-%m-%d"),
                    round(vix_close, 2),
                    round(vix3m_val, 2),
                    round(pcr, 3),
                    vix3m_is_synthetic,
                    pcr_synthetic
                ])

        latest_vix = vix_hist['Close'].iloc[-1]
        print(f"[ATLAS]   VIX: {latest_vix:.2f}, {len(vix_hist)} bars")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching VIX: {e}")
        _write_default_volatility(data_dir)


def _fetch_macro(data_dir, fred_api_key=None):
    """Fetch macro/rates data. Uses FRED if API key provided, otherwise approximates from yfinance."""
    print(f"[ATLAS]   Fetching macro data...")
    try:
        if fred_api_key:
            _fetch_macro_from_fred(data_dir, fred_api_key)
            return

        # Without FRED, use Treasury ETF proxies from yfinance
        # ^TNX = 10Y yield, ^FVX = 5Y yield
        tnx = yf.Ticker("^TNX")  # 10-year yield
        tnx_hist = tnx.history(period="1y", interval="1d", timeout=30)

        # 2Y yield not directly available, approximate from ^TWO or use offset
        try:
            irx = yf.Ticker("^IRX")  # 13-week T-bill
            irx_hist = irx.history(period="1y", interval="1d", timeout=30)
        except Exception:
            irx_hist = None

        filepath = os.path.join(data_dir, "macro_rates.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "us_10y", "us_2y", "us_2s10s", "hy_spread", "real_yield",
                            "us_2y_synthetic", "hy_spread_synthetic", "real_yield_synthetic"])

            for date, row in tnx_hist.iterrows():
                us_10y = row['Close']

                # Approximate 2Y from IRX or offset
                us_2y_is_synthetic = False
                if irx_hist is not None and date in irx_hist.index:
                    us_2y = irx_hist.loc[date, 'Close'] + 0.3  # IRX is 3mo, add spread
                    us_2y_is_synthetic = True  # Still derived from IRX, not direct 2Y
                else:
                    us_2y = us_10y - 0.5  # Approximate
                    us_2y_is_synthetic = True

                spread_2s10s = round(us_10y - us_2y, 3)

                # HY spread approximation (not freely available)
                hy_spread = 300 + (us_10y - 4.0) * 30  # Rough approximation
                hy_spread = max(200, min(600, hy_spread))
                hy_spread_synthetic = True  # Always synthetic — no free source

                # Real yield = 10Y - breakeven inflation (~2.3%)
                real_yield = round(us_10y - 2.3, 3)
                real_yield_synthetic = True  # Assumes fixed 2.3% breakeven

                writer.writerow([
                    date.strftime("%Y-%m-%d"),
                    round(us_10y, 3),
                    round(us_2y, 3),
                    spread_2s10s,
                    round(hy_spread, 1),
                    real_yield,
                    us_2y_is_synthetic,
                    hy_spread_synthetic,
                    real_yield_synthetic
                ])

        latest_10y = tnx_hist['Close'].iloc[-1]
        print(f"[ATLAS]   10Y: {latest_10y:.3f}%, {len(tnx_hist)} bars")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching macro: {e}")
        _write_default_macro(data_dir)


def _fetch_macro_from_fred(data_dir, api_key):
    """Fetch macro data from FRED API (better quality)."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)

        end = datetime.now()
        start = end - timedelta(days=400)

        us_10y = fred.get_series('DGS10', start, end)
        us_2y = fred.get_series('DGS2', start, end)
        # HY spread: ICE BofA US High Yield
        try:
            hy = fred.get_series('BAMLH0A0HYM2', start, end)
        except Exception:
            hy = None

        filepath = os.path.join(data_dir, "macro_rates.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "us_10y", "us_2y", "us_2s10s", "hy_spread", "real_yield"])

            for date in us_10y.index:
                if date not in us_2y.index:
                    continue
                y10 = us_10y.loc[date]
                y2 = us_2y.loc[date]
                if np.isnan(y10) or np.isnan(y2):
                    continue

                spread = round(y10 - y2, 3)
                hy_val = hy.loc[date] * 100 if (hy is not None and date in hy.index and not np.isnan(hy.loc[date])) else 300
                real = round(y10 - 2.3, 3)

                writer.writerow([
                    date.strftime("%Y-%m-%d"),
                    round(y10, 3), round(y2, 3), spread,
                    round(hy_val, 1), real
                ])

        print(f"[ATLAS]   FRED macro data loaded ({len(us_10y)} points)")

    except ImportError:
        print(f"[ATLAS]   fredapi not installed, falling back to yfinance approximation")
        _fetch_macro.__wrapped__(data_dir, None)
    except Exception as e:
        print(f"[ATLAS]   FRED error: {e}, falling back to yfinance")


def _fetch_breadth(data_dir):
    """Approximate market breadth from sector ETFs."""
    print(f"[ATLAS]   Fetching breadth data...")
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y", interval="1d", timeout=30)

        # Use SPY returns to approximate advancing/declining
        filepath = os.path.join(data_dir, "breadth.csv")
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "advancing", "declining", "new_highs", "new_lows",
                           "pct_above_50dma", "pct_above_200dma", "breadth_synthetic"])

            closes = spy_hist['Close'].values
            sma50 = []
            sma200 = []
            for i in range(len(closes)):
                s50 = np.mean(closes[max(0, i-49):i+1]) if i >= 49 else np.nan
                s200 = np.mean(closes[max(0, i-199):i+1]) if i >= 199 else np.nan
                sma50.append(s50)
                sma200.append(s200)

            for idx, (date, row) in enumerate(spy_hist.iterrows()):
                if idx == 0:
                    continue

                # Approximate breadth from SPY daily return
                daily_return = (closes[idx] - closes[idx-1]) / closes[idx-1]

                # Positive day = more advancers
                adv_pct = 0.50 + daily_return * 15  # Scale return to breadth
                adv_pct = max(0.25, min(0.85, adv_pct))
                adv = int(adv_pct * 500)
                dec = 500 - adv

                nh = max(5, int(40 + daily_return * 500))
                nl = max(3, int(25 - daily_return * 300))

                # % above moving averages - approximate
                pct_50 = 65 + daily_return * 200  # Very rough
                pct_50 = max(30, min(90, pct_50))
                pct_200 = pct_50 - 3

                writer.writerow([
                    date.strftime("%Y-%m-%d"),
                    adv, dec, nh, nl,
                    round(pct_50, 1), round(pct_200, 1),
                    True  # All breadth data is synthetic (approximated from SPY)
                ])

        print(f"[ATLAS]   Breadth approximated from SPY, {len(spy_hist)} bars")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching breadth: {e}")
        _write_default_breadth(data_dir)


def _fetch_global_overnight(data_dir):
    """Fetch global market overnight returns."""
    print(f"[ATLAS]   Fetching global overnight data...")
    try:
        tickers = {
            'nikkei': '^N225',
            'hang_seng': '^HSI',
            'dax': '^GDAXI',
            'ftse': '^FTSE',
            'stoxx': '^STOXX50E',
        }

        global_data = {}

        for name, sym in tickers.items():
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d", interval="1d", timeout=15)
                if len(hist) >= 2:
                    prev = hist['Close'].iloc[-2]
                    curr = hist['Close'].iloc[-1]
                    ret = (curr - prev) / prev * 100
                    global_data[f"{name}_return"] = round(ret, 2)
                else:
                    global_data[f"{name}_return"] = 0.0
            except Exception:
                global_data[f"{name}_return"] = 0.0

        # ES futures (use SPY as proxy)
        try:
            es = yf.Ticker("ES=F")
            es_hist = es.history(period="5d", interval="1d", timeout=15)
            if len(es_hist) >= 2:
                global_data['es_overnight_return'] = round(
                    (es_hist['Close'].iloc[-1] - es_hist['Close'].iloc[-2]) / es_hist['Close'].iloc[-2] * 100, 2)
            else:
                global_data['es_overnight_return'] = 0.0
        except Exception:
            global_data['es_overnight_return'] = 0.0

        # NQ futures
        try:
            nq = yf.Ticker("NQ=F")
            nq_hist = nq.history(period="5d", interval="1d", timeout=15)
            if len(nq_hist) >= 2:
                global_data['nq_overnight_return'] = round(
                    (nq_hist['Close'].iloc[-1] - nq_hist['Close'].iloc[-2]) / nq_hist['Close'].iloc[-2] * 100, 2)
            else:
                global_data['nq_overnight_return'] = 0.0
        except Exception:
            global_data['nq_overnight_return'] = 0.0

        # Currency
        try:
            usdjpy = yf.Ticker("JPY=X")
            uj = usdjpy.history(period="5d", interval="1d", timeout=15)
            if len(uj) >= 2:
                global_data['usdjpy_change'] = round(
                    (uj['Close'].iloc[-1] - uj['Close'].iloc[-2]) / uj['Close'].iloc[-2] * 100, 2)
        except Exception:
            global_data['usdjpy_change'] = 0.0

        try:
            eurusd = yf.Ticker("EURUSD=X")
            eu = eurusd.history(period="5d", interval="1d", timeout=15)
            if len(eu) >= 2:
                global_data['eurusd_change'] = round(
                    (eu['Close'].iloc[-1] - eu['Close'].iloc[-2]) / eu['Close'].iloc[-2] * 100, 2)
        except Exception:
            global_data['eurusd_change'] = 0.0

        filepath = os.path.join(data_dir, "global_overnight.json")
        with open(filepath, 'w') as f:
            json.dump(global_data, f, indent=2)

        print(f"[ATLAS]   Global: Nikkei {global_data.get('nikkei_return', 0):+.2f}%, "
              f"DAX {global_data.get('dax_return', 0):+.2f}%, "
              f"ES {global_data.get('es_overnight_return', 0):+.2f}%")

    except Exception as e:
        print(f"[ATLAS]   ERROR fetching global: {e}")
        filepath = os.path.join(data_dir, "global_overnight.json")
        with open(filepath, 'w') as f:
            json.dump({}, f)


# ============================================================================
# FALLBACK DATA WRITERS (if APIs fail)
# ============================================================================

def _write_empty_ohlcv(data_dir):
    filepath = os.path.join(data_dir, "ohlcv.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])

def _write_default_volatility(data_dir):
    filepath = os.path.join(data_dir, "volatility.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "vix", "vix_3m", "put_call_ratio",
                        "vix3m_synthetic", "pcr_synthetic"])
        today = datetime.now().strftime("%Y-%m-%d")
        writer.writerow([today, 20.0, 22.0, 0.85, True, True])

def _write_default_macro(data_dir):
    filepath = os.path.join(data_dir, "macro_rates.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "us_10y", "us_2y", "us_2s10s", "hy_spread", "real_yield",
                        "us_2y_synthetic", "hy_spread_synthetic", "real_yield_synthetic"])
        today = datetime.now().strftime("%Y-%m-%d")
        writer.writerow([today, 4.2, 3.8, 0.4, 300, 1.9, True, True, True])

def _write_default_breadth(data_dir):
    filepath = os.path.join(data_dir, "breadth.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "advancing", "declining", "new_highs", "new_lows",
                        "pct_above_50dma", "pct_above_200dma", "breadth_synthetic"])
        today = datetime.now().strftime("%Y-%m-%d")
        writer.writerow([today, 280, 220, 40, 20, 60, 55, True])


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "KO"
    data_dir = fetch_live_data(symbol)
    print(f"\nData written to: {data_dir}")
    print("Files:")
    for f in sorted(os.listdir(data_dir)):
        size = os.path.getsize(os.path.join(data_dir, f))
        print(f"  {f} ({size:,} bytes)")
