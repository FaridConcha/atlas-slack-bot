#!/usr/bin/env python3
"""
ATLAS V12+ Phase 3 — 11-Layer Hierarchical Trading Engine
Importable Module with Regime-Aware Execution + Probabilistic Framework
Phase 1: Regime-conditioned interactions, half-Kelly, CVaR gate, fat-tailed growth, WACC regime scaling
Phase 2: Soft GMM regime, EMA smoothing, improved variance, regime correlations, dynamic risk blend
Phase 3: EWMA tracking, realized correlations, dynamic thresholds, cold-start inflation, shrinkage

Layers:
  0: Data Integrity Check
  1: Regime Vector (10 features)
  2: Score Normalization
  3: Meta-Regime Learning
  4: Dynamic Weight Matrix
  5: Composite Score
  6: Risk Governor
  7: Trade Quality Gate
  8: Portfolio Meta Policy
  9: Execution Microstructure
  10: Pyramid Report Output

Usage:
    from atlas_engine import run_atlas
    summary = run_atlas(symbol='SPY', data_path='./data/live')
"""

import json
import csv
import numpy as np
from math import erf, log
from datetime import datetime
from pathlib import Path
try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    # Lightweight fallback: approximate Student-t CDF for small ν using normal CDF
    class _StudentTFallback:
        @staticmethod
        def cdf(x, df):
            """Approximate Student-t CDF via normal CDF with df correction."""
            # Cornish-Fisher-style correction for small df
            z = x * (1 - 1 / (4 * max(df, 1)))
            return 0.5 * (1.0 + erf(z / np.sqrt(2)))
    student_t = _StudentTFallback()

try:
    from valuation_config import CONFIG as VAL_CONFIG
    _FF = VAL_CONFIG.flags
except ImportError:
    # Standalone mode: create default flags
    class _DefaultFlags:
        native_engine_uncertainty = True
        anisotropic_gmm = True
        regime_transitions = True
        online_garch = True
        distributional_engines = True
        coskewness_tensor = True
        copula_tail_dep = True
        mc_feedback = True
        dpmm_regime = False
    _FF = _DefaultFlags()

# ============================================================================
# SECTION 1: IMPORTS + ATLAS_CONFIG
# ============================================================================

ATLAS_CONFIG = {
    'normalization': {
        'trend': 100,
        'valuation': 40,
        'consensus': 50,
        'volatility': 50,
        'macro': 50,
        'liquidity': 50,
        'global': 50,
        'correlation': 50,
    },
    'affinity_alpha': 0.25,
    'meta_learning_eta': 0.15,
    'meta_learning_lambda': 0.70,
    'meta_learning_floor': 0.04,
    'meta_learning_ceiling': 0.30,
    'risk_governor_tau': 0.40,
    'risk_governor_s': 0.15,
    'decision': {
        'tau_base': 15.0,
        'uncertainty_sensitivity': 0.5,
        'regime_tau_multiplier': {
            'Calm': 1.0, 'Chop': 1.5, 'Tightening Shock': 1.3,
            'Crisis Trend': 0.8, 'Credit Stress': 1.4,
        },
    },
}

# ============================================================================
# SECTION 1b: PROBABILISTIC ENGINE CONSTANTS
# ============================================================================

ENGINE_VARIANCE_BASE = {
    'trend': 100, 'valuation': 25, 'consensus': 50, 'volatility': 40,
    'macro': 40, 'liquidity': 50, 'global': 50, 'correlation': 50,
}

VARIANCE_DATA_PENALTIES = {
    'price_insufficient': 1.8,
    'macro_insufficient': 1.5,
    'vol_insufficient': 1.4,
    'fundamentals_missing': 1.3,
    'consensus_missing': 1.2,
    'global_overnight_missing': 1.2,
    'data_stale': 1.6,
}

ENGINE_DATA_DEPS = {
    'trend': ['price_insufficient', 'data_stale'],
    'valuation': ['fundamentals_missing', 'data_stale'],
    'consensus': ['consensus_missing', 'data_stale'],
    'volatility': ['vol_insufficient', 'price_insufficient', 'data_stale'],
    'macro': ['macro_insufficient', 'data_stale'],
    'liquidity': ['price_insufficient', 'data_stale'],
    'global': ['global_overnight_missing', 'data_stale'],
    'correlation': ['price_insufficient', 'vol_insufficient', 'macro_insufficient', 'data_stale'],
}

# 8×8 structural correlation matrix (engine order: trend, valuation, consensus,
# volatility, macro, liquidity, global, correlation)
ENGINE_STRUCTURAL_CORRELATIONS = np.array([
    [ 1.00,  0.10,  0.15, -0.30,  0.10,  0.35,  0.15,  0.10],  # trend
    [ 0.10,  1.00,  0.20,  0.05,  0.30,  0.10,  0.05,  0.05],  # valuation
    [ 0.15,  0.20,  1.00,  0.05,  0.10,  0.10,  0.10, -0.20],  # consensus (herding)
    [-0.30,  0.05,  0.05,  1.00,  0.15,  0.10,  0.20,  0.40],  # volatility
    [ 0.10,  0.30,  0.10,  0.15,  1.00,  0.15,  0.20,  0.15],  # macro
    [ 0.35,  0.10,  0.10,  0.10,  0.15,  1.00,  0.15,  0.10],  # liquidity
    [ 0.15,  0.05,  0.10,  0.20,  0.20,  0.15,  1.00,  0.15],  # global
    [ 0.10,  0.05, -0.20,  0.40,  0.15,  0.10,  0.15,  1.00],  # correlation
])

# Engine order: trend=0, valuation=1, consensus=2, volatility=3,
#               macro=4, liquidity=5, global=6, correlation=7

# Legacy sparse interaction matrix (preserved for backward compat / tests)
INTERACTION_MATRIX = np.zeros((8, 8))
INTERACTION_MATRIX[0, 3] = -0.003  # trend × volatility
INTERACTION_MATRIX[3, 0] = -0.003
INTERACTION_MATRIX[1, 4] = -0.002  # valuation × macro
INTERACTION_MATRIX[4, 1] = -0.002
INTERACTION_MATRIX[2, 7] = -0.002  # consensus × correlation
INTERACTION_MATRIX[7, 2] = -0.002

# --------------------------------------------------------------------------
# Regime-conditioned interaction matrices  A(r) = Σ_k π_k(r) A_k
# 6 economically motivated pairs × 5 regimes
# Spectral bound ‖A_k‖₂ ≤ 0.05 enforced at load time
# --------------------------------------------------------------------------
_REGIME_INTERACTION_PAIRS = {
    # (i, j): {regime: coefficient}
    # trend × volatility — momentum dampened in high vol
    (0, 3): {'Calm': 0.00, 'Chop': -0.02, 'Tightening Shock': -0.01,
             'Crisis Trend': -0.03, 'Credit Stress': -0.02},
    # valuation × macro — value traps in tightening
    (1, 4): {'Calm': 0.00, 'Chop': -0.01, 'Tightening Shock': -0.03,
             'Crisis Trend': -0.02, 'Credit Stress': -0.02},
    # consensus × correlation — herding risk
    (2, 7): {'Calm': 0.00, 'Chop': -0.01, 'Tightening Shock': -0.01,
             'Crisis Trend': -0.02, 'Credit Stress': -0.03},
    # trend × liquidity — liquidity-momentum coupling
    (0, 5): {'Calm': 0.01, 'Chop': 0.00, 'Tightening Shock': 0.00,
             'Crisis Trend': 0.02, 'Credit Stress': 0.00},
    # volatility × correlation — vol clustering
    (3, 7): {'Calm': 0.00, 'Chop': -0.01, 'Tightening Shock': -0.01,
             'Crisis Trend': -0.02, 'Credit Stress': -0.03},
    # macro × global — global contagion
    (4, 6): {'Calm': 0.00, 'Chop': 0.00, 'Tightening Shock': -0.02,
             'Crisis Trend': -0.01, 'Credit Stress': -0.02},
}

REGIME_NAMES = ['Calm', 'Chop', 'Tightening Shock', 'Crisis Trend', 'Credit Stress']

def _enforce_spectral_bound(A, alpha_max=0.05):
    """Clip eigenvalues of symmetric A so ‖A‖₂ ≤ alpha_max."""
    eigvals, eigvecs = np.linalg.eigh(A)
    max_abs = np.max(np.abs(eigvals))
    if max_abs > alpha_max:
        eigvals = np.clip(eigvals, -alpha_max, alpha_max)
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A

def _build_regime_interaction_matrix(regime_name):
    """Build 8×8 symmetric interaction matrix for a given regime."""
    A = np.zeros((8, 8))
    for (i, j), regime_vals in _REGIME_INTERACTION_PAIRS.items():
        val = regime_vals.get(regime_name, 0.0)
        A[i, j] = val
        A[j, i] = val
    return _enforce_spectral_bound(A, alpha_max=0.05)

# Pre-build all 5 regime matrices at import time (zero runtime cost)
REGIME_INTERACTION_MATRICES = {
    regime: _build_regime_interaction_matrix(regime)
    for regime in REGIME_NAMES
}

# Kelly criterion risk aversion: λ = 2.0 → half-Kelly
# Full Kelly (λ=1) is provably optimal for log-utility but empirically
# over-bets; half-Kelly sacrifices ~25% return for ~50% variance reduction.
KELLY_LAMBDA = 2.0

# Kelly regime caps
KELLY_REGIME_CAPS = {
    'Calm': 0.15, 'Chop': 0.08, 'Tightening Shock': 0.10,
    'Crisis Trend': 0.05, 'Credit Stress': 0.06,
}

# --------------------------------------------------------------------------
# Soft GMM Regime Model (Phase 2a)
# 5 centroids in 10D feature space [TS,CH,VL,VS,CI,RS,CS,GR,BM_f,BEI]
# Expert-calibrated from empirical regime behavior
# --------------------------------------------------------------------------
GMM_REGIME_CENTROIDS = np.array([
    # Calm: low vol, low chop, moderate trend, low stress
    [ 0.30, 0.25, 0.15, 0.10, 0.10, 0.10, 0.05, 0.10, 0.05, 0.05],
    # Chop: near-zero trend, high choppiness, moderate vol
    [ 0.00, 0.70, 0.35, 0.25, 0.30, 0.15, 0.10, 0.15, 0.15, 0.20],
    # Tightening Shock: rates shock, moderate stress, some vol
    [ 0.10, 0.40, 0.40, 0.35, 0.25, 0.70, 0.30, 0.20, 0.10, 0.25],
    # Crisis Trend: strong negative trend, high vol, high stress
    [-0.70, 0.30, 0.75, 0.70, 0.60, 0.40, 0.40, 0.60, 0.50, 0.60],
    # Credit Stress: high credit spreads, moderate vol, correlation breakdown
    [-0.20, 0.45, 0.55, 0.45, 0.50, 0.35, 0.80, 0.40, 0.30, 0.40],
])

# Prior probabilities: reflects typical market state distribution
GMM_REGIME_PRIORS = np.array([0.45, 0.20, 0.15, 0.10, 0.10])

# Isotropic bandwidth per regime (legacy — used when anisotropic_gmm=False)
GMM_REGIME_BANDWIDTHS = np.array([0.20, 0.25, 0.25, 0.30, 0.25])

# Anisotropic 10D diagonal bandwidths per regime (Step 3)
# Rows: [Calm, Chop, Tightening, Crisis, Credit], Cols: [TS,CH,VL,VS,CI,RS,CS,GR,BM_f,BEI]
GMM_REGIME_BANDWIDTHS_DIAG = np.array([
    [0.20, 0.25, 0.15, 0.15, 0.20, 0.20, 0.15, 0.20, 0.20, 0.20],  # Calm
    [0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],  # Chop
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.20, 0.25, 0.25, 0.25],  # Tight
    [0.15, 0.25, 0.20, 0.20, 0.25, 0.30, 0.30, 0.25, 0.30, 0.25],  # Crisis
    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.25, 0.25, 0.25],  # Credit
])

# EMA smoothing factor for regime transitions (Phase 2b)
REGIME_EMA_ALPHA = 0.3

# 5×5 regime transition matrix (Step 4): rows = from, cols = to
# Encodes regime persistence (high diagonal) and transition probabilities
REGIME_TRANSITION_MATRIX = np.array([
    [0.85, 0.08, 0.04, 0.02, 0.01],  # Calm →
    [0.10, 0.70, 0.10, 0.05, 0.05],  # Chop →
    [0.05, 0.10, 0.70, 0.08, 0.07],  # Tightening →
    [0.03, 0.07, 0.05, 0.75, 0.10],  # Crisis →
    [0.02, 0.08, 0.10, 0.10, 0.70],  # Credit →
])

# Regime-conditioned engine variance multiplier (Phase 2c)
ENGINE_VARIANCE_REGIME_MULT = {
    'Calm': 1.0, 'Chop': 1.4, 'Tightening Shock': 1.3,
    'Crisis Trend': 1.6, 'Credit Stress': 1.5,
}

# Regime-conditioned engine correlation matrices (Phase 2d)
# In stress regimes, cross-engine correlations increase (contagion effect)
_REGIME_CORR_SHIFTS = {
    'Calm':              0.00,   # structural prior as-is
    'Chop':              0.05,   # mild correlation increase
    'Tightening Shock':  0.08,   # moderate shift
    'Crisis Trend':      0.15,   # strong contagion
    'Credit Stress':     0.12,   # credit-driven contagion
}

# Dynamic structural/tactical risk blend (Phase 2e)
RISK_BLEND_ALPHA = {
    'Calm':              0.70,   # 70% structural, 30% tactical
    'Chop':              0.55,   # more balanced
    'Tightening Shock':  0.50,   # balanced
    'Crisis Trend':      0.30,   # 30% structural, 70% tactical (fast-moving)
    'Credit Stress':     0.40,   # lean tactical
}

# Phase 3c: Dynamic threshold constants
DYNAMIC_THRESHOLD_K1 = 0.5   # uncertainty sensitivity (same as existing)
DYNAMIC_THRESHOLD_K2 = 0.3   # regime entropy sensitivity

# Phase 3d: Cold-start σ inflation schedule
COLD_START_RUN_THRESHOLD = 20   # runs needed for full convergence
COLD_START_SIGMA_MULT = 2.0     # max inflation at run_count=0

# Phase 3e: Uncertainty shrinkage
SHRINKAGE_LAMBDA_BASE = 0.8   # blend weight toward prior at cold start
SHRINKAGE_DECAY_RUNS = 50     # runs to decay shrinkage to near-zero

# ============================================================================
# SECTION 2: HELPERS (KEEP EXACTLY AS-IS)
# ============================================================================

def safe_float(val, default=0.0):
    """Convert value to float safely."""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def sma(prices, period):
    """Calculate simple moving average."""
    if len(prices) < period:
        return [np.nan] * len(prices)
    sma_vals = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_vals.append(np.nan)
        else:
            sma_vals.append(np.mean(prices[i-period+1:i+1]))
    return sma_vals

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr_list = []
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_list.append(tr)

    atr_vals = []
    for i in range(len(tr_list)):
        if i < period - 1:
            atr_vals.append(np.nan)
        else:
            atr_vals.append(np.mean(tr_list[i-period+1:i+1]))
    return atr_vals

def z_score(values, window=None):
    """Calculate z-score of values."""
    if window is None:
        window = len([v for v in values if not np.isnan(v)])

    valid_vals = [v for v in values if not np.isnan(v)]
    if not valid_vals:
        return 0.0

    mean = np.mean(valid_vals[-window:])
    std = np.std(valid_vals[-window:])
    if std == 0:
        return 0.0

    current = valid_vals[-1]
    return (current - mean) / std

def slope(prices, period=5):
    """Calculate slope of prices over period."""
    valid_prices = [p for p in prices if not np.isnan(p)]
    if len(valid_prices) < period:
        return 0.0

    x = np.arange(period)
    y = np.array(valid_prices[-period:])
    try:
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    except (ValueError, TypeError):
        return 0.0

# ============================================================================
# SECTION 3: DATA LOADING (KEEP EXACTLY AS-IS)
# ============================================================================

def load_csv(filepath):
    """Load CSV file and return rows as list of dicts."""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return []
    return data

def load_json(filepath):
    """Load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return {}

def setup_data(data_path=None):
    """Load all data files from the given directory."""
    if data_path is None:
        data_path = Path(__file__).parent / "data" / "default"
    base_path = Path(data_path)

    price_data = load_csv(base_path / "ohlcv.csv")
    macro_data = load_csv(base_path / "macro_rates.csv")
    vol_data = load_csv(base_path / "volatility.csv")
    breadth_data = load_csv(base_path / "breadth.csv")
    fundamentals = load_json(base_path / "fundamentals.json")
    consensus = load_json(base_path / "consensus.json")
    global_overnight = load_json(base_path / "global_overnight.json")

    return {
        'price': price_data,
        'macro': macro_data,
        'vol': vol_data,
        'breadth': breadth_data,
        'fundamentals': fundamentals,
        'consensus': consensus,
        'global_overnight': global_overnight
    }

# ============================================================================
# SECTION 4: 8 RAW ENGINES (KEEP EXACTLY AS-IS, FIX VALUATION SIGN BUG)
# ============================================================================

def engine_trend(price_data, breadth_data):
    """
    Trend Engine: MA-based scoring with breakout detection.
    Range: -100 to +100
    """
    if not price_data or len(price_data) < 200:
        return 0, {}

    closes = [safe_float(row['close']) for row in price_data]
    volumes = [safe_float(row['volume']) for row in price_data]
    highs = [safe_float(row['high']) for row in price_data]
    lows = [safe_float(row['low']) for row in price_data]

    sma20_vals = sma(closes, 20)
    sma50_vals = sma(closes, 50)
    sma200_vals = sma(closes, 200)

    price = closes[-1]
    vol = volumes[-1]
    vol_20ma = np.mean(volumes[-20:])

    score = 0
    details = {}

    if price > sma20_vals[-1]:
        score += 20
    else:
        score -= 20

    if price > sma50_vals[-1]:
        score += 30
    else:
        score -= 30

    if price > sma200_vals[-1]:
        score += 20
    else:
        score -= 20

    if sma20_vals[-1] > sma50_vals[-1] > sma200_vals[-1]:
        score += 15
    elif sma20_vals[-1] < sma50_vals[-1] < sma200_vals[-1]:
        score -= 15

    ma_slope = slope(sma20_vals, 5)
    score += np.clip(ma_slope * 100, -30, 30)

    highest_5 = max(highs[-5:])
    if price >= highest_5 * 0.98:
        score += 10
    if price >= highest_5:
        score += 10

    if vol > vol_20ma * 1.2:
        if price > closes[-2]:
            score += 10
        else:
            score += 5
    elif vol < vol_20ma * 0.6:
        score -= 5

    if breadth_data and len(breadth_data) > 0:
        last_breadth = breadth_data[-1]
        advancing = safe_float(last_breadth.get('advancing', 0))
        declining = safe_float(last_breadth.get('declining', 0))
        if advancing + declining > 0:
            adv_ratio = advancing / (advancing + declining)
            if adv_ratio > 0.65:
                score += 10
            elif adv_ratio < 0.35:
                score -= 10

    score = np.clip(score, -100, 100)

    details = {
        'price': price,
        'sma20': sma20_vals[-1],
        'sma50': sma50_vals[-1],
        'sma200': sma200_vals[-1],
        'vol_vs_avg': vol / vol_20ma if vol_20ma > 0 else 1.0
    }

    # Native variance: bootstrap variance from 5 overlapping 20-day return windows
    if _FF.native_engine_uncertainty and len(closes) >= 40:
        window_returns = []
        for start in range(0, min(100, len(closes) - 20), 20):
            w = closes[-(start + 20):len(closes) - start] if start > 0 else closes[-20:]
            if len(w) >= 2:
                r = (w[-1] - w[0]) / w[0] * 100 if w[0] > 0 else 0
                window_returns.append(r)
        details['native_var'] = float(np.var(window_returns)) if len(window_returns) >= 2 else 0.0
    else:
        details['native_var'] = 0.0

    # Distributional: skewness/kurtosis from return distribution
    if _FF.distributional_engines and len(closes) >= 21:
        daily_r = np.diff(np.log(np.maximum(np.array(closes[-21:]), 1e-10)))
        mu_r = np.mean(daily_r)
        std_r = np.std(daily_r)
        if std_r > 1e-10:
            z = (daily_r - mu_r) / std_r
            details['skewness'] = float(np.mean(z ** 3))
            details['kurtosis'] = float(np.mean(z ** 4) - 3.0)  # excess kurtosis
        else:
            details['skewness'] = 0.0
            details['kurtosis'] = 0.0
    else:
        details['skewness'] = 0.0
        details['kurtosis'] = 0.0

    return score, details

def engine_valuation(fundamentals, macro_data):
    """
    Valuation Engine: PE, EV/EBITDA, FCF scoring.
    Range: -40 to +40 (weak timing signal)
    FIX: score -= pe_z * 8, score -= ev_z * 5
    """
    score = 0
    details = {}

    if not fundamentals:
        return 0, details

    current_pe = safe_float(fundamentals.get('trailing_pe', 25))
    pe_hist = fundamentals.get('trailing_pe_history', [current_pe])

    current_ev = safe_float(fundamentals.get('ev_ebitda', 18))
    ev_hist = fundamentals.get('ev_ebitda_history', [current_ev])

    current_fcf = safe_float(fundamentals.get('fcf_yield', 3.5))
    fcf_hist = fundamentals.get('fcf_yield_history', [current_fcf])

    pe_mean = np.mean([safe_float(x) for x in pe_hist])
    pe_std = np.std([safe_float(x) for x in pe_hist])
    if pe_std > 0:
        pe_z = (current_pe - pe_mean) / pe_std
    else:
        pe_z = 0

    if pe_z > 1.5:
        score -= 20
    elif pe_z < -1.5:
        score += 20
    else:
        score -= pe_z * 8  # FIX: negative sign

    ev_mean = np.mean([safe_float(x) for x in ev_hist])
    ev_std = np.std([safe_float(x) for x in ev_hist])
    if ev_std > 0:
        ev_z = (current_ev - ev_mean) / ev_std
    else:
        ev_z = 0

    if ev_z > 1.5:
        score -= 12
    elif ev_z < -1.5:
        score += 12
    else:
        score -= ev_z * 5  # FIX: negative sign

    fcf_mean = np.mean([safe_float(x) for x in fcf_hist])
    if current_fcf > fcf_mean:
        score += 8
    else:
        score -= 8

    if macro_data and len(macro_data) > 0:
        real_yield = safe_float(macro_data[-1].get('real_yield', 0))
        if real_yield > 2.0 and current_pe > pe_mean:
            score -= 10

    score = np.clip(score, -40, 40)

    # Compute per-component contributions for native variance
    pe_component = -pe_z * 8 if abs(pe_z) <= 1.5 else (-20 if pe_z > 1.5 else 20)
    ev_component = -ev_z * 5 if abs(ev_z) <= 1.5 else (-12 if ev_z > 1.5 else 12)
    fcf_component = 8 if current_fcf > fcf_mean else -8

    details = {
        'pe_z_score': pe_z,
        'ev_z_score': ev_z,
        'pe_current': current_pe,
        'pe_mean': pe_mean
    }

    # Native variance: cross-metric disagreement
    if _FF.native_engine_uncertainty:
        details['native_var'] = float(np.var([pe_component, ev_component, fcf_component]))
    else:
        details['native_var'] = 0.0

    # Distributional: skewness from PE z-score (value trap asymmetry), kurtosis from PE distribution
    if _FF.distributional_engines:
        details['skewness'] = float(np.sign(pe_z) * abs(pe_z) / 3.0)
        pe_vals = [safe_float(x) for x in pe_hist]
        if len(pe_vals) >= 4:
            pe_arr = np.array(pe_vals)
            pe_mu = np.mean(pe_arr)
            pe_sd = np.std(pe_arr)
            if pe_sd > 1e-10:
                z = (pe_arr - pe_mu) / pe_sd
                details['kurtosis'] = float(np.mean(z ** 4) - 3.0)
            else:
                details['kurtosis'] = 0.0
        else:
            details['kurtosis'] = 0.0
    else:
        details['skewness'] = 0.0
        details['kurtosis'] = 0.0

    return score, details

def engine_consensus(consensus_data):
    """
    Consensus Engine: Revisions, surprises, target price.
    Range: -50 to +50
    """
    score = 0
    details = {}

    if not consensus_data:
        return 0, details

    up_1m = safe_float(consensus_data.get('revisions', {}).get('uprevisions_1m', 0))
    down_1m = safe_float(consensus_data.get('revisions', {}).get('downrevisions_1m', 0))

    if up_1m + down_1m > 0:
        revision_ratio = up_1m / (up_1m + down_1m)
    else:
        revision_ratio = 0.5

    if revision_ratio > 0.65:
        score += 30
    elif revision_ratio < 0.35:
        score -= 30
    else:
        score += (revision_ratio - 0.5) * 60

    surprise_hist = consensus_data.get('surprise_history', [])
    if len(surprise_hist) >= 3:
        if all(s > 0 for s in surprise_hist[-3:]):
            score += 25
        elif all(s < 0 for s in surprise_hist[-3:]):
            score -= 25

    target_price = safe_float(consensus_data.get('target_price', 0))
    current_price = safe_float(consensus_data.get('current_price', 0))

    if current_price > 0 and target_price > 0:
        target_premium = (target_price - current_price) / current_price
        score += np.clip(target_premium * 100, -20, 20)

    ratings = consensus_data.get('analyst_ratings', {})
    buy = safe_float(ratings.get('buy', 0))
    hold = safe_float(ratings.get('hold', 0))
    sell = safe_float(ratings.get('sell', 0))
    total = buy + hold + sell

    if total > 0:
        buy_pct = buy / total
        if buy_pct > 0.85:
            score -= 15

    score = np.clip(score, -50, 50)

    details = {
        'revision_ratio': revision_ratio,
        'target_price': target_price,
        'current_price': current_price,
        'buy_pct': buy_pct if total > 0 else 0
    }

    # Native variance: revision ratio uncertainty (Bernoulli variance × 100)
    if _FF.native_engine_uncertainty:
        details['native_var'] = float(revision_ratio * (1 - revision_ratio) * 100)
    else:
        details['native_var'] = 0.0

    # Distributional: default (insufficient data for reliable higher moments)
    details['skewness'] = 0.0
    details['kurtosis'] = 0.0

    return score, details

def engine_volatility(vol_data, price_data):
    """
    Volatility Engine: VIX regime, IV vs RV, put/call.
    Range: -50 to +30
    """
    score = 0
    details = {}

    if not vol_data or len(vol_data) < 2:
        return 0, details

    vix = safe_float(vol_data[-1].get('vix', 20))
    vix_prev = safe_float(vol_data[-2].get('vix', 20))
    vix_3m = safe_float(vol_data[-1].get('vix_3m', 20))
    put_call = safe_float(vol_data[-1].get('put_call_ratio', 1.0))

    details['vix'] = vix

    if vix < 15:
        score += 15
    elif vix < 22:
        score -= 5
    elif vix < 30:
        if vix < vix_prev:
            score += 15
        else:
            score -= 25
    else:
        score -= 35

    if vix > vix_3m:
        score -= 20
    elif vix_3m > vix + 5:
        score -= 5
    else:
        score += 5

    if price_data and len(price_data) > 1:
        closes = [safe_float(row['close']) for row in price_data]
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0 and closes[i] > 0:
                returns.append(np.log(closes[i] / closes[i-1]))

        if len(returns) >= 20:
            rv_20 = np.std(returns[-20:]) * np.sqrt(252)
            iv_rv_ratio = vix / (rv_20 * 100) if rv_20 > 0 else 1.0
        else:
            iv_rv_ratio = 1.0

        details['rv_20d'] = rv_20 if len(returns) >= 20 else np.nan
        details['iv_rv_ratio'] = iv_rv_ratio  # D20 FIX: store for diagnostics

    if put_call > 1.3:
        score += 20
    elif put_call < 0.5:
        score -= 20

    score = np.clip(score, -50, 30)

    # Native variance: IV-RV spread variance
    if _FF.native_engine_uncertainty:
        iv_rv = details.get('iv_rv_ratio', 1.0)
        details['native_var'] = float((iv_rv - 1.0) ** 2 * 50)
    else:
        details['native_var'] = 0.0

    # Distributional: vol has positive skew (spikes up), kurtosis from VIX history
    if _FF.distributional_engines and vol_data and len(vol_data) >= 20:
        vix_hist = [safe_float(row.get('vix', 20)) for row in vol_data[-20:]]
        vix_arr = np.array(vix_hist)
        vix_mu = np.mean(vix_arr)
        vix_sd = np.std(vix_arr)
        if vix_sd > 1e-10:
            z = (vix_arr - vix_mu) / vix_sd
            details['skewness'] = float(np.mean(z ** 3))
            details['kurtosis'] = float(np.mean(z ** 4) - 3.0)
        else:
            details['skewness'] = 0.0
            details['kurtosis'] = 0.0
    else:
        details['skewness'] = 0.0
        details['kurtosis'] = 0.0

    return score, details

def engine_macro(macro_data):
    """
    Macro Engine: Yields, real yield, curve shape, credit spreads.
    Range: -50 to +30
    """
    score = 0
    details = {}

    if not macro_data or len(macro_data) < 5:
        return 0, details

    us_10y = safe_float(macro_data[-1].get('us_10y', 2.5))
    us_2y = safe_float(macro_data[-1].get('us_2y', 2.5))
    real_yield = safe_float(macro_data[-1].get('real_yield', 0.5))
    hy_spread = safe_float(macro_data[-1].get('hy_spread', 300))

    us_10y_5d = safe_float(macro_data[-6].get('us_10y', us_10y)) if len(macro_data) >= 6 else us_10y
    us_10y_20d = safe_float(macro_data[-21].get('us_10y', us_10y)) if len(macro_data) >= 21 else us_10y
    hy_spread_5d = safe_float(macro_data[-6].get('hy_spread', hy_spread)) if len(macro_data) >= 6 else hy_spread

    yield_5d_change = us_10y - us_10y_5d
    yield_20d_change = us_10y - us_10y_20d
    spread_5d_change = hy_spread - hy_spread_5d

    details['us_10y'] = us_10y
    details['real_yield'] = real_yield
    details['hy_spread'] = hy_spread

    if yield_5d_change > 0.2:
        score -= 20
    elif yield_5d_change > 0.05:
        score -= 10

    if real_yield > 2.0:
        score -= 15
    elif real_yield < 0:
        score += 10

    spread = us_10y - us_2y
    if spread < -0.5:
        score -= 20
    elif spread > 1.0:
        score += 15

    if spread_5d_change > 50:
        score -= 30
    elif spread_5d_change > 30:
        score -= 15
    elif spread_5d_change < -20:
        score += 10

    score = np.clip(score, -50, 30)

    # Native variance: rate change variance from 20d rolling std of yield changes
    if _FF.native_engine_uncertainty and len(macro_data) >= 21:
        yield_vals = [safe_float(row.get('us_10y', 2.5)) for row in macro_data[-21:]]
        yield_changes = [yield_vals[i] - yield_vals[i-1] for i in range(1, len(yield_vals))]
        details['native_var'] = float(np.var(yield_changes) * 1000) if yield_changes else 0.0
    else:
        details['native_var'] = 0.0

    # Distributional: skewness/kurtosis from yield change distribution
    if _FF.distributional_engines and len(macro_data) >= 21:
        yield_vals = [safe_float(row.get('us_10y', 2.5)) for row in macro_data[-21:]]
        yc = np.diff(yield_vals)
        yc_mu = np.mean(yc)
        yc_sd = np.std(yc)
        if yc_sd > 1e-10:
            z = (yc - yc_mu) / yc_sd
            details['skewness'] = float(np.mean(z ** 3))
            details['kurtosis'] = float(np.mean(z ** 4) - 3.0)
        else:
            details['skewness'] = 0.0
            details['kurtosis'] = 0.0
    else:
        details['skewness'] = 0.0
        details['kurtosis'] = 0.0

    return score, details

def engine_liquidity(price_data, breadth_data):
    """
    Liquidity Engine: Volume, breadth, new highs/lows.
    Range: -50 to +50
    """
    score = 0
    details = {}

    if not price_data or not breadth_data or len(price_data) < 20:
        return 0, details

    volumes = [safe_float(row['volume']) for row in price_data]
    vol_20ma = np.mean(volumes[-20:])
    vol_current = volumes[-1]

    vol_ratio = vol_current / vol_20ma if vol_20ma > 0 else 1.0

    if vol_ratio > 2.0:
        score += 20
    elif vol_ratio > 1.5:
        score += 10
    elif vol_ratio < 0.5:
        score -= 15

    details['vol_ratio'] = vol_ratio

    if breadth_data and len(breadth_data) > 0:
        last_breadth = breadth_data[-1]
        advancing = safe_float(last_breadth.get('advancing', 0))
        declining = safe_float(last_breadth.get('declining', 0))
        new_highs = safe_float(last_breadth.get('new_highs', 0))
        new_lows = safe_float(last_breadth.get('new_lows', 0))
        pct_50dma = safe_float(last_breadth.get('pct_above_50dma', 50))
        pct_200dma = safe_float(last_breadth.get('pct_above_200dma', 50))

        total = advancing + declining
        if total > 0:
            adv_pct = advancing / total

            if adv_pct > 0.8:
                score += 30
            elif adv_pct > 0.65:
                score += 15
            elif adv_pct < 0.35:
                score -= 15
            elif adv_pct < 0.5:
                score -= 8

        if new_highs > 0 and new_lows > 0:
            nh_nl_ratio = new_highs / new_lows if new_lows > 0 else 10
            if nh_nl_ratio > 2:
                score += 15
            elif nh_nl_ratio < 0.5:
                score -= 15

        if pct_50dma > 80:
            score += 10
        elif pct_50dma < 35:
            score -= 10

        details['breadth_advancing_pct'] = adv_pct if total > 0 else 0.5

    score = np.clip(score, -50, 50)

    # Native variance: volume ratio variance
    if _FF.native_engine_uncertainty and len(volumes) >= 20 and vol_20ma > 0:
        details['native_var'] = float(np.var(volumes[-20:]) / vol_20ma ** 2 * 50)
    else:
        details['native_var'] = 0.0

    # Distributional: default (insufficient data for reliable higher moments)
    details['skewness'] = 0.0
    details['kurtosis'] = 0.0

    return score, details

def engine_global_overnight(global_overnight, vol_data):
    """
    Global Overnight Engine: Asia, Europe, US futures, VIX context.
    Range: -50 to +50
    """
    score = 0
    details = {}

    if not global_overnight:
        return 0, details

    nikkei = safe_float(global_overnight.get('nikkei_return', 0))
    hang_seng = safe_float(global_overnight.get('hang_seng_return', 0))
    usdjpy = safe_float(global_overnight.get('usdjpy_change', 0))

    asia_score = 0
    if nikkei > 0.5:
        asia_score += 10
    elif nikkei < -0.5:
        asia_score -= 10

    if hang_seng > 0.5:
        asia_score += 10
    elif hang_seng < -0.5:
        asia_score -= 10

    dax = safe_float(global_overnight.get('dax_return', 0))
    ftse = safe_float(global_overnight.get('ftse_return', 0))
    stoxx = safe_float(global_overnight.get('stoxx_return', 0))
    eurusd = safe_float(global_overnight.get('eurusd_change', 0))

    europe_score = 0
    if dax > 0.5:
        europe_score += 8
    elif dax < -0.5:
        europe_score -= 8

    if ftse > 0.5:
        europe_score += 8
    elif ftse < -0.5:
        europe_score -= 8

    es_overnight = safe_float(global_overnight.get('es_overnight_return', 0))
    nq_overnight = safe_float(global_overnight.get('nq_overnight_return', 0))

    us_score = 0
    if es_overnight > 0.3:
        us_score += 15
    elif es_overnight < -0.3:
        us_score -= 15

    if nq_overnight > 0.3:
        us_score += 10
    elif nq_overnight < -0.3:
        us_score -= 10

    vix = 20.0
    if vol_data and len(vol_data) > 0:
        vix = safe_float(vol_data[-1].get('vix', 20))

    multiplier = 1.3 if vix > 25 else 0.6 if vix < 15 else 1.0

    details['asia_score'] = asia_score
    details['europe_score'] = europe_score
    details['us_score'] = us_score
    details['vix_multiplier'] = multiplier

    score = (asia_score + europe_score + us_score) * multiplier
    score = np.clip(score, -50, 50)

    # Native variance: cross-region disagreement
    if _FF.native_engine_uncertainty:
        details['native_var'] = float(np.var([asia_score, europe_score, us_score]))
    else:
        details['native_var'] = 0.0

    # Distributional: default (insufficient data for reliable higher moments)
    details['skewness'] = 0.0
    details['kurtosis'] = 0.0

    return score, details

def engine_correlation(price_data, vol_data, macro_data):
    """
    Correlation Engine: SPX vs bonds, SPX vs VIX, correlation instability.
    Outputs RISK SCORE 0-100 (higher = more risk).
    """
    risk_score = 50
    details = {}

    if not price_data or len(price_data) < 60:
        return risk_score, details

    closes = np.array([safe_float(row['close']) for row in price_data])
    closes = np.maximum(closes, 1e-10)  # D12: prevent log(0) on missing closes
    returns = np.diff(np.log(closes))

    if macro_data and len(macro_data) >= 60:
        yields = np.array([safe_float(row.get('us_10y', 2.5)) for row in macro_data[-len(price_data):]])
        yield_changes = np.diff(yields)
    else:
        yield_changes = np.zeros(len(returns))

    if vol_data and len(vol_data) >= 60:
        vix_vals = np.array([safe_float(row.get('vix', 20)) for row in vol_data[-len(price_data):]])
    else:
        vix_vals = np.ones(len(returns)) * 20

    if len(returns) >= 60:
        corr_20_spx_yield = []
        corr_60_spx_yield = []

        for i in range(20, len(returns)):
            if i >= 60:
                c60 = np.corrcoef(returns[i-60:i], yield_changes[i-60:i])[0, 1]
                if not np.isnan(c60):
                    corr_60_spx_yield.append(c60)
            if i >= 20:
                c20 = np.corrcoef(returns[i-20:i], yield_changes[i-20:i])[0, 1]
                if not np.isnan(c20):
                    corr_20_spx_yield.append(c20)

        if corr_20_spx_yield:
            current_corr_20 = corr_20_spx_yield[-1]
            current_corr_60 = corr_60_spx_yield[-1] if corr_60_spx_yield else current_corr_20

            if current_corr_20 > 0.4 and returns[-1] < 0:
                risk_score += 20

            if len(corr_20_spx_yield) >= 5:
                recent_mean = np.mean(corr_20_spx_yield[-5:])
                if abs(current_corr_20 - recent_mean) > 0.3:
                    risk_score += 15

            details['corr_spx_yield_20d'] = current_corr_20
            details['corr_spx_yield_60d'] = current_corr_60

    if macro_data and len(macro_data) > 0:
        hy_spread = safe_float(macro_data[-1].get('hy_spread', 300))
        if hy_spread > 350:
            risk_score += 10

    if vol_data and len(vol_data) > 0:
        vix = safe_float(vol_data[-1].get('vix', 20))
        if vix > 25:
            risk_score += 10

    risk_score = np.clip(risk_score, 0, 100)

    # Native variance: rolling correlation std × 100
    if _FF.native_engine_uncertainty and 'corr_spx_yield_20d' in details:
        if len(returns) >= 60:
            corr_std = np.std(corr_20_spx_yield) if corr_20_spx_yield else 0.0
            details['native_var'] = float(corr_std * 100)
        else:
            details['native_var'] = 0.0
    else:
        details['native_var'] = 0.0

    # Distributional: default (insufficient data for reliable higher moments)
    details['skewness'] = 0.0
    details['kurtosis'] = 0.0

    return risk_score, details

# ============================================================================
# SECTION 4b: PROBABILISTIC ENGINE FRAMEWORK (P1–P6)
# ============================================================================

def _compute_engine_variance(engine_name, score, dc_details, score_range,
                              ensemble_sign=None, regime_label=None, mid=0.0,
                              native_var=None, run_count=0):
    """
    Compute variance estimate for a single engine.
    σ²_i = σ²_base × f_data × f_extremity × f_agreement × f_regime

    Phase 2c additions:
      f_agreement: penalizes engines disagreeing with ensemble direction
      f_regime: regime-conditioned variance multiplier
    D23 FIX: mid parameter allows per-engine midpoint (correlation: 50, others: 0).
    Redesign G1: blend native_var with heuristic when available.
    """
    sigma2_base = ENGINE_VARIANCE_BASE.get(engine_name, 50)

    # f_data: product of penalties for missing data dependencies
    f_data = 1.0
    for dep in ENGINE_DATA_DEPS.get(engine_name, []):
        if dc_details.get(dep, False):
            f_data *= VARIANCE_DATA_PENALTIES.get(dep, 1.0)

    # Additional penalty if data confidence is low
    dc = dc_details.get('final_dc', 100)
    if dc < 70:
        f_data *= (100.0 / max(dc, 10))

    # f_extremity: higher variance at score extremes (measured from engine midpoint)
    half_range = score_range / 2.0 if score_range > 0 else 1.0
    f_extremity = 1.0 + 0.5 * ((score - mid) / half_range) ** 2

    # f_agreement (Phase 2c): engines disagreeing with ensemble have higher uncertainty
    f_agreement = 1.0
    if ensemble_sign is not None and ensemble_sign != 0:
        engine_sign = 1 if score > 0 else (-1 if score < 0 else 0)
        if engine_sign != 0 and engine_sign != ensemble_sign:
            f_agreement = 1.5  # 50% variance inflation for disagreement

    # f_regime (Phase 2c): regime-conditioned variance scaling
    f_regime = ENGINE_VARIANCE_REGIME_MULT.get(regime_label, 1.0) if regime_label else 1.0

    sigma2_heuristic = sigma2_base * f_data * f_extremity * f_agreement * f_regime

    # Redesign G1: blend native variance with heuristic
    if _FF.native_engine_uncertainty and native_var is not None and native_var > 0:
        blend_w = min(0.7, run_count / 50.0)
        return (1 - blend_w) * sigma2_heuristic + blend_w * native_var
    return sigma2_heuristic


def compute_all_engine_variances(scores_dict, dc_details, regime_label=None,
                                  engine_details=None, run_count=0):
    """
    Compute variance estimates for all 8 engines.
    Phase 2c: includes f_agreement (ensemble disagreement) and f_regime.
    Redesign G1: passes native_var from engine details for blending.
    """
    score_ranges = {
        'trend': 200, 'valuation': 80, 'consensus': 100, 'volatility': 80,
        'macro': 80, 'liquidity': 100, 'global': 100, 'correlation': 100,
    }
    # D23 FIX: per-engine midpoints (correlation is 0-100 centered at 50, others symmetric around 0)
    score_mids = {
        'trend': 0, 'valuation': 0, 'consensus': 0, 'volatility': 0,
        'macro': 0, 'liquidity': 0, 'global': 0, 'correlation': 50,
    }

    # Determine ensemble direction (sign of mean normalized score)
    ensemble_mean = np.mean([s for s in scores_dict.values()])
    ensemble_sign = 1 if ensemble_mean > 0 else (-1 if ensemble_mean < 0 else 0)

    variances = {}
    for engine_name, score in scores_dict.items():
        sr = score_ranges.get(engine_name, 100)
        native_var = None
        if engine_details is not None:
            native_var = engine_details.get(engine_name, {}).get('native_var', None)
        variances[engine_name] = _compute_engine_variance(
            engine_name, score, dc_details, sr,
            ensemble_sign=ensemble_sign, regime_label=regime_label,
            mid=score_mids.get(engine_name, 0.0),
            native_var=native_var, run_count=run_count
        )
    return variances


def transform_engine_variances_to_normalized(engine_variances):
    """
    D1 FIX: Transform engine variances from raw score space to tanh-normalized space.

    The composite score uses tanh-normalized engines: e_norm = tanh(score/s).
    Engine variances must be in the same units for σ²_C = wᵀΣw to be
    dimensionally consistent with μ_C = wᵀe_norm.

    Linear rescaling (delta method at score=0):
        σ²_norm = σ²_raw / s²

    where s is the normalization scale from ATLAS_CONFIG['normalization'].
    """
    norm_scales = ATLAS_CONFIG['normalization']
    engines_standard = ['trend', 'valuation', 'consensus', 'volatility',
                        'macro', 'liquidity', 'global']

    transformed = {}

    for eng in engines_standard:
        s = norm_scales.get(eng, 50)
        sigma2_raw = engine_variances.get(eng, 50)
        # σ²_norm = σ²_raw / s²  (linear approximation of delta method)
        transformed[eng] = sigma2_raw / (s ** 2) if s > 0 else sigma2_raw

    # Correlation engine: signal = -(risk - 50), then tanh(signal/s_corr)
    # The inversion doesn't affect variance (sign change squares out)
    s_corr = norm_scales.get('correlation', 50)
    sigma2_raw = engine_variances.get('correlation', 50)
    transformed['correlation'] = sigma2_raw / (s_corr ** 2) if s_corr > 0 else sigma2_raw

    return transformed


def build_covariance_matrix(engine_variances, correlation_matrix=None):
    """
    Build engine covariance matrix Σ = ρ ⊙ (σσᵀ), PSD-corrected.
    Phase 2d: accepts optional regime-conditioned correlation matrix.
    Returns (Σ, engine_order).
    """
    engine_order = ['trend', 'valuation', 'consensus', 'volatility',
                    'macro', 'liquidity', 'global', 'correlation']
    n = len(engine_order)

    sigmas = np.array([np.sqrt(engine_variances.get(e, 50)) for e in engine_order])

    R = correlation_matrix if correlation_matrix is not None else ENGINE_STRUCTURAL_CORRELATIONS

    # Σ[i,j] = ρ[i,j] × σ_i × σ_j
    sigma_outer = np.outer(sigmas, sigmas)
    cov = R * sigma_outer

    # PSD enforcement: eigenvalue floor at 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Symmetry enforcement
    cov = (cov + cov.T) / 2.0

    return cov, engine_order


def compute_composite_variance(w_dynamic, cov_matrix, engine_order):
    """
    Compute composite score variance: σ²_C = wᵀΣw.
    """
    w = np.array([w_dynamic.get(e, 1/8) for e in engine_order])
    sigma2_C = float(w @ cov_matrix @ w)
    return max(sigma2_C, 1e-6)  # floor for numerical stability


def compute_confidence_adjusted_composite(c_raw, sigma2_C):
    """
    Confidence-adjusted composite: C_conf = μ × confidence_ratio × 100.
    Uses Gaussian CDF via math.erf.
    """
    mu = c_raw / 100.0
    sigma_C = np.sqrt(sigma2_C)

    if sigma_C < 1e-10:
        p_positive = 1.0 if mu > 0 else (0.0 if mu < 0 else 0.5)
    else:
        # Φ(x) = 0.5 × (1 + erf(x / √2))
        p_positive = 0.5 * (1.0 + erf(mu / (sigma_C * np.sqrt(2))))

    confidence_ratio = 2.0 * p_positive - 1.0
    c_conf = mu * confidence_ratio * 100.0

    return c_conf, p_positive, confidence_ratio


def compute_smooth_verdict(c_raw, c_adjusted, sigma_C, regime_label, tq, regime_entropy=0.0):
    """
    Smooth decision mapping with sigmoid probability.
    Phase 3c: τ now includes regime entropy term.
    θ = θ_base × regime_mult × (1 + k₁σ_C) × (1 + k₂ H_π/H_max)
    Returns both smooth verdict and legacy verdict for backward compatibility.
    """
    cfg = ATLAS_CONFIG['decision']
    tau_base = cfg['tau_base']
    k1 = cfg['uncertainty_sensitivity']
    k2 = DYNAMIC_THRESHOLD_K2
    regime_mult = cfg['regime_tau_multiplier'].get(regime_label, 1.0)

    tau_effective = max(
        tau_base * regime_mult * (1.0 + k1 * sigma_C) * (1.0 + k2 * regime_entropy),
        1.0
    )

    # Sigmoid: P_buy = 1 / (1 + exp(-C_adjusted / τ))
    x = c_adjusted / tau_effective
    x = np.clip(x, -20, 20)  # prevent overflow
    p_buy = 1.0 / (1.0 + np.exp(-x))
    p_sell = 1.0 - p_buy

    # Smooth verdict — O3: TQ gate mirrors legacy CASH classification
    if tq < 0.12:
        verdict_smooth = "CASH / STAND ASIDE"
    elif p_buy > 0.70:
        verdict_smooth = "BUY / LONG BIAS"
    elif p_sell > 0.70:
        verdict_smooth = "SELL / SHORT BIAS"
    elif p_buy > 0.55:
        verdict_smooth = "LONG BIAS (Moderate)"
    elif p_sell > 0.55:
        verdict_smooth = "SHORT BIAS (Moderate)"
    else:
        verdict_smooth = "NEUTRAL / FLAT"

    # Legacy verdict (exact old threshold logic preserved)
    if tq < 0.12:
        verdict_legacy = "CASH / STAND ASIDE"
    elif c_raw > 20 and c_adjusted > 10:
        verdict_legacy = "BUY / LONG BIAS"
    elif c_raw < -20 and c_adjusted < -10:
        verdict_legacy = "SELL / SHORT BIAS"
    elif c_raw > 0:
        verdict_legacy = "LONG BIAS (Moderate)"
    elif c_raw < 0:
        verdict_legacy = "SHORT BIAS (Moderate)"
    else:
        verdict_legacy = "NEUTRAL / FLAT"

    return {
        'verdict_smooth': verdict_smooth,
        'verdict_legacy': verdict_legacy,
        'p_buy': float(p_buy),
        'p_sell': float(p_sell),
        'tau_effective': float(tau_effective),
    }


def compute_cvar_gate(mu_C, sigma_C, alpha=0.05, theta=0.30):
    """
    CVaR (Conditional Value at Risk) gate for tail-risk control.

    CVaR_α(C) = μ_C - σ_C × φ(z_α) / α
    where z_α = Φ⁻¹(α) = 1.6449 for α=0.05
    and φ(z_α) = 0.10314 (standard normal PDF at z_α).

    θ is in normalized composite units (0.30 ≡ 30 raw composite points).
    Returns (pass: bool, cvar: float).
    Gate passes when CVaR > -θ (i.e., tail loss is within acceptable bound).
    """
    if sigma_C < 1e-10:
        return True, float(mu_C)

    z_alpha = 1.6449   # Φ⁻¹(0.95) for α=0.05
    phi_z = 0.10314    # φ(1.6449) = standard normal PDF

    cvar = mu_C - sigma_C * phi_z / alpha
    return cvar > -theta, float(cvar)


def compute_kelly_position(c_raw, sigma2_C, g, dc, capital, regime_label, cvar_pass=True):
    """
    Half-Kelly position sizing: f* = μ_C / (λ × σ²_C), clipped by regime.
    λ = KELLY_LAMBDA (default 2.0) gives half-Kelly fraction.
    If cvar_pass=False (CVaR gate failed), Kelly fraction is zeroed.
    Informational only — does not override existing position sizing.
    """
    mu = c_raw / 100.0

    if sigma2_C < 1e-10:
        kelly_raw = 0.0
    else:
        kelly_raw = mu / (KELLY_LAMBDA * sigma2_C)

    f_max = KELLY_REGIME_CAPS.get(regime_label, 0.10)
    kelly_clipped = float(np.clip(kelly_raw, -f_max, f_max))

    # CVaR gate: zero position if tail risk exceeds threshold
    if not cvar_pass:
        kelly_clipped = 0.0

    dc_factor = min(1.0, dc / 100.0)
    position = capital * abs(kelly_clipped) * g * dc_factor

    return {
        'kelly_fraction': float(kelly_raw),
        'kelly_clipped': kelly_clipped,
        'kelly_position': float(position),
        'kelly_pct': float(position / capital * 100) if capital > 0 else 0.0,
        'cvar_pass': cvar_pass,
    }


# ============================================================================
# SECTION 5: LAYER 0 — DATA INTEGRITY
# ============================================================================

def compute_data_confidence(data):
    """
    Compute data confidence score (DC) from 0-100.
    Check: ohlcv rows >= 200, macro rows >= 5, vol rows >= 2
    Check staleness and missing signals.
    """
    dc = 100.0
    details = {}

    # Price data check
    if not data.get('price') or len(data['price']) < 200:
        dc -= 40
        details['price_insufficient'] = True
    else:
        details['price_rows'] = len(data['price'])

    # Macro data check
    if not data.get('macro') or len(data['macro']) < 5:
        dc -= 25
        details['macro_insufficient'] = True
    else:
        details['macro_rows'] = len(data['macro'])

    # Vol data check
    if not data.get('vol') or len(data['vol']) < 2:
        dc -= 20
        details['vol_insufficient'] = True
    else:
        details['vol_rows'] = len(data['vol'])

    # Missing signals
    if not data.get('fundamentals'):
        dc -= 10
        details['fundamentals_missing'] = True

    if not data.get('consensus'):
        dc -= 10
        details['consensus_missing'] = True

    if not data.get('global_overnight'):
        dc -= 5
        details['global_overnight_missing'] = True

    # Data staleness detection (G7): parse last OHLCV date, penalize if stale
    details['data_stale'] = False
    details['stale_days'] = 0
    if data.get('price') and len(data['price']) > 0:
        last_row = data['price'][-1]
        date_str = last_row.get('date', '') or last_row.get('Date', '')
        if date_str:
            try:
                last_date = datetime.strptime(str(date_str).strip()[:10], '%Y-%m-%d')
                today = datetime.now()
                calendar_days = (today - last_date).days
                # Business-day adjustment: subtract weekends (~2 per 7 calendar days)
                stale_bdays = max(0, calendar_days - 2 * (calendar_days // 7))
                if stale_bdays > 3:
                    penalty = min(20, stale_bdays * 4)
                    dc -= penalty
                    details['data_stale'] = True
                    details['stale_days'] = stale_bdays
            except (ValueError, TypeError):
                pass

    dc = np.clip(dc, 0, 100)
    details['final_dc'] = dc

    return dc, details

# ============================================================================
# SECTION 6: LAYER 1 — REGIME VECTOR
# ============================================================================

def compute_soft_regime(regime_vector):
    """
    Soft GMM regime classification (Phase 2a).

    Computes posterior probabilities π_k for each of 5 regimes using
    Gaussian kernels with expert-calibrated centroids.
    Step 3: Uses anisotropic (10D diagonal) bandwidths when enabled.

    Input: regime_vector dict with keys [TS,CH,VL,VS,CI,RS,CS,GR,BM_f,BEI]
    Returns: (regime_probs: dict, regime_label: str)
      - regime_probs: {regime_name: probability} summing to 1.0
      - regime_label: argmax regime (for backward compat)
    """
    feature_order = ['TS', 'CH', 'VL', 'VS', 'CI', 'RS', 'CS', 'GR', 'BM_f', 'BEI']
    x = np.array([regime_vector.get(f, 0.0) for f in feature_order])

    n_regimes = len(REGIME_NAMES)
    log_probs = np.zeros(n_regimes)
    use_anisotropic = _FF.anisotropic_gmm

    for k in range(n_regimes):
        diff = x - GMM_REGIME_CENTROIDS[k]
        if use_anisotropic:
            # Anisotropic: -0.5 * Σ_d (diff_d / σ_kd)² + log(prior)
            sigma_diag = GMM_REGIME_BANDWIDTHS_DIAG[k]
            log_probs[k] = -0.5 * np.sum(diff ** 2 / sigma_diag ** 2) + np.log(GMM_REGIME_PRIORS[k])
        else:
            # Legacy isotropic: -0.5 * ||diff||² / σ² + log(prior)
            sigma = GMM_REGIME_BANDWIDTHS[k]
            log_probs[k] = -0.5 * np.sum(diff ** 2) / (sigma ** 2) + np.log(GMM_REGIME_PRIORS[k])

    # Numerically stable softmax
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= np.sum(probs)

    regime_probs = {REGIME_NAMES[k]: float(probs[k]) for k in range(n_regimes)}
    regime_label = REGIME_NAMES[int(np.argmax(probs))]

    return regime_probs, regime_label


def smooth_regime_probs(regime_probs_new, regime_probs_prev, alpha=None):
    """
    EMA-smooth regime probabilities to prevent whipsawing (Phase 2b).
    Step 4: When regime_transitions enabled, applies HMM-style prediction
    via transition matrix before EMA blend.
    π_smooth = α × π_combined + (1-α) × π_prev
    """
    if alpha is None:
        alpha = REGIME_EMA_ALPHA

    if regime_probs_prev is None:
        return regime_probs_new

    pi_prev = np.array([regime_probs_prev.get(r, 0.2) for r in REGIME_NAMES])
    pi_new = np.array([regime_probs_new.get(r, 0.2) for r in REGIME_NAMES])

    if _FF.regime_transitions:
        # HMM-style: π_predict = T^T × π_prev, then update with observation
        pi_predict = REGIME_TRANSITION_MATRIX.T @ pi_prev
        pi_predict = np.maximum(pi_predict, 1e-10)
        # Combine prediction with new observation (Bayesian update)
        pi_combined = pi_new * pi_predict
        total_combined = pi_combined.sum()
        if total_combined > 0:
            pi_combined /= total_combined
        else:
            pi_combined = pi_new
    else:
        pi_combined = pi_new

    # EMA smooth
    smoothed_arr = alpha * pi_combined + (1.0 - alpha) * pi_prev

    # Re-normalize after smoothing
    total = smoothed_arr.sum()
    if total > 0:
        smoothed_arr /= total

    smoothed = {REGIME_NAMES[k]: float(smoothed_arr[k]) for k in range(len(REGIME_NAMES))}
    return smoothed


def compute_regime_entropy(regime_probs):
    """
    Compute Shannon entropy of regime probability vector (Phase 3c).
    H(π) = -Σ π_k log(π_k), normalized to [0, 1] via H_max = log(K).
    High entropy → uncertain regime → widen thresholds.
    """
    probs = np.array([regime_probs.get(r, 0.0) for r in REGIME_NAMES])
    probs = np.maximum(probs, 1e-10)  # avoid log(0)
    H = -np.sum(probs * np.log(probs))
    H_max = np.log(len(REGIME_NAMES))  # log(5) ≈ 1.609
    return float(H / H_max) if H_max > 0 else 0.0


def build_regime_conditioned_correlations(regime_probs):
    """
    Build regime-conditioned engine correlation matrix (Phase 2d).
    Blends structural correlations with regime-specific shifts.
    In stress, off-diagonal correlations increase (contagion).
    """
    # Weighted shift based on regime probabilities
    shift = sum(
        regime_probs.get(r, 0.0) * _REGIME_CORR_SHIFTS.get(r, 0.0)
        for r in REGIME_NAMES
    )

    # Shift off-diagonals toward 1.0 (contagion)
    R = ENGINE_STRUCTURAL_CORRELATIONS.copy()
    n = R.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                R[i, j] = R[i, j] + shift * (1.0 - abs(R[i, j]))
                R[i, j] = np.clip(R[i, j], -0.95, 0.95)

    # PSD enforcement
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = eigvecs @ np.diag(eigvals) @ eigvecs.T
    R = (R + R.T) / 2.0

    # Force unit diagonal
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)

    return R


def compute_dynamic_risk_blend(regime_probs):
    """
    Compute dynamic structural/tactical risk blend α(r) (Phase 2e).
    Returns α ∈ [0,1]: weight on structural risk (1-α on tactical).
    """
    alpha = sum(
        regime_probs.get(r, 0.0) * RISK_BLEND_ALPHA.get(r, 0.60)
        for r in REGIME_NAMES
    )
    return float(np.clip(alpha, 0.1, 0.9))


def compute_regime_vector(data, scores_dict, engine_details):
    """
    Compute 10-feature regime vector and reliability score.

    Features:
      TS - Trend Score normalized [-1, 1]
      CH - Choppiness Index
      VL - VIX Level normalized [0, 1]
      VS - VIX Stress (momentum)
      CI - Correlation Instability
      RS - Rates Shock
      CS - Credit Stress
      GR - Global Risk
      BM_f - Bad Mix frequency
      BEI - Bond-Equity flip
    """
    regime_vector = {}

    # TS: Trend normalized
    ts_raw = scores_dict.get('trend', 0) / 100.0
    regime_vector['TS'] = np.clip(ts_raw, -1, 1)

    # CH: Choppiness Index
    if data['price'] and len(data['price']) >= 14:
        closes = [safe_float(row['close']) for row in data['price']]
        highs = [safe_float(row['high']) for row in data['price']]
        lows = [safe_float(row['low']) for row in data['price']]

        atr_14d_vals = atr(highs, lows, closes, 14)
        if atr_14d_vals and not np.isnan(atr_14d_vals[-1]):
            sum_atr = sum([v for v in atr_14d_vals[-14:] if not np.isnan(v)])
            highest = max(highs[-14:])
            lowest = min(lows[-14:])
            range_14 = highest - lowest

            if range_14 > 0:
                ch = sum_atr / range_14
                regime_vector['CH'] = np.clip(ch / 3.0, 0, 1)  # normalized
            else:
                regime_vector['CH'] = 0.0
        else:
            regime_vector['CH'] = 0.0
    else:
        regime_vector['CH'] = 0.0

    # VL: VIX Level
    vix = 20.0
    if data['vol'] and len(data['vol']) > 0:
        vix = safe_float(data['vol'][-1].get('vix', 20))
    regime_vector['VL'] = np.clip((vix - 12) / 40.0, 0, 1)

    # VS: VIX Stress (momentum)
    if data['vol'] and len(data['vol']) >= 20:
        vix_vals = [safe_float(row.get('vix', 20)) for row in data['vol'][-20:]]
        vix_sma20 = np.mean(vix_vals)
        vs_raw = max(0, vix - vix_sma20) / (vix_sma20 + 0.01)
        regime_vector['VS'] = np.clip(vs_raw, 0, 1)
    else:
        regime_vector['VS'] = 0.0

    # CI: Correlation Instability (std of 20d rolling corr over 60d window)
    if data['price'] and len(data['price']) >= 60 and data['macro'] and len(data['macro']) >= 60:
        closes = np.array([safe_float(row['close']) for row in data['price'][-60:]])
        closes = np.maximum(closes, 1e-10)  # D12: prevent log(0)
        yields = np.array([safe_float(row.get('us_10y', 2.5)) for row in data['macro'][-60:]])

        returns = np.diff(np.log(closes))
        yield_changes = np.diff(yields)

        rolling_corrs = []
        for i in range(20, len(returns)):
            c = np.corrcoef(returns[i-20:i], yield_changes[i-20:i])[0, 1]
            if not np.isnan(c):
                rolling_corrs.append(c)

        if rolling_corrs:
            ci = np.std(rolling_corrs)
            regime_vector['CI'] = np.clip(ci, 0, 1)
        else:
            regime_vector['CI'] = 0.0
    else:
        regime_vector['CI'] = 0.0

    # RS: Rates Shock
    if data['macro'] and len(data['macro']) >= 6:
        us_10y_now = safe_float(data['macro'][-1].get('us_10y', 2.5))
        us_10y_5d = safe_float(data['macro'][-6].get('us_10y', us_10y_now))
        rs_raw = abs(us_10y_now - us_10y_5d) / 0.50
        regime_vector['RS'] = np.clip(rs_raw, 0, 1)
    else:
        regime_vector['RS'] = 0.0

    # CS: Credit Stress
    if data['macro'] and len(data['macro']) > 0:
        hy_spread = safe_float(data['macro'][-1].get('hy_spread', 300))
        cs_raw = (hy_spread - 300) / 400.0
        regime_vector['CS'] = np.clip(cs_raw, 0, 1)
    else:
        regime_vector['CS'] = 0.0

    # GR: Global Risk (mean absolute overnight returns)
    if data['global_overnight']:
        es = safe_float(data['global_overnight'].get('es_overnight_return', 0))
        gr_raw = abs(es) / 1.5
        regime_vector['GR'] = np.clip(gr_raw, 0, 1)
    else:
        regime_vector['GR'] = 0.0

    # BM_f: Bad Mix frequency (VIX > 25 AND breadth < 40% in last 20 days)
    if data['vol'] and data['breadth'] and len(data['vol']) >= 20:
        bad_mix_count = 0
        for i in range(-20, 0):
            vix_check = safe_float(data['vol'][i].get('vix', 20)) > 25
            if data['breadth'] and len(data['breadth']) >= abs(i):
                adv = safe_float(data['breadth'][i].get('advancing', 1000))
                dec = safe_float(data['breadth'][i].get('declining', 1000))
                breadth_check = (adv / (adv + dec)) < 0.40 if (adv + dec) > 0 else False
            else:
                breadth_check = False

            if vix_check and breadth_check:
                bad_mix_count += 1

        regime_vector['BM_f'] = bad_mix_count / 20.0
    else:
        regime_vector['BM_f'] = 0.0

    # BEI: Bond-Equity correlation flip
    if data['price'] and len(data['price']) >= 5 and data['macro'] and len(data['macro']) >= 5:
        closes_now = np.array([safe_float(row['close']) for row in data['price'][-5:]])
        closes_now = np.maximum(closes_now, 1e-10)  # D12: prevent log(0)
        yields_now = np.array([safe_float(row.get('us_10y', 2.5)) for row in data['macro'][-5:]])

        returns_now = np.diff(np.log(closes_now))
        yield_changes_now = np.diff(yields_now)

        corr_now = np.corrcoef(returns_now, yield_changes_now)[0, 1] if len(returns_now) >= 2 else 0.0
        if np.isnan(corr_now):
            corr_now = 0.0

        if data['price'] and len(data['price']) >= 10 and data['macro'] and len(data['macro']) >= 10:
            closes_old = np.array([safe_float(row['close']) for row in data['price'][-10:-5]])
            closes_old = np.maximum(closes_old, 1e-10)  # D12: prevent log(0)
            yields_old = np.array([safe_float(row.get('us_10y', 2.5)) for row in data['macro'][-10:-5]])

            returns_old = np.diff(np.log(closes_old))
            yield_changes_old = np.diff(yields_old)

            corr_old = np.corrcoef(returns_old, yield_changes_old)[0, 1] if len(returns_old) >= 2 else 0.0
            if np.isnan(corr_old):
                corr_old = 0.0

            sign_flip = (np.sign(corr_now) != np.sign(corr_old)) and (corr_old != 0)
            regime_vector['BEI'] = 1.0 if sign_flip else 0.0
        else:
            regime_vector['BEI'] = 0.0  # D17: fallback — no old data to detect sign flip
    else:
        regime_vector['BEI'] = 0.0

    # Reliability: Rel = (1-CH)^1.2 * (1-VS)^1.0 * (1-CI)^0.8, clipped [0.05, 1.0]
    ch = regime_vector['CH']
    vs = regime_vector['VS']
    ci = regime_vector['CI']
    rel = ((1 - ch) ** 1.2) * ((1 - vs) ** 1.0) * ((1 - ci) ** 0.8)
    rel = np.clip(rel, 0.05, 1.0)

    # Regime label
    cs = regime_vector['CS']
    ts = regime_vector['TS']
    bei = regime_vector['BEI']
    rs = regime_vector['RS']

    if cs > 0.6:
        regime_label = "Credit Stress"
    elif abs(ts) > 0.7 and (vs > 0.7 or bei > 0.8):
        regime_label = "Crisis Trend"
    elif ch > 0.6:
        regime_label = "Chop"
    elif rs > 0.6 and cs < 0.4:
        regime_label = "Tightening Shock"
    else:
        regime_label = "Calm"

    return regime_vector, rel, regime_label

# ============================================================================
# SECTION 7: LAYER 2 — SCORE NORMALIZATION
# ============================================================================

def normalize_engine_scores(scores_dict):
    """
    Apply tanh normalization to each engine score.
    For correlation, invert first (signal = -(corr_risk - 50))
    """
    e_norm = {}

    for engine in ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global']:
        raw_score = scores_dict.get(engine, 0)
        s = ATLAS_CONFIG['normalization'].get(engine, 50)

        if s > 0:
            normalized = np.tanh(raw_score / s)
        else:
            normalized = 0.0

        e_norm[f'{engine}_norm'] = normalized

    # Correlation: invert first
    corr_risk = scores_dict.get('correlation', 50)
    corr_signal = -(corr_risk - 50)  # Convert risk to signal
    s = ATLAS_CONFIG['normalization'].get('correlation', 50)
    corr_norm = np.tanh(corr_signal / s) if s > 0 else 0.0
    e_norm['correlation_norm'] = corr_norm

    return e_norm

# ============================================================================
# SECTION 8: LAYER 3 — META-REGIME LEARNING
# ============================================================================

def load_meta_state(state_dir, symbol=None):
    """Load meta-learning state from state_dir/atlas_meta_state_{symbol}.json.
    Per-ticker state prevents cross-contamination of learned weights."""
    suffix = f"_{symbol.upper()}" if symbol else ""
    state_path = Path(state_dir) / f"atlas_meta_state{suffix}.json"

    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except (ValueError, TypeError):
            pass

    # Cold start defaults (includes Phase 2b/3a/3b state fields + Redesign state)
    engines = ['trend', 'valuation', 'consensus', 'volatility',
               'macro', 'liquidity', 'global', 'correlation']
    return {
        'w0': {e: 1/8 for e in engines},
        'Q': {e: 0.0 for e in engines},
        'run_count': 0,
        'regime_probs_prev': None,       # Phase 2b: previous smoothed π
        'ewma_var': {e: 0.0 for e in engines},   # Phase 3a
        'ewma_mean': {e: 0.0 for e in engines},  # Phase 3a
        'ewma_cross': None,              # Phase 3b: cross-product matrix
        'ewma_enorm_mean': None,         # Phase 3b: EWMA of e_norm means
        # Redesign state additions
        'garch_params': {e: {'omega': 1.0, 'alpha': 0.06, 'beta': 0.94} for e in engines},  # Step 5
        'garch_innovations': {e: [] for e in engines},  # Step 5
        'ewma_cube': None,              # Step 7: 8×8×8 coskewness EWMA
        'dpmm_centroids': None,         # Step 10: DPMM cluster centers (opt-in)
        'dpmm_counts': None,            # Step 10: DPMM assignment counts
        'dpmm_stick_weights': None,     # Step 10: DPMM stick-breaking weights
    }

def save_meta_state(state, state_dir, symbol=None):
    """Save meta-learning state to state_dir/atlas_meta_state_{symbol}.json.
    Uses atomic write (write-tmp-then-rename) for crash safety."""
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{symbol.upper()}" if symbol else ""
    target = state_path / f"atlas_meta_state{suffix}.json"
    tmp = target.with_suffix('.json.tmp')

    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    tmp.replace(target)  # atomic on POSIX

def update_meta_learning(state, e_norm, regime_vector):
    """
    Update meta-learning weights based on engine performance.
    Cold start if run_count < 10.
    """
    run_count = state.get('run_count', 0)

    if run_count < 10:
        # Cold start: return default weights, just increment counter
        state['run_count'] += 1
        return state, {}

    # Compute utilities Q_i
    ch = regime_vector.get('CH', 0.5)
    rel = regime_vector.get('Rel', 0.5)  # passed separately

    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']
    Q = {}

    for engine in engines:
        norm_key = f'{engine}_norm'
        e_norm_val = e_norm.get(norm_key, 0.0)

        Q[engine] = abs(e_norm_val) * (1 - ch) * 0.5 + abs(e_norm_val) * rel * 0.5

    # Exponentiated gradient update
    w0_prev = state.get('w0', {engine: 1/8 for engine in engines})
    eta = ATLAS_CONFIG['meta_learning_eta']

    # log(w0) + eta * Q
    logits = {}
    for engine in engines:
        w_prev = w0_prev.get(engine, 1/8)
        q = Q.get(engine, 0.0)
        logits[engine] = np.log(max(w_prev, 0.01)) + eta * q

    # Softmax
    logits_vals = np.array(list(logits.values()))
    logits_vals = logits_vals - np.max(logits_vals)  # numerical stability
    exp_logits = np.exp(logits_vals)
    w0_prime = dict(zip(engines, exp_logits / np.sum(exp_logits)))

    # Inertia
    lam = ATLAS_CONFIG['meta_learning_lambda']
    w0_inertia = {}
    for engine in engines:
        w0_inertia[engine] = lam * w0_prev.get(engine, 1/8) + (1 - lam) * w0_prime[engine]

    # Floors and ceilings
    floor = ATLAS_CONFIG['meta_learning_floor']
    ceiling = ATLAS_CONFIG['meta_learning_ceiling']

    w0_bounded = {}
    for engine in engines:
        w0_bounded[engine] = np.clip(w0_inertia[engine], floor, ceiling)

    # Re-normalize
    total = sum(w0_bounded.values())
    w0_new = {engine: w0_bounded[engine] / total for engine in engines}

    # Update state
    state['w0'] = w0_new
    state['Q'] = Q
    state['run_count'] += 1

    return state, Q


def update_ewma_engine_variances(state, scores_dict, decay=0.94):
    """
    Phase 3a: EWMA volatility tracking per engine.
    Step 5: Online GARCH(1,1) upgrade when enabled and run_count >= 20.
    σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}  where ε_t = s_t - μ_t
    Falls back to EWMA (IGARCH) when run_count < 20 or feature disabled.
    """
    engines = ['trend', 'valuation', 'consensus', 'volatility',
               'macro', 'liquidity', 'global', 'correlation']

    ewma_var = state.get('ewma_var', {e: 0.0 for e in engines})
    ewma_mean = state.get('ewma_mean', {e: 0.0 for e in engines})
    run_count = state.get('run_count', 0)

    use_garch = _FF.online_garch and run_count >= 20
    garch_params = state.get('garch_params', {e: {'omega': 1.0, 'alpha': 0.06, 'beta': 0.94} for e in engines})
    garch_innovations = state.get('garch_innovations', {e: [] for e in engines})

    for eng in engines:
        score = scores_dict.get(eng, 0.0)
        prev_mean = ewma_mean.get(eng, 0.0)
        prev_var = ewma_var.get(eng, 0.0)

        # Innovation (surprise)
        innovation = score - prev_mean

        if use_garch:
            # GARCH(1,1): σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
            params = garch_params.get(eng, {'omega': 1.0, 'alpha': 0.06, 'beta': 0.94})
            omega = params.get('omega', 1.0)
            alpha_g = params.get('alpha', 0.06)
            beta_g = params.get('beta', 0.94)

            new_var = omega + alpha_g * (innovation ** 2) + beta_g * prev_var
            new_var = max(new_var, 1e-6)  # floor for numerical stability

            # Store innovation for periodic MLE refit
            innov_list = garch_innovations.get(eng, [])
            innov_list.append(float(innovation))
            if len(innov_list) > 50:
                innov_list = innov_list[-50:]
            garch_innovations[eng] = innov_list

            # Online MLE refit every 20 runs (approximate moment-matching)
            if len(innov_list) >= 20 and run_count % 20 == 0:
                innov_arr = np.array(innov_list)
                innov2 = innov_arr ** 2
                # Moment matching: E[ε²] = ω/(1-α-β), autocorr of ε² gives α+β
                mean_innov2 = np.mean(innov2)
                if len(innov2) > 1:
                    autocorr = np.corrcoef(innov2[:-1], innov2[1:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0.9
                else:
                    autocorr = 0.9
                # Constrain α+β < 1 for stationarity
                persistence = np.clip(autocorr, 0.5, 0.98)
                alpha_new = np.clip(persistence * 0.06 / 0.94, 0.01, 0.15)
                beta_new = np.clip(persistence - alpha_new, 0.5, 0.97)
                omega_new = max(0.01, mean_innov2 * (1 - alpha_new - beta_new))
                garch_params[eng] = {
                    'omega': float(omega_new),
                    'alpha': float(alpha_new),
                    'beta': float(beta_new),
                }
        else:
            # Legacy EWMA (IGARCH): σ²_t = λ σ²_{t-1} + (1-λ) ε²_t
            new_var = decay * prev_var + (1 - decay) * (innovation ** 2)

        # Update EWMA mean
        new_mean = decay * prev_mean + (1 - decay) * score

        ewma_mean[eng] = float(new_mean)
        ewma_var[eng] = float(new_var)

    state['ewma_var'] = ewma_var
    state['ewma_mean'] = ewma_mean
    state['garch_params'] = garch_params
    state['garch_innovations'] = garch_innovations
    return state


def update_realized_correlations(state, e_norm, shrinkage_weight=0.3, decay=0.94):
    """
    Phase 3b: Realized cross-engine correlation tracking with shrinkage.
    Tracks EWMA cross-product matrix, computes realized correlations,
    then shrinks toward structural prior.
    """
    engines = ['trend', 'valuation', 'consensus', 'volatility',
               'macro', 'liquidity', 'global', 'correlation']
    n = len(engines)

    # Current normalized score vector
    e_vec = np.array([e_norm.get(f'{eng}_norm', 0.0) for eng in engines])

    # Load or initialize EWMA cross-product and means
    cross_key = 'ewma_cross'
    if cross_key in state and state[cross_key] is not None:
        cross = np.array(state[cross_key])
    else:
        cross = np.zeros((n, n))

    mean_key = 'ewma_enorm_mean'
    if mean_key in state and state[mean_key] is not None:
        means = np.array(state[mean_key])
    else:
        means = np.zeros(n)

    # Update EWMA
    means = decay * means + (1 - decay) * e_vec
    cross = decay * cross + (1 - decay) * np.outer(e_vec, e_vec)

    # Compute realized covariance
    realized_cov = cross - np.outer(means, means)

    # Extract realized correlation
    diag = np.sqrt(np.maximum(np.diag(realized_cov), 1e-10))
    realized_corr = realized_cov / np.outer(diag, diag)
    np.fill_diagonal(realized_corr, 1.0)
    realized_corr = np.clip(realized_corr, -0.95, 0.95)

    # Shrink toward structural prior
    R_shrunk = (1 - shrinkage_weight) * realized_corr + shrinkage_weight * ENGINE_STRUCTURAL_CORRELATIONS

    # PSD enforcement
    eigvals, eigvecs = np.linalg.eigh(R_shrunk)
    eigvals = np.maximum(eigvals, 1e-6)
    R_shrunk = eigvecs @ np.diag(eigvals) @ eigvecs.T
    R_shrunk = (R_shrunk + R_shrunk.T) / 2.0
    d = np.sqrt(np.diag(R_shrunk))
    R_shrunk = R_shrunk / np.outer(d, d)

    # Store in state
    state[cross_key] = cross.tolist()
    state[mean_key] = means.tolist()

    return state, R_shrunk


def compute_cold_start_sigma_mult(run_count):
    """
    Phase 3d: Cold-start σ inflation.
    When run_count is low, inflate variance estimates to express uncertainty.
    Returns multiplier ∈ [1.0, COLD_START_SIGMA_MULT].
    Decays to 1.0 as run_count → COLD_START_RUN_THRESHOLD.
    """
    if run_count >= COLD_START_RUN_THRESHOLD:
        return 1.0
    t = run_count / COLD_START_RUN_THRESHOLD  # 0→1
    # Exponential decay from max to 1.0
    return 1.0 + (COLD_START_SIGMA_MULT - 1.0) * (1.0 - t) ** 2


def compute_shrinkage_weight(run_count):
    """
    Phase 3e: Uncertainty shrinkage weight.
    Blends prior variance toward realized data as evidence accumulates.
    Returns λ ∈ [~0, SHRINKAGE_LAMBDA_BASE]: weight on structural prior.
    At cold start, λ → SHRINKAGE_LAMBDA_BASE (heavy prior).
    As runs accumulate, λ → 0 (data dominates).
    """
    return SHRINKAGE_LAMBDA_BASE * np.exp(-run_count / SHRINKAGE_DECAY_RUNS)


# ============================================================================
# SECTION 9: LAYER 4 — DYNAMIC WEIGHT MATRIX
# ============================================================================

# Pre-defined affinity matrix: 8 engines x 10 regime features (TS, CH, VL, VS, CI, RS, CS, GR, BM_f, BEI)
AFFINITY_MATRIX = np.array([
    [0.40, 0.50, -0.1, 0.30, 0.10, 0.10, -0.2, 0.10, 0.20, 0.15],  # trend
    [0.05, -0.2, 0.20, 0.30, 0.20, 0.30, 0.10, 0.00, -0.1, 0.10],  # valuation
    [0.10, 0.10, 0.05, 0.10, 0.15, -0.2, 0.00, 0.05, 0.10, 0.08],  # consensus
    [0.20, -0.3, 0.50, 0.30, 0.40, 0.20, 0.10, 0.15, 0.20, 0.25],  # volatility
    [0.10, 0.00, 0.10, 0.20, 0.20, 0.30, 0.20, 0.00, 0.10, 0.12],  # macro
    [0.15, 0.20, 0.10, 0.05, 0.10, -0.1, 0.05, 0.20, 0.05, 0.08],  # liquidity
    [0.10, 0.05, 0.10, 0.15, 0.10, 0.10, 0.20, 0.10, 0.05, 0.10],  # global
    [0.05, -0.4, 0.20, 0.30, 0.30, 0.40, 0.30, 0.20, 0.30, 0.35],  # correlation
])

def compute_dynamic_weights(w0, regime_vector):
    """
    Compute dynamic weights from base weights and regime features.
    w_dynamic = softmax(log(w0) + alpha * A @ r_bar)
    """
    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']

    # Extract regime vector as numpy array, in same order as features
    feature_order = ['TS', 'CH', 'VL', 'VS', 'CI', 'RS', 'CS', 'GR', 'BM_f', 'BEI']
    r_vec = np.array([regime_vector.get(f, 0.0) for f in feature_order])

    # Zero-mean
    r_bar = r_vec - np.mean(r_vec)

    # Logits
    alpha = ATLAS_CONFIG['affinity_alpha']
    logits = np.zeros(8)

    for i, engine in enumerate(engines):
        w0_val = w0.get(engine, 1/8)
        logits[i] = np.log(max(w0_val, 0.01)) + alpha * np.dot(AFFINITY_MATRIX[i], r_bar)

    # Softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    w_dyn = exp_logits / np.sum(exp_logits)

    # Apply floors/ceilings
    floor = ATLAS_CONFIG['meta_learning_floor']
    ceiling = ATLAS_CONFIG['meta_learning_ceiling']

    w_bounded = np.clip(w_dyn, floor, ceiling)
    w_normalized = w_bounded / np.sum(w_bounded)

    return dict(zip(engines, w_normalized))

# ============================================================================
# SECTION 10: LAYER 5 — COMPOSITE SCORE
# ============================================================================

def compute_composite(e_norm, w_dynamic, interaction_matrix=None, regime_label=None, regime_probs=None):
    """
    Compute composite score from normalized engine scores and dynamic weights.
    C_raw = 100 * (wᵀe + eᵀA(r)e)  where A(r) is regime-conditioned.

    D3 FIX: If regime_probs is provided, uses soft blend A(r) = Σ_k π_k A_k.
    Falls back to hard argmax (regime_label) or legacy static matrix.
    """
    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']

    e_vec = np.array([e_norm.get(f'{eng}_norm', 0.0) for eng in engines])
    w_vec = np.array([w_dynamic.get(eng, 1/8) for eng in engines])

    linear_term = float(np.dot(w_vec, e_vec))
    quadratic_term = 0.0

    # D3: Soft probability-weighted blend A(r) = Σ_k π_k(r) A_k
    A = None
    if regime_probs is not None:
        A = sum(
            regime_probs.get(r, 0.0) * REGIME_INTERACTION_MATRICES[r]
            for r in REGIME_NAMES
        )
    elif regime_label is not None and regime_label in REGIME_INTERACTION_MATRICES:
        A = REGIME_INTERACTION_MATRICES[regime_label]
    elif interaction_matrix is not None:
        A = interaction_matrix

    if A is not None:
        quadratic_term = float(e_vec @ A @ e_vec)

    c_raw = (linear_term + quadratic_term) * 100

    if c_raw > 0:
        direction = "LONG"
    elif c_raw < 0:
        direction = "SHORT"
    else:
        direction = "FLAT"

    strength = abs(c_raw) / 100

    details = {
        'direction': direction,
        'strength': strength,
        'raw_value': c_raw,
        'composite_linear': round(linear_term * 100, 4),
        'composite_quadratic': round(quadratic_term * 100, 4),
    }

    return c_raw, details

# ============================================================================
# SECTION 11: LAYER 6 — RISK GOVERNOR
# ============================================================================

def compute_risk_governor(regime_vector, c_raw, regime_label, dc, e_norm, risk_blend_alpha=None):
    """
    Compute risk governor gate and adjusted composite score.
    Phase 2e: dynamic structural/tactical blend via risk_blend_alpha.
    """
    # Structural risk
    ci = regime_vector.get('CI', 0.0)
    cs = regime_vector.get('CS', 0.0)
    rs = regime_vector.get('RS', 0.0)
    bei = regime_vector.get('BEI', 0.0)

    sr_s = 0.5 * (ci + cs) + 0.3 * rs + 0.2 * bei

    # Tactical risk
    ts = regime_vector.get('TS', 0.0)
    bm_f = regime_vector.get('BM_f', 0.0)
    vs = regime_vector.get('VS', 0.0)

    sr_t = 0.4 * abs(ts) + 0.3 * bm_f + 0.3 * (vs / 2.0)

    # Total risk (Phase 2e: dynamic blend)
    alpha = risk_blend_alpha if risk_blend_alpha is not None else 0.6
    sr = alpha * sr_s + (1.0 - alpha) * sr_t

    # Gate function (sigmoid-like)
    tau = ATLAS_CONFIG['risk_governor_tau']
    s = ATLAS_CONFIG['risk_governor_s']

    g = 1.0 - 1.0 / (1.0 + np.exp(-(sr - tau) / s))

    # Smooth gate attenuation below threshold (D13: no discontinuity)
    g_floor = 0.35
    if g < g_floor:
        if regime_label == "Crisis Trend":
            g = max(g, 0.30)  # crisis trend-following floor
        else:
            # Quadratic fade: continuous at g_floor, approaches 0 smoothly
            g = g * (g / g_floor)

    # Data confidence cap
    dc_cap = min(1.0, dc / 100.0)

    # Adjusted composite with explicit adjustment chain
    c_after_gate = c_raw * g
    c_adjusted = c_after_gate * dc_cap

    # Risk drivers (list of top risk factors)
    risk_drivers = []
    if sr_s > 0.5:
        risk_drivers.append("Structural Risk High")
    if sr_t > 0.5:
        risk_drivers.append("Tactical Risk High")
    if cs > 0.6:
        risk_drivers.append("Credit Stress")
    if bei > 0.7:
        risk_drivers.append("Bond-Equity Flip")
    if ci > 0.5:
        risk_drivers.append("Correlation Instability")

    details = {
        'SR_s': sr_s,
        'SR_t': sr_t,
        'SR': sr,
        'G': g,
        'risk_blend_alpha': alpha,
        'risk_drivers': risk_drivers
    }

    return c_adjusted, g, details

# ============================================================================
# SECTION 12: LAYER 7 — TRADE QUALITY
# ============================================================================

def compute_trade_quality(c_raw, rel, g, dc):
    """
    Compute trade quality score and category.
    TQ = (abs(C_raw)/100) * Rel * G * (DC/100), clipped [0,1]
    """
    tq = (abs(c_raw) / 100.0) * rel * g * (dc / 100.0)
    tq = np.clip(tq, 0, 1)

    if tq < 0.12:
        category = "CASH"
    elif tq < 0.20:
        category = "SMALL_DIRECTIONAL"
    elif tq < 0.35:
        category = "NORMAL_DIRECTIONAL"
    else:
        category = "STRONG_DIRECTIONAL"

    return tq, category

# ============================================================================
# SECTION 13: LAYER 8 — PORTFOLIO META POLICY
# ============================================================================

REGIME_POLICIES = {
    "Calm": {"CASH": 0.05, "SMALL": 0.10, "NORMAL": 0.60, "LARGE": 0.20, "HEDGE": 0.05},
    "Chop": {"CASH": 0.20, "SMALL": 0.25, "NORMAL": 0.40, "LARGE": 0.10, "HEDGE": 0.05},
    "Tightening Shock": {"CASH": 0.15, "SMALL": 0.15, "NORMAL": 0.50, "LARGE": 0.10, "HEDGE": 0.10},
    "Crisis Trend": {"CASH": 0.40, "SMALL": 0.15, "NORMAL": 0.20, "LARGE": 0.05, "HEDGE": 0.20},
    "Credit Stress": {"CASH": 0.35, "SMALL": 0.20, "NORMAL": 0.25, "LARGE": 0.10, "HEDGE": 0.10},
}

EXPOSURE_MULTIPLIERS = {
    "CASH": 0.0,
    "SMALL": 0.20,
    "NORMAL": 0.50,
    "LARGE": 1.00,
    "HEDGE": -0.50,
}

def compute_portfolio_policy(regime_label, tq, c_adjusted, capital):
    """
    Compute portfolio policy based on regime and trade quality.
    """
    policy = REGIME_POLICIES.get(regime_label, REGIME_POLICIES["Calm"]).copy()

    # Adjust policy based on TQ
    if tq < 0.12:
        policy = {"CASH": 1.0, "SMALL": 0.0, "NORMAL": 0.0, "LARGE": 0.0, "HEDGE": 0.0}
    elif tq > 0.35:
        # Boost LARGE, reduce CASH
        policy["LARGE"] = min(policy.get("LARGE", 0.10) * 1.5, 0.40)
        policy["CASH"] = max(policy.get("CASH", 0.05) * 0.5, 0.01)
        total = sum(policy.values())
        policy = {k: v / total for k, v in policy.items()}

    # Compute exposure multiplier
    mu = sum(policy.get(action, 0) * EXPOSURE_MULTIPLIERS.get(action, 0) for action in policy.keys())

    # Risk budget
    b = 0.01 * capital * mu

    # Selected action (highest probability)
    selected_action = max(policy.items(), key=lambda x: x[1])[0]

    return policy, mu, b

# ============================================================================
# SECTION 14: LAYER 9 — EXECUTION MICROSTRUCTURE
# ============================================================================

def compute_execution_micro(data, regime_vector, regime_label, c_adjusted, tq, b, engine_details):
    """
    Compute execution parameters: mode, continuation probability, penalties, etc.
    """
    # Mode selection
    if regime_label == "Calm" and c_adjusted > 10:
        mode = "Momentum"
    elif regime_label == "Chop":
        mode = "MeanRev"
    else:
        mode = "Confirm"

    # Continuation probability
    p_cont_map = {"Momentum": 0.65, "MeanRev": 0.50, "Confirm": 0.55}
    p_cont = p_cont_map.get(mode, 0.55)

    # Execution penalty
    exec_pen_map = {"Momentum": 0.02, "MeanRev": 0.05, "Confirm": 0.04}
    exec_pen = exec_pen_map.get(mode, 0.04)

    vs = regime_vector.get('VS', 0.0)
    exec_pen += vs * 0.10
    exec_pen = np.clip(exec_pen, 0, 0.30)

    # Liquidity window
    vol_ratio = engine_details.get('trend', {}).get('vol_vs_avg', 1.0)
    liq_win = np.clip(0.5 + vol_ratio * 0.4, 0.5, 1.5)

    # Opening hazard
    haz_open = 0.02 + vs * 0.15
    haz_open = np.clip(haz_open, 0.02, 0.20)

    # Execution gate
    g_exec = (1.0 - exec_pen) * liq_win * (1.0 - haz_open)

    # Final size
    size_final = b * g_exec if b > 0 else 0

    # Compute trade levels (reuse existing logic)
    if data['price'] and len(data['price']) >= 200:
        closes = [safe_float(row['close']) for row in data['price']]
        highs = [safe_float(row['high']) for row in data['price']]
        lows = [safe_float(row['low']) for row in data['price']]

        current_price = closes[-1]

        sma20_vals = sma(closes, 20)
        sma50_vals = sma(closes, 50)
        sma200_vals = sma(closes, 200)

        sma20_now = sma20_vals[-1] if not np.isnan(sma20_vals[-1]) else current_price
        sma50_now = sma50_vals[-1] if not np.isnan(sma50_vals[-1]) else current_price
        sma200_now = sma200_vals[-1] if not np.isnan(sma200_vals[-1]) else current_price

        atr_vals = atr(highs, lows, closes, 14)
        atr_now = atr_vals[-1] if atr_vals and not np.isnan(atr_vals[-1]) else 1.0

        atr_stop = current_price - (atr_now * 2.0)
        structural_stop = sma200_now * 0.97

        entry = sma50_now * 1.003
        stop_loss = min(atr_stop, structural_stop)

        # Take profit zones
        hi_52w = max(closes[-min(252, len(closes)):])
        tp_low = round(hi_52w * 0.98, 2)
        tp_high = round(hi_52w * 1.05, 2)

        exec_params = {
            'mode': mode,
            'entry': entry,
            'stop_loss': stop_loss,
            'tp_low': tp_low,
            'tp_high': tp_high,
            'sma20': sma20_now,
            'sma50': sma50_now,
            'sma200': sma200_now,
            'atr_14': atr_now,
            'p_continuation': p_cont,
            'exec_penalty': exec_pen,
            'liq_window': liq_win,
            'hazard_open': haz_open,
            'g_exec': g_exec,
        }
    else:
        exec_params = {
            'mode': mode,
            'entry': 0,
            'stop_loss': 0,
            'tp_low': 0,
            'tp_high': 0,
            'p_continuation': p_cont,
            'exec_penalty': exec_pen,
            'liq_window': liq_win,
            'hazard_open': haz_open,
            'g_exec': g_exec,
        }

    return exec_params, size_final

# ============================================================================
# SECTION 15: LAYER 10 — PYRAMID REPORT
# ============================================================================

def generate_pyramid_report(symbol, summary):
    """
    Generate the complete ATLAS pyramid report from summary dict.
    CLI-only — not called during bot/API execution.
    """
    report = []

    price = summary.get('price', 0)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report.append("=" * 88)
    report.append(f"ATLAS V12+ — {symbol} | ${price:.2f} | {timestamp}")
    report.append("=" * 88)
    report.append("")

    # TOP SECTION
    report.append("┌─ TOP ──────────────────────────────────────────────────────────────────────────┐")
    report.append(f"│ Verdict: {summary.get('verdict', 'N/A'):<78}│")
    report.append(f"│ Confidence: {summary.get('tq_category', 'N/A'):<19} | Size: ${summary.get('position_size', 0):>10,.0f} | Mode: {summary.get('execution_mode', 'N/A'):<10}│")
    report.append(f"│ Direction: {summary.get('direction', 'N/A'):<19} | Gate: {summary.get('gate_value', 0):.2f}                          │")
    report.append("└─────────────────────────────────────────────────────────────────────────────────┘")
    report.append("")

    # RISK SECTION
    report.append("── RISK ────────────────────────────────────────────────────────────────────────────")
    report.append(f"Structural Risk (SR_s): {summary.get('risk_structural', 0):.3f}")
    report.append(f"Tactical Risk (SR_t):   {summary.get('risk_tactical', 0):.3f}")
    report.append(f"Total Risk (SR):        {summary.get('risk_total', 0):.3f}")
    report.append(f"Gate G(t):              {summary.get('gate_value', 0):.3f}")
    risk_drivers = summary.get('risk_drivers', [])
    if risk_drivers:
        report.append(f"Risk Drivers: {', '.join(risk_drivers)}")
    report.append("")

    # SIGNAL SECTION
    report.append("── SIGNAL ──────────────────────────────────────────────────────────────────────────")
    report.append(f"Composite Raw:      {summary.get('composite_raw', 0):+.1f}/100")
    report.append(f"Composite Adjusted: {summary.get('composite_adjusted', 0):+.1f}/100")
    report.append(f"Trade Quality:      {summary.get('trade_quality', 0):.4f} [{summary.get('tq_category', 'N/A')}]")
    report.append(f"Data Confidence:    {summary.get('data_confidence', 0):.1f}%")
    report.append("")

    # FORCES (Engines)
    report.append("── FORCES ──────────────────────────────────────────────────────────────────────────")
    report.append(f"{'Engine':<15} {'Base w0':>10} {'Dynamic w(t)':>13} {'Shift':>8} {'Score':>8} {'Contribution':>12}")
    report.append("-" * 88)

    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']
    w0 = summary.get('w0', {})
    w_dyn = summary.get('w_dynamic', {})
    scores = summary.get('scores', {})
    scores_norm = summary.get('scores_norm', {})

    for engine in engines:
        w0_val = w0.get(engine, 1/8)
        w_dyn_val = w_dyn.get(engine, 1/8)
        shift = w_dyn_val - w0_val
        score = scores.get(engine, 0)
        norm_val = scores_norm.get(engine, 0)
        contrib = w_dyn_val * norm_val * 100

        report.append(f"{engine:<15} {w0_val:>10.3f} {w_dyn_val:>13.3f} {shift:>+8.3f} {score:>+8.1f} {contrib:>+12.2f}")

    report.append("")

    # REGIME
    report.append("── REGIME ──────────────────────────────────────────────────────────────────────────")
    report.append(f"Classification: {summary.get('regime_label', 'N/A')}")
    report.append(f"Reliability:    {summary.get('regime_reliability', 0):.3f}")
    report.append("")

    regime_vector = summary.get('regime_vector', {})
    feature_labels = {
        'TS': 'Trend Score',
        'CH': 'Choppiness',
        'VL': 'Vol Level',
        'VS': 'Vol Stress',
        'CI': 'Corr Instab',
        'RS': 'Rates Shock',
        'CS': 'Credit Stress',
        'GR': 'Global Risk',
        'BM_f': 'Bad Mix Freq',
        'BEI': 'Bond-Eq Flip'
    }

    report.append(f"{'Feature':<20} {'Value':>10} {'Status':<20}")
    report.append("-" * 50)

    for feature_key, feature_label in feature_labels.items():
        val = regime_vector.get(feature_key, 0)
        if val < 0.3:
            status = "LOW"
        elif val > 0.7:
            status = "HIGH"
        else:
            status = "MODERATE"
        report.append(f"{feature_label:<20} {val:>10.3f} {status:<20}")

    report.append("")

    # META LEARNING
    report.append("── META LEARNING ───────────────────────────────────────────────────────────────────")
    run_count = summary.get('run_count', 0)
    report.append(f"Run Count: {run_count}")

    learning_status = "Cold Start" if run_count < 10 else "Learning" if run_count < 50 else "Converged"
    report.append(f"Learning: {learning_status}")

    q_metrics = summary.get('q_metrics', {})
    if q_metrics:
        report.append("Engine Utilities (Q):")
        for engine in engines:
            q_val = q_metrics.get(engine, 0.0)
            report.append(f"  {engine:<15} {q_val:.4f}")

    report.append("")

    # EXECUTION
    report.append("── EXECUTION ───────────────────────────────────────────────────────────────────────")
    report.append(f"Mode:             {summary.get('execution_mode', 'N/A')}")
    report.append(f"Entry:            ${summary.get('entry', 0):.2f}")
    report.append(f"Stop Loss:        ${summary.get('stop_loss', 0):.2f}")
    report.append(f"Take Profit:      ${summary.get('tp_low', 0):.2f} – ${summary.get('tp_high', 0):.2f}")

    # Risk/reward
    entry = summary.get('entry', 0)
    tp_mid = (summary.get('tp_low', 0) + summary.get('tp_high', 0)) / 2
    stop = summary.get('stop_loss', 0)

    if entry > stop and entry > 0:
        reward = tp_mid - entry
        risk = entry - stop
        rr = reward / risk if risk > 0 else 0
        report.append(f"Risk/Reward:      {rr:.1f}:1")

    report.append(f"Continuation P:   {summary.get('p_cont', 0):.0%}")
    report.append(f"Exec Penalty:     {summary.get('exec_pen', 0):.1%}")
    report.append(f"Liq Window:       {summary.get('liq_win', 1.0):.2f}")
    report.append(f"Open Hazard:      {summary.get('haz_open', 0):.1%}")
    report.append(f"Exec Gate:        {summary.get('g_exec', 0):.3f}")
    report.append(f"Position Size:    ${summary.get('position_size', 0):,.0f}")
    report.append("")

    # PROBABILISTIC FRAMEWORK
    report.append("── PROBABILISTIC ───────────────────────────────────────────────────────────────────")
    report.append(f"Composite Variance (σ²_C): {summary.get('sigma2_C', 0):.4f}")
    report.append(f"Composite Std Dev (σ_C):   {summary.get('sigma_C', 0):.4f}")
    report.append(f"Confidence-Adj Composite:  {summary.get('composite_confidence_adjusted', 0):+.2f}")
    report.append(f"P(Signal > 0):             {summary.get('p_positive_signal', 0.5):.1%}")
    report.append(f"Confidence Ratio:          {summary.get('confidence_ratio', 0):.4f}")
    report.append(f"Linear Component:          {summary.get('composite_linear', 0):+.4f}")
    report.append(f"Quadratic Component:       {summary.get('composite_quadratic', 0):+.4f}")
    report.append("")

    # SMOOTH VERDICT
    report.append(f"P(Buy):    {summary.get('p_buy', 0.5):.1%}  |  P(Sell): {summary.get('p_sell', 0.5):.1%}")
    report.append(f"Smooth Verdict: {summary.get('verdict_smooth', 'N/A')}")
    report.append(f"Tau Effective:  {summary.get('tau_effective', 15):.2f}")
    report.append("")

    # KELLY SIZING
    report.append(f"Kelly Fraction (raw):    {summary.get('kelly_fraction', 0):+.6f}")
    report.append(f"Kelly Fraction (clipped):{summary.get('kelly_clipped', 0):+.6f}")
    report.append(f"Kelly Position:          ${summary.get('kelly_position', 0):,.0f} ({summary.get('kelly_pct', 0):.2f}%)")
    report.append("")

    # SELF-AUDIT
    report.append("── SELF-AUDIT ──────────────────────────────────────────────────────────────────────")

    contradictions = summary.get('contradictions', [])
    if contradictions:
        report.append("Contradictions Detected:")
        for c in contradictions:
            report.append(f"  * {c}")
    else:
        report.append("Contradictions: None detected")

    max_w = max(w_dyn.values()) if w_dyn else 0.2
    max_engine = max(w_dyn.items(), key=lambda x: x[1])[0] if w_dyn else "N/A"
    report.append(f"Weight Concentration: Max weight = {max_w:.1%} ({max_engine})")
    report.append(f"Regime Stability: Reliability = {summary.get('regime_reliability', 0):.2f}")
    report.append(f"Data Quality: DC = {summary.get('data_confidence', 0):.0f}%")

    report.append("")
    report.append("=" * 88)

    return "\n".join(report)

# ============================================================================
# SECTION 16: SELF-AUDIT
# ============================================================================

def detect_contradictions(layers_data):
    """
    Detect contradictions in the model output.
    """
    contradictions = []

    c_raw = layers_data.get('c_raw', 0)
    tq = layers_data.get('tq', 0)
    dc = layers_data.get('dc', 0)

    # High TQ but low DC
    if tq > 0.25 and dc < 50:
        contradictions.append("High TQ but low data confidence — caution advised")

    # Large position but high risk
    size = layers_data.get('size_final', 0)
    sr = layers_data.get('SR', 0)

    if size > 50000 and sr > 0.6:
        contradictions.append("Large position with high systemic risk — consider reducing")

    # Check weight concentration
    w_dyn = layers_data.get('w_dynamic', {})
    max_w = max(w_dyn.values()) if w_dyn else 0.2

    if max_w > 0.35:
        max_engine = max(w_dyn.items(), key=lambda x: x[1])[0]
        contradictions.append(f"Weight concentration excessive ({max_engine} = {max_w:.1%})")

    return contradictions

# ============================================================================
# SECTION 16b: COSKEWNESS TENSOR (Step 7)
# ============================================================================

def compute_coskewness_tensor(state, e_norm, w_dynamic, sigma_C, decay=0.94):
    """
    Step 7: Compute 3rd-order coskewness tensor for composite skewness.
    S[i,j,k] = E[e_i × e_j × e_k] tracked via EWMA of triple outer products.
    Composite skewness: γ_C = Σ_ijk w_i w_j w_k S_ijk / σ_C^3
    Adjusts P(C>0) via Cornish-Fisher expansion.
    Memory: 8×8×8 = 512 floats ≈ 4KB.
    """
    engines = ['trend', 'valuation', 'consensus', 'volatility',
               'macro', 'liquidity', 'global', 'correlation']
    n = len(engines)

    e_vec = np.array([e_norm.get(f'{eng}_norm', 0.0) for eng in engines])
    w_vec = np.array([w_dynamic.get(eng, 1.0/n) for eng in engines])

    # Load or initialize EWMA cube
    ewma_cube = state.get('ewma_cube', None)
    if ewma_cube is not None:
        cube = np.array(ewma_cube)
        if cube.shape != (n, n, n):
            cube = np.zeros((n, n, n))
    else:
        cube = np.zeros((n, n, n))

    # Update EWMA of triple outer product
    triple = np.einsum('i,j,k->ijk', e_vec, e_vec, e_vec)
    cube = decay * cube + (1 - decay) * triple

    state['ewma_cube'] = cube.tolist()

    # Composite skewness: γ_C = Σ_ijk w_i w_j w_k S_ijk / σ_C^3
    if sigma_C > 1e-10:
        gamma_C = float(np.einsum('i,j,k,ijk->', w_vec, w_vec, w_vec, cube) / (sigma_C ** 3))
    else:
        gamma_C = 0.0

    # Cornish-Fisher adjusted P(C>0)
    mu_C_norm = float(np.dot(w_vec, e_vec))
    if sigma_C > 1e-10:
        z = mu_C_norm / sigma_C
        z_cf = z + (z ** 2 - 1) * gamma_C / 6.0
        # Φ(z_cf) via erf
        p_positive_cf = 0.5 * (1.0 + erf(z_cf / np.sqrt(2)))
    else:
        p_positive_cf = 0.5 if mu_C_norm == 0 else (1.0 if mu_C_norm > 0 else 0.0)

    return state, float(gamma_C), float(p_positive_cf)


# ============================================================================
# SECTION 16c: STUDENT-T COPULA (Step 8)
# ============================================================================

def compute_copula_tail_dependence(cov_matrix, engine_order, nu=5):
    """
    Step 8: Student-t copula tail dependence coefficient.
    λ_L = 2 × t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
    where ρ is pairwise correlation and ν is degrees of freedom.
    Returns (tail_dep_max, tail_dep_matrix).
    """
    n = len(engine_order)

    # Extract correlation matrix from covariance
    diag = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-10))
    corr = cov_matrix / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)

    tail_dep = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = np.clip(corr[i, j], -0.999, 0.999)
            arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            # Student-t CDF: use scipy if available
            lambda_L = 2.0 * student_t.cdf(arg, df=nu + 1)
            tail_dep[i, j] = lambda_L
            tail_dep[j, i] = lambda_L

    tail_dep_max = float(np.max(tail_dep))

    return tail_dep_max, tail_dep


# ============================================================================
# SECTION 16d: DPMM REGIME MODEL (Step 10)
# ============================================================================

def compute_dpmm_regime(regime_vector, state, alpha_dp=1.0, max_components=10):
    """
    Step 10: Dirichlet Process Mixture Model with truncated stick-breaking.
    Online DPMM: V_k ~ Beta(1, α), π_k = V_k × Π_{j<k}(1-V_j)
    Warm-starts from GMM centroids. New clusters form when observation
    is far from all existing centroids (Mahalanobis distance > threshold).
    Feature flag: dpmm_regime (opt-in, default False).
    """
    feature_order = ['TS', 'CH', 'VL', 'VS', 'CI', 'RS', 'CS', 'GR', 'BM_f', 'BEI']
    x = np.array([regime_vector.get(f, 0.0) for f in feature_order])
    n_features = len(feature_order)

    # Load or warm-start state
    centroids = state.get('dpmm_centroids', None)
    counts = state.get('dpmm_counts', None)
    stick_weights = state.get('dpmm_stick_weights', None)

    if centroids is None:
        # Warm start from GMM centroids
        centroids = GMM_REGIME_CENTROIDS.tolist()
        counts = [10.0] * len(REGIME_NAMES)  # pseudo-counts for warm start
        stick_weights = [0.3, 0.2, 0.2, 0.15, 0.15]

    centroids = [np.array(c) for c in centroids]
    n_clusters = len(centroids)

    # Compute distances to all existing centroids
    distances = np.array([np.sqrt(np.sum((x - c) ** 2)) for c in centroids])

    # Threshold for creating new cluster (adaptive based on mean inter-centroid distance)
    if n_clusters >= 2:
        inter_dists = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                inter_dists.append(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
        threshold = np.mean(inter_dists) * 1.5
    else:
        threshold = 2.0

    min_dist_idx = int(np.argmin(distances))
    min_dist = distances[min_dist_idx]

    # Assignment or new cluster creation
    if min_dist > threshold and n_clusters < max_components:
        # Create new cluster
        centroids.append(x.copy())
        counts.append(1.0)
        assigned = n_clusters
        n_clusters += 1
        # Stick-breaking: add new weight from remaining mass
        remaining = 1.0 - sum(stick_weights)
        new_w = remaining * alpha_dp / (1.0 + alpha_dp)
        stick_weights.append(max(new_w, 0.01))
    else:
        assigned = min_dist_idx
        counts[assigned] += 1.0

    # Online centroid update (learning rate decays with count)
    lr = 1.0 / (counts[assigned] + 1.0)
    centroids[assigned] = (1 - lr) * centroids[assigned] + lr * x

    # Update stick-breaking weights based on counts
    total_counts = sum(counts)
    if total_counts > 0:
        stick_weights = [c / total_counts for c in counts]

    # Normalize to get regime probabilities
    total_sw = sum(stick_weights)
    if total_sw > 0:
        probs = [w / total_sw for w in stick_weights]
    else:
        probs = [1.0 / n_clusters] * n_clusters

    # Build regime_probs dict
    regime_probs = {}
    for k in range(n_clusters):
        if k < len(REGIME_NAMES):
            regime_probs[REGIME_NAMES[k]] = probs[k]
        else:
            regime_probs[f'DPMM_Cluster_{k}'] = probs[k]

    # Store state
    state['dpmm_centroids'] = [c.tolist() for c in centroids]
    state['dpmm_counts'] = counts
    state['dpmm_stick_weights'] = stick_weights

    # Regime label: argmax
    regime_label = max(regime_probs, key=regime_probs.get)

    return regime_probs, regime_label, n_clusters


# ============================================================================
# SECTION 17: ORCHESTRATOR
# ============================================================================

def run_atlas(symbol='SPY', data_path=None, capital=250000, state_dir=None):
    """
    Run the complete 11-layer ATLAS pipeline.

    Args:
        symbol: Ticker symbol
        data_path: Path to data directory
        capital: Portfolio capital
        state_dir: Path to state directory for meta-learning

    Returns:
        dict: summary_dict with all engine outputs
    """

    # Create state directory if needed
    if state_dir is None:
        state_dir = './state'

    Path(state_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = setup_data(data_path)

    if not data['price']:
        raise ValueError(f"Could not load price data for {symbol}")

    # Run 8 raw engines
    trend_score, trend_details = engine_trend(data['price'], data['breadth'])
    valuation_score, val_details = engine_valuation(data['fundamentals'], data['macro'])
    consensus_score, cons_details = engine_consensus(data['consensus'])
    vol_score, vol_details = engine_volatility(data['vol'], data['price'])
    macro_score, macro_details = engine_macro(data['macro'])
    liquidity_score, liq_details = engine_liquidity(data['price'], data['breadth'])
    global_score, global_details = engine_global_overnight(data['global_overnight'], data['vol'])
    corr_score, corr_details = engine_correlation(data['price'], data['vol'], data['macro'])

    scores_dict = {
        'trend': trend_score,
        'valuation': valuation_score,
        'consensus': consensus_score,
        'volatility': vol_score,
        'macro': macro_score,
        'liquidity': liquidity_score,
        'global': global_score,
        'correlation': corr_score
    }

    engine_details = {
        'trend': trend_details,
        'valuation': val_details,
        'consensus': cons_details,
        'volatility': vol_details,
        'macro': macro_details,
        'liquidity': liq_details,
        'global': global_details,
        'correlation': corr_details
    }

    # Layer 0: Data Integrity
    dc, dc_details = compute_data_confidence(data)

    # Layer 1: Regime Vector (hard classification)
    regime_vector, rel, regime_label_hard = compute_regime_vector(data, scores_dict, engine_details)

    # [Phase 2a] Soft GMM Regime Classification
    regime_probs, _ = compute_soft_regime(regime_vector)

    # [Phase 2b] EMA Smoothing for regime transitions (Step 4: HMM-style if enabled)
    state = load_meta_state(state_dir, symbol=symbol)
    regime_probs_prev = state.get('regime_probs_prev', None)
    regime_probs = smooth_regime_probs(regime_probs, regime_probs_prev)
    regime_label = max(regime_probs, key=regime_probs.get)  # argmax of smoothed π
    state['regime_probs_prev'] = regime_probs

    # [Step 10] DPMM Regime (opt-in alternative)
    dpmm_active = False
    dpmm_n_clusters = len(REGIME_NAMES)
    if _FF.dpmm_regime:
        dpmm_probs, dpmm_label, dpmm_n_clusters = compute_dpmm_regime(regime_vector, state)
        regime_probs = dpmm_probs
        regime_label = dpmm_label
        dpmm_active = True

    # [Phase 3c] Regime entropy for dynamic threshold
    regime_entropy = compute_regime_entropy(regime_probs)

    # [P1 + Phase 2c + G1] Engine Variance Estimates (with native_var blending)
    run_count = state.get('run_count', 0)
    engine_variances = compute_all_engine_variances(
        scores_dict, dc_details, regime_label=regime_label,
        engine_details=engine_details, run_count=run_count
    )

    # [Step 9: MC-to-Composite Feedback (G3)] — widen valuation variance from MC spread
    if _FF.mc_feedback and data.get('fundamentals'):
        mc_spread = safe_float(data['fundamentals'].get('mc_iv_spread', 0))
        if mc_spread > 0:
            mc_uncertainty_mult = 1.0 + mc_spread / 100.0
            engine_variances['valuation'] *= mc_uncertainty_mult

    # [Phase 3d] Cold-start σ inflation
    cold_start_mult = compute_cold_start_sigma_mult(run_count)
    if cold_start_mult > 1.0:
        engine_variances = {e: v * cold_start_mult for e, v in engine_variances.items()}

    # Layer 2: Score Normalization
    e_norm = normalize_engine_scores(scores_dict)

    # Layer 3: Meta-Learning
    state, q_metrics = update_meta_learning(state, e_norm, {**regime_vector, 'Rel': rel})

    # [Phase 3a] EWMA Volatility Tracking
    state = update_ewma_engine_variances(state, scores_dict)

    # [Phase 3b] Realized Correlation Tracking
    shrinkage_w = compute_shrinkage_weight(run_count)
    state, realized_corr = update_realized_correlations(state, e_norm, shrinkage_weight=shrinkage_w)

    # NOTE: save_meta_state deferred to after coskewness/DPMM updates below

    # Layer 4: Dynamic Weights
    w_dynamic = compute_dynamic_weights(state['w0'], regime_vector)

    # Layer 5: Composite Score [P4: regime-conditioned interaction matrix A(r) = Σ π_k A_k]
    c_raw, comp_details = compute_composite(e_norm, w_dynamic, regime_label=regime_label, regime_probs=regime_probs)

    # [Phase 2d] Regime-conditioned correlation matrix
    regime_corr = build_regime_conditioned_correlations(regime_probs)

    # [D19 FIX] Blend regime-conditioned and realized correlations
    # At cold start (high shrinkage_w), lean on regime theory (regime_corr)
    # As evidence accumulates (low shrinkage_w), lean on realized data (realized_corr)
    corr_blended = shrinkage_w * regime_corr + (1.0 - shrinkage_w) * realized_corr
    # Force unit diagonal after blending
    d_blend = np.sqrt(np.diag(corr_blended))
    corr_blended = corr_blended / np.outer(d_blend, d_blend)

    # [D1 FIX] Transform variances to normalized space for probabilistic framework
    # σ²_norm = σ²_raw / s²  — ensures σ²_C and μ_C are dimensionally consistent
    engine_variances_norm = transform_engine_variances_to_normalized(engine_variances)

    # [P2] Covariance Matrix + Composite Variance (using normalized variances + blended correlations)
    cov_matrix, engine_order = build_covariance_matrix(engine_variances_norm, correlation_matrix=corr_blended)
    sigma2_C = compute_composite_variance(w_dynamic, cov_matrix, engine_order)
    sigma_C = np.sqrt(sigma2_C)

    # [Step 8: Student-t Copula] — inflate variance for tail dependence
    tail_dep_max = 0.0
    copula_nu = 5
    if _FF.copula_tail_dep:
        tail_dep_max, _tail_dep_matrix = compute_copula_tail_dependence(cov_matrix, engine_order, nu=copula_nu)
        if tail_dep_max > 0.3:
            sigma2_C *= (1 + 0.2 * tail_dep_max)
            sigma_C = np.sqrt(sigma2_C)

    # [Step 7: Coskewness Tensor] — compute composite skewness
    composite_skewness = 0.0
    p_positive_cf = 0.5
    if _FF.coskewness_tensor:
        state, composite_skewness, p_positive_cf = compute_coskewness_tensor(
            state, e_norm, w_dynamic, sigma_C
        )

    # Save meta state (deferred from earlier to include coskewness/DPMM updates)
    save_meta_state(state, state_dir, symbol=symbol)

    # [P3] Confidence-Adjusted Composite
    c_conf, p_positive, confidence_ratio = compute_confidence_adjusted_composite(c_raw, sigma2_C)

    # [Phase 2e] Dynamic structural/tactical risk blend
    risk_blend_alpha = compute_dynamic_risk_blend(regime_probs)

    # Layer 6: Risk Governor (with dynamic blend)
    c_adjusted, g, risk_details = compute_risk_governor(
        regime_vector, c_raw, regime_label, dc, e_norm,
        risk_blend_alpha=risk_blend_alpha
    )

    # Layer 7: Trade Quality
    tq, tq_category = compute_trade_quality(c_raw, rel, g, dc)

    # Layer 8: Portfolio Policy
    policy, mu, b = compute_portfolio_policy(regime_label, tq, c_adjusted, capital)

    # [CVaR Gate] Tail-risk control — blocks Kelly when CVaR breaches threshold
    cvar_pass, cvar_value = compute_cvar_gate(c_raw / 100.0, sigma_C)

    # [P6] Half-Kelly Position Sizing (λ=2.0, CVaR-gated)
    kelly_data = compute_kelly_position(c_raw, sigma2_C, g, dc, capital, regime_label, cvar_pass=cvar_pass)

    # Layer 9: Execution Microstructure
    exec_params, size_final = compute_execution_micro(data, regime_vector, regime_label, c_adjusted, tq, b, engine_details)

    # [P5 + Phase 3c] Smooth Decision Mapping (with regime entropy)
    verdict_data = compute_smooth_verdict(c_raw, c_adjusted, sigma_C, regime_label, tq,
                                          regime_entropy=regime_entropy)
    verdict = verdict_data['verdict_smooth']  # D10: unified on probabilistic verdict

    # Self-Audit
    contradictions = detect_contradictions({
        'c_raw': c_raw,
        'tq': tq,
        'dc': dc,
        'size_final': size_final,
        'SR': risk_details.get('SR', 0),
        'w_dynamic': w_dynamic
    })

    # Prepare data for report
    current_price = safe_float(data['price'][-1].get('close', 0))

    direction = comp_details.get('direction', 'FLAT')

    # Build summary dict
    summary = {
        'symbol': symbol,
        'price': current_price,
        'composite_raw': round(c_raw, 1),
        'composite_adjusted': round(c_adjusted, 1),
        'verdict': verdict,
        'verdict_detail': f"{direction} with TQ={tq:.2f}",
        'trade_quality': round(tq, 4),
        'tq_category': tq_category,
        'gate_value': round(g, 2),
        'regime_label': regime_label,
        'regime_reliability': round(rel, 2),
        'data_confidence': round(dc, 1),
        'execution_mode': exec_params.get('mode', 'N/A'),
        'position_size': round(size_final, 0),
        'position_pct': round((size_final / capital * 100) if capital > 0 else 0, 2),
        'risk_structural': round(risk_details.get('SR_s', 0), 3),
        'risk_tactical': round(risk_details.get('SR_t', 0), 3),
        'buy_zone': (round(exec_params.get('sma200', 0) * 0.97, 2), round(exec_params.get('sma200', 0), 2)),
        'take_profit': (round(exec_params.get('tp_low', 0), 2), round(exec_params.get('tp_high', 0), 2)),
        'stop_loss': round(exec_params.get('stop_loss', 0), 2),
        'triggers_fired': 0,
        'regime_trend': regime_label,
        'regime_risk': 'RISK-ON' if c_adjusted > 15 else 'RISK-OFF' if c_adjusted < -15 else 'STAND-ASIDE',
        'regime_vol': 'HIGH' if regime_vector.get('VL', 0) > 0.6 else 'LOW' if regime_vector.get('VL', 0) < 0.3 else 'MODERATE',
    }

    # Extended fields for trader abstraction layer (Layer 11)
    # No math changes — data transport only
    engines = ['trend', 'valuation', 'consensus', 'volatility', 'macro', 'liquidity', 'global', 'correlation']
    summary['scores'] = dict(scores_dict)
    summary['scores_norm'] = {}
    summary['contributions'] = {}
    for eng in engines:
        w_val = w_dynamic.get(eng, 1/8)
        e_val = e_norm.get(f'{eng}_norm', 0)
        summary['scores_norm'][eng] = round(float(e_val), 4)
        summary['contributions'][eng] = round(float(w_val * e_val * 100), 2)
    summary['w0'] = {k: float(v) for k, v in state['w0'].items()}
    summary['w_dynamic'] = {k: float(v) for k, v in w_dynamic.items()}
    summary['regime_vector'] = {k: float(v) for k, v in regime_vector.items()}
    summary['run_count'] = state['run_count']
    summary['entry'] = round(exec_params.get('entry', 0), 2)
    summary['sma20'] = round(exec_params.get('sma20', 0), 2)
    summary['sma50'] = round(exec_params.get('sma50', 0), 2)
    summary['sma200'] = round(exec_params.get('sma200', 0), 2)
    summary['atr'] = round(exec_params.get('atr_14', 0), 2)
    summary['vix'] = round(safe_float(data['vol'][-1].get('vix', 20)) if data.get('vol') else 20.0, 1)
    summary['contradictions'] = contradictions
    summary['risk_drivers'] = risk_details.get('risk_drivers', [])
    summary['direction'] = direction
    summary['risk_total'] = round(risk_details.get('SR', 0), 3)
    summary['tp_low'] = round(exec_params.get('tp_low', 0), 2)
    summary['tp_high'] = round(exec_params.get('tp_high', 0), 2)
    summary['p_cont'] = round(exec_params.get('p_continuation', 0), 4)
    summary['exec_pen'] = round(exec_params.get('exec_penalty', 0), 4)
    summary['liq_win'] = round(exec_params.get('liq_window', 1.0), 4)
    summary['haz_open'] = round(exec_params.get('hazard_open', 0), 4)
    summary['g_exec'] = round(exec_params.get('g_exec', 0), 4)
    summary['q_metrics'] = {k: round(float(v), 4) for k, v in q_metrics.items()}

    # Probabilistic framework summary fields
    summary['engine_variances'] = {k: round(float(v), 2) for k, v in engine_variances.items()}
    summary['engine_variances_norm'] = {k: round(float(v), 6) for k, v in engine_variances_norm.items()}
    summary['sigma2_C'] = round(float(sigma2_C), 4)
    summary['sigma_C'] = round(float(sigma_C), 4)
    summary['composite_confidence_adjusted'] = round(float(c_conf), 2)
    summary['p_positive_signal'] = round(float(p_positive), 4)
    summary['confidence_ratio'] = round(float(confidence_ratio), 4)
    summary['composite_linear'] = round(float(comp_details.get('composite_linear', c_raw)), 4)
    summary['composite_quadratic'] = round(float(comp_details.get('composite_quadratic', 0)), 4)
    summary['p_buy'] = round(float(verdict_data['p_buy']), 4)
    summary['p_sell'] = round(float(verdict_data['p_sell']), 4)
    summary['verdict_smooth'] = verdict_data['verdict_smooth']
    summary['verdict_legacy'] = verdict_data['verdict_legacy']  # diagnostic
    summary['tau_effective'] = round(float(verdict_data['tau_effective']), 2)
    summary['kelly_fraction'] = round(float(kelly_data['kelly_fraction']), 6)
    summary['kelly_clipped'] = round(float(kelly_data['kelly_clipped']), 6)
    summary['kelly_position'] = round(float(kelly_data['kelly_position']), 0)
    summary['kelly_pct'] = round(float(kelly_data['kelly_pct']), 2)
    summary['kelly_lambda'] = KELLY_LAMBDA
    summary['cvar_pass'] = cvar_pass
    summary['cvar_value'] = round(float(cvar_value), 4)

    # Phase 2+3 summary fields
    summary['regime_probs'] = {k: round(float(v), 4) for k, v in regime_probs.items()}
    summary['regime_entropy'] = round(float(regime_entropy), 4)
    summary['regime_label_hard'] = regime_label_hard
    summary['risk_blend_alpha'] = round(float(risk_blend_alpha), 4)
    summary['cold_start_mult'] = round(float(cold_start_mult), 4)
    summary['shrinkage_weight'] = round(float(shrinkage_w), 4)

    # Redesign summary fields
    engines_list = ['trend', 'valuation', 'consensus', 'volatility',
                    'macro', 'liquidity', 'global', 'correlation']
    summary['data_stale'] = dc_details.get('data_stale', False)
    summary['stale_days'] = dc_details.get('stale_days', 0)
    summary['engine_native_var'] = {
        eng: round(float(engine_details.get(eng, {}).get('native_var', 0.0)), 4)
        for eng in engines_list
    }
    summary['engine_skewness'] = {
        eng: round(float(engine_details.get(eng, {}).get('skewness', 0.0)), 4)
        for eng in engines_list
    }
    summary['engine_kurtosis'] = {
        eng: round(float(engine_details.get(eng, {}).get('kurtosis', 0.0)), 4)
        for eng in engines_list
    }
    summary['composite_skewness'] = round(float(composite_skewness), 4)
    summary['tail_dependence_max'] = round(float(tail_dep_max), 4)
    summary['copula_nu'] = copula_nu
    summary['garch_params'] = state.get('garch_params', {})
    summary['regime_transition_used'] = bool(_FF.regime_transitions)
    summary['dpmm_active'] = dpmm_active
    summary['dpmm_n_clusters'] = dpmm_n_clusters

    # Composite adjustment chain — full auditability
    dc_cap = min(1.0, dc / 100.0)
    summary['adjustment_chain'] = [
        {
            'name': 'C_raw',
            'formula': 'Σ(wᵢ × Sᵢ)',
            'value': round(c_raw, 4),
        },
        {
            'name': 'Gate (risk governor)',
            'formula': 'sigmoid(SR, τ, s)',
            'value': round(g, 4),
            'result': round(c_raw * g, 4),
        },
        {
            'name': 'Data confidence cap',
            'formula': 'min(1.0, DC/100)',
            'value': round(dc_cap, 4),
            'result': round(c_adjusted, 4),
        },
    ]
    summary['c_after_gate'] = round(c_raw * g, 4)
    summary['dc_cap'] = round(dc_cap, 4)

    return summary

# ============================================================================
# SECTION 18: CLI MODE
# ============================================================================

if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    symbol = sys.argv[2] if len(sys.argv) > 2 else 'SPY'
    state_dir = sys.argv[3] if len(sys.argv) > 3 else './state'

    summary = run_atlas(symbol=symbol, data_path=data_path, state_dir=state_dir)

    # CLI-only: generate text report from summary for terminal display
    report = generate_pyramid_report(symbol, summary)
    print(report)
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for key, val in summary.items():
        print(f"{key:<25} {val}")
