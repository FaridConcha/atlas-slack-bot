#!/usr/bin/env python3
"""
ATLAS V12+ Phase 3 — Valuation Governance Configuration

Single source of truth for all institutional valuation constants.
Every governance module imports from here. No magic numbers elsewhere.
V12+: Added MonteCarloConfig, SECTOR_GROWTH_SIGMA, MC_CORRELATION_MATRIX.
Phase 1: Added SECTOR_TAIL_DF (Student's t df), REGIME_VARIANCE_MULTIPLIER.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np


# ============================================================================
# REPORT MODE (Canonical Suppression)
# ============================================================================

class ReportMode(Enum):
    """Determines how fundamental-derived outputs are rendered."""
    NORMAL = 'NORMAL'                             # All data available
    FUNDAMENTALS_SUPPRESSED = 'FUNDAMENTALS_SUPPRESSED'  # Core fundamentals missing → suppress valuation
    PARTIAL = 'PARTIAL'                           # Some data degraded → show with caveats


# ============================================================================
# BETA POLICY (P2)
# ============================================================================

class BetaPath(Enum):
    """Tracks how beta was determined."""
    MEASURED = 'MEASURED'          # Provider-reported beta → stabilized
    PROXY = 'PROXY'               # No beta available → default 1.0 proxy
    DISABLED = 'DISABLED'         # Feature flag off → raw passthrough


# ============================================================================
# SECTOR PROVENANCE (P1)
# ============================================================================

class SectorProvenance(Enum):
    """Tracks how sector classification was determined."""
    PROVIDER = 'PROVIDER'         # Real sector from data provider
    DEFAULTED = 'DEFAULTED'       # Sector unknown → governance uses general defaults
    OVERRIDE = 'OVERRIDE'         # Manually overridden (future)


# ============================================================================
# FUNDAMENTALS QUALITY
# ============================================================================

@dataclass
class FundamentalsQuality:
    """Tracks which core fields are real vs defaulted/missing."""
    market_cap_available: bool = False
    shares_available: bool = False
    revenue_available: bool = False
    beta_available: bool = False      # False means beta was defaulted to 1.0
    sector_available: bool = False    # False means sector is unknown/defaulted
    price_available: bool = False
    high_52w_available: bool = False
    low_52w_available: bool = False
    data_status: str = 'OK'          # OK / DEGRADED / INVALID
    data_reasons: List[str] = field(default_factory=list)

    @property
    def report_mode(self) -> ReportMode:
        if self.data_status == 'INVALID':
            return ReportMode.FUNDAMENTALS_SUPPRESSED
        elif self.data_status == 'DEGRADED':
            return ReportMode.PARTIAL
        return ReportMode.NORMAL

    @property
    def beta_defaulted(self) -> bool:
        return not self.beta_available

    @property
    def sector_defaulted(self) -> bool:
        return not self.sector_available


# ============================================================================
# FEATURE FLAGS
# ============================================================================

@dataclass
class FeatureFlags:
    """Toggle each governance fix independently."""
    beta_stabilization: bool = True          # A1: Sector beta floors/caps
    wacc_governance: bool = True             # A2: Multi-layer WACC clamp
    dynamic_mos: bool = True                 # C: Additive MOS model
    fragility_scoring: bool = True           # D: Fragility contributor detection
    terminal_governance: bool = True         # G: Enhanced terminal penalty
    reconciliation_checks: bool = True       # J: Cross-module reconciliation
    ca_evidence: bool = True                 # Capital allocation evidence block
    industry_priors: bool = True             # Stage 6: Industry prior system for missing data
    monte_carlo_dcf: bool = True             # V12+: Monte Carlo DCF simulations


# ============================================================================
# BETA GOVERNANCE
# ============================================================================

@dataclass(frozen=True)
class BetaBounds:
    floor: float
    cap: float


SECTOR_BETA_BOUNDS: Dict[str, BetaBounds] = {
    'Industrials':          BetaBounds(0.80, 1.60),
    'Consumer Cyclical':    BetaBounds(0.80, 1.60),
    'Energy':               BetaBounds(0.90, 1.80),
    'Basic Materials':      BetaBounds(0.85, 1.70),
    'Technology':           BetaBounds(0.90, 1.80),
    'Financial Services':   BetaBounds(0.80, 1.60),
    'Healthcare':           BetaBounds(0.70, 1.40),
    'Consumer Defensive':   BetaBounds(0.60, 1.20),
    'Utilities':            BetaBounds(0.50, 1.10),
    'Real Estate':          BetaBounds(0.60, 1.30),
    'Communication Services': BetaBounds(0.75, 1.50),
}

# Default for unknown sectors
DEFAULT_BETA_BOUNDS = BetaBounds(0.80, 1.60)


# ============================================================================
# WACC GOVERNANCE
# ============================================================================

@dataclass(frozen=True)
class WACCGovernance:
    risk_free_rate: float = 0.04       # 10Y treasury proxy
    equity_risk_premium: float = 0.05  # long-run ERP
    wacc_floor_spread: float = 0.025   # Rf + spread = general floor
    wacc_ceiling: float = 0.15         # absolute maximum
    pre_tax_cost_of_debt: float = 0.05
    tax_rate: float = 0.25

    @property
    def wacc_general_floor(self) -> float:
        return self.risk_free_rate + self.wacc_floor_spread


SECTOR_WACC_FLOORS: Dict[str, float] = {
    'Industrials':          0.085,
    'Consumer Cyclical':    0.085,
    'Energy':               0.090,
    'Basic Materials':      0.090,
    'Technology':           0.080,
    'Financial Services':   0.080,
    'Healthcare':           0.075,
    'Consumer Defensive':   0.070,
    'Utilities':            0.065,
    'Real Estate':          0.070,
    'Communication Services': 0.080,
}


# ============================================================================
# TERMINAL VALUE GOVERNANCE
# ============================================================================

@dataclass(frozen=True)
class TerminalGovernance:
    warning_threshold: float = 70.0    # % — start warning
    penalty_threshold: float = 70.0    # % — start conviction penalty
    severe_threshold: float = 80.0     # % — bear_mult haircut
    extreme_threshold: float = 90.0    # % — iv_confidence = LOW
    penalty_per_pct: float = 0.50      # bps per pct above threshold (50 bps = 0.5 pts)
    penalty_cap: float = 20.0          # max conviction penalty
    bear_mult_haircut: float = 0.90    # multiply bear_mult when severe


# ============================================================================
# MARGIN OF SAFETY GOVERNANCE
# ============================================================================

@dataclass(frozen=True)
class MOSConfig:
    """Additive MOS model: base + uplift components."""

    # Base MOS by business type (decimal)
    base_very_stable: float = 0.20
    base_normal: float = 0.30
    base_cyclical: float = 0.45

    # Uplift amounts (percentage points)
    uplift_low_data_confidence: float = 10.0   # DC < 70%
    uplift_high_terminal_dep: float = 10.0     # terminal_pct >= 80%
    uplift_wacc_clamp: float = 5.0             # any WACC clamp applied
    uplift_high_leverage: float = 5.0          # D/E > 2.0
    uplift_extreme_leverage: float = 10.0      # D/E > 3.0
    uplift_value_destruction: float = 10.0     # ROE < WACC proxy (poor value creation)
    uplift_fragility_per: float = 3.0          # per extra fragility contributor

    # Thresholds
    data_confidence_threshold: float = 70.0
    terminal_dep_threshold: float = 80.0
    leverage_high_threshold: float = 2.0
    leverage_extreme_threshold: float = 3.0

    def base_for_type(self, business_type: str) -> float:
        if business_type == 'Very Stable':
            return self.base_very_stable
        elif business_type == 'Normal':
            return self.base_normal
        else:
            return self.base_cyclical


# ============================================================================
# FRAGILITY SCORING
# ============================================================================

@dataclass(frozen=True)
class FragilityConfig:
    low_wacc_threshold: float = 7.0          # discount_rate < 7% is fragile
    high_terminal_dep_threshold: float = 80.0  # terminal_pct >= 80%
    low_data_confidence_threshold: float = 70.0  # DC < 70%


# Fragility contributor codes
FRAGILITY_LOW_WACC = 'LOW_WACC'
FRAGILITY_HIGH_TERMINAL_DEP = 'HIGH_TERMINAL_DEP'
FRAGILITY_LOW_DATA_CONF = 'LOW_DATA_CONF'
FRAGILITY_FLAT_MARGINS = 'FLAT_MARGINS_ASSUMPTION'


# ============================================================================
# NARRATIVE GATING
# ============================================================================

# Industries with structural moat characteristics (government contracts, etc.)
MOAT_PROTECTED_INDUSTRIES = {
    'Aerospace & Defense',
    'Defense',
    'Aerospace',
}

# Margin thresholds for "thin margins" narrative
THIN_MARGIN_NET_THRESHOLD = 5.0    # NM < 5% → thin
THIN_MARGIN_OP_THRESHOLD = 8.0     # OM < 8% → thin


# ============================================================================
# INDUSTRY PRIOR SYSTEM (Stage 6: G1)
# ============================================================================

@dataclass(frozen=True)
class IndustryPrior:
    """Structural prior for an industry when metric data is missing."""
    bq_prior: float = 0.0       # Business quality baseline adjustment
    moat_prior: float = 0.0     # Moat durability baseline adjustment
    ca_prior: float = 0.0       # Capital allocation baseline adjustment
    rationale: str = ''         # Human-readable justification


INDUSTRY_PRIORS: Dict[str, IndustryPrior] = {
    'Aerospace & Defense': IndustryPrior(
        bq_prior=0.5, moat_prior=0.6, ca_prior=0.3,
        rationale='Government contracts, program lock-in, high switching costs, certification barriers'),
    'Utilities—Regulated': IndustryPrior(
        bq_prior=0.4, moat_prior=0.5, ca_prior=0.2,
        rationale='Regulated monopoly, guaranteed returns, high capital barriers'),
    'Railroads': IndustryPrior(
        bq_prior=0.5, moat_prior=0.7, ca_prior=0.3,
        rationale='Duopoly structure, irreplaceable infrastructure, regulatory moat'),
    'Drug Manufacturers—General': IndustryPrior(
        bq_prior=0.4, moat_prior=0.5, ca_prior=0.2,
        rationale='Patent protection, high R&D barriers, regulatory approval moat'),
    'Banks—Diversified': IndustryPrior(
        bq_prior=0.3, moat_prior=0.3, ca_prior=0.2,
        rationale='Regulatory barriers, deposit base, network effects'),
    'Semiconductors': IndustryPrior(
        bq_prior=0.3, moat_prior=0.4, ca_prior=0.2,
        rationale='High capital barriers, design IP, ecosystem lock-in'),
}

# Max absolute impact on 0-5 scale from industry prior
PRIOR_CAP = 0.7


# ============================================================================
# MONTE CARLO DCF CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for Monte Carlo DCF simulations."""
    n_simulations: int = 1000
    wacc_sigma: float = 0.01              # 100 bps std dev
    terminal_growth_sigma: float = 0.005
    terminal_growth_floor: float = 0.01
    terminal_growth_cap: float = 0.04
    margin_cv: float = 0.15              # coefficient of variation
    margin_sigma_floor: float = 0.02
    growth_floor: float = -0.10
    growth_ar1_rho: float = 0.5          # year-to-year persistence
    proxy_sigma_multiplier: float = 1.5  # wider uncertainty for NI/EBITDA proxies
    bear_percentile: float = 5.0
    base_percentile: float = 50.0
    bull_percentile: float = 95.0
    monte_carlo_dcf: bool = True         # feature flag


# Sector-specific revenue growth volatility
SECTOR_GROWTH_SIGMA: Dict[str, float] = {
    'Technology':           0.08,
    'Consumer Cyclical':    0.07,
    'Energy':               0.09,
    'Basic Materials':      0.07,
    'Financial Services':   0.06,
    'Industrials':          0.05,
    'Healthcare':           0.06,
    'Consumer Defensive':   0.03,
    'Utilities':            0.03,
    'Real Estate':          0.04,
    'Communication Services': 0.06,
}

# Sector-specific Student's t degrees of freedom for fat-tailed growth shocks.
# Lower ν → fatter tails. Empirically: Energy/Cyclical have heavier tails,
# Defensive/Utilities are near-Gaussian.
SECTOR_TAIL_DF: Dict[str, int] = {
    'Technology':               5,
    'Consumer Cyclical':        5,
    'Energy':                   4,
    'Basic Materials':          5,
    'Financial Services':       5,
    'Industrials':              6,
    'Healthcare':               6,
    'Consumer Defensive':       8,
    'Utilities':                10,
    'Real Estate':              6,
    'Communication Services':   6,
}

# Regime-conditioned WACC sigma multiplier.
# Wider uncertainty in stressed regimes → fatter MC WACC distribution.
REGIME_VARIANCE_MULTIPLIER: Dict[str, float] = {
    'Calm':              1.0,
    'Chop':              1.4,
    'Tightening Shock':  1.3,
    'Crisis Trend':      1.6,
    'Credit Stress':     1.5,
}

# 4×4 correlation matrix for MC draws: [growth, margin, wacc, terminal_g]
MC_CORRELATION_MATRIX = np.array([
    [ 1.00,  0.30,  0.15,  0.00],   # growth
    [ 0.30,  1.00,  0.00,  0.00],   # margin
    [ 0.15,  0.00,  1.00,  0.10],   # wacc
    [ 0.00,  0.00,  0.10,  1.00],   # terminal_g
])


# ============================================================================
# MASTER CONFIG SINGLETON
# ============================================================================

@dataclass
class ValuationConfig:
    """Master configuration aggregating all governance sub-configs."""
    flags: FeatureFlags = field(default_factory=FeatureFlags)
    wacc: WACCGovernance = field(default_factory=WACCGovernance)
    terminal: TerminalGovernance = field(default_factory=TerminalGovernance)
    mos: MOSConfig = field(default_factory=MOSConfig)
    fragility: FragilityConfig = field(default_factory=FragilityConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)

    def __repr__(self):
        return (
            f"ValuationConfig(\n"
            f"  flags={self.flags},\n"
            f"  wacc={self.wacc},\n"
            f"  terminal={self.terminal},\n"
            f"  mos={self.mos},\n"
            f"  fragility={self.fragility},\n"
            f"  monte_carlo={self.monte_carlo}\n"
            f")"
        )


CONFIG = ValuationConfig()
