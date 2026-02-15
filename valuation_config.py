#!/usr/bin/env python3
"""
ATLAS V10 — Valuation Governance Configuration

Single source of truth for all institutional valuation constants.
Every governance module imports from here. No magic numbers elsewhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
    narrative_gating: bool = True            # I: Sector-aware narratives
    reconciliation_checks: bool = True       # J: Cross-module reconciliation
    tq_precision: bool = True                # E: 4dp TQ display
    engine_table_x100: bool = True           # F: Contribution column clarity
    ca_evidence: bool = True                 # Capital allocation evidence block
    use_institutional_dcf: bool = False      # Part II: NOT YET IMPLEMENTED


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

    def __repr__(self):
        return (
            f"ValuationConfig(\n"
            f"  flags={self.flags},\n"
            f"  wacc={self.wacc},\n"
            f"  terminal={self.terminal},\n"
            f"  mos={self.mos},\n"
            f"  fragility={self.fragility}\n"
            f")"
        )


CONFIG = ValuationConfig()
