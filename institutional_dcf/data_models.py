"""
Institutional DCF — Data Models

Dataclasses for structured data flow between phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RevenueProjection:
    """Projected revenue for a single year."""
    year: int
    revenue: float
    growth_rate: float
    growth_source: str  # 'roic_based', 'analyst', 'fade', 'terminal'


@dataclass
class MarginProjection:
    """Projected margins for a single year."""
    year: int
    fcf_margin: float
    operating_margin: float
    margin_source: str  # 'historical', 'fade', 'terminal'


@dataclass
class WACCResult:
    """Governed WACC computation result."""
    wacc_raw: float
    discount_rate: float
    cost_of_equity: float
    cost_of_debt_after_tax: float
    equity_weight: float
    debt_weight: float
    beta_raw: float
    beta_governed: float
    reason_codes: List[str] = field(default_factory=list)
    sector: str = ''


@dataclass
class ScenarioResult:
    """Single scenario valuation result."""
    wacc: float
    terminal_growth: float
    enterprise_value: float
    equity_value: float
    per_share_value: float
    terminal_pct: float


@dataclass
class ScenarioGrid:
    """Full scenario grid with percentile statistics."""
    scenarios: List[ScenarioResult] = field(default_factory=list)
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    base_case: Optional[ScenarioResult] = None


@dataclass
class InstitutionalDCFResult:
    """Complete result from the institutional DCF engine."""
    bear: float = 0.0
    base: float = 0.0
    bull: float = 0.0
    scenario_grid: Optional[ScenarioGrid] = None
    wacc_result: Optional[WACCResult] = None
    revenue_projections: List[RevenueProjection] = field(default_factory=list)
    margin_projections: List[MarginProjection] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)
