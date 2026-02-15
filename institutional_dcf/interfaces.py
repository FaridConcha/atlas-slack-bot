"""
Institutional DCF — Protocol Interfaces

Defines the contracts that each phase must implement.
"""

from typing import Protocol, runtime_checkable, Dict, List, Any


@runtime_checkable
class RevenueForecaster(Protocol):
    """Phase 1: Revenue projection from ROIC-based growth."""
    def forecast(self, financials: Dict[str, Any], years: int) -> List[float]:
        """Return list of projected revenues for each forecast year."""
        ...


@runtime_checkable
class MarginModeler(Protocol):
    """Phase 1/6: Margin projection with fade dynamics."""
    def project_margins(self, financials: Dict[str, Any], years: int) -> List[float]:
        """Return list of projected FCF margins for each forecast year."""
        ...


@runtime_checkable
class MultistageValuator(Protocol):
    """Phase 2: Three-stage valuation (explicit → fade → terminal)."""
    def compute(self, projections: List[Dict], wacc: float, terminal_g: float) -> Dict[str, Any]:
        """Return valuation result with EV, equity value, per-share values."""
        ...


@runtime_checkable
class ScenarioGenerator(Protocol):
    """Phase 3: IV grid over (WACC, terminal_g)."""
    def generate_grid(self, base_projections: List[Dict],
                      wacc_range: List[float],
                      terminal_g_range: List[float]) -> Dict[str, Any]:
        """Return scenario grid with P25/P50/P75 percentiles."""
        ...


@runtime_checkable
class CapitalAnalyzer(Protocol):
    """Phase 4: Capital structure and working capital analysis."""
    def analyze(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """Return backlog/revenue ratio, WC volatility, capital metrics."""
        ...


@runtime_checkable
class ReturnModeler(Protocol):
    """Phase 5: Required return model."""
    def compute_required_return(self, beta: float, sector: str,
                                 financials: Dict[str, Any]) -> float:
        """Return required rate of return for the equity."""
        ...


@runtime_checkable
class FullFinancialModeler(Protocol):
    """Phase 6: Complete financial model with depreciation, capex, WC, taxes."""
    def build_model(self, financials: Dict[str, Any],
                    assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Return full financial model projections."""
        ...
