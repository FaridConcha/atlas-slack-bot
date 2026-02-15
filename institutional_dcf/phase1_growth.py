"""
Institutional DCF — Phase 1: ROIC-Based Growth

Scope: Replace blanket revenue growth with ROIC × Reinvestment Rate.
Growth = ReinvestmentRate × ROIC, where:
  - ReinvestmentRate = (CapEx - Depreciation + ΔWC) / NOPAT
  - ROIC = NOPAT / Invested Capital

Status: STUB — Not yet implemented.
"""

from typing import Dict, Any, List


def compute_roic_growth(financials: Dict[str, Any], years: int = 5) -> List[float]:
    """
    Compute ROIC-based growth rates for each forecast year.

    Args:
        financials: Company financial data dict
        years: Number of forecast years

    Returns:
        List of growth rates (decimals) per year

    Raises:
        NotImplementedError: Phase 1 is not yet implemented
    """
    raise NotImplementedError(
        "Phase 1 (ROIC-based growth) is not yet implemented. "
        "This phase will replace blanket revenue growth assumptions with "
        "fundamental-driven growth = ReinvestmentRate × ROIC."
    )
