"""
Institutional DCF — Phase 6: Full Financial Model

Scope: Build complete three-statement financial model:
  - Income statement projection (revenue, COGS, SGA, D&A, interest, taxes)
  - Balance sheet projection (assets, liabilities, equity)
  - Cash flow statement (operating, investing, financing)
  - UFCF derivation from NOPAT + D&A - CapEx - ΔWC
  - Tax shield modeling
  - Depreciation schedule

Status: STUB — Not yet implemented.
"""

from typing import Dict, Any


def build_full_model(
    financials: Dict[str, Any],
    assumptions: Dict[str, Any],
    years: int = 10,
) -> Dict[str, Any]:
    """
    Build complete three-statement financial model.

    Args:
        financials: Historical financial data
        assumptions: Growth, margin, capex, WC assumptions
        years: Number of projection years

    Returns:
        Dict with income_statement, balance_sheet, cash_flow,
        ufcf_per_year, audit

    Raises:
        NotImplementedError: Phase 6 is not yet implemented
    """
    raise NotImplementedError(
        "Phase 6 (full financial model) is not yet implemented. "
        "This phase builds a complete three-statement model to derive "
        "UFCF from NOPAT + D&A - CapEx - ΔWC with explicit tax shield modeling."
    )
