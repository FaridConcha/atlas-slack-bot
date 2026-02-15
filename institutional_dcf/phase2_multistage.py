"""
Institutional DCF — Phase 2: Multi-Stage Valuation

Scope: Replace single-stage DCF with 3-stage model:
  Stage 1 (Years 1-5): Explicit year-by-year projections
  Stage 2 (Years 6-10): Fade period (growth declines linearly to terminal)
  Stage 3 (Year 11+): Terminal value with Gordon Growth Model

Status: STUB — Not yet implemented.
"""

from typing import Dict, Any, List


def compute_multistage_dcf(
    projections: List[Dict[str, Any]],
    wacc: float,
    terminal_growth: float,
    fade_years: int = 5,
) -> Dict[str, Any]:
    """
    3-stage DCF: explicit → fade → terminal.

    Args:
        projections: Year-by-year projection dicts from Phase 1
        wacc: Weighted average cost of capital (decimal)
        terminal_growth: Long-run growth rate (decimal)
        fade_years: Number of years for growth fade (default 5)

    Returns:
        Dict with enterprise_value, equity_value, terminal_pct, audit

    Raises:
        NotImplementedError: Phase 2 is not yet implemented
    """
    raise NotImplementedError(
        "Phase 2 (multi-stage valuation) is not yet implemented. "
        "This phase introduces a 3-stage model: explicit growth (5yr), "
        "fade period (5yr), and terminal value."
    )
