"""
Institutional DCF — Phase 3: Scenario Analysis

Scope: Generate IV grid over (WACC, terminal_g) pairs.
Compute P25, P50, P75 from the grid to produce probabilistic IV range.

Status: STUB — Not yet implemented.
"""

from typing import Dict, Any, List


def generate_scenario_grid(
    base_projections: List[Dict[str, Any]],
    wacc_range: List[float],
    terminal_g_range: List[float],
    shares: float,
    net_debt: float,
) -> Dict[str, Any]:
    """
    Generate IV grid and percentile statistics.

    Args:
        base_projections: Year-by-year projection dicts
        wacc_range: List of WACC values to test (e.g. [0.07, 0.08, ..., 0.12])
        terminal_g_range: List of terminal growth rates (e.g. [0.02, 0.025, 0.03])
        shares: Shares outstanding
        net_debt: Net debt for EV→equity bridge

    Returns:
        Dict with grid, p25, p50, p75, base_case

    Raises:
        NotImplementedError: Phase 3 is not yet implemented
    """
    raise NotImplementedError(
        "Phase 3 (scenario grid) is not yet implemented. "
        "This phase generates a WACC × terminal_growth IV grid "
        "and computes P25/P50/P75 percentile values."
    )
