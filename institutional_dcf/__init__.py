"""
ATLAS Institutional DCF Package — Phase II Scaffold

This package contains stubs for the institutional-grade DCF valuation engine.
All phases are behind the `use_institutional_dcf` feature flag (default: False).

When enabled, this replaces the single-stage DCF in v8_data.py with:
  Phase 1: ROIC-based growth projection
  Phase 2: Multi-stage valuation (explicit → fade → terminal)
  Phase 3: Scenario grid (WACC × terminal_g), P25/P50/P75
  Phase 4: Capital structure (backlog, working capital)
  Phase 5: Required return model
  Phase 6: Full financial model (depreciation, capex, WC, taxes)
"""

from valuation_config import CONFIG


def is_enabled() -> bool:
    """Check if the institutional DCF is enabled via feature flag."""
    return CONFIG.flags.use_institutional_dcf
