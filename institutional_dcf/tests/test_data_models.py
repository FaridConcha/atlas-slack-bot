"""
Tests for institutional DCF data models.
Verifies dataclass construction and field defaults.
"""

import pytest
from institutional_dcf.data_models import (
    RevenueProjection, MarginProjection, WACCResult,
    ScenarioResult, ScenarioGrid, InstitutionalDCFResult,
)


class TestRevenueProjection:
    def test_construction(self):
        rp = RevenueProjection(year=1, revenue=1e9, growth_rate=0.05, growth_source='roic_based')
        assert rp.year == 1
        assert rp.revenue == 1e9
        assert rp.growth_source == 'roic_based'


class TestMarginProjection:
    def test_construction(self):
        mp = MarginProjection(year=1, fcf_margin=0.12, operating_margin=0.15, margin_source='historical')
        assert mp.fcf_margin == 0.12


class TestWACCResult:
    def test_defaults(self):
        wr = WACCResult(
            wacc_raw=0.08, discount_rate=0.085, cost_of_equity=0.09,
            cost_of_debt_after_tax=0.0375, equity_weight=0.6, debt_weight=0.4,
            beta_raw=0.42, beta_governed=0.80,
        )
        assert wr.reason_codes == []
        assert wr.sector == ''


class TestScenarioGrid:
    def test_defaults(self):
        sg = ScenarioGrid()
        assert sg.scenarios == []
        assert sg.p50 == 0.0
        assert sg.base_case is None


class TestInstitutionalDCFResult:
    def test_defaults(self):
        result = InstitutionalDCFResult()
        assert result.bear == 0.0
        assert result.base == 0.0
        assert result.bull == 0.0
        assert result.scenario_grid is None
        assert result.audit == {}

    def test_full_construction(self):
        result = InstitutionalDCFResult(
            bear=80.0, base=100.0, bull=120.0,
            audit={'model_version': 'institutional_v1'}
        )
        assert result.base == 100.0
        assert result.audit['model_version'] == 'institutional_v1'
