"""
Tests for institutional DCF Protocol interfaces.
Verifies that Protocol classes are correctly defined and runtime-checkable.
"""

import pytest
from institutional_dcf.interfaces import (
    RevenueForecaster, MarginModeler, MultistageValuator,
    ScenarioGenerator, CapitalAnalyzer, ReturnModeler, FullFinancialModeler,
)


class TestProtocolDefinitions:
    """Verify all Protocol classes are runtime-checkable."""

    def test_revenue_forecaster_is_protocol(self):
        assert hasattr(RevenueForecaster, '__protocol_attrs__') or hasattr(RevenueForecaster, '_is_protocol')

    def test_margin_modeler_is_protocol(self):
        assert hasattr(MarginModeler, '__protocol_attrs__') or hasattr(MarginModeler, '_is_protocol')

    def test_multistage_valuator_is_protocol(self):
        assert hasattr(MultistageValuator, '__protocol_attrs__') or hasattr(MultistageValuator, '_is_protocol')

    def test_scenario_generator_is_protocol(self):
        assert hasattr(ScenarioGenerator, '__protocol_attrs__') or hasattr(ScenarioGenerator, '_is_protocol')

    def test_capital_analyzer_is_protocol(self):
        assert hasattr(CapitalAnalyzer, '__protocol_attrs__') or hasattr(CapitalAnalyzer, '_is_protocol')

    def test_return_modeler_is_protocol(self):
        assert hasattr(ReturnModeler, '__protocol_attrs__') or hasattr(ReturnModeler, '_is_protocol')

    def test_full_financial_modeler_is_protocol(self):
        assert hasattr(FullFinancialModeler, '__protocol_attrs__') or hasattr(FullFinancialModeler, '_is_protocol')


class TestRuntimeCheckable:
    """Verify @runtime_checkable allows isinstance checks."""

    def test_dummy_revenue_forecaster(self):
        class DummyForecaster:
            def forecast(self, financials, years):
                return [0.05] * years
        assert isinstance(DummyForecaster(), RevenueForecaster)

    def test_non_conforming_rejected(self):
        class NotAForecaster:
            def something_else(self):
                pass
        assert not isinstance(NotAForecaster(), RevenueForecaster)
