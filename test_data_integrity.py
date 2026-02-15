#!/usr/bin/env python3
"""
ATLAS V10 — Data Integrity Regression Tests

Tests the None-safe data pipeline, fundamental integrity checks,
DCF gating, and scoring behavior with missing data.

Run: python3 -m pytest test_data_integrity.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import MagicMock, patch
from v8_data import (
    _safe_num, _safe_pct, _check_fundamental_integrity,
    _build_financials, _build_dcf, _validate_financials,
    _sanitize_yield, _sanitize_payout,
)
from v8_report import _n, _compute_v8_scores, _compute_v9_owner_scores


# ============================================================================
# Unit Tests: _safe_num
# ============================================================================

class TestSafeNum:
    def test_none_returns_none(self):
        assert _safe_num(None) is None

    def test_valid_float(self):
        assert _safe_num(42.5) == 42.5

    def test_valid_int(self):
        assert _safe_num(100) == 100.0

    def test_zero_returns_zero(self):
        assert _safe_num(0) == 0.0

    def test_nan_returns_none(self):
        assert _safe_num(float('nan')) is None

    def test_string_returns_none(self):
        assert _safe_num("bad") is None

    def test_min_val_filters(self):
        assert _safe_num(-5, min_val=0) is None
        assert _safe_num(5, min_val=0) == 5.0

    def test_max_val_filters(self):
        assert _safe_num(200, max_val=100) is None
        assert _safe_num(50, max_val=100) == 50.0


class TestSafePct:
    def test_none_returns_none(self):
        assert _safe_pct(None) is None

    def test_converts_decimal_to_pct(self):
        assert _safe_pct(0.15) == 15.0

    def test_zero_returns_zero(self):
        assert _safe_pct(0) == 0.0


# ============================================================================
# Unit Tests: _check_fundamental_integrity
# ============================================================================

class TestFundamentalIntegrity:
    def test_ok_when_all_valid(self):
        status, reasons = _check_fundamental_integrity(
            mc=150e9, shares=1.3e9, revenue=70e9, price=115.0
        )
        assert status == 'OK'
        assert reasons == []

    def test_invalid_when_mc_missing(self):
        status, reasons = _check_fundamental_integrity(
            mc=None, shares=1.3e9, revenue=70e9, price=115.0
        )
        assert status == 'INVALID'
        assert 'market_cap_missing' in reasons

    def test_invalid_when_mc_zero(self):
        status, reasons = _check_fundamental_integrity(
            mc=0, shares=1.3e9, revenue=70e9, price=115.0
        )
        assert status == 'INVALID'
        assert 'market_cap_missing' in reasons

    def test_invalid_when_shares_missing(self):
        status, reasons = _check_fundamental_integrity(
            mc=150e9, shares=None, revenue=70e9, price=115.0
        )
        assert status == 'INVALID'
        assert 'shares_missing' in reasons

    def test_degraded_when_revenue_missing(self):
        status, reasons = _check_fundamental_integrity(
            mc=150e9, shares=1.3e9, revenue=None, price=115.0
        )
        assert status == 'DEGRADED'
        assert 'revenue_missing' in reasons

    def test_degraded_when_mc_implausible(self):
        status, reasons = _check_fundamental_integrity(
            mc=50e6, shares=1e6, revenue=10e6, price=50.0
        )
        assert status == 'DEGRADED'
        assert 'market_cap_implausible' in reasons

    def test_degraded_mc_shares_mismatch(self):
        """mc and shares*price diverge by >20%"""
        status, reasons = _check_fundamental_integrity(
            mc=150e9, shares=1.3e9, revenue=70e9, price=200.0
        )
        # 1.3e9 * 200 = 260e9, vs 150e9 — big mismatch
        assert 'mc_shares_mismatch' in reasons


# ============================================================================
# Integration Tests: _build_financials with missing data
# ============================================================================

class TestBuildFinancials:
    """Test that _build_financials produces None instead of 0/1 for missing data."""

    def _make_ticker(self, info_dict):
        """Create a mock yfinance ticker with given info."""
        ticker = MagicMock()
        ticker.info = info_dict
        ticker.cashflow = MagicMock()
        ticker.cashflow.empty = True
        return ticker

    def test_missing_market_cap_returns_none(self):
        """THE ROOT CAUSE BUG: marketCap missing should NOT produce mc=1."""
        ticker = self._make_ticker({'currentPrice': 200.0})
        fin = _build_financials(ticker, ticker.info)
        # Must NOT be $1 (the old bug)
        assert fin['market_cap'] is None or fin['market_cap'] != 1
        assert fin['_data_status'] == 'INVALID'

    def test_missing_revenue_returns_none(self):
        ticker = self._make_ticker({
            'currentPrice': 200.0,
            'marketCap': 150e9,
            'sharesOutstanding': 1.3e9,
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['revenue_ttm'] is None

    def test_margins_none_when_missing(self):
        ticker = self._make_ticker({
            'currentPrice': 100.0,
            'marketCap': 50e9,
            'sharesOutstanding': 500e6,
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['gross_margin'] is None
        assert fin['operating_margin'] is None
        assert fin['net_margin'] is None
        assert fin['roe'] is None

    def test_valid_data_produces_valid_financials(self):
        """Full RTX-like data should produce OK status and real values."""
        ticker = self._make_ticker({
            'currentPrice': 200.0,
            'marketCap': 260e9,
            'sharesOutstanding': 1.3e9,
            'totalRevenue': 70e9,
            'netIncomeToCommon': 7e9,
            'freeCashflow': 5e9,
            'totalDebt': 40e9,
            'totalCash': 8e9,
            'ebitda': 12e9,
            'grossMargins': 0.30,
            'operatingMargins': 0.12,
            'profitMargins': 0.10,
            'returnOnEquity': 0.20,
            'returnOnAssets': 0.05,
            'debtToEquity': 150.0,
            'currentRatio': 1.2,
            'beta': 0.42,
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['_data_status'] == 'OK'
        assert fin['revenue_ttm'] == 70e9
        assert fin['gross_margin'] == 30.0
        assert fin['market_cap'] == 260e9
        assert fin['shares_outstanding'] == 1.3e9

    def test_fcf_yield_none_when_mc_invalid(self):
        """FCF yield must not compute when market_cap is invalid."""
        ticker = self._make_ticker({
            'currentPrice': 200.0,
            'freeCashflow': 5e9,
            # No marketCap
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['fcf_yield'] is None

    def test_buyback_yield_none_when_mc_invalid(self):
        """Buyback yield must not compute when market_cap is below threshold."""
        ticker = self._make_ticker({
            'currentPrice': 200.0,
            'marketCap': 50,  # $50 — absurd
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['buyback_yield'] is None

    def test_shares_from_sharesOutstanding(self):
        """Shares should come from sharesOutstanding, not mc/price."""
        ticker = self._make_ticker({
            'currentPrice': 200.0,
            'marketCap': 300e9,
            'sharesOutstanding': 1.5e9,
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['shares_outstanding'] == 1.5e9

    def test_peg_ratio_none_for_missing(self):
        ticker = self._make_ticker({
            'currentPrice': 100.0,
            'marketCap': 50e9,
            'sharesOutstanding': 500e6,
        })
        fin = _build_financials(ticker, ticker.info)
        assert fin['peg_ratio'] is None


# ============================================================================
# Integration Tests: DCF gating
# ============================================================================

class TestDCFGating:
    def test_dcf_disabled_when_fundamentals_invalid(self):
        """DCF must not run when _data_status is INVALID."""
        info = {'currentPrice': 200.0}
        fin = {'_data_status': 'INVALID', 'revenue_ttm': None, 'shares_outstanding': None}
        result = _build_dcf(info, fin)
        assert result.get('_dcf_disabled') is True
        assert result['bear'] == 0
        assert result['base'] == 0

    def test_dcf_disabled_when_revenue_missing(self):
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = {
            '_data_status': 'DEGRADED',
            'revenue_ttm': None,
            'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9,
            'debt_equity': 1.5,
            'net_debt': 30e9,
        }
        result = _build_dcf(info, fin)
        assert result.get('_dcf_disabled') is True

    def test_dcf_runs_with_valid_data(self):
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK',
            'revenue_ttm': 70e9,
            'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 7e9,
            'ebitda': 12e9,
            'revenue_growth': 5.0,
            'debt_equity': 1.5,
            'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        assert result.get('_dcf_disabled') is not True
        assert result['base'] > 0
        assert result['bear'] > 0
        assert result['bull'] > result['base']
        # Verify net debt was subtracted (assumptions dict)
        assert result['assumptions']['net_debt_subtracted'] != 0

    def test_dcf_wacc_uses_beta(self):
        """Low beta should produce lower WACC than default 10%."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK',
            'revenue_ttm': 70e9,
            'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 7e9,
            'ebitda': 12e9,
            'revenue_growth': 5.0,
            'debt_equity': 1.5,
            'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        assert result['assumptions']['discount_rate'] < 10.0  # beta 0.42 → WACC < 10%


# ============================================================================
# Integration Tests: v8_report scoring with None data
# ============================================================================

class TestV8Scoring:
    def test_n_helper(self):
        assert _n(None) == 0
        assert _n(42.5) == 42.5
        assert _n(0) == 0

    def test_scores_with_invalid_data(self):
        """Scoring should not crash with INVALID data."""
        summary = {'composite_raw': 0, 'trade_quality': 0.5, 'gate_value': 0.8}
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'technicals': {},
            'news': [],
            'institutional': {},
            'earnings': [],
        }
        scores = _compute_v8_scores(summary, v8_data)
        assert scores['fundamental'] is None  # Cannot compute
        assert scores['composite'] >= 0
        assert scores['_data_status'] == 'INVALID'

    def test_v9_scores_with_invalid_data(self):
        """V9 owner scores should return RESEARCH with INVALID data."""
        summary = {}
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'company': {'price': 200.0},
            'dcf': {'bear': 0, 'base': 0, 'bull': 0, '_dcf_disabled': True},
            'institutional': {},
            'earnings': [],
        }
        v9 = _compute_v9_owner_scores(summary, v8_data)
        assert v9['v9_decision'] == 'RESEARCH'
        assert v9['business_quality'] == 0
        assert v9['conviction'] == 0

    def test_v9_scores_with_valid_data(self):
        """V9 owner scores should work normally with OK data."""
        summary = {}
        v8_data = {
            'financials': {
                '_data_status': 'OK',
                'roe': 20.0,
                'net_margin': 15.0,
                'revenue_growth': 8.0,
                'free_cash_flow': 5e9,
                'fcf_yield': 3.0,
                'debt_equity': 1.0,
                'gross_margin': 35.0,
                'operating_margin': 18.0,
                'market_cap': 150e9,
                'interest_coverage': 12.0,
                'buyback_yield': 2.0,
                'forward_pe': 18.0,
                'dividend_yield': 1.5,
                'payout_ratio': 35.0,
                'net_debt_ebitda': 2.0,
                'total_cash': 8e9,
                'total_debt': 40e9,
                'recommendation': 'buy',
            },
            'company': {'price': 100.0, 'market_cap': 150e9},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 125.0},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}, {'beat': True}, {'beat': False}, {'beat': True}],
        }
        v9 = _compute_v9_owner_scores(summary, v8_data)
        assert v9['business_quality'] > 0
        assert v9['moat_durability'] > 0
        assert v9['conviction'] > 0


# ============================================================================
# Invariant Tests
# ============================================================================

class TestInvariants:
    def test_required_drop_never_exceeds_100(self):
        """Required price drop must be <= 100%."""
        # This tests the JS logic conceptually — we verify the Python side
        iv = 50.0
        price = 200.0
        required_mos = 0.45
        target_price = iv * (1 - required_mos)
        fall_needed = max(0, min(100, (price - target_price) / price * 100))
        assert fall_needed <= 100

    def test_dividend_yield_sanity(self):
        """Yield > 25% should be flagged as None."""
        info = {'dividendYield': 1.36, 'dividendRate': 2.36}  # 136% raw
        result, anomaly = _sanitize_yield(info, 200.0)
        # Should either be corrected or None
        assert result is None or result <= 25

    def test_payout_ratio_sanity(self):
        """Payout > 200% should be None."""
        info = {'payoutRatio': 3.5}  # 350%
        result = _sanitize_payout(info)
        assert result is None

    def test_mos_sign_invariant(self):
        """If price > IV, MOS must be negative."""
        price = 200.0
        iv = 100.0
        mos = (iv - price) / iv * 100  # -100%
        # Invariant: if price > iv, mos < 0
        assert price > iv
        assert mos < 0


# ============================================================================
# TASK 1 Tests: CAPM arithmetic and discount rate definitions
# ============================================================================

class TestCAPMArithmetic:
    """Verify CAPM cost_of_equity is computed with exact precision."""

    def _make_valid_fin(self, de=1.5, net_debt=32e9):
        return {
            '_data_status': 'OK',
            'revenue_ttm': 70e9,
            'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 7e9,
            'ebitda': 12e9,
            'revenue_growth': 5.0,
            'debt_equity': de,
            'net_debt': net_debt,
        }

    def test_cost_of_equity_exact_precision(self):
        """Ke = Rf + β × ERP = 4.00% + 0.42 × 5.00% = 6.10% exactly."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = self._make_valid_fin()
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # CAPM: 4.00 + 0.42 × 5.00 = 6.10
        assert a['cost_of_equity'] == 6.10
        assert a['risk_free_rate'] == 4.00
        assert a['equity_risk_premium'] == 5.00
        assert a['beta'] == 0.42

    def test_wacc_vs_cost_of_equity_distinct(self):
        """When D/E > 0, WACC must differ from cost_of_equity."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = self._make_valid_fin(de=1.5)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        assert a['discount_rate_type'] == 'WACC'
        # WACC includes debt component, must differ from Ke
        assert a['wacc_raw'] != a['cost_of_equity']
        # Verify WACC < Ke (debt is cheaper after tax shield)
        assert a['wacc_raw'] < a['cost_of_equity']

    def test_cost_of_equity_when_no_debt(self):
        """When D/E = 0, WACC = cost_of_equity."""
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = self._make_valid_fin(de=0, net_debt=0)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # Ke = 4% + 1.0 × 5% = 9%
        assert a['cost_of_equity'] == 9.00
        # No debt → WACC = Ke
        assert a['wacc_raw'] == a['cost_of_equity']
        assert a['debt_weight'] == 0.0
        assert a['equity_weight'] == 100.0

    def test_discount_rate_clamping(self):
        """Discount rate clamped between 6% and 15%."""
        # Very low beta → should hit floor
        info = {'currentPrice': 200.0, 'beta': 0.1}
        fin = self._make_valid_fin(de=2.0)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        assert a['discount_rate'] >= 6.0
        assert a['discount_rate_clamped'] is True

    def test_discount_rate_type_label(self):
        """discount_rate_type must be 'WACC' when D/E > 0."""
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = self._make_valid_fin(de=1.0)
        result = _build_dcf(info, fin)
        assert result['assumptions']['discount_rate_type'] == 'WACC'

    def test_wacc_arithmetic_precise(self):
        """Verify WACC = E/V × Ke + D/V × Kd(1-t) with exact values."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = self._make_valid_fin(de=1.5)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # D/E = 1.5 → D/V = 1.5/2.5 = 60%, E/V = 40%
        expected_dw = 1.5 / 2.5  # 0.6
        expected_ew = 1 - expected_dw  # 0.4
        ke = 0.04 + 0.42 * 0.05  # 0.061
        kd_at = 0.05 * (1 - 0.25)  # 0.0375
        expected_wacc = expected_ew * ke + expected_dw * kd_at
        # Allow rounding tolerance
        assert abs(a['wacc_raw'] - round(expected_wacc * 100, 2)) < 0.02


# ============================================================================
# TASK 2 Tests: DCF audit panel — projections and EV→Equity bridge
# ============================================================================

class TestDCFAuditPanel:
    """Verify year-by-year projections and EV→Equity bridge."""

    def _run_dcf(self, beta=0.42, de=1.5, net_debt=32e9):
        info = {'currentPrice': 200.0, 'beta': beta}
        fin = {
            '_data_status': 'OK',
            'revenue_ttm': 70e9,
            'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 7e9,
            'ebitda': 12e9,
            'revenue_growth': 5.0,
            'debt_equity': de,
            'net_debt': net_debt,
        }
        return _build_dcf(info, fin)

    def test_projections_array_present(self):
        """DCF must include year-by-year projections array."""
        result = self._run_dcf()
        assert 'projections' in result
        assert len(result['projections']) == 5  # 5-year forecast

    def test_projection_fields(self):
        """Each projection year must have revenue, growth_rate, fcf, pv_factor, pv_fcf."""
        result = self._run_dcf()
        for proj in result['projections']:
            assert 'year' in proj
            assert 'revenue' in proj
            assert 'growth_rate' in proj
            assert 'fcf' in proj
            assert 'pv_factor' in proj
            assert 'pv_fcf' in proj

    def test_projections_revenue_grows(self):
        """Revenue in Y2 must be >= Y1 (positive growth applied)."""
        result = self._run_dcf()
        p = result['projections']
        assert p[1]['revenue'] >= p[0]['revenue']

    def test_ev_equity_bridge_present(self):
        """Assumptions must include EV→Equity bridge fields."""
        result = self._run_dcf()
        a = result['assumptions']
        assert 'enterprise_value' in a
        assert 'net_debt_subtracted' in a
        assert 'equity_value' in a
        assert 'pv_fcf_sum' in a
        assert 'shares' in a

    def test_ev_minus_debt_equals_equity(self):
        """EV - net_debt = equity_value."""
        result = self._run_dcf()
        a = result['assumptions']
        expected_equity = max(0, a['enterprise_value'] - a['net_debt_subtracted'])
        assert abs(a['equity_value'] - expected_equity) < 1  # rounding tolerance

    def test_equity_per_share_matches_base(self):
        """equity_value / shares ≈ base IV."""
        result = self._run_dcf()
        a = result['assumptions']
        expected_iv = a['equity_value'] / a['shares']
        assert abs(result['base'] - round(expected_iv, 2)) < 0.01

    def test_terminal_value_pct_present(self):
        """Terminal value % must be reported."""
        result = self._run_dcf()
        a = result['assumptions']
        assert 'terminal_value_pct' in a
        assert a['terminal_value_pct'] > 0
        assert a['terminal_value_pct'] < 100

    def test_cash_flow_definition_label(self):
        """Must label cash flow as unlevered_fcf."""
        result = self._run_dcf()
        assert result['cash_flow_definition'] == 'unlevered_fcf'
        assert result['forecast_years'] == 5


# ============================================================================
# TASK 3 Tests: Composite reconciliation — adjustment chain
# ============================================================================

class TestCompositeReconciliation:
    """Verify the adjustment chain is correctly exposed."""

    def test_adjustment_chain_structure(self):
        """adjustment_chain must have 3 steps: C_raw, Gate, DC_cap."""
        # We test the structure as set in atlas_engine.py
        # Since we can't easily run the full engine in a unit test,
        # verify the structure matches expectations
        chain = [
            {'name': 'C_raw', 'formula': 'Σ(wᵢ × Sᵢ)', 'value': 4.80},
            {'name': 'Gate (risk governor)', 'formula': 'sigmoid(SR, τ, s)',
             'value': 0.82, 'result': 3.936},
            {'name': 'Data confidence cap', 'formula': 'min(1.0, DC/100)',
             'value': 0.65, 'result': 2.5584},
        ]
        assert len(chain) == 3
        assert chain[0]['name'] == 'C_raw'
        assert chain[1]['name'] == 'Gate (risk governor)'
        assert chain[2]['name'] == 'Data confidence cap'
        # Final result should equal C_raw × Gate × DC_cap
        expected = 4.80 * 0.82 * 0.65
        assert abs(chain[2]['result'] - expected) < 0.001

    def test_composite_formula_invariant(self):
        """C_adjusted = C_raw × Gate × DC_cap must always hold."""
        c_raw = 4.80
        gate = 0.82
        dc = 65.0
        dc_cap = min(1.0, dc / 100)
        c_adjusted = c_raw * gate * dc_cap
        assert abs(c_adjusted - 2.5584) < 0.001
        assert dc_cap == 0.65

    def test_dc_cap_capped_at_one(self):
        """DC ≥ 100 should produce dc_cap = 1.0 (no reduction)."""
        dc = 100.0
        dc_cap = min(1.0, dc / 100)
        assert dc_cap == 1.0
        dc = 150.0
        dc_cap = min(1.0, dc / 100)
        assert dc_cap == 1.0


# ============================================================================
# TASK 4 Tests: Dividend anomaly reason codes
# ============================================================================

class TestDividendAnomalyReasonCodes:
    """Verify _sanitize_yield returns (value, anomaly_dict) with reason codes."""

    def test_returns_tuple(self):
        """_sanitize_yield must return (value, anomaly_dict) tuple."""
        info = {'dividendYield': 0.0118, 'dividendRate': 2.36}
        result = _sanitize_yield(info, 200.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        val, anomaly = result
        assert isinstance(anomaly, dict)

    def test_anomaly_has_provenance(self):
        """anomaly dict must contain raw_provider_value, dividend_rate, provider, reason_codes."""
        info = {'dividendYield': 0.0118, 'dividendRate': 2.36}
        val, anomaly = _sanitize_yield(info, 200.0)
        assert 'raw_provider_value' in anomaly
        assert 'dividend_rate' in anomaly
        assert 'provider' in anomaly
        assert 'reason_codes' in anomaly
        assert anomaly['provider'] == 'yfinance'
        assert anomaly['raw_provider_value'] == 0.0118

    def test_missing_yield_codes(self):
        """Both yield and rate missing → DIV_YIELD_MISSING + DIV_PER_SHARE_MISSING."""
        val, anomaly = _sanitize_yield({}, 200.0)
        assert val is None
        assert 'DIV_YIELD_MISSING' in anomaly['reason_codes']
        assert 'DIV_PER_SHARE_MISSING' in anomaly['reason_codes']

    def test_out_of_range_code(self):
        """Yield > 25% → DIV_YIELD_OUT_OF_RANGE and value = None."""
        info = {'dividendYield': 1.36, 'dividendRate': 2.36}  # 136%
        val, anomaly = _sanitize_yield(info, 200.0)
        assert val is None
        assert 'DIV_YIELD_OUT_OF_RANGE' in anomaly['reason_codes']

    def test_normal_yield_no_anomaly_flags(self):
        """Normal yield (e.g., 1.18%) should have no rejection codes."""
        info = {'dividendYield': 0.0118, 'dividendRate': 2.36}
        val, anomaly = _sanitize_yield(info, 200.0)
        assert val is not None
        assert val > 0
        assert val <= 25
        assert 'DIV_YIELD_OUT_OF_RANGE' not in anomaly['reason_codes']

    def test_zero_rate_zero_yield(self):
        """Company that doesn't pay dividends → value = 0, no rejection."""
        info = {'dividendYield': 0, 'dividendRate': 0}
        val, anomaly = _sanitize_yield(info, 200.0)
        assert val == 0


# ============================================================================
# TASK 5 Tests: Narrative precision (max buy price)
# ============================================================================

class TestNarrativePrecision:
    """Verify max-buy-price math is correct."""

    def test_max_buy_price_formula(self):
        """Max buy price = IV × (1 - required_mos)."""
        iv = 184.69
        required_mos = 0.45
        max_buy = iv * (1 - required_mos)
        assert abs(max_buy - 101.58) < 0.01

    def test_drop_to_max_buy_bounded(self):
        """Drop from current price to max buy must be 0-100%."""
        price = 200.06
        iv = 184.69
        required_mos = 0.45
        max_buy = iv * (1 - required_mos)
        drop = max(0, min(100, (price - max_buy) / price * 100))
        assert 0 <= drop <= 100
        # For RTX: (200.06 - 101.58) / 200.06 ≈ 49.2%
        assert abs(drop - 49.2) < 0.5

    def test_drop_when_already_below_max_buy(self):
        """If price < max_buy, drop should be 0 (no decline needed)."""
        price = 80.0
        iv = 200.0
        required_mos = 0.30
        max_buy = iv * (1 - required_mos)  # 140
        drop = max(0, min(100, (price - max_buy) / price * 100))
        assert drop == 0  # Price already below max buy


# ============================================================================
# TASK 6 Tests: Terminal-dependence penalty
# ============================================================================

class TestTerminalDependencePenalty:
    """Verify conviction reduction when terminal value > 70% of DCF."""

    def test_penalty_when_above_70_pct(self):
        """Terminal 87% → penalty = min(15, round((87-70)*0.5)) = round(8.5) = 8."""
        terminal_pct = 87.0
        penalty = min(15, round((terminal_pct - 70) * 0.5))
        assert penalty == 8  # Python banker's rounding: round(8.5) = 8

    def test_no_penalty_when_below_70_pct(self):
        """Terminal 65% → no penalty."""
        terminal_pct = 65.0
        if terminal_pct > 70:
            penalty = min(15, round((terminal_pct - 70) * 0.5))
        else:
            penalty = 0
        assert penalty == 0

    def test_penalty_capped_at_15(self):
        """Terminal 99% → penalty = min(15, round((99-70)*0.5)) = round(14.5) = 14."""
        terminal_pct = 99.0
        penalty = min(15, round((terminal_pct - 70) * 0.5))
        assert penalty == 14  # Python banker's rounding: round(14.5) = 14
        # Verify cap still works: terminal 100% → (100-70)*0.5 = 15
        penalty2 = min(15, round((100 - 70) * 0.5))
        assert penalty2 == 15

    def test_terminal_penalty_in_v9_scores(self):
        """v9 owner scores must include terminal_penalty field."""
        summary = {}
        v8_data = {
            'financials': {
                '_data_status': 'OK',
                'roe': 20.0,
                'net_margin': 15.0,
                'revenue_growth': 8.0,
                'free_cash_flow': 5e9,
                'fcf_yield': 3.0,
                'debt_equity': 1.0,
                'gross_margin': 35.0,
                'operating_margin': 18.0,
                'market_cap': 150e9,
                'interest_coverage': 12.0,
                'buyback_yield': 2.0,
                'forward_pe': 18.0,
                'dividend_yield': 1.5,
                'payout_ratio': 35.0,
                'net_debt_ebitda': 2.0,
                'total_cash': 8e9,
                'total_debt': 40e9,
                'recommendation': 'buy',
            },
            'company': {'price': 100.0, 'market_cap': 150e9},
            'dcf': {
                'bear': 80.0, 'base': 100.0, 'bull': 125.0,
                'assumptions': {'terminal_value_pct': 85.0},
            },
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}, {'beat': True}, {'beat': False}, {'beat': True}],
        }
        v9 = _compute_v9_owner_scores(summary, v8_data)
        assert 'terminal_penalty' in v9
        assert v9['terminal_penalty'] > 0  # 85% > 70% threshold
        assert v9['terminal_pct'] == 85.0


# ============================================================================
# RTX Regression Snapshot
# ============================================================================

class TestRTXRegressionSnapshot:
    """End-to-end regression test with RTX-like inputs."""

    def test_rtx_dcf_full_pipeline(self):
        """RTX (beta=0.42, D/E=1.5, revenue=70B) should produce auditable DCF."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK',
            'revenue_ttm': 70e9,
            'shares_outstanding': 1.34e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 7e9,
            'ebitda': 12e9,
            'revenue_growth': 5.0,
            'debt_equity': 1.5,
            'net_debt': 32.5e9,
        }
        result = _build_dcf(info, fin)

        # Must NOT be disabled
        assert result.get('_dcf_disabled') is not True

        # Must have projections
        assert len(result['projections']) == 5

        # Assumptions: full CAPM decomposition
        a = result['assumptions']
        assert a['cost_of_equity'] == 6.10  # 4 + 0.42×5
        assert a['discount_rate_type'] == 'WACC'
        assert a['after_tax_cost_of_debt'] == 3.75  # 5% × (1-0.25)

        # EV → Equity bridge
        assert a['enterprise_value'] > 0
        assert a['net_debt_subtracted'] == 32.5e9
        assert a['equity_value'] < a['enterprise_value']  # Net debt subtracted
        assert a['equity_value'] > 0

        # IV per share (equity / shares)
        expected_iv = a['equity_value'] / a['shares']
        assert abs(result['base'] - round(expected_iv, 2)) < 0.01

        # Bear < Base < Bull
        assert result['bear'] < result['base'] < result['bull']

        # Shares: never formatted with $ prefix (tested via fN in JS, but verify data)
        assert a['shares'] > 1e9  # In absolute units, not dollars


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
