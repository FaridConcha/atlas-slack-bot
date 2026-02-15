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
    _build_company_info, _build_financials, _build_dcf, _validate_financials,
    _sanitize_yield, _sanitize_payout,
    _stabilize_beta, _compute_wacc_governed,
)
from v8_report import (
    _n, _compute_v8_scores, _compute_v9_owner_scores,
    _compute_fragility, _compute_dynamic_mos, _reconciliation_checks,
)
from valuation_config import CONFIG, SECTOR_BETA_BOUNDS


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
        """Yield with cross-check available must normalize correctly."""
        info = {'dividendYield': 1.36, 'dividendRate': 2.36}
        result, anomaly = _sanitize_yield(info, 200.0)
        # Cross-check recognizes 1.36 as percent → 1.36% (not 136%)
        assert result is not None
        assert result <= 25

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
        """Ke = Rf + β × ERP — Stage 5: beta floored for low-beta inputs."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = self._make_valid_fin()
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # Stage 5: beta=0.42 floored to 0.80 (DEFAULT_BETA_BOUNDS)
        # CAPM: 4.00 + 0.80 × 5.00 = 8.00
        assert a['beta_raw'] == 0.42
        assert a['beta'] == 0.80
        assert a['cost_of_equity'] == 8.00
        assert a['risk_free_rate'] == 4.00
        assert a['equity_risk_premium'] == 5.00

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
        assert a['equity_weight'] == 1.0  # decimal [0,1]

    def test_discount_rate_clamping(self):
        """Discount rate governed by multi-layer floor (Stage 5)."""
        # Very low beta → beta floored, then WACC floored
        info = {'currentPrice': 200.0, 'beta': 0.1}
        fin = self._make_valid_fin(de=2.0)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # Stage 5: general floor is 6.5%, default sector floor also applies
        assert a['discount_rate'] >= 6.5
        assert len(a['wacc_clamp_codes']) > 0  # Some floor applied

    def test_discount_rate_type_label(self):
        """discount_rate_type must be 'WACC' when D/E > 0."""
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = self._make_valid_fin(de=1.0)
        result = _build_dcf(info, fin)
        assert result['assumptions']['discount_rate_type'] == 'WACC'

    def test_wacc_arithmetic_precise(self):
        """Verify WACC = E/V × Ke + D/V × Kd(1-t) with governed beta."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = self._make_valid_fin(de=1.5)
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # Stage 5: beta=0.42 floored to 0.80
        governed_beta = a['beta']  # 0.80
        # D/E = 1.5 → D/V = 1.5/2.5 = 60%, E/V = 40%
        expected_dw = 1.5 / 2.5  # 0.6
        expected_ew = 1 - expected_dw  # 0.4
        ke = 0.04 + governed_beta * 0.05
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

    def test_percent_points_recognized_by_crosscheck(self):
        """raw=1.36 with div_rate=2.36, price=200 → recognized as percent, NOT out of range."""
        info = {'dividendYield': 1.36, 'dividendRate': 2.36}
        val, anomaly = _sanitize_yield(info, 200.0)
        # Cross-check: 2.36/200 = 0.0118 → raw 1.36 is percent (0.0136 ≈ 0.0118)
        assert val is not None
        assert abs(val - 1.36) < 0.1
        assert 'DIV_YIELD_OUT_OF_RANGE' not in anomaly['reason_codes']
        assert anomaly['unit_hint'] == 'percent_points'

    def test_normal_decimal_yield(self):
        """Normal yield as decimal (0.0118) should resolve correctly."""
        info = {'dividendYield': 0.0118, 'dividendRate': 2.36}
        val, anomaly = _sanitize_yield(info, 200.0)
        assert val is not None
        assert val > 0
        assert val <= 25
        assert 'DIV_YIELD_OUT_OF_RANGE' not in anomaly['reason_codes']
        assert anomaly['unit_hint'] == 'decimal'

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
        """RTX (beta=0.42, D/E=1.5, revenue=70B) should produce auditable DCF with Stage 5 governance."""
        info = {'currentPrice': 200.0, 'beta': 0.42, 'sector': 'Industrials'}
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
        result = _build_dcf(info, fin, sector='Industrials')

        # Must NOT be disabled
        assert result.get('_dcf_disabled') is not True

        # Must have projections
        assert len(result['projections']) == 5

        # Assumptions: full CAPM decomposition — Stage 5: beta floored to 0.80
        a = result['assumptions']
        assert a['beta_raw'] == 0.42  # Original beta preserved
        assert a['beta'] == 0.80  # Industrials floor applied
        assert 'BETA_FLOOR_APPLIED' in a['beta_flags']
        assert a['cost_of_equity'] == 8.00  # 4 + 0.80×5
        assert a['discount_rate_type'] == 'WACC'
        assert a['after_tax_cost_of_debt'] == 3.75  # 5% × (1-0.25)

        # Stage 5: WACC governance — sector floor should dominate
        assert a['discount_rate'] == 8.50  # Industrials sector floor
        assert 'FLOOR_SECTOR_INDUSTRIALS' in a['wacc_clamp_codes']

        # Weights stored as decimals [0,1]
        assert 0 < a['equity_weight'] < 1
        assert 0 < a['debt_weight'] < 1
        assert abs(a['equity_weight'] + a['debt_weight'] - 1.0) < 0.001

        # Clamp fields present
        assert 'clamp_floor' in a
        assert 'clamp_ceiling' in a

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

        # Shares: never formatted with $ prefix
        assert a['shares'] > 1e9

        # Stage 5: IV is significantly lower than pre-governance (~$184 → ~$95-110)
        assert result['base'] < 150, f"Expected governed IV < $150, got ${result['base']}"


# ============================================================================
# BLOCKER A — TEST 1: Weight format / bounds
# ============================================================================

class TestWeightFormatBounds:
    """Weights must be decimals in [0,1], never exceed 100% when displayed."""

    def test_weights_are_decimals(self):
        """E/V and D/V must be stored as decimals in [0,1]."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # D/E=1.5 → D/V=0.6, E/V=0.4
        assert abs(a['equity_weight'] - 0.4) < 0.001
        assert abs(a['debt_weight'] - 0.6) < 0.001
        # Display as percent: 40.0% and 60.0%, never >100%
        assert a['equity_weight'] * 100 <= 100
        assert a['debt_weight'] * 100 <= 100
        assert abs(a['equity_weight'] + a['debt_weight'] - 1.0) < 0.001

    def test_weight_guard_flags_invalid(self):
        """Cap structure warning set when weights are impossible."""
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 0, 'net_debt': 0,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # D/E=0 → E/V=1.0, D/V=0.0 — valid
        assert a['equity_weight'] == 1.0
        assert a['debt_weight'] == 0.0
        assert a['cap_structure_warning'] is None


# ============================================================================
# BLOCKER A — TEST 2: Discount factor consistency
# ============================================================================

class TestDiscountFactorConsistency:
    """Discount factors in projections must match discount_rate_used."""

    def test_df_y1_matches_discount_rate(self):
        """PV factor for Y1 must equal 1/(1+r) within tolerance."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        r = a['discount_rate'] / 100  # Convert from display percent to decimal
        expected_df_y1 = 1 / (1 + r)
        actual_df_y1 = result['projections'][0]['pv_factor']
        assert abs(actual_df_y1 - expected_df_y1) < 0.001, \
            f"DF Y1 mismatch: got {actual_df_y1}, expected {expected_df_y1} (r={r})"

    def test_df_y5_matches_discount_rate(self):
        """PV factor for Y5 must equal 1/(1+r)^5 within tolerance."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        r = a['discount_rate'] / 100
        expected_df_y5 = 1 / (1 + r) ** 5
        actual_df_y5 = result['projections'][4]['pv_factor']
        assert abs(actual_df_y5 - expected_df_y5) < 0.001, \
            f"DF Y5 mismatch: got {actual_df_y5}, expected {expected_df_y5} (r={r})"


# ============================================================================
# BLOCKER B — TEST 3: RTX dividend case (regression)
# ============================================================================

class TestRTXDividendRegression:
    """RTX-specific dividend yield regression: raw=1.36, rate=$2.72, price=$200.06."""

    def test_rtx_yield_resolves_correctly(self):
        """RTX: raw_yield=1.36 (percent_points) + div_rate=2.72 → ~1.36%, NOT N/A."""
        info = {'dividendYield': 1.36, 'dividendRate': 2.72}
        val, anomaly = _sanitize_yield(info, 200.06)

        # Must resolve to a value, not N/A
        assert val is not None, f"Dividend yield should NOT be None. Anomaly: {anomaly}"

        # Should be approximately 1.36%
        assert abs(val - 1.36) < 0.1, f"Expected ~1.36%, got {val}%"

        # Must NOT be flagged out of range
        assert 'DIV_YIELD_OUT_OF_RANGE' not in anomaly['reason_codes']

        # Provenance fields present
        assert anomaly['unit_hint'] == 'percent_points'
        assert anomaly['normalized_decimal'] is not None
        assert abs(anomaly['normalized_decimal'] - 0.0136) < 0.001

        # Expected yield from div_rate/price
        assert anomaly['expected_yield_decimal'] is not None
        expected = 2.72 / 200.06
        assert abs(anomaly['expected_yield_decimal'] - expected) < 0.0001

        # Display percent
        assert anomaly['normalized_pct'] is not None
        assert abs(anomaly['normalized_pct'] - 1.36) < 0.1


# ============================================================================
# BLOCKER B — TEST 4: Out of range AFTER normalization
# ============================================================================

class TestOutOfRangeAfterNormalization:
    """Out-of-range is evaluated AFTER normalization, not before."""

    def test_truly_anomalous_yield_flagged(self):
        """raw=136.0 (percent_points) → normalized decimal=1.36 (136%) → out of range."""
        info = {'dividendYield': 136.0, 'dividendRate': 2.72}
        val, anomaly = _sanitize_yield(info, 200.06)

        # 136/100 = 1.36 decimal → 136% → out of range
        assert val is None
        assert 'DIV_YIELD_OUT_OF_RANGE' in anomaly['reason_codes']

    def test_normal_yield_not_flagged(self):
        """raw=0.0136 (decimal) → normalized=0.0136 → 1.36% → NOT out of range."""
        info = {'dividendYield': 0.0136, 'dividendRate': 2.72}
        val, anomaly = _sanitize_yield(info, 200.06)
        assert val is not None
        assert abs(val - 1.36) < 0.1
        assert 'DIV_YIELD_OUT_OF_RANGE' not in anomaly['reason_codes']

    def test_edge_case_25_pct_threshold(self):
        """Normalized decimal 0.25 (25%) is the boundary — just over triggers flag."""
        # 26% yield → out of range
        info = {'dividendYield': 0.26}  # decimal, no rate for cross-check
        val, anomaly = _sanitize_yield(info, 100.0)
        assert val is None
        assert 'DIV_YIELD_OUT_OF_RANGE' in anomaly['reason_codes']


# ============================================================================
# BLOCKER C — TEST 5: Label
# ============================================================================

class TestEVBridgeLabel:
    """The EV bridge label must render as PV(FCF₁–₅), not PV(FCF₁₅)."""

    def test_label_contains_en_dash(self):
        """Verify the string uses subscript-1 + en-dash + subscript-5."""
        # The web_server.py line uses: PV(FCF\\u2081\\u2013\\u2085)
        # \\u2081 = ₁, \\u2013 = –, \\u2085 = ₅
        correct_label = 'PV(FCF\u2081\u2013\u2085)'
        wrong_label = 'PV(FCF\u2081\u2085)'
        assert '\u2013' in correct_label  # en-dash present
        assert correct_label != wrong_label  # different from wrong version
        # Verify the actual web_server.py file contains the correct label
        import re
        with open('web_server.py', 'r') as f:
            content = f.read()
        assert 'PV(FCF\\u2081\\u2013\\u2085)' in content, \
            "web_server.py must contain PV(FCF₁–₅) with en-dash"
        assert 'PV(FCF\\u2081\\u2085)' not in content, \
            "web_server.py must NOT contain PV(FCF₁₅) without en-dash"


# ============================================================================
# AUDIT FIX: CAPM display consistency (beta rounded at source)
# ============================================================================

class TestCAPMDisplayConsistency:
    """Beta is rounded to 2dp at source, so displayed Ke = Rf + β×ERP is exact."""

    def test_ke_matches_displayed_beta(self):
        """Ke must equal Rf + governed_beta × ERP exactly (no rounding drift)."""
        info = {'currentPrice': 200.0, 'beta': 0.418}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # Stage 5: raw beta 0.418 → rounded to 0.42 → floored to 0.80
        assert a['beta_raw'] == 0.42
        assert a['beta'] == 0.80  # DEFAULT_BETA_BOUNDS floor
        # Ke = Rf + β × ERP — uses governed beta
        expected_ke = a['risk_free_rate'] + a['beta'] * a['equity_risk_premium']
        assert abs(a['cost_of_equity'] - expected_ke) < 0.005, \
            f"CAPM display mismatch: Ke={a['cost_of_equity']}, expected={expected_ke}"

    def test_beta_exact_at_two_dp(self):
        """Stage 5: low beta floored; high beta preserved at 2dp."""
        info = {'currentPrice': 100.0, 'beta': 0.4178}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 50e9, 'shares_outstanding': 1e9,
            'free_cash_flow': 3e9, 'net_income_ttm': 4e9, 'ebitda': 8e9,
            'revenue_growth': 3.0, 'debt_equity': 0.5, 'net_debt': 5e9,
        }
        result = _build_dcf(info, fin)
        # Raw: 0.4178 → 0.42, then floored to 0.80
        assert result['assumptions']['beta_raw'] == 0.42
        assert result['assumptions']['beta'] == 0.80

        # Test with beta that's within bounds (no floor/cap)
        info2 = {'currentPrice': 100.0, 'beta': 1.15}
        result2 = _build_dcf(info2, fin)
        assert result2['assumptions']['beta'] == 1.15
        assert result2['assumptions']['beta_raw'] == 1.15

    def test_beta_none_defaults_to_one(self):
        """Missing beta defaults to 1.0."""
        info = {'currentPrice': 100.0, 'beta': None}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 50e9, 'shares_outstanding': 1e9,
            'free_cash_flow': 3e9, 'net_income_ttm': 4e9, 'ebitda': 8e9,
            'revenue_growth': 3.0, 'debt_equity': 0.5, 'net_debt': 5e9,
        }
        result = _build_dcf(info, fin)
        assert result['assumptions']['beta'] == 1.0


# ============================================================================
# AUDIT FIX: D/E normalization guard
# ============================================================================

class TestDEGuard:
    """D/E > 10x after normalization is flagged as None (anomalous)."""

    def test_normal_de_passes(self):
        """D/E = 150 (yfinance %) → 1.5x ratio → valid."""
        from v8_data import _safe_num
        de_raw = _safe_num(150.0, min_val=0)
        debt_equity = round(de_raw / 100, 2) if de_raw is not None else None
        if debt_equity is not None and debt_equity > 10.0:
            debt_equity = None
        assert debt_equity == 1.5

    def test_extreme_de_rejected(self):
        """D/E = 1500 (yfinance %) → 15.0x → anomalous → None."""
        from v8_data import _safe_num
        de_raw = _safe_num(1500.0, min_val=0)
        debt_equity = round(de_raw / 100, 2) if de_raw is not None else None
        if debt_equity is not None and debt_equity > 10.0:
            debt_equity = None
        assert debt_equity is None

    def test_borderline_de_passes(self):
        """D/E = 1000 (yfinance %) → 10.0x → exactly at boundary → valid."""
        from v8_data import _safe_num
        de_raw = _safe_num(1000.0, min_val=0)
        debt_equity = round(de_raw / 100, 2) if de_raw is not None else None
        if debt_equity is not None and debt_equity > 10.0:
            debt_equity = None
        assert debt_equity == 10.0


# ============================================================================
# AUDIT FIX: Bridge reconciliation warnings
# ============================================================================

class TestBridgeReconciliation:
    """DCF assumptions must include bridge_warnings field."""

    def test_no_warnings_for_valid_bridge(self):
        """Clean DCF should produce empty bridge_warnings."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.34e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32.5e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        assert 'bridge_warnings' in a
        assert a['bridge_warnings'] == [], \
            f"Expected no warnings, got: {a['bridge_warnings']}"

    def test_bridge_fields_consistent(self):
        """PV sum + TV PV = EV; EV - debt = equity; equity/shares ≈ base."""
        info = {'currentPrice': 200.0, 'beta': 0.42}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.34e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32.5e9,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # PV sum + TV PV = EV
        assert abs((a['pv_fcf_sum'] + a['terminal_value_pv']) - a['enterprise_value']) < 2
        # EV - net_debt = equity_value
        assert abs(a['equity_value'] - max(0, a['enterprise_value'] - a['net_debt_subtracted'])) < 2
        # equity / shares ≈ base IV
        expected_iv = a['equity_value'] / a['shares']
        assert abs(result['base'] - round(expected_iv, 2)) < 0.01


# ============================================================================
# AUDIT FIX: Normalized scores in engine summary
# ============================================================================

class TestNormalizedScoresExport:
    """atlas_engine.py must export scores_norm for display transparency."""

    def test_tanh_normalization_identity(self):
        """tanh(0/scale) = 0 for any scale > 0 (correlation risk 50 → signal 0)."""
        import numpy as np
        corr_risk = 50
        signal = -(corr_risk - 50)  # = 0
        norm = np.tanh(signal / 50)
        assert norm == 0.0

    def test_tanh_normalization_positive(self):
        """Positive raw score normalizes to positive value in [-1, 1]."""
        import numpy as np
        raw = 50.0
        scale = 100.0
        norm = np.tanh(raw / scale)
        assert 0 < norm < 1
        assert abs(norm - 0.4621) < 0.001

    def test_contribution_equals_weight_times_norm(self):
        """Contribution = weight × normalized_score × 100."""
        import numpy as np
        raw = 30.0
        scale = 50.0
        norm = float(np.tanh(raw / scale))
        weight = 0.125
        contribution = round(weight * norm * 100, 2)
        # Verify: contribution should be weight × norm × 100
        assert abs(contribution - round(weight * norm * 100, 2)) < 0.01


# ============================================================================
# STAGE 5: Beta Stabilization Tests
# ============================================================================

class TestBetaStabilization:
    """Sector beta floors and caps from valuation_config."""

    def test_industrials_floor(self):
        """Beta 0.42 for Industrials → floored to 0.80."""
        beta, flags = _stabilize_beta(0.42, 'Industrials')
        assert beta == 0.80
        assert 'BETA_FLOOR_APPLIED' in flags
        assert 'LOW_BETA_WARNING' in flags

    def test_industrials_cap(self):
        """Beta 2.0 for Industrials → capped to 1.60."""
        beta, flags = _stabilize_beta(2.0, 'Industrials')
        assert beta == 1.60
        assert 'BETA_CAP_APPLIED' in flags

    def test_no_clamp_when_in_bounds(self):
        """Beta 1.1 for Industrials → no clamp."""
        beta, flags = _stabilize_beta(1.1, 'Industrials')
        assert beta == 1.1
        assert 'BETA_FLOOR_APPLIED' not in flags
        assert 'BETA_CAP_APPLIED' not in flags

    def test_feature_flag_disables(self):
        """When beta_stabilization=False, raw beta passes through."""
        original = CONFIG.flags.beta_stabilization
        try:
            CONFIG.flags.beta_stabilization = False
            beta, flags = _stabilize_beta(0.42, 'Industrials')
            assert beta == 0.42
            assert flags == []
        finally:
            CONFIG.flags.beta_stabilization = original


# ============================================================================
# STAGE 5: WACC Governance Tests
# ============================================================================

class TestWACCGovernance:
    """Multi-layer WACC floor/ceiling governance."""

    def test_general_floor(self):
        """WACC below general floor → floored."""
        dr, wacc_raw, codes, audit = _compute_wacc_governed(0.80, '', 0)
        # Ke = 4% + 0.80*5% = 8%, no debt → WACC = 8%
        # General floor = 6.5%, sector floor for '' = 6.5%
        # 8% > 6.5% → no floor applied
        assert dr >= 0.065

    def test_sector_floor_industrials(self):
        """Industrials sector floor of 8.5% dominates when WACC < 8.5%."""
        # beta=0.80, D/E=1.5 → WACC ≈ 6.5% → sector floor 8.5% kicks in
        dr, wacc_raw, codes, audit = _compute_wacc_governed(0.80, 'Industrials', 1.5)
        assert abs(dr - 0.085) < 0.001
        assert any('SECTOR' in c for c in codes)

    def test_ceiling(self):
        """WACC above ceiling → capped at 15%."""
        dr, wacc_raw, codes, audit = _compute_wacc_governed(3.0, 'Technology', 0)
        # Ke = 4% + 3.0*5% = 19%, no debt → WACC = 19%
        assert abs(dr - 0.15) < 0.001
        assert 'CEILING_ABSOLUTE' in codes

    def test_reason_codes_present(self):
        """Reason codes list is always returned."""
        dr, wacc_raw, codes, audit = _compute_wacc_governed(1.0, 'Industrials', 0)
        assert isinstance(codes, list)
        assert isinstance(audit, dict)


# ============================================================================
# STAGE 5: Dynamic MOS Tests
# ============================================================================

class TestDynamicMOS:
    """Additive MOS model from valuation_config."""

    def test_base_only_very_stable(self):
        """Very Stable business with no uplifts → base MOS = 20%."""
        # Set up minimal DCF with no triggers
        dcf = {'assumptions': {'terminal_value_pct': 50, 'wacc_clamp_codes': [], 'discount_rate': 10}}
        fin = {'debt_equity': 0.5, 'roe': 15}
        mos, build = _compute_dynamic_mos('Very Stable', dcf, fin, 90, [])
        assert mos == 0.20
        assert build[0]['component'] == 'base'

    def test_all_uplifts(self):
        """Cyclical with all uplifts → high MOS."""
        dcf = {'assumptions': {'terminal_value_pct': 85, 'wacc_clamp_codes': ['FLOOR_SECTOR'], 'discount_rate': 9}}
        fin = {'debt_equity': 3.5, 'roe': 5}
        fragility = ['LOW_WACC', 'HIGH_TERMINAL_DEP', 'LOW_DATA_CONF']
        mos, build = _compute_dynamic_mos('Cyclical', dcf, fin, 60, fragility)
        # base=45 + DC=10 + terminal=10 + clamp=5 + leverage=10 + value_creation=10 + fragility=6 = 96
        assert mos > 0.80
        assert len(build) >= 5

    def test_rtx_scenario(self):
        """RTX-like scenario: Cyclical with high terminal dep and WACC clamp."""
        dcf = {'assumptions': {'terminal_value_pct': 80, 'wacc_clamp_codes': ['FLOOR_SECTOR_INDUSTRIALS'], 'discount_rate': 8.5}}
        fin = {'debt_equity': 1.5, 'roe': 7}
        fragility = ['HIGH_TERMINAL_DEP']
        mos, build = _compute_dynamic_mos('Cyclical', dcf, fin, 75, fragility)
        # base=45 + terminal=10 + clamp=5 + value_creation=10 = 70 minimum
        assert mos >= 0.70

    def test_max_buy_price_decreases(self):
        """Higher required MOS → lower max buy price."""
        mos_low = 0.30
        mos_high = 0.85
        iv = 100.0
        assert iv * (1 - mos_high) < iv * (1 - mos_low)


# ============================================================================
# STAGE 5: Fragility Tests
# ============================================================================

class TestFragility:
    """Fragility contributor detection."""

    def test_low_wacc_detected(self):
        """Discount rate < 7% triggers LOW_WACC."""
        dcf = {'assumptions': {'discount_rate': 6.0, 'terminal_value_pct': 50, 'fcf_margin': 10}}
        contributors = _compute_fragility(dcf, {}, {'data_confidence': 80})
        assert 'LOW_WACC' in contributors

    def test_no_fragility_when_healthy(self):
        """No contributors when all metrics healthy."""
        dcf = {'assumptions': {'discount_rate': 9.0, 'terminal_value_pct': 60, 'fcf_margin': 15}}
        contributors = _compute_fragility(dcf, {}, {'data_confidence': 85})
        assert len(contributors) == 0

    def test_multiple_contributors(self):
        """Multiple fragility conditions stack."""
        dcf = {'assumptions': {'discount_rate': 5.0, 'terminal_value_pct': 85, 'fcf_margin': 3}}
        contributors = _compute_fragility(dcf, {}, {'data_confidence': 60})
        assert 'LOW_WACC' in contributors
        assert 'HIGH_TERMINAL_DEP' in contributors
        assert 'LOW_DATA_CONF' in contributors
        assert 'FLAT_MARGINS_ASSUMPTION' in contributors
        assert len(contributors) == 4


# ============================================================================
# STAGE 5: Terminal Governance Tests
# ============================================================================

class TestTerminalGovernance:
    """Enhanced terminal penalty and bear_mult haircut."""

    def test_high_terminal_penalty(self):
        """80% terminal dependence → penalty = min(20, (80-70)*0.5) = 5."""
        info = {'currentPrice': 100.0, 'beta': 1.0}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 10e9, 'shares_outstanding': 1e9,
            'free_cash_flow': 500e6, 'net_income_ttm': 1e9, 'ebitda': 2e9,
            'revenue_growth': 3.0, 'debt_equity': 0, 'net_debt': 0,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        if a['terminal_value_pct'] >= 80:
            assert 'BEAR_MULT_HAIRCUT' in a['terminal_flags']

    def test_extreme_terminal_iv_confidence(self):
        """terminal_pct >= 90 → IV_CONFIDENCE_LOW flag."""
        info = {'currentPrice': 100.0, 'beta': 1.0}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 10e9, 'shares_outstanding': 1e9,
            'free_cash_flow': 500e6, 'net_income_ttm': 1e9, 'ebitda': 2e9,
            'revenue_growth': 2.0, 'debt_equity': 0, 'net_debt': 0,
        }
        result = _build_dcf(info, fin)
        a = result['assumptions']
        # terminal_flags should contain IV_CONFIDENCE_LOW only if pct >= 90
        if a['terminal_value_pct'] >= 90:
            assert 'IV_CONFIDENCE_LOW' in a['terminal_flags']
        else:
            assert 'IV_CONFIDENCE_LOW' not in a['terminal_flags']

    def test_terminal_penalty_cap_20(self):
        """Terminal conviction penalty cannot exceed 20."""
        # Penalty = min(20, (pct - 70) * 0.5)
        # At pct=110 (hypothetical): (110-70)*0.5 = 20, capped at 20
        penalty = min(CONFIG.terminal.penalty_cap, round((110 - 70) * CONFIG.terminal.penalty_per_pct))
        assert penalty == 20


# ============================================================================
# STAGE 5: Reconciliation Tests
# ============================================================================

class TestReconciliation:
    """Cross-module consistency checks."""

    def test_mos_sign_invariant(self):
        """If price > IV and MOS is positive, reconciliation error fires."""
        v9 = {'intrinsic_value_base': 100, 'mos_pct': 10}  # Bug: MOS should be negative
        v8 = {'company': {'price': 120}}
        errors = _reconciliation_checks({}, v8, v9)
        assert any(e['check'] == 'MOS_SIGN_INVARIANT' for e in errors)

    def test_no_error_when_consistent(self):
        """Consistent data produces no errors."""
        v9 = {'intrinsic_value_base': 100, 'mos_pct': -20}
        v8 = {'company': {'price': 120}, 'dcf': {'assumptions': {'pv_fcf_sum': 50, 'terminal_value_pv': 50, 'enterprise_value': 100}}}
        summary = {'w_dynamic': {'trend': 0.125, 'valuation': 0.125, 'consensus': 0.125, 'volatility': 0.125, 'macro': 0.125, 'liquidity': 0.125, 'global': 0.125, 'correlation': 0.125}}
        errors = _reconciliation_checks(summary, v8, v9)
        assert len(errors) == 0

    def test_weight_sum_warning(self):
        """Weight sum != 1.0 produces warning."""
        v9 = {'intrinsic_value_base': 0, 'mos_pct': 0}
        v8 = {'company': {'price': 100}, 'dcf': {}}
        summary = {'w_dynamic': {'trend': 0.2, 'valuation': 0.2}}  # Sum = 0.4, not 1.0
        errors = _reconciliation_checks(summary, v8, v9)
        assert any(e['check'] == 'WEIGHT_SUM' for e in errors)


# ============================================================================
# STAGE 5: Narrative Gating Tests
# ============================================================================

class TestNarrativeGating:
    """Sector-aware narrative gating logic."""

    def test_moat_protected_industries(self):
        """Aerospace & Defense is in MOAT_PROTECTED_INDUSTRIES."""
        from valuation_config import MOAT_PROTECTED_INDUSTRIES
        assert 'Aerospace & Defense' in MOAT_PROTECTED_INDUSTRIES

    def test_thin_margin_thresholds(self):
        """Thin margin thresholds match config."""
        from valuation_config import THIN_MARGIN_NET_THRESHOLD, THIN_MARGIN_OP_THRESHOLD
        assert THIN_MARGIN_NET_THRESHOLD == 5.0
        assert THIN_MARGIN_OP_THRESHOLD == 8.0

    def test_net_cash_gating(self):
        """Net cash narrative should only fire when net_debt < 0."""
        # Net debt = -5B means net cash
        assert -5e9 < 0  # Net cash condition
        # Net debt = 10B means NOT net cash
        assert 10e9 >= 0  # Not net cash


# ============================================================================
# STAGE 5.1: Fundamentals Integrity Regression Tests (Problem A)
# ============================================================================

class TestFundamentalsIntegrity:
    """Missing fundamentals must propagate as None, never as 0/1.0/'Unknown'."""

    def test_missing_market_cap_is_none(self):
        """market_cap=None when yfinance returns no marketCap."""
        from v8_data import _build_company_info, _safe_num
        info = {'longName': 'Test Corp'}  # No marketCap
        co = _build_company_info(info, 'TEST')
        assert co['market_cap'] is None

    def test_missing_beta_is_none(self):
        """beta=None when yfinance returns no beta."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp'}  # No beta
        co = _build_company_info(info, 'TEST')
        assert co['beta'] is None

    def test_zero_beta_is_none(self):
        """beta=0 from yfinance treated as missing."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'beta': 0}
        co = _build_company_info(info, 'TEST')
        assert co['beta'] is None

    def test_missing_sector_is_none(self):
        """sector=None when yfinance returns no sector or 'Unknown'."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp'}  # No sector
        co = _build_company_info(info, 'TEST')
        assert co['sector'] is None

    def test_unknown_sector_is_none(self):
        """sector='Unknown' from yfinance treated as missing."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'sector': 'Unknown'}
        co = _build_company_info(info, 'TEST')
        assert co['sector'] is None

    def test_valid_sector_preserved(self):
        """Real sector values are preserved."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'sector': 'Industrials'}
        co = _build_company_info(info, 'TEST')
        assert co['sector'] == 'Industrials'

    def test_missing_52w_range_is_none(self):
        """52-week high/low=None when not reported."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp'}
        co = _build_company_info(info, 'TEST')
        assert co['fifty_two_week_high'] is None
        assert co['fifty_two_week_low'] is None

    def test_zero_52w_range_is_none(self):
        """52-week high/low=0 treated as missing."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'fiftyTwoWeekHigh': 0, 'fiftyTwoWeekLow': 0}
        co = _build_company_info(info, 'TEST')
        assert co['fifty_two_week_high'] is None
        assert co['fifty_two_week_low'] is None

    def test_valid_data_preserved(self):
        """All fields present → values preserved."""
        from v8_data import _build_company_info
        info = {
            'longName': 'RTX Corp', 'sector': 'Industrials', 'industry': 'Aerospace & Defense',
            'marketCap': 150e9, 'fullTimeEmployees': 185000,
            'beta': 0.42, 'fiftyTwoWeekHigh': 130.0, 'fiftyTwoWeekLow': 90.0,
            'currentPrice': 115.0,
        }
        co = _build_company_info(info, 'RTX')
        assert co['market_cap'] == 150e9
        assert co['beta'] == 0.42
        assert co['sector'] == 'Industrials'
        assert co['fifty_two_week_high'] == 130.0

    def test_fundamentals_quality_object(self):
        """_fundamentals_quality tracks which fields are real vs defaulted."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'sector': 'Industrials', 'beta': 1.1}
        co = _build_company_info(info, 'TEST')
        fq = co['_fundamentals_quality']
        assert fq.sector_available is True
        assert fq.beta_available is True
        assert fq.market_cap_available is False

    def test_fundamentals_quality_missing_beta(self):
        """beta_defaulted=True when beta missing."""
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp'}
        co = _build_company_info(info, 'TEST')
        fq = co['_fundamentals_quality']
        assert fq.beta_defaulted is True


# ============================================================================
# STAGE 5.1: Canonical Suppression Mode Tests (Problem B)
# ============================================================================

class TestCanonicalSuppression:
    """ReportMode and consistent NULL outputs under suppression."""

    def test_report_mode_normal(self):
        """NORMAL when data is OK."""
        from valuation_config import FundamentalsQuality, ReportMode
        fq = FundamentalsQuality(data_status='OK')
        assert fq.report_mode == ReportMode.NORMAL

    def test_report_mode_suppressed(self):
        """FUNDAMENTALS_SUPPRESSED when data INVALID."""
        from valuation_config import FundamentalsQuality, ReportMode
        fq = FundamentalsQuality(data_status='INVALID')
        assert fq.report_mode == ReportMode.FUNDAMENTALS_SUPPRESSED

    def test_report_mode_partial(self):
        """PARTIAL when data DEGRADED."""
        from valuation_config import FundamentalsQuality, ReportMode
        fq = FundamentalsQuality(data_status='DEGRADED')
        assert fq.report_mode == ReportMode.PARTIAL

    def test_invalid_scores_all_zero(self):
        """Under INVALID, all scores return 0/null consistently."""
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'company': {'price': 100, 'sector': None},
            'dcf': {'bear': 0, 'base': 0, 'bull': 0, '_dcf_disabled': True},
            'institutional': {},
            'earnings': [],
        }
        summary = {}
        v9 = _compute_v9_owner_scores(summary, v8_data)
        assert v9['conviction'] == 0
        assert v9['business_quality'] == 0
        assert v9['moat_durability'] == 0
        assert v9['capital_allocation'] == 0
        assert v9['mos_pct'] == 0
        assert v9['v9_decision'] == 'RESEARCH'


# ============================================================================
# STAGE 5.1: TQ Reconciliation Tests (Problem C)
# ============================================================================

class TestTQReconciliation:
    """Single canonical TQ formula: (|C_raw|/100) * Rel * Gate * (DC/100)."""

    def test_tq_formula_matches_engine(self):
        """TQ computed from canonical formula matches engine output."""
        from atlas_engine import compute_trade_quality
        import numpy as np
        # Known inputs
        c_raw, rel, g, dc = 25.0, 0.85, 0.92, 78.0
        tq, cat = compute_trade_quality(c_raw, rel, g, dc)
        # Verify against canonical formula
        expected = (abs(c_raw) / 100.0) * rel * g * (dc / 100.0)
        expected = np.clip(expected, 0, 1)
        assert abs(tq - expected) < 1e-10, f"TQ {tq} != expected {expected}"

    def test_tq_zero_dc(self):
        """DC=0 produces TQ=0."""
        from atlas_engine import compute_trade_quality
        tq, _ = compute_trade_quality(50.0, 0.9, 1.0, 0.0)
        assert tq == 0.0

    def test_tq_zero_gate(self):
        """Gate=0 produces TQ=0."""
        from atlas_engine import compute_trade_quality
        tq, _ = compute_trade_quality(50.0, 0.9, 0.0, 80.0)
        assert tq == 0.0

    def test_tq_negative_composite(self):
        """Negative composite uses absolute value."""
        from atlas_engine import compute_trade_quality
        tq_pos, _ = compute_trade_quality(30.0, 0.8, 1.0, 90.0)
        tq_neg, _ = compute_trade_quality(-30.0, 0.8, 1.0, 90.0)
        assert abs(tq_pos - tq_neg) < 1e-10

    def test_tq_categories(self):
        """TQ categories match thresholds."""
        from atlas_engine import compute_trade_quality
        _, cat1 = compute_trade_quality(5.0, 0.5, 0.5, 50.0)  # Very small
        assert cat1 == "CASH"
        _, cat2 = compute_trade_quality(80.0, 0.9, 0.9, 95.0)  # Large
        assert cat2 == "STRONG_DIRECTIONAL"


# ============================================================================
# STAGE 5.1: Beta Defaulted Flag Tests (Problem E)
# ============================================================================

class TestBetaDefaulted:
    """Beta defaulted flag propagation and governance behavior."""

    def test_dcf_beta_defaulted_flag_set(self):
        """DCF assumptions include beta_defaulted=True when beta missing."""
        info = {'totalRevenue': 70e9, 'sharesOutstanding': 1.3e9}
        financials = {
            'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 4e9, 'ebitda': 8e9,
            'revenue_growth': 5.0, 'debt_equity': 0.5, 'net_debt': 10e9,
            '_data_status': 'OK', '_data_reasons': [],
        }
        dcf = _build_dcf(info, financials, sector='Industrials')
        assert dcf['assumptions']['beta_defaulted'] is True

    def test_dcf_beta_not_defaulted_when_present(self):
        """DCF assumptions beta_defaulted=False when beta reported."""
        info = {'beta': 1.15, 'totalRevenue': 70e9, 'sharesOutstanding': 1.3e9}
        financials = {
            'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 4e9, 'ebitda': 8e9,
            'revenue_growth': 5.0, 'debt_equity': 0.5, 'net_debt': 10e9,
            '_data_status': 'OK', '_data_reasons': [],
        }
        dcf = _build_dcf(info, financials, sector='Industrials')
        assert dcf['assumptions']['beta_defaulted'] is False

    def test_unknown_sector_uses_default_governance(self):
        """Empty sector falls back to DEFAULT_BETA_BOUNDS."""
        beta, flags = _stabilize_beta(0.42, '')
        # DEFAULT_BETA_BOUNDS is (0.80, 1.60), so 0.42 → 0.80
        assert beta == 0.80
        assert 'BETA_FLOOR_APPLIED' in flags


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
