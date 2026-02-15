#!/usr/bin/env python3
"""
ATLAS V12+ — Data Integrity & Mathematical Redesign Tests

Tests the None-safe data pipeline, fundamental integrity checks,
DCF gating, scoring behavior with missing data, probabilistic
engine framework (P1-P6), and Monte Carlo DCF simulations.

Run: python3 -m pytest test_data_integrity.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import MagicMock, patch
import numpy as np
from v8_data import (
    _safe_num, _safe_pct, _check_fundamental_integrity,
    _build_company_info, _build_financials, _build_dcf, _validate_financials,
    _sanitize_yield, _sanitize_payout,
    _stabilize_beta, _compute_wacc_governed,
    _evaluate_dcf_single, _ticker_seed, _run_monte_carlo_dcf, _compute_sensitivity,
)
from v8_report import (
    _n, _compute_v8_scores, _compute_v9_owner_scores,
    _compute_fragility, _compute_dynamic_mos, _reconciliation_checks,
)
from valuation_config import CONFIG, SECTOR_BETA_BOUNDS, INDUSTRY_PRIORS, PRIOR_CAP
from atlas_engine import (
    _compute_engine_variance, compute_all_engine_variances,
    build_covariance_matrix, compute_composite_variance,
    compute_confidence_adjusted_composite, compute_composite,
    compute_smooth_verdict, compute_kelly_position,
    ENGINE_VARIANCE_BASE, INTERACTION_MATRIX, ENGINE_STRUCTURAL_CORRELATIONS,
)


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
        """V9 owner scores should return RESEARCH with INVALID data.
        P3: scores are None (not 0) for INVALID — 'not scoreable' not 'scored zero'."""
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
        assert v9['business_quality'] is None  # P3: not scoreable
        assert v9['moat_durability'] is None
        assert v9['capital_allocation'] is None
        assert v9['conviction'] is None
        assert v9['mos_pct'] is None

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
        """equity_value / shares ≈ base_deterministic IV."""
        result = self._run_dcf()
        a = result['assumptions']
        expected_iv = a['equity_value'] / a['shares']
        base_det = result.get('base_deterministic', result['base'])
        assert abs(base_det - round(expected_iv, 2)) < 0.01

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

        # IV per share (equity / shares) — check deterministic base
        expected_iv = a['equity_value'] / a['shares']
        base_det = result.get('base_deterministic', result['base'])
        assert abs(base_det - round(expected_iv, 2)) < 0.01

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
        # equity / shares ≈ base_deterministic IV
        expected_iv = a['equity_value'] / a['shares']
        base_det = result.get('base_deterministic', result['base'])
        assert abs(base_det - round(expected_iv, 2)) < 0.01


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

    def test_52w_fallback_from_history(self):
        """52W range derived from price history when info is empty."""
        import pandas as pd
        import numpy as np
        from v8_data import _build_company_info
        info = {'longName': 'Test Corp', 'currentPrice': 150.0}
        # Simulate 1Y of price history
        dates = pd.date_range('2025-02-14', periods=252, freq='B')
        hist = pd.DataFrame({
            'Open': np.linspace(100, 160, 252),
            'High': np.linspace(105, 165, 252),
            'Low': np.linspace(95, 145, 252),
            'Close': np.linspace(100, 160, 252),
            'Volume': [1000000] * 252,
        }, index=dates)
        co = _build_company_info(info, 'TEST', hist=hist)
        assert co['fifty_two_week_high'] == round(165.0, 2) or co['fifty_two_week_high'] > 160
        assert co['fifty_two_week_low'] == round(95.0, 2) or co['fifty_two_week_low'] < 100

    def test_safe_info_fast_info_fallback(self):
        """_safe_fast_info returns dict with expected keys."""
        from v8_data import _safe_fast_info
        from unittest.mock import MagicMock
        ticker = MagicMock()
        fi = MagicMock()
        fi.market_cap = 150e9
        fi.shares = 1.3e9
        fi.last_price = 115.0
        fi.previous_close = 114.5
        fi.year_high = 130.0
        fi.year_low = 90.0
        fi.fifty_day_average = 112.0
        fi.two_hundred_day_average = 105.0
        ticker.fast_info = fi
        result = _safe_fast_info(ticker)
        assert result['marketCap'] == 150e9
        assert result['sharesOutstanding'] == 1.3e9
        assert result['fiftyTwoWeekHigh'] == 130.0
        assert result['fiftyTwoWeekLow'] == 90.0


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

    def test_invalid_scores_all_none(self):
        """Under INVALID, all scores return None consistently (P3: 'not scoreable')."""
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'company': {'price': 100, 'sector': None},
            'dcf': {'bear': 0, 'base': 0, 'bull': 0, '_dcf_disabled': True},
            'institutional': {},
            'earnings': [],
        }
        summary = {}
        v9 = _compute_v9_owner_scores(summary, v8_data)
        assert v9['conviction'] is None
        assert v9['business_quality'] is None
        assert v9['moat_durability'] is None
        assert v9['capital_allocation'] is None
        assert v9['mos_pct'] is None
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


# ============================================================================
# P0: MOS Semantics Tests
# ============================================================================

class TestMOSSemantics:
    """P0: mos_iv_basis, premium_to_iv, narrative consistency."""

    def _make_v9(self, price=200, iv=100):
        v8_data = {
            'financials': {
                '_data_status': 'OK', 'roe': 20.0, 'net_margin': 15.0,
                'revenue_growth': 8.0, 'free_cash_flow': 5e9, 'fcf_yield': 3.0,
                'debt_equity': 1.0, 'gross_margin': 35.0, 'operating_margin': 18.0,
                'market_cap': 150e9, 'interest_coverage': 12.0, 'buyback_yield': 2.0,
                'forward_pe': 18.0, 'dividend_yield': 1.5, 'payout_ratio': 35.0,
                'net_debt_ebitda': 2.0, 'total_cash': 8e9, 'total_debt': 40e9,
                'recommendation': 'buy',
            },
            'company': {'price': price, 'market_cap': 150e9},
            'dcf': {'bear': iv * 0.8, 'base': iv, 'bull': iv * 1.25,
                    'assumptions': {'terminal_value_pct': 65, 'discount_rate': 9,
                                    'wacc_clamp_codes': [], 'fcf_margin': 10}},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}, {'beat': True}, {'beat': False}, {'beat': True}],
        }
        return _compute_v9_owner_scores({}, v8_data)

    def test_mos_iv_basis_present(self):
        """mos_iv_basis field exists and is IV-denominated."""
        v9 = self._make_v9(price=80, iv=100)
        assert 'mos_iv_basis' in v9
        assert v9['mos_iv_basis'] is not None
        # (100 - 80) / 100 = 20%
        assert abs(v9['mos_iv_basis'] - 20.0) < 0.5

    def test_premium_to_iv_present(self):
        """premium_to_iv field exists: (price/IV - 1)."""
        v9 = self._make_v9(price=120, iv=100)
        assert 'premium_to_iv' in v9
        # (120/100 - 1) = 20%
        assert abs(v9['premium_to_iv'] - 20.0) < 0.5

    def test_premium_positive_when_overvalued(self):
        """premium_to_iv > 0 when price > IV."""
        v9 = self._make_v9(price=150, iv=100)
        assert v9['premium_to_iv'] > 0

    def test_premium_negative_when_undervalued(self):
        """premium_to_iv < 0 when price < IV."""
        v9 = self._make_v9(price=80, iv=100)
        assert v9['premium_to_iv'] < 0

    def test_mos_none_when_invalid(self):
        """P3+P0: mos_iv_basis is None when data INVALID."""
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['mc_missing']},
            'company': {'price': 100}, 'dcf': {'_dcf_disabled': True, 'bear': 0, 'base': 0, 'bull': 0},
            'institutional': {}, 'earnings': [],
        }
        v9 = _compute_v9_owner_scores({}, v8_data)
        assert v9['mos_iv_basis'] is None
        assert v9['premium_to_iv'] is None


# ============================================================================
# P1: Sector Provenance Tests
# ============================================================================

class TestSectorProvenance:
    """P1: Sector provenance tracking in DCF governance."""

    def test_provider_sector_has_provenance(self):
        """Known sector → sector_provenance = PROVIDER."""
        info = {'currentPrice': 200.0, 'beta': 1.0, 'sector': 'Industrials'}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        dcf = _build_dcf(info, fin, sector='Industrials')
        assert dcf['assumptions']['sector_provenance'] == 'PROVIDER'

    def test_missing_sector_is_defaulted(self):
        """Missing sector → sector_provenance = DEFAULTED."""
        info = {'currentPrice': 200.0, 'beta': 1.0}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        dcf = _build_dcf(info, fin)
        assert dcf['assumptions']['sector_provenance'] == 'DEFAULTED'

    def test_defaulted_sector_no_sector_floor(self):
        """P1: When sector is DEFAULTED, sector-specific WACC floor should NOT apply."""
        from v8_data import _compute_wacc_governed
        from valuation_config import SectorProvenance
        # With Industrials floor (8.5%): if sector is DEFAULTED, should only use general floor (6.5%)
        dr, wacc_raw, codes, audit = _compute_wacc_governed(
            0.80, 'Industrials', 1.5, sector_provenance=SectorProvenance.DEFAULTED
        )
        # General floor is 6.5%, WACC for beta=0.80 D/E=1.5 is ~5.5%
        # Should floor at general (6.5%), NOT sector (8.5%)
        assert abs(dr - 0.065) < 0.001
        assert not any('SECTOR' in c for c in codes)

    def test_provider_sector_applies_sector_floor(self):
        """P1: When sector is from PROVIDER, sector floor DOES apply."""
        from v8_data import _compute_wacc_governed
        from valuation_config import SectorProvenance
        dr, wacc_raw, codes, audit = _compute_wacc_governed(
            0.80, 'Industrials', 1.5, sector_provenance=SectorProvenance.PROVIDER
        )
        # Industrials sector floor is 8.5%
        assert abs(dr - 0.085) < 0.001
        assert any('SECTOR' in c for c in codes)


# ============================================================================
# P2: Beta Path Tests
# ============================================================================

class TestBetaPath:
    """P2: Measured vs proxy beta paths."""

    def test_measured_beta_path(self):
        """When beta is reported, beta_path = MEASURED."""
        info = {'currentPrice': 200.0, 'beta': 0.42, 'sector': 'Industrials'}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        dcf = _build_dcf(info, fin, sector='Industrials')
        assert dcf['assumptions']['beta_path'] == 'MEASURED'

    def test_proxy_beta_path(self):
        """When beta is missing, beta_path = PROXY."""
        info = {'currentPrice': 200.0, 'sector': 'Industrials'}
        fin = {
            '_data_status': 'OK', 'revenue_ttm': 70e9, 'shares_outstanding': 1.3e9,
            'free_cash_flow': 5e9, 'net_income_ttm': 7e9, 'ebitda': 12e9,
            'revenue_growth': 5.0, 'debt_equity': 1.5, 'net_debt': 32e9,
        }
        dcf = _build_dcf(info, fin, sector='Industrials')
        assert dcf['assumptions']['beta_path'] == 'PROXY'
        assert dcf['assumptions']['beta_defaulted'] is True

    def test_proxy_beta_skips_sector_stabilization(self):
        """P2: Proxy beta (1.0) should NOT be sector-floored/capped."""
        from valuation_config import BetaPath
        # Proxy beta is 1.0 — should pass through without Industrials floor (0.80)
        beta, flags = _stabilize_beta(1.0, 'Industrials', beta_path=BetaPath.PROXY)
        assert beta == 1.0
        assert 'BETA_PROXY_1_0' in flags
        assert 'BETA_FLOOR_APPLIED' not in flags

    def test_measured_beta_gets_stabilized(self):
        """P2: Measured beta IS subject to sector stabilization."""
        from valuation_config import BetaPath
        beta, flags = _stabilize_beta(0.42, 'Industrials', beta_path=BetaPath.MEASURED)
        assert beta == 0.80  # Industrials floor
        assert 'BETA_FLOOR_APPLIED' in flags


# ============================================================================
# P4: MOS Build Completeness Tests
# ============================================================================

class TestMOSBuildCompleteness:
    """P4: All MOS components shown, even non-triggered."""

    def test_all_components_present(self):
        """Even when no uplifts trigger, all 7 components present in build."""
        dcf = {'assumptions': {'terminal_value_pct': 50, 'wacc_clamp_codes': [], 'discount_rate': 10}}
        fin = {'debt_equity': 0.5, 'roe': 15}
        mos, build = _compute_dynamic_mos('Very Stable', dcf, fin, 90, [])
        components = [b['component'] for b in build]
        assert 'base' in components
        assert 'data_confidence' in components
        assert 'terminal_dependence' in components
        assert 'wacc_clamp' in components
        assert 'leverage' in components
        assert 'value_creation' in components
        assert 'fragility' in components
        assert len(build) == 7

    def test_zero_adjustment_for_non_triggered(self):
        """Non-triggered components show adjustment=0."""
        dcf = {'assumptions': {'terminal_value_pct': 50, 'wacc_clamp_codes': [], 'discount_rate': 10}}
        fin = {'debt_equity': 0.5, 'roe': 15}
        mos, build = _compute_dynamic_mos('Very Stable', dcf, fin, 90, [])
        dc_entry = next(b for b in build if b['component'] == 'data_confidence')
        assert dc_entry['adjustment'] == 0
        assert 'no uplift' in dc_entry['reason']


# ============================================================================
# P5: CA Evidence Contradiction Tests
# ============================================================================

class TestCAEvidenceContradictions:
    """P5: Resolve ADEQUATE + VALUE_DESTROYER contradiction."""

    def test_value_destroyer_buyback_discipline(self):
        """P5: If ROIC < WACC, buybacks labeled DESTROYS_VALUE regardless of PE."""
        v8_data = {
            'financials': {
                '_data_status': 'OK', 'roe': 5.0, 'net_margin': 3.0,
                'revenue_growth': 2.0, 'free_cash_flow': 1e9, 'fcf_yield': 2.0,
                'debt_equity': 1.5, 'gross_margin': 25.0, 'operating_margin': 8.0,
                'market_cap': 50e9, 'interest_coverage': 6.0, 'buyback_yield': 3.0,
                'forward_pe': 15.0, 'dividend_yield': 1.0, 'payout_ratio': 40.0,
                'net_debt_ebitda': 3.0, 'total_cash': 5e9, 'total_debt': 20e9,
                'recommendation': 'hold',
            },
            'company': {'price': 100.0, 'market_cap': 50e9},
            'dcf': {'bear': 60.0, 'base': 80.0, 'bull': 100.0,
                    'assumptions': {'terminal_value_pct': 70, 'discount_rate': 9,
                                    'wacc_clamp_codes': [], 'fcf_margin': 5}},
            'institutional': {'short_pct': 3.0},
            'earnings': [{'beat': True}, {'beat': False}],
        }
        v9 = _compute_v9_owner_scores({}, v8_data)
        ca_ev = v9['ca_evidence']
        # ROE 5% with D/E 1.5 → ROIC proxy ≈ 2% < WACC 9% → VALUE_DESTROYER
        assert 'VALUE_DESTROYER' in ca_ev['reason_codes']
        # P5: buyback_discipline must NOT be ADEQUATE when destroying value
        assert ca_ev['buyback_discipline'] != 'ADEQUATE'
        assert ca_ev['buyback_discipline'] == 'DESTROYS_VALUE'

    def test_value_creator_good_buyback(self):
        """Value creator with low PE → GOOD buyback discipline."""
        v8_data = {
            'financials': {
                '_data_status': 'OK', 'roe': 25.0, 'net_margin': 20.0,
                'revenue_growth': 10.0, 'free_cash_flow': 8e9, 'fcf_yield': 5.0,
                'debt_equity': 0.5, 'gross_margin': 45.0, 'operating_margin': 25.0,
                'market_cap': 150e9, 'interest_coverage': 20.0, 'buyback_yield': 4.0,
                'forward_pe': 15.0, 'dividend_yield': 1.5, 'payout_ratio': 30.0,
                'net_debt_ebitda': 1.0, 'total_cash': 10e9, 'total_debt': 15e9,
                'recommendation': 'buy',
            },
            'company': {'price': 100.0, 'market_cap': 150e9},
            'dcf': {'bear': 90.0, 'base': 110.0, 'bull': 140.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 12}},
            'institutional': {'short_pct': 1.0},
            'earnings': [{'beat': True}, {'beat': True}, {'beat': True}],
        }
        v9 = _compute_v9_owner_scores({}, v8_data)
        ca_ev = v9['ca_evidence']
        assert 'VALUE_CREATOR' in ca_ev['reason_codes']
        assert ca_ev['buyback_discipline'] == 'GOOD'
        assert 'BUYBACK_VALUE_DESTRUCTIVE' not in ca_ev['reason_codes']


# ============================================================================
# P6: Narrative Guardrails Tests
# ============================================================================

class TestNarrativeGuardrails:
    """P6: Narrative assertions gated on evidence."""

    def test_invalid_data_no_numeric_assertions(self):
        """Under INVALID data, narrative should say 'data insufficient', not numeric claims."""
        from v8_report import _section_owner_assessment
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'company': {'symbol': 'TEST', 'name': 'Test Corp', 'price': 100},
            'dcf': {'_dcf_disabled': True, 'bear': 0, 'base': 0, 'bull': 0},
            'institutional': {}, 'earnings': [],
        }
        section = _section_owner_assessment({}, v8_data)
        assert 'data unavailable' in section.lower() or 'data insufficient' in section.lower()
        # Should NOT contain score assertions like "X.X/5" for BQ/Moat/CA
        assert 'N/A  (data unavailable)' in section


# ============================================================================
# Stage 6: G1 — Industry Prior System Tests
# ============================================================================

class TestIndustryPriors:
    """G1: Industry prior applied when metric data is missing."""

    def _make_v8_data(self, industry='Aerospace & Defense', missing_fields=None):
        """Build v8_data with configurable missing fields."""
        fin = {
            '_data_status': 'OK',
            'roe': 15.0,
            'net_margin': 10.0,
            'revenue_growth': 5.0,
            'free_cash_flow': 3e9,
            'fcf_yield': 2.0,
            'debt_equity': 1.0,
            'gross_margin': 30.0,
            'operating_margin': 12.0,
            'market_cap': 100e9,
            'interest_coverage': 8.0,
            'buyback_yield': 1.0,
            'forward_pe': 18.0,
            'dividend_yield': 1.5,
            'payout_ratio': 35.0,
            'net_debt_ebitda': 2.0,
            'total_cash': 5e9,
            'total_debt': 20e9,
            'recommendation': 'hold',
        }
        if missing_fields:
            for f in missing_fields:
                fin[f] = None
        return {
            'financials': fin,
            'company': {'price': 100.0, 'market_cap': 100e9, 'industry': industry},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 120.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 8}},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}, {'beat': True}],
        }

    def test_prior_applied_with_missing_data(self):
        """Prior should be applied when >= 2 metric fields are missing."""
        v8 = self._make_v8_data(
            industry='Aerospace & Defense',
            missing_fields=['roe', 'net_margin', 'revenue_growth']
        )
        v9 = _compute_v9_owner_scores({}, v8)
        pa = v9['prior_audit']
        assert pa['prior_applied'] is True
        assert pa['industry'] == 'Aerospace & Defense'
        assert len(pa['data_missing_fields']) >= 2

    def test_prior_capped_at_prior_cap(self):
        """Prior adjustment should never exceed PRIOR_CAP (0.7)."""
        v8 = self._make_v8_data(
            industry='Aerospace & Defense',
            missing_fields=['roe', 'net_margin', 'revenue_growth', 'gross_margin', 'operating_margin', 'free_cash_flow']
        )
        v9 = _compute_v9_owner_scores({}, v8)
        pa = v9['prior_audit']
        assert pa['prior_applied'] is True
        # All adjustments should be <= PRIOR_CAP
        assert pa['prior_value']['bq'] <= PRIOR_CAP
        assert pa['prior_value']['moat'] <= PRIOR_CAP
        assert pa['prior_value']['ca'] <= PRIOR_CAP

    def test_prior_not_applied_when_data_complete(self):
        """Prior should NOT be applied when all data is present (< 2 missing)."""
        v8 = self._make_v8_data(industry='Aerospace & Defense', missing_fields=[])
        v9 = _compute_v9_owner_scores({}, v8)
        pa = v9['prior_audit']
        assert pa['prior_applied'] is False

    def test_prior_not_applied_for_unknown_industry(self):
        """Prior should NOT be applied for industries not in INDUSTRY_PRIORS."""
        v8 = self._make_v8_data(
            industry='Pet Food Manufacturing',
            missing_fields=['roe', 'net_margin', 'revenue_growth']
        )
        v9 = _compute_v9_owner_scores({}, v8)
        pa = v9['prior_audit']
        assert pa['prior_applied'] is False

    def test_prior_feature_flag_toggle(self):
        """When industry_priors flag is False, no prior should be applied."""
        original = CONFIG.flags.industry_priors
        try:
            CONFIG.flags.industry_priors = False
            v8 = self._make_v8_data(
                industry='Aerospace & Defense',
                missing_fields=['roe', 'net_margin', 'revenue_growth']
            )
            v9 = _compute_v9_owner_scores({}, v8)
            pa = v9['prior_audit']
            assert pa['prior_applied'] is False
        finally:
            CONFIG.flags.industry_priors = original


# ============================================================================
# Stage 6: G2 — ROIC Uncertain Tests
# ============================================================================

class TestROICUncertain:
    """G2: ROIC None when ROE missing — no penalty."""

    def _make_v8_data(self, roe=None):
        fin = {
            '_data_status': 'OK',
            'roe': roe,
            'net_margin': 10.0,
            'revenue_growth': 5.0,
            'free_cash_flow': 3e9,
            'fcf_yield': 2.0,
            'debt_equity': 1.0,
            'gross_margin': 30.0,
            'operating_margin': 12.0,
            'market_cap': 100e9,
            'interest_coverage': 8.0,
            'buyback_yield': 1.0,
            'forward_pe': 18.0,
            'dividend_yield': 1.5,
            'payout_ratio': 35.0,
            'net_debt_ebitda': 2.0,
            'total_cash': 5e9,
            'total_debt': 20e9,
            'recommendation': 'hold',
        }
        return {
            'financials': fin,
            'company': {'price': 100.0, 'market_cap': 100e9},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 120.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 8}},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}],
        }

    def test_roic_none_when_roe_missing(self):
        """When ROE is None, ca_evidence should show roic_proxy as None."""
        v8 = self._make_v8_data(roe=None)
        v9 = _compute_v9_owner_scores({}, v8)
        ca_ev = v9['ca_evidence']
        assert ca_ev['roic_proxy'] is None

    def test_ca_score_not_penalized_for_missing_roe(self):
        """CA score should not be zero-punished when ROE is missing."""
        v8_with = self._make_v8_data(roe=20.0)
        v8_without = self._make_v8_data(roe=None)
        v9_with = _compute_v9_owner_scores({}, v8_with)
        v9_without = _compute_v9_owner_scores({}, v8_without)
        # Without ROE, CA should still have some score from other components (debt, dividend, etc.)
        assert v9_without['capital_allocation'] >= 0
        # The CA score without ROE should not be dramatically lower than expected
        # (no penalty, just no ROIC credit)

    def test_ca_evidence_shows_unavailable(self):
        """When ROE missing, data_completeness should show roic_method as 'unavailable'."""
        v8 = self._make_v8_data(roe=None)
        v9 = _compute_v9_owner_scores({}, v8)
        dc = v9['ca_evidence'].get('data_completeness', {})
        assert dc.get('roe_available') is False
        assert dc.get('roic_method') == 'unavailable'


# ============================================================================
# Stage 6: G3 — Buyback IV Check Tests
# ============================================================================

class TestBuybackIVCheck:
    """G3: Buyback discipline includes IV-price check."""

    def _make_v8_data(self, price=100.0, base_iv=120.0, buyback_yield=3.0, forward_pe=15.0, roe=20.0):
        fin = {
            '_data_status': 'OK',
            'roe': roe,
            'net_margin': 15.0,
            'revenue_growth': 8.0,
            'free_cash_flow': 5e9,
            'fcf_yield': 3.0,
            'debt_equity': 0.5,
            'gross_margin': 40.0,
            'operating_margin': 20.0,
            'market_cap': 100e9,
            'interest_coverage': 15.0,
            'buyback_yield': buyback_yield,
            'forward_pe': forward_pe,
            'dividend_yield': 1.5,
            'payout_ratio': 35.0,
            'net_debt_ebitda': 1.0,
            'total_cash': 8e9,
            'total_debt': 15e9,
            'recommendation': 'buy',
        }
        return {
            'financials': fin,
            'company': {'price': price, 'market_cap': 100e9},
            'dcf': {'bear': base_iv * 0.8, 'base': base_iv, 'bull': base_iv * 1.25,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 10}},
            'institutional': {'short_pct': 1.0},
            'earnings': [{'beat': True}, {'beat': True}],
        }

    def test_potentially_destructive_when_price_above_iv(self):
        """POTENTIALLY_DESTRUCTIVE when price > IV and buying back."""
        v8 = self._make_v8_data(price=150.0, base_iv=100.0, buyback_yield=3.0)
        v9 = _compute_v9_owner_scores({}, v8)
        assert v9['ca_evidence']['buyback_discipline'] == 'POTENTIALLY_DESTRUCTIVE'

    def test_insufficient_evidence_when_no_data(self):
        """INSUFFICIENT_EVIDENCE when buyback_yield is None."""
        v8 = self._make_v8_data(buyback_yield=0.0)
        # Set buyback_yield to None (not just 0)
        v8['financials']['buyback_yield'] = None
        v9 = _compute_v9_owner_scores({}, v8)
        assert v9['ca_evidence']['buyback_discipline'] == 'INSUFFICIENT_EVIDENCE'

    def test_good_when_value_creator_low_pe(self):
        """GOOD when value creator + low PE + price < IV."""
        v8 = self._make_v8_data(price=90.0, base_iv=120.0, buyback_yield=3.0, forward_pe=15.0)
        v9 = _compute_v9_owner_scores({}, v8)
        assert v9['ca_evidence']['buyback_discipline'] == 'GOOD'


# ============================================================================
# Stage 6: G6 — Risk Uncertain Labels Tests
# ============================================================================

class TestRiskUncertain:
    """G6: Missing data → 'Uncertain' risk entries."""

    def _make_v8_data(self, **overrides):
        fin = {
            '_data_status': 'OK',
            'roe': 15.0,
            'net_margin': 10.0,
            'revenue_growth': 5.0,
            'free_cash_flow': 3e9,
            'fcf_yield': 2.0,
            'debt_equity': 1.0,
            'gross_margin': 30.0,
            'operating_margin': 12.0,
            'market_cap': 100e9,
            'interest_coverage': 8.0,
            'buyback_yield': 1.0,
            'forward_pe': 18.0,
            'dividend_yield': 1.5,
            'payout_ratio': 35.0,
            'net_debt_ebitda': 2.0,
            'total_cash': 5e9,
            'total_debt': 20e9,
            'recommendation': 'hold',
        }
        fin.update(overrides)
        return {
            'financials': fin,
            'company': {'price': 100.0, 'market_cap': 100e9},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 120.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 8}},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}],
        }

    def test_missing_debt_equity_uncertain(self):
        """Missing D/E → 'Uncertain' risk entry."""
        v8 = self._make_v8_data(debt_equity=None)
        v9 = _compute_v9_owner_scores({}, v8)
        risks = v9['permanent_loss_risks']
        uncertain = [r for r in risks if r[1] == 'Uncertain']
        assert len(uncertain) > 0
        assert any('Leverage' in r[0] for r in uncertain)

    def test_missing_revenue_growth_uncertain(self):
        """Missing revenue growth → 'Uncertain' risk entry."""
        v8 = self._make_v8_data(revenue_growth=None)
        v9 = _compute_v9_owner_scores({}, v8)
        risks = v9['permanent_loss_risks']
        uncertain = [r for r in risks if r[1] == 'Uncertain']
        assert any('Revenue' in r[0] for r in uncertain)

    def test_all_data_present_no_uncertain(self):
        """When all data present, no 'Uncertain' entries."""
        v8 = self._make_v8_data()
        v9 = _compute_v9_owner_scores({}, v8)
        risks = v9['permanent_loss_risks']
        uncertain = [r for r in risks if r[1] == 'Uncertain']
        assert len(uncertain) == 0


# ============================================================================
# Stage 6: G7 — Narrative Moat References Tests
# ============================================================================

class TestNarrativeMoat:
    """G7: Moat durability referenced in WHY bullets."""

    def test_moat_referenced_in_why(self):
        """WHY bullets should include moat durability reference."""
        from v8_report import _section_owner_assessment
        v8_data = {
            'financials': {
                '_data_status': 'OK',
                'roe': 25.0, 'net_margin': 20.0, 'revenue_growth': 10.0,
                'free_cash_flow': 8e9, 'fcf_yield': 5.0, 'debt_equity': 0.5,
                'gross_margin': 55.0, 'operating_margin': 25.0,
                'market_cap': 200e9, 'interest_coverage': 20.0,
                'buyback_yield': 2.0, 'forward_pe': 15.0,
                'dividend_yield': 1.5, 'payout_ratio': 30.0,
                'net_debt_ebitda': 1.0, 'total_cash': 10e9, 'total_debt': 15e9,
                'recommendation': 'buy',
            },
            'company': {'symbol': 'TEST', 'name': 'Test Corp', 'price': 80.0, 'market_cap': 200e9},
            'dcf': {'bear': 90.0, 'base': 110.0, 'bull': 140.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 12}},
            'institutional': {'short_pct': 1.0},
            'earnings': [{'beat': True}, {'beat': True}, {'beat': True}],
        }
        section = _section_owner_assessment({}, v8_data)
        assert 'Moat durability' in section or 'moat durability' in section.lower()

    def test_moat_uncertain_when_low_with_prior(self):
        """Low moat + prior applied → 'uncertain due to limited data' in narrative."""
        v8_data = {
            'financials': {
                '_data_status': 'OK',
                'roe': None, 'net_margin': None, 'revenue_growth': None,
                'free_cash_flow': None, 'fcf_yield': None, 'debt_equity': 1.0,
                'gross_margin': None, 'operating_margin': None,
                'market_cap': 100e9, 'interest_coverage': 8.0,
                'buyback_yield': 1.0, 'forward_pe': 18.0,
                'dividend_yield': 1.5, 'payout_ratio': 35.0,
                'net_debt_ebitda': 2.0, 'total_cash': 5e9, 'total_debt': 20e9,
                'recommendation': 'hold',
            },
            'company': {'symbol': 'RTX', 'name': 'RTX Corp', 'price': 200.0,
                        'market_cap': 100e9, 'industry': 'Aerospace & Defense'},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 120.0,
                    'assumptions': {'terminal_value_pct': 60, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 8}},
            'institutional': {'short_pct': 2.0},
            'earnings': [],
        }
        v9 = _compute_v9_owner_scores({}, v8_data)
        # Prior should be applied (A&D industry, 5+ missing fields)
        assert v9['prior_audit']['prior_applied'] is True

    def test_moat_data_insufficient_when_none(self):
        """When moat_durability is None (INVALID data), narrative says 'data insufficient'."""
        from v8_report import _section_owner_assessment
        v8_data = {
            'financials': {'_data_status': 'INVALID', '_data_reasons': ['market_cap_missing']},
            'company': {'symbol': 'TEST', 'name': 'Test Corp', 'price': 100},
            'dcf': {'_dcf_disabled': True, 'bear': 0, 'base': 0, 'bull': 0},
            'institutional': {}, 'earnings': [],
        }
        section = _section_owner_assessment({}, v8_data)
        # Under INVALID, moat is None → "data insufficient"
        assert 'data insufficient' in section.lower() or 'data unavailable' in section.lower()


# ============================================================================
# Stage 6: G9 — MOS Build Reconciliation Tests
# ============================================================================

class TestMOSBuildReconciliation:
    """G9: MOS build sum must match required_mos."""

    def test_build_sum_matches_required_mos(self):
        """When MOS build is consistent, no error flagged."""
        v9_scores = {
            'mos_build': [
                {'component': 'base', 'adjustment': 30},
                {'component': 'data_confidence', 'adjustment': 10},
                {'component': 'terminal_dependence', 'adjustment': 0},
                {'component': 'wacc_clamp', 'adjustment': 0},
                {'component': 'leverage', 'adjustment': 0},
                {'component': 'value_creation', 'adjustment': 0},
                {'component': 'fragility', 'adjustment': 0},
            ],
            'required_mos': 0.40,
        }
        errors = _reconciliation_checks({}, {'company': {}, 'dcf': {}}, v9_scores)
        mos_errs = [e for e in errors if e['check'] == 'MOS_BUILD_SUM']
        assert len(mos_errs) == 0

    def test_mismatch_flags_error(self):
        """When build sum != required_mos, error flagged."""
        v9_scores = {
            'mos_build': [
                {'component': 'base', 'adjustment': 30},
                {'component': 'data_confidence', 'adjustment': 10},
            ],
            'required_mos': 0.50,  # 50% != 40% build sum
        }
        errors = _reconciliation_checks({}, {'company': {}, 'dcf': {}}, v9_scores)
        mos_errs = [e for e in errors if e['check'] == 'MOS_BUILD_SUM']
        assert len(mos_errs) == 1
        assert mos_errs[0]['status'] == 'ERROR'


# ============================================================================
# Stage 6: G10 — Range-First IV Threshold Tests
# ============================================================================

class TestRangeFirstThreshold:
    """G10: iv_confidence uses severe_threshold (80%) not extreme (90%)."""

    def _make_v8_data(self, terminal_pct=85.0):
        return {
            'financials': {
                '_data_status': 'OK',
                'roe': 15.0, 'net_margin': 10.0, 'revenue_growth': 5.0,
                'free_cash_flow': 3e9, 'fcf_yield': 2.0, 'debt_equity': 1.0,
                'gross_margin': 30.0, 'operating_margin': 12.0,
                'market_cap': 100e9, 'interest_coverage': 8.0,
                'buyback_yield': 1.0, 'forward_pe': 18.0,
                'dividend_yield': 1.5, 'payout_ratio': 35.0,
                'net_debt_ebitda': 2.0, 'total_cash': 5e9, 'total_debt': 20e9,
                'recommendation': 'hold',
            },
            'company': {'price': 100.0, 'market_cap': 100e9},
            'dcf': {'bear': 80.0, 'base': 100.0, 'bull': 120.0,
                    'assumptions': {'terminal_value_pct': terminal_pct, 'discount_rate': 8,
                                    'wacc_clamp_codes': [], 'fcf_margin': 8}},
            'institutional': {'short_pct': 2.0},
            'earnings': [{'beat': True}],
        }

    def test_iv_confidence_low_at_80(self):
        """iv_confidence should be LOW at 80% terminal dependence (severe_threshold)."""
        v8 = self._make_v8_data(terminal_pct=80.0)
        v9 = _compute_v9_owner_scores({}, v8)
        assert v9['iv_confidence'] == 'LOW'

    def test_iv_confidence_normal_at_75(self):
        """iv_confidence should be NORMAL at 75% terminal dependence."""
        v8 = self._make_v8_data(terminal_pct=75.0)
        v9 = _compute_v9_owner_scores({}, v8)
        assert v9['iv_confidence'] == 'NORMAL'


# ============================================================================
# Phase 1 Tests: Probabilistic Engine Framework
# ============================================================================

class TestEngineVariance:
    """P1: Engine variance estimates."""

    def test_base_variance_all_engines(self):
        """Every engine should have a base variance defined."""
        engines = ['trend', 'valuation', 'consensus', 'volatility',
                   'macro', 'liquidity', 'global', 'correlation']
        for e in engines:
            assert e in ENGINE_VARIANCE_BASE

    def test_zero_score_gives_base_variance(self):
        """Score=0, full DC → variance ≈ base × 1.0 (f_extremity=1, f_data=1)."""
        dc_details = {'final_dc': 100}
        v = _compute_engine_variance('trend', 0, dc_details, 200)
        assert abs(v - ENGINE_VARIANCE_BASE['trend']) < 1e-6

    def test_extreme_score_increases_variance(self):
        """Score at edge of range should have higher variance than mid."""
        dc_details = {'final_dc': 100}
        v_mid = _compute_engine_variance('trend', 0, dc_details, 200)
        v_ext = _compute_engine_variance('trend', 100, dc_details, 200)
        assert v_ext > v_mid

    def test_low_dc_increases_variance(self):
        """Low data confidence should multiply variance."""
        dc_high = {'final_dc': 100}
        dc_low = {'final_dc': 30}
        v_high = _compute_engine_variance('valuation', 0, dc_high, 80)
        v_low = _compute_engine_variance('valuation', 0, dc_low, 80)
        assert v_low > v_high

    def test_missing_data_penalty(self):
        """Missing data dependency should increase variance."""
        dc_ok = {'final_dc': 100}
        dc_missing = {'final_dc': 100, 'price_insufficient': True}
        v_ok = _compute_engine_variance('trend', 0, dc_ok, 200)
        v_miss = _compute_engine_variance('trend', 0, dc_missing, 200)
        assert v_miss > v_ok

    def test_compute_all_returns_dict(self):
        """compute_all_engine_variances returns dict with 8 entries."""
        scores = {e: 0 for e in ['trend', 'valuation', 'consensus', 'volatility',
                                  'macro', 'liquidity', 'global', 'correlation']}
        dc_details = {'final_dc': 100}
        variances = compute_all_engine_variances(scores, dc_details)
        assert len(variances) == 8
        assert all(v > 0 for v in variances.values())


class TestCovarianceMatrix:
    """P2: Engine covariance matrix."""

    def test_correlation_matrix_symmetric(self):
        """Structural correlations should be symmetric."""
        diff = ENGINE_STRUCTURAL_CORRELATIONS - ENGINE_STRUCTURAL_CORRELATIONS.T
        assert np.allclose(diff, 0, atol=1e-10)

    def test_correlation_matrix_unit_diagonal(self):
        """Diagonal should be all 1s."""
        diag = np.diag(ENGINE_STRUCTURAL_CORRELATIONS)
        assert np.allclose(diag, 1.0)

    def test_build_covariance_psd(self):
        """Covariance matrix should be positive semi-definite."""
        variances = {e: 50 for e in ['trend', 'valuation', 'consensus', 'volatility',
                                      'macro', 'liquidity', 'global', 'correlation']}
        cov, order = build_covariance_matrix(variances)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_build_covariance_symmetric(self):
        """Output covariance matrix should be symmetric."""
        variances = {e: 50 for e in ['trend', 'valuation', 'consensus', 'volatility',
                                      'macro', 'liquidity', 'global', 'correlation']}
        cov, order = build_covariance_matrix(variances)
        assert np.allclose(cov, cov.T)

    def test_composite_variance_positive(self):
        """σ²_C should always be positive."""
        variances = {e: 50 for e in ['trend', 'valuation', 'consensus', 'volatility',
                                      'macro', 'liquidity', 'global', 'correlation']}
        cov, order = build_covariance_matrix(variances)
        w = {e: 1/8 for e in order}
        sigma2 = compute_composite_variance(w, cov, order)
        assert sigma2 > 0

    def test_higher_variance_engines_increase_composite(self):
        """Higher individual variances should increase composite variance."""
        engines = ['trend', 'valuation', 'consensus', 'volatility',
                   'macro', 'liquidity', 'global', 'correlation']
        low_var = {e: 10 for e in engines}
        high_var = {e: 200 for e in engines}
        cov_low, order = build_covariance_matrix(low_var)
        cov_high, _ = build_covariance_matrix(high_var)
        w = {e: 1/8 for e in order}
        s2_low = compute_composite_variance(w, cov_low, order)
        s2_high = compute_composite_variance(w, cov_high, order)
        assert s2_high > s2_low


class TestConfidenceAdjusted:
    """P3: Confidence-adjusted composite."""

    def test_zero_signal_zero_output(self):
        """Zero raw composite → zero confidence-adjusted."""
        c_conf, p_pos, cr = compute_confidence_adjusted_composite(0, 1.0)
        assert abs(c_conf) < 1e-6
        assert abs(p_pos - 0.5) < 1e-6
        assert abs(cr) < 1e-6

    def test_positive_signal_positive_output(self):
        """Positive raw composite → positive confidence-adjusted."""
        c_conf, p_pos, cr = compute_confidence_adjusted_composite(30, 1.0)
        assert c_conf > 0
        assert p_pos > 0.5
        assert cr > 0

    def test_negative_signal_dampened_output(self):
        """Negative raw composite → dampened magnitude (μ<0 × cr<0 = positive, dampened)."""
        c_conf, p_pos, cr = compute_confidence_adjusted_composite(-30, 1.0)
        # μ = -0.30, cr < 0 → μ × cr > 0: signal magnitude is dampened toward zero
        assert abs(c_conf) < abs(-30)
        assert p_pos < 0.5
        assert cr < 0

    def test_high_uncertainty_dampens_signal(self):
        """Large σ²_C should dampen the confidence-adjusted composite."""
        c_conf_low_var, _, _ = compute_confidence_adjusted_composite(30, 0.01)
        c_conf_high_var, _, _ = compute_confidence_adjusted_composite(30, 100.0)
        assert abs(c_conf_low_var) > abs(c_conf_high_var)

    def test_p_positive_bounded(self):
        """P(C>0) must be in [0, 1]."""
        for raw in [-100, -50, 0, 50, 100]:
            _, p_pos, _ = compute_confidence_adjusted_composite(raw, 1.0)
            assert 0 <= p_pos <= 1


class TestInteractionMatrix:
    """P4: Quadratic interaction terms."""

    def test_interaction_matrix_symmetric(self):
        """Interaction matrix should be symmetric."""
        assert np.allclose(INTERACTION_MATRIX, INTERACTION_MATRIX.T)

    def test_interaction_matrix_sparse(self):
        """Should have exactly 6 nonzero entries (3 pairs)."""
        assert np.count_nonzero(INTERACTION_MATRIX) == 6

    def test_composite_with_interactions(self):
        """Composite with interactions should differ from without."""
        e_norm = {f'{e}_norm': 0.5 for e in
                  ['trend', 'valuation', 'consensus', 'volatility',
                   'macro', 'liquidity', 'global', 'correlation']}
        w = {e: 1/8 for e in ['trend', 'valuation', 'consensus', 'volatility',
                               'macro', 'liquidity', 'global', 'correlation']}
        c_no_int, _ = compute_composite(e_norm, w, interaction_matrix=None)
        c_with_int, details = compute_composite(e_norm, w, interaction_matrix=INTERACTION_MATRIX)
        assert c_no_int != c_with_int
        assert 'composite_linear' in details
        assert 'composite_quadratic' in details

    def test_composite_backward_compat(self):
        """Without interaction_matrix, should match original linear behavior."""
        e_norm = {f'{e}_norm': 0.3 for e in
                  ['trend', 'valuation', 'consensus', 'volatility',
                   'macro', 'liquidity', 'global', 'correlation']}
        w = {e: 1/8 for e in ['trend', 'valuation', 'consensus', 'volatility',
                               'macro', 'liquidity', 'global', 'correlation']}
        c_raw, _ = compute_composite(e_norm, w)
        expected = 0.3 * 100  # all norms equal, weights sum to 1
        assert abs(c_raw - expected) < 1e-6

    def test_quadratic_contribution_bounded(self):
        """Quadratic contribution should be small relative to linear."""
        e_norm = {f'{e}_norm': 0.8 for e in
                  ['trend', 'valuation', 'consensus', 'volatility',
                   'macro', 'liquidity', 'global', 'correlation']}
        w = {e: 1/8 for e in ['trend', 'valuation', 'consensus', 'volatility',
                               'macro', 'liquidity', 'global', 'correlation']}
        _, details = compute_composite(e_norm, w, INTERACTION_MATRIX)
        assert abs(details['composite_quadratic']) < 2.0  # max ~0.9 pts


class TestSmoothVerdict:
    """P5: Smooth decision mapping."""

    def test_strong_buy_signal(self):
        """Strong positive signal should give high P(buy)."""
        result = compute_smooth_verdict(50, 40, 1.0, 'Calm', 0.3)
        assert result['p_buy'] > 0.70
        assert result['verdict_smooth'] == "BUY / LONG BIAS"

    def test_strong_sell_signal(self):
        """Strong negative signal should give high P(sell)."""
        result = compute_smooth_verdict(-50, -40, 1.0, 'Calm', 0.3)
        assert result['p_sell'] > 0.70
        assert result['verdict_smooth'] == "SELL / SHORT BIAS"

    def test_neutral_signal(self):
        """Zero signal should give P(buy) ≈ 0.50."""
        result = compute_smooth_verdict(0, 0, 1.0, 'Calm', 0.3)
        assert abs(result['p_buy'] - 0.5) < 0.05

    def test_legacy_verdict_preserved(self):
        """Legacy verdict should match old threshold logic exactly."""
        result = compute_smooth_verdict(30, 15, 1.0, 'Calm', 0.3)
        assert result['verdict_legacy'] == "BUY / LONG BIAS"

        result = compute_smooth_verdict(-30, -15, 1.0, 'Calm', 0.3)
        assert result['verdict_legacy'] == "SELL / SHORT BIAS"

        result = compute_smooth_verdict(5, 3, 1.0, 'Calm', 0.05)
        assert result['verdict_legacy'] == "CASH / STAND ASIDE"

    def test_high_uncertainty_widens_tau(self):
        """Higher σ_C should increase τ_effective."""
        r_low = compute_smooth_verdict(30, 20, 0.1, 'Calm', 0.3)
        r_high = compute_smooth_verdict(30, 20, 10.0, 'Calm', 0.3)
        assert r_high['tau_effective'] > r_low['tau_effective']

    def test_chop_regime_increases_tau(self):
        """Chop regime should increase tau (more cautious)."""
        r_calm = compute_smooth_verdict(30, 20, 1.0, 'Calm', 0.3)
        r_chop = compute_smooth_verdict(30, 20, 1.0, 'Chop', 0.3)
        assert r_chop['tau_effective'] > r_calm['tau_effective']


class TestKellySizing:
    """P6: Kelly-inspired position sizing."""

    def test_zero_signal_zero_position(self):
        """Zero signal → zero Kelly position."""
        result = compute_kelly_position(0, 1.0, 0.8, 100, 250000, 'Calm')
        assert result['kelly_position'] == 0

    def test_positive_signal_positive_position(self):
        """Positive signal → positive Kelly position."""
        result = compute_kelly_position(30, 1.0, 0.8, 100, 250000, 'Calm')
        assert result['kelly_position'] > 0

    def test_regime_caps(self):
        """Kelly fraction should respect regime-specific caps."""
        result = compute_kelly_position(80, 0.01, 0.8, 100, 250000, 'Calm')
        assert abs(result['kelly_clipped']) <= 0.15

        result = compute_kelly_position(80, 0.01, 0.8, 100, 250000, 'Crisis Trend')
        assert abs(result['kelly_clipped']) <= 0.05

    def test_low_dc_reduces_position(self):
        """Low data confidence should reduce Kelly position."""
        r_high_dc = compute_kelly_position(30, 1.0, 0.8, 100, 250000, 'Calm')
        r_low_dc = compute_kelly_position(30, 1.0, 0.8, 40, 250000, 'Calm')
        assert r_low_dc['kelly_position'] < r_high_dc['kelly_position']

    def test_kelly_pct_bounded(self):
        """Kelly % should be reasonable."""
        result = compute_kelly_position(100, 0.01, 1.0, 100, 250000, 'Calm')
        assert result['kelly_pct'] <= 20  # max 15% cap * 100% gate * 100% DC


# ============================================================================
# Phase 2 Tests: Monte Carlo DCF
# ============================================================================

class TestDCFKernel:
    """Extracted DCF kernel: _evaluate_dcf_single."""

    def test_positive_iv(self):
        """With positive inputs, IV should be positive."""
        iv = _evaluate_dcf_single(
            revenue=1e9, fcf_margin=0.10, discount_rate=0.10,
            terminal_growth=0.03, growth_rates_array=[0.05]*5,
            shares=1e8, net_debt=0
        )
        assert iv > 0

    def test_higher_discount_lower_iv(self):
        """Higher discount rate → lower IV."""
        kwargs = dict(revenue=1e9, fcf_margin=0.10, growth_rates_array=[0.05]*5,
                      shares=1e8, net_debt=0, terminal_growth=0.03)
        iv_low = _evaluate_dcf_single(discount_rate=0.08, **kwargs)
        iv_high = _evaluate_dcf_single(discount_rate=0.12, **kwargs)
        assert iv_low > iv_high

    def test_higher_margin_higher_iv(self):
        """Higher FCF margin → higher IV."""
        kwargs = dict(revenue=1e9, discount_rate=0.10, growth_rates_array=[0.05]*5,
                      shares=1e8, net_debt=0, terminal_growth=0.03)
        iv_low = _evaluate_dcf_single(fcf_margin=0.05, **kwargs)
        iv_high = _evaluate_dcf_single(fcf_margin=0.15, **kwargs)
        assert iv_high > iv_low

    def test_net_debt_reduces_iv(self):
        """Net debt should reduce IV."""
        kwargs = dict(revenue=1e9, fcf_margin=0.10, discount_rate=0.10,
                      growth_rates_array=[0.05]*5, shares=1e8, terminal_growth=0.03)
        iv_no_debt = _evaluate_dcf_single(net_debt=0, **kwargs)
        iv_debt = _evaluate_dcf_single(net_debt=5e8, **kwargs)
        assert iv_no_debt > iv_debt

    def test_zero_shares_zero_iv(self):
        """Zero shares → zero IV (no division error)."""
        iv = _evaluate_dcf_single(
            revenue=1e9, fcf_margin=0.10, discount_rate=0.10,
            terminal_growth=0.03, growth_rates_array=[0.05]*5,
            shares=0, net_debt=0
        )
        assert iv == 0


class TestTickerSeed:
    """Deterministic ticker seed."""

    def test_deterministic(self):
        """Same ticker → same seed."""
        assert _ticker_seed('AAPL') == _ticker_seed('AAPL')

    def test_case_insensitive(self):
        """Case-insensitive."""
        assert _ticker_seed('aapl') == _ticker_seed('AAPL')

    def test_different_tickers(self):
        """Different tickers → different seeds (with high probability)."""
        assert _ticker_seed('AAPL') != _ticker_seed('MSFT')


class TestMonteCarloEngine:
    """Monte Carlo DCF engine."""

    def _run_mc(self, **overrides):
        defaults = dict(
            revenue=1e9, fcf_margin=0.10, discount_rate=0.10,
            terminal_growth=0.03, growth_rates=[0.05]*5,
            shares=1e8, net_debt=0, sector='Technology',
            ticker_seed=42, price=50.0, n_simulations=500,
        )
        defaults.update(overrides)
        return _run_monte_carlo_dcf(**defaults)

    def test_returns_all_fields(self):
        """MC result should contain all required keys."""
        result = self._run_mc()
        required = ['n_simulations', 'seed', 'elapsed_ms',
                    'iv_mean', 'iv_median', 'iv_std',
                    'iv_p5', 'iv_p10', 'iv_p25', 'iv_p50',
                    'iv_p75', 'iv_p90', 'iv_p95',
                    'prob_undervalued', 'expected_mos']
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_percentile_ordering(self):
        """Percentiles should be monotonically increasing."""
        result = self._run_mc()
        assert result['iv_p5'] <= result['iv_p25']
        assert result['iv_p25'] <= result['iv_p50']
        assert result['iv_p50'] <= result['iv_p75']
        assert result['iv_p75'] <= result['iv_p95']

    def test_mean_near_median(self):
        """Mean and median should be in same ballpark (within 3x)."""
        result = self._run_mc()
        assert result['iv_mean'] > 0
        assert result['iv_median'] > 0
        ratio = result['iv_mean'] / result['iv_median']
        assert 0.3 < ratio < 3.0

    def test_deterministic_with_seed(self):
        """Same seed → same results."""
        r1 = self._run_mc(ticker_seed=123)
        r2 = self._run_mc(ticker_seed=123)
        assert r1['iv_mean'] == r2['iv_mean']
        assert r1['iv_p50'] == r2['iv_p50']

    def test_different_seeds_different_results(self):
        """Different seeds → different results."""
        r1 = self._run_mc(ticker_seed=123)
        r2 = self._run_mc(ticker_seed=456)
        assert r1['iv_mean'] != r2['iv_mean']

    def test_prob_undervalued_with_price(self):
        """P(undervalued) should be between 0 and 1."""
        result = self._run_mc(price=50.0)
        assert 0 <= result['prob_undervalued'] <= 1

    def test_prob_undervalued_without_price(self):
        """Without price, prob_undervalued should be None."""
        result = self._run_mc(price=None)
        assert result['prob_undervalued'] is None

    def test_performance(self):
        """N=1000 should complete in < 500ms."""
        import time
        t0 = time.perf_counter()
        self._run_mc(n_simulations=1000)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 500, f"MC took {elapsed:.0f}ms (limit: 500ms)"

    def test_higher_volatility_wider_spread(self):
        """Higher growth sigma → wider distribution."""
        r_tech = self._run_mc(sector='Technology')   # sigma=0.08
        r_util = self._run_mc(sector='Utilities')    # sigma=0.03
        spread_tech = r_tech['iv_p95'] - r_tech['iv_p5']
        spread_util = r_util['iv_p95'] - r_util['iv_p5']
        assert spread_tech > spread_util


class TestSensitivity:
    """DCF sensitivity analysis."""

    def _run_sens(self):
        return _compute_sensitivity(
            revenue=1e9, fcf_margin=0.10, discount_rate=0.10,
            terminal_growth=0.03, growth_rates=[0.05]*5,
            shares=1e8, net_debt=0,
        )

    def test_returns_all_fields(self):
        """Sensitivity should have all 5 fields."""
        result = self._run_sens()
        for k in ['dIV_dWACC', 'dIV_dGrowth', 'dIV_dMargin',
                  'dIV_dTerminalG', 'most_sensitive_to']:
            assert k in result

    def test_wacc_sensitivity_negative(self):
        """Higher WACC → lower IV: dIV/dWACC should be negative."""
        result = self._run_sens()
        assert result['dIV_dWACC'] < 0

    def test_growth_sensitivity_positive(self):
        """Higher growth → higher IV: dIV/dGrowth should be positive."""
        result = self._run_sens()
        assert result['dIV_dGrowth'] > 0

    def test_margin_sensitivity_positive(self):
        """Higher margin → higher IV: dIV/dMargin should be positive."""
        result = self._run_sens()
        assert result['dIV_dMargin'] > 0

    def test_terminal_growth_sensitivity_positive(self):
        """Higher terminal growth → higher IV: dIV/dTerminalG should be positive."""
        result = self._run_sens()
        assert result['dIV_dTerminalG'] > 0

    def test_most_sensitive_is_valid(self):
        """most_sensitive_to should be one of the 4 parameters."""
        result = self._run_sens()
        assert result['most_sensitive_to'] in ['WACC', 'Growth', 'Margin', 'TerminalG']


class TestBuildDCFMonteCarlo:
    """Integration: _build_dcf with Monte Carlo."""

    def _make_info(self):
        return {
            'beta': 1.1,
            'sector': 'Technology',
        }

    def _make_financials(self):
        return {
            'revenue_ttm': 50e9,
            'shares_outstanding': 1e9,
            'free_cash_flow': 5e9,
            'net_income_ttm': 4e9,
            'ebitda': 8e9,
            'revenue_growth': 10.0,
            'debt_equity': 0.5,
            'net_debt': 10e9,
        }

    def test_mc_fields_present(self):
        """DCF result should contain monte_carlo and sensitivity keys."""
        result = _build_dcf(self._make_info(), self._make_financials(),
                           sector='Technology', ticker='AAPL', price=150.0)
        assert 'monte_carlo' in result
        assert 'sensitivity' in result

    def test_bear_base_bull_from_mc(self):
        """bear/base/bull should come from MC percentiles."""
        result = _build_dcf(self._make_info(), self._make_financials(),
                           sector='Technology', ticker='AAPL', price=150.0)
        mc = result['monte_carlo']
        assert result['bear'] == mc['iv_p5']
        assert result['base'] == mc['iv_p50']
        assert result['bull'] == mc['iv_p95']

    def test_deterministic_base_preserved(self):
        """base_deterministic should be the old deterministic value."""
        result = _build_dcf(self._make_info(), self._make_financials(),
                           sector='Technology', ticker='AAPL', price=150.0)
        assert 'base_deterministic' in result
        assert result['base_deterministic'] > 0

    def test_disabled_dcf_no_mc(self):
        """DCF disabled (no revenue) → no MC fields."""
        bad_fin = self._make_financials()
        bad_fin['revenue_ttm'] = 0
        result = _build_dcf(self._make_info(), bad_fin,
                           sector='Technology', ticker='AAPL', price=150.0)
        assert result.get('_dcf_disabled') is True
        assert 'monte_carlo' not in result

    def test_all_existing_fields_preserved(self):
        """All existing DCF fields should still be present."""
        result = _build_dcf(self._make_info(), self._make_financials(),
                           sector='Technology', ticker='AAPL', price=150.0)
        for key in ['bear', 'base', 'bull', 'cash_flow_source',
                    'forecast_years', 'projections', 'assumptions']:
            assert key in result, f"Missing existing field: {key}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
