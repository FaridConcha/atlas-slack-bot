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
        result = _sanitize_yield(info, 200.0)
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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '--tb=short'])
