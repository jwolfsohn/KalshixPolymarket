"""Tests for fee calculators against known platform examples."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator


class TestKalshiFees:
    def setup_method(self):
        self.calc = KalshiFeeCalculator()

    def test_taker_fee_at_fifty_cents(self):
        # P=0.50: 0.07 * 0.5 * 0.5 = 0.0175, ceil to 0.02
        fee = self.calc.fee_per_contract(0.50, is_taker=True)
        assert fee == 0.02

    def test_taker_fee_capped_at_two_cents(self):
        # P=0.50 is the maximum fee point; should never exceed $0.02
        fee = self.calc.fee_per_contract(0.50, is_taker=True)
        assert fee <= 0.02

    def test_taker_fee_at_extreme_price(self):
        # P=0.95: 0.07 * 0.95 * 0.05 = 0.003325, ceil = 0.01
        fee = self.calc.fee_per_contract(0.95, is_taker=True)
        assert fee == 0.01

    def test_taker_fee_at_very_extreme_price(self):
        # P=0.99: 0.07 * 0.99 * 0.01 = 0.000693, ceil = 0.01
        fee = self.calc.fee_per_contract(0.99, is_taker=True)
        assert fee == 0.01

    def test_taker_fee_at_ten_cents(self):
        # P=0.10: 0.07 * 0.10 * 0.90 = 0.0063, ceil = 0.01
        fee = self.calc.fee_per_contract(0.10, is_taker=True)
        assert fee == 0.01

    def test_maker_fee_at_fifty_cents(self):
        # P=0.50: 0.0175 * 0.5 * 0.5 = 0.004375, ceil = 0.01
        fee = self.calc.fee_per_contract(0.50, is_taker=False)
        assert fee == 0.01

    def test_maker_fee_at_extreme(self):
        # P=0.95: 0.0175 * 0.95 * 0.05 = 0.00083125, ceil = 0.01
        fee = self.calc.fee_per_contract(0.95, is_taker=False)
        assert fee == 0.01

    def test_total_fee_multiple_contracts(self):
        fee = self.calc.total_fee(0.50, contracts=100, is_taker=True)
        assert fee == 2.00  # 0.02 * 100

    def test_fee_at_zero_price(self):
        fee = self.calc.fee_per_contract(0.0, is_taker=True)
        assert fee == 0.0

    def test_fee_at_one_dollar(self):
        fee = self.calc.fee_per_contract(1.0, is_taker=True)
        assert fee == 0.0

    def test_fee_symmetry(self):
        # Fee at P should equal fee at 1-P (bell curve property)
        fee_30 = self.calc.fee_per_contract(0.30, is_taker=True)
        fee_70 = self.calc.fee_per_contract(0.70, is_taker=True)
        assert fee_30 == fee_70


class TestPolymarketFees:
    def setup_method(self):
        self.calc = PolymarketFeeCalculator()

    def test_maker_always_zero(self):
        for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
            fee = self.calc.fee_per_contract(p, category="crypto", is_taker=False)
            assert fee == 0.0

    def test_crypto_at_fifty_cents(self):
        # rate=0.036, P=0.50: 0.036 * 0.5 * 0.5 * 2 = 0.018 -> ceil = 0.02
        fee = self.calc.fee_per_contract(0.50, category="crypto")
        assert fee == 0.02

    def test_sports_at_fifty_cents(self):
        # rate=0.015, P=0.50: 0.015 * 0.5 * 0.5 * 2 = 0.0075 -> ceil = 0.01
        fee = self.calc.fee_per_contract(0.50, category="sports")
        assert fee == 0.01

    def test_geopolitics_always_zero(self):
        fee = self.calc.fee_per_contract(0.50, category="geopolitics")
        assert fee == 0.0

    def test_fee_lower_at_extremes(self):
        fee_50 = self.calc.fee_per_contract(0.50, category="politics")
        fee_90 = self.calc.fee_per_contract(0.90, category="politics")
        # fee at 0.90 should be less than or equal to fee at 0.50
        assert fee_90 <= fee_50

    def test_fee_at_ninety_cents_crypto(self):
        # rate=0.036, P=0.90: 0.036 * 0.9 * 0.1 * 2 = 0.00648 -> ceil = 0.01
        fee = self.calc.fee_per_contract(0.90, category="crypto")
        assert fee == 0.01

    def test_total_fee(self):
        per = self.calc.fee_per_contract(0.50, category="politics")
        total = self.calc.total_fee(0.50, contracts=50, category="politics")
        assert total == per * 50

    def test_unknown_category_uses_default(self):
        # Unknown category should use conservative default (0.030)
        fee = self.calc.fee_per_contract(0.50, category="unknown_category")
        fee_default = self.calc.fee_per_contract(0.50, category="economics")
        assert fee == fee_default  # both use rate 0.030

    def test_fee_symmetry(self):
        fee_30 = self.calc.fee_per_contract(0.30, category="politics")
        fee_70 = self.calc.fee_per_contract(0.70, category="politics")
        assert fee_30 == fee_70


class TestCrossplatformFeeInteraction:
    """Test combined fee calculations for arbitrage scenarios."""

    def setup_method(self):
        self.kalshi = KalshiFeeCalculator()
        self.poly = PolymarketFeeCalculator()

    def test_combined_fees_at_fifty_fifty(self):
        """At P=0.50 on both sides, combined fees are highest."""
        k_fee = self.kalshi.fee_per_contract(0.50, is_taker=True)
        p_fee = self.poly.fee_per_contract(0.50, category="politics", is_taker=True)
        combined = k_fee + p_fee
        # A 1c gross spread minus 3c in fees is unprofitable
        assert combined > 0.01

    def test_combined_fees_at_extreme(self):
        """At extreme prices, combined fees are much lower."""
        k_fee = self.kalshi.fee_per_contract(0.95, is_taker=True)
        p_fee = self.poly.fee_per_contract(0.05, category="politics", is_taker=True)
        combined = k_fee + p_fee
        # Should be much less than at 50/50
        combined_50 = (
            self.kalshi.fee_per_contract(0.50, is_taker=True)
            + self.poly.fee_per_contract(0.50, category="politics", is_taker=True)
        )
        assert combined <= combined_50

    def test_profitable_spread_example(self):
        """Verify a 5% spread at extreme prices is profitable after fees."""
        # YES on Kalshi at 93c, NO on Polymarket at 2c
        # Total cost: 0.95, gross profit: 0.05
        k_fee = self.kalshi.fee_per_contract(0.93, is_taker=True)
        p_fee = self.poly.fee_per_contract(0.02, category="politics", is_taker=True)
        gross = 1.00 - 0.93 - 0.02
        net = gross - k_fee - p_fee
        assert net > 0, f"5% spread should be profitable, got net={net}"
