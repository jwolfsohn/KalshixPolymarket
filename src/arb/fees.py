"""Fee calculators for Kalshi and Polymarket.

Accuracy here is the #1 determinant of profitability. Fees are always
rounded conservatively (up for costs, down for rebates).
"""

from __future__ import annotations

import math

from src.arb.models import FeeEstimate


class KalshiFeeCalculator:
    """
    Kalshi fee model (as of 2026):
    - Taker: roundup(0.07 * P * (1-P)) per contract, capped at $0.02
    - Maker: roundup(0.0175 * P * (1-P)) per contract
    - No settlement fees, no withdrawal fees (ACH)

    P is the contract price in [0, 1].
    """

    TAKER_RATE = 0.07
    MAKER_RATE = 0.0175
    MAX_FEE_PER_CONTRACT = 0.02

    def fee_per_contract(self, price: float, is_taker: bool = True) -> float:
        """Compute fee per contract in dollars.

        Uses ceil to round up to the nearest cent (conservative).
        """
        rate = self.TAKER_RATE if is_taker else self.MAKER_RATE
        raw = rate * price * (1.0 - price)
        # Round up to nearest cent
        per_contract = math.ceil(raw * 100) / 100
        if is_taker:
            per_contract = min(per_contract, self.MAX_FEE_PER_CONTRACT)
        return per_contract

    def total_fee(self, price: float, contracts: int, is_taker: bool = True) -> float:
        return self.fee_per_contract(price, is_taker) * contracts

    def estimate(
        self, price: float, contracts: int, is_taker: bool = True
    ) -> FeeEstimate:
        per = self.fee_per_contract(price, is_taker)
        return FeeEstimate(
            platform="kalshi",
            fee_per_contract=per,
            total_fee=per * contracts,
            fee_type="taker" if is_taker else "maker",
        )


class PolymarketFeeCalculator:
    """
    Polymarket fee model (as of 2026):
    - Makers: 0% fees (+ daily USDC rebates, not modeled here)
    - Takers: category-dependent, bell-curve shaped
      fee_per_contract = rate * P * (1 - P) * 2
      where rate varies by category (max at P=0.50, near-zero at extremes)

    The *2 factor normalizes so that the rate represents the fee at P=0.50.
    At P=0.50: fee = rate * 0.25 * 2 = rate * 0.5
    """

    # Category -> effective taker rate. These produce the documented category
    # percentages at P=0.50.  E.g. crypto: 0.036 * 0.50 * 0.50 * 2 = 0.018 = 1.8%
    CATEGORY_RATES: dict[str, float] = {
        "crypto": 0.036,
        "economics": 0.030,
        "mentions": 0.0312,
        "culture": 0.025,
        "weather": 0.025,
        "finance": 0.020,
        "politics": 0.020,
        "tech": 0.020,
        "sports": 0.015,
        "geopolitics": 0.0,
    }
    DEFAULT_RATE = 0.030  # conservative default

    def _rate_for_category(self, category: str) -> float:
        return self.CATEGORY_RATES.get(category.lower(), self.DEFAULT_RATE)

    def fee_per_contract(
        self, price: float, category: str = "", is_taker: bool = True
    ) -> float:
        """Compute fee per contract in dollars (contract pays $1 on win)."""
        if not is_taker:
            return 0.0
        rate = self._rate_for_category(category)
        # Bell curve: maximal at P=0.50, zero at P=0 and P=1
        raw = rate * price * (1.0 - price) * 2
        # Round up to nearest cent
        return math.ceil(raw * 100) / 100

    def total_fee(
        self, price: float, contracts: int, category: str = "", is_taker: bool = True
    ) -> float:
        return self.fee_per_contract(price, category, is_taker) * contracts

    def estimate(
        self,
        price: float,
        contracts: int,
        category: str = "",
        is_taker: bool = True,
    ) -> FeeEstimate:
        per = self.fee_per_contract(price, category, is_taker)
        return FeeEstimate(
            platform="polymarket",
            fee_per_contract=per,
            total_fee=per * contracts,
            fee_type="taker" if is_taker else "maker",
        )


# Module-level singletons
kalshi_fees = KalshiFeeCalculator()
polymarket_fees = PolymarketFeeCalculator()
