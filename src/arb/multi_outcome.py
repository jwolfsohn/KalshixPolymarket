"""Strategy 3: Multi-outcome arbitrage.

For markets with N outcomes (e.g., "Who will win?"), the sum of all
outcome YES prices should equal $1.00. When it doesn't, arbitrage exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.arb.models import Market, Opportunity, OrderLeg

logger = logging.getLogger(__name__)


def detect_multi_outcome(
    markets: list[Market],
    kalshi_fees: KalshiFeeCalculator,
    poly_fees: PolymarketFeeCalculator,
    min_profit_after_fees_cents: float = 0.5,
) -> list[Opportunity]:
    """Scan multi-outcome markets where sum of asks != $1.00."""
    opportunities: list[Opportunity] = []

    for market in markets:
        if market.outcome_count <= 2:
            continue

        outcomes_with_prices = [o for o in market.outcomes if o.best_ask > 0]
        if len(outcomes_with_prices) < market.outcome_count:
            continue

        # Sum of all YES asks
        total_ask = sum(o.best_ask for o in outcomes_with_prices)
        gross_profit = 1.0 - total_ask

        if gross_profit <= 0:
            continue

        # Compute fees per outcome
        fees = []
        total_fees_per_contract = 0.0
        for o in outcomes_with_prices:
            if market.platform == "kalshi":
                fee = kalshi_fees.estimate(o.best_ask, 1, is_taker=True)
            else:
                fee = poly_fees.estimate(
                    o.best_ask, 1, category=market.category, is_taker=True
                )
            fees.append(fee)
            total_fees_per_contract += fee.fee_per_contract

        net_profit = gross_profit - total_fees_per_contract

        if net_profit * 100 < min_profit_after_fees_cents:
            continue

        max_contracts = min(
            (o.ask_depth if o.ask_depth > 0 else 1000)
            for o in outcomes_with_prices
        )

        legs = [
            OrderLeg(
                platform=market.platform,
                market_id=market.market_id,
                outcome_id=o.outcome_id,
                side="buy",
                price=o.best_ask,
                size=1,
            )
            for o in outcomes_with_prices
        ]

        days = 30
        if market.settlement_date:
            days = max((market.settlement_date - datetime.now(timezone.utc)).days, 1)

        capital = total_ask
        annualized = (net_profit / capital) * (365 / days) if capital > 0 else 0

        outcome_prices = ", ".join(
            f"{o.label}={o.best_ask:.2f}" for o in outcomes_with_prices
        )
        opportunities.append(
            Opportunity(
                strategy="multi_outcome",
                legs=legs,
                fees=fees,
                gross_profit_per_contract=gross_profit,
                total_fees_per_contract=total_fees_per_contract,
                net_profit_per_contract=net_profit,
                max_contracts=max_contracts,
                total_net_profit=net_profit * max_contracts,
                matched_market=None,
                settlement_date=market.settlement_date,
                days_to_settlement=days,
                annualized_return=annualized,
                description=(
                    f"{market.title} [{market.platform}] — "
                    f"Sum={total_ask:.2f} ({outcome_prices})"
                ),
            )
        )

    return opportunities
