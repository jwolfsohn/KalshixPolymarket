"""Strategy 2: Bundle arbitrage (intra-platform).

YES + NO on the same platform don't sum to $1.00.
Buy both when sum < $1.00, sell both when sum > $1.00 (requires position).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.arb.models import Market, Opportunity, OrderLeg

logger = logging.getLogger(__name__)


def detect_bundle(
    markets: list[Market],
    kalshi_fees: KalshiFeeCalculator,
    poly_fees: PolymarketFeeCalculator,
    min_profit_after_fees_cents: float = 0.5,
) -> list[Opportunity]:
    """Scan for YES + NO mispricing on a single platform."""
    opportunities: list[Opportunity] = []

    for market in markets:
        by_side = {o.side: o for o in market.outcomes}
        if "yes" not in by_side or "no" not in by_side:
            continue

        yes_ask = by_side["yes"].best_ask
        no_ask = by_side["no"].best_ask

        if yes_ask <= 0 or no_ask <= 0:
            continue

        total_cost = yes_ask + no_ask
        gross_profit = 1.0 - total_cost

        if gross_profit <= 0:
            continue

        # Compute fees
        if market.platform == "kalshi":
            yes_fee = kalshi_fees.estimate(yes_ask, 1, is_taker=True)
            no_fee = kalshi_fees.estimate(no_ask, 1, is_taker=True)
        else:
            yes_fee = poly_fees.estimate(
                yes_ask, 1, category=market.category, is_taker=True
            )
            no_fee = poly_fees.estimate(
                no_ask, 1, category=market.category, is_taker=True
            )

        total_fees = yes_fee.fee_per_contract + no_fee.fee_per_contract
        net_profit = gross_profit - total_fees

        if net_profit * 100 < min_profit_after_fees_cents:
            continue

        max_contracts = min(
            by_side["yes"].ask_depth if by_side["yes"].ask_depth > 0 else 1000,
            by_side["no"].ask_depth if by_side["no"].ask_depth > 0 else 1000,
        )

        legs = [
            OrderLeg(
                platform=market.platform,
                market_id=market.market_id,
                outcome_id=by_side["yes"].outcome_id,
                side="buy",
                price=yes_ask,
                size=1,
            ),
            OrderLeg(
                platform=market.platform,
                market_id=market.market_id,
                outcome_id=by_side["no"].outcome_id,
                side="buy",
                price=no_ask,
                size=1,
            ),
        ]

        days = 30
        if market.settlement_date:
            days = max((market.settlement_date - datetime.now(timezone.utc)).days, 1)

        capital = total_cost
        annualized = (net_profit / capital) * (365 / days) if capital > 0 else 0

        opportunities.append(
            Opportunity(
                strategy="bundle",
                legs=legs,
                fees=[yes_fee, no_fee],
                gross_profit_per_contract=gross_profit,
                total_fees_per_contract=total_fees,
                net_profit_per_contract=net_profit,
                max_contracts=max_contracts,
                total_net_profit=net_profit * max_contracts,
                matched_market=None,
                settlement_date=market.settlement_date,
                days_to_settlement=days,
                annualized_return=annualized,
                description=(
                    f"{market.title} [{market.platform}] — "
                    f"YES({yes_ask:.2f}) + NO({no_ask:.2f}) = {total_cost:.2f}"
                ),
            )
        )

    return opportunities
