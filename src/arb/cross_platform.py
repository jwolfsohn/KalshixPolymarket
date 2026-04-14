"""Strategy 1: Cross-platform arbitrage.

Buy YES on one platform + NO on the other when the combined cost < $1.00.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.arb.models import FeeEstimate, MatchedMarket, Opportunity, OrderLeg

logger = logging.getLogger(__name__)


def detect_cross_platform(
    matched: list[MatchedMarket],
    kalshi_fees: KalshiFeeCalculator,
    poly_fees: PolymarketFeeCalculator,
    min_edge_pct: float = 0.02,
    min_profit_after_fees_cents: float = 0.5,
    max_days_to_settlement: int = 90,
) -> list[Opportunity]:
    """Scan matched markets for cross-platform arbitrage.

    For each matched pair, checks both directions:
    - Buy YES on Kalshi + Buy NO on Polymarket
    - Buy YES on Polymarket + Buy NO on Kalshi
    """
    opportunities: list[Opportunity] = []

    for match in matched:
        k = match.kalshi
        p = match.polymarket

        # Skip if settlement risk is too high
        if match.settlement_risk == "different_criteria":
            continue

        # Skip if settlement is too far out
        if match.kalshi.settlement_date:
            days = (match.kalshi.settlement_date - datetime.now(timezone.utc)).days
            if days > max_days_to_settlement:
                continue
        else:
            days = 30  # default assumption

        # Get YES/NO outcomes from each platform
        k_by_side = {o.side: o for o in k.outcomes}
        p_by_side = {o.side: o for o in p.outcomes}

        if "yes" not in k_by_side or "no" not in k_by_side:
            continue
        if "yes" not in p_by_side or "no" not in p_by_side:
            continue

        # Direction A: YES on Kalshi + NO on Polymarket
        opp_a = _check_direction(
            yes_platform="kalshi",
            no_platform="polymarket",
            yes_outcome=k_by_side["yes"],
            no_outcome=p_by_side["no"],
            kalshi_fees=kalshi_fees,
            poly_fees=poly_fees,
            poly_category=p.category,
            match=match,
            days_to_settlement=days,
            min_edge_pct=min_edge_pct,
            min_profit_cents=min_profit_after_fees_cents,
        )
        if opp_a:
            opportunities.append(opp_a)

        # Direction B: YES on Polymarket + NO on Kalshi
        opp_b = _check_direction(
            yes_platform="polymarket",
            no_platform="kalshi",
            yes_outcome=p_by_side["yes"],
            no_outcome=k_by_side["no"],
            kalshi_fees=kalshi_fees,
            poly_fees=poly_fees,
            poly_category=p.category,
            match=match,
            days_to_settlement=days,
            min_edge_pct=min_edge_pct,
            min_profit_cents=min_profit_after_fees_cents,
        )
        if opp_b:
            opportunities.append(opp_b)

    return opportunities


def _check_direction(
    yes_platform: str,
    no_platform: str,
    yes_outcome,
    no_outcome,
    kalshi_fees: KalshiFeeCalculator,
    poly_fees: PolymarketFeeCalculator,
    poly_category: str,
    match: MatchedMarket,
    days_to_settlement: int,
    min_edge_pct: float,
    min_profit_cents: float,
) -> Opportunity | None:
    """Check one direction of cross-platform arb."""
    yes_price = yes_outcome.best_ask
    no_price = no_outcome.best_ask

    if yes_price <= 0 or no_price <= 0:
        return None

    total_cost = yes_price + no_price
    gross_profit = 1.0 - total_cost

    if gross_profit <= 0:
        return None

    gross_pct = gross_profit / total_cost
    if gross_pct < min_edge_pct:
        return None

    # Compute fees for each leg
    if yes_platform == "kalshi":
        yes_fee = kalshi_fees.estimate(yes_price, 1, is_taker=True)
        no_fee = poly_fees.estimate(no_price, 1, category=poly_category, is_taker=True)
    else:
        yes_fee = poly_fees.estimate(yes_price, 1, category=poly_category, is_taker=True)
        no_fee = kalshi_fees.estimate(no_price, 1, is_taker=True)

    total_fees = yes_fee.fee_per_contract + no_fee.fee_per_contract
    net_profit = gross_profit - total_fees

    if net_profit * 100 < min_profit_cents:
        return None

    # Max contracts limited by order book depth
    max_contracts = min(
        yes_outcome.ask_depth if yes_outcome.ask_depth > 0 else 1000,
        no_outcome.ask_depth if no_outcome.ask_depth > 0 else 1000,
    )

    # Annualized return
    capital = total_cost
    days = max(days_to_settlement, 1)
    annualized = (net_profit / capital) * (365 / days) if capital > 0 else 0

    legs = [
        OrderLeg(
            platform=yes_platform,
            market_id=yes_outcome.market_id,
            outcome_id=yes_outcome.outcome_id,
            side="buy",
            price=yes_price,
            size=1,
        ),
        OrderLeg(
            platform=no_platform,
            market_id=no_outcome.market_id,
            outcome_id=no_outcome.outcome_id,
            side="buy",
            price=no_price,
            size=1,
        ),
    ]

    return Opportunity(
        strategy="cross_platform",
        legs=legs,
        fees=[yes_fee, no_fee],
        gross_profit_per_contract=gross_profit,
        total_fees_per_contract=total_fees,
        net_profit_per_contract=net_profit,
        max_contracts=max_contracts,
        total_net_profit=net_profit * max_contracts,
        matched_market=match,
        settlement_date=match.kalshi.settlement_date,
        days_to_settlement=days,
        annualized_return=annualized,
        description=(
            f"{match.kalshi.title} — "
            f"YES@{yes_platform}({yes_price:.2f}) + NO@{no_platform}({no_price:.2f})"
        ),
    )
