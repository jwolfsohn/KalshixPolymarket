"""Smart money copy-trade detector — generate signals from tracked insider wallets."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.arb.fees import PolymarketFeeCalculator
from src.arb.models import FeeEstimate, Opportunity, OrderLeg
from src.signals.smart_money.polygon_client import PolygonClient
from src.signals.smart_money.tracker import refresh_tracked_wallets
from src.signals.smart_money.wallet_db import WalletDB

logger = logging.getLogger(__name__)


def detect_smart_money(
    wallet_db: WalletDB,
    poly_client: PolygonClient,
    poly_fees: PolymarketFeeCalculator,
    min_wallet_score: float = 0.7,
    min_position_usd: float = 500.0,
    max_market_price: float = 0.70,
    min_market_price: float = 0.05,
    lookback_hours: int = 24,
    excluded_categories: list[str] | None = None,
    wallet_refresh_hours: int = 6,
    max_tracked_wallets: int = 50,
) -> list[Opportunity]:
    """Detect copy-trade opportunities from smart money wallet activity.

    1. Refresh wallet tracking data if stale
    2. Query tracked wallets above score threshold
    3. Find their recent positions
    4. Filter by category, price range, and position size
    5. Generate Opportunity objects for positions worth copying
    """
    excluded = set(c.lower() for c in (excluded_categories or []))
    opportunities: list[Opportunity] = []

    # Get high-scoring wallets from the database
    wallets = wallet_db.get_tracked_wallets(
        min_score=min_wallet_score, limit=max_tracked_wallets
    )

    if not wallets:
        logger.info("Smart money: no tracked wallets above threshold, attempting refresh...")
        refresh_tracked_wallets(
            poly_client, wallet_db, max_wallets=max_tracked_wallets
        )
        wallets = wallet_db.get_tracked_wallets(
            min_score=min_wallet_score, limit=max_tracked_wallets
        )

    if not wallets:
        logger.info("Smart money: no qualifying wallets found")
        return []

    logger.info(f"Smart money: evaluating {len(wallets)} tracked wallets")

    # Collect all recent positions across tracked wallets
    all_positions = wallet_db.get_all_recent_positions(hours=lookback_hours)

    # Deduplicate: if multiple wallets hold the same position, take the strongest signal
    seen_markets: dict[str, dict] = {}  # market_id -> best position info

    for pos in all_positions:
        category = pos.get("category", "").lower()
        if category in excluded:
            continue

        current_price = pos.get("current_price", 0)
        if current_price < min_market_price or current_price > max_market_price:
            continue

        size_usd = pos.get("size", 0) * pos.get("avg_entry_price", 0)
        if size_usd < min_position_usd:
            continue

        market_id = pos.get("market_id", "")
        wallet_score = pos.get("score", 0)

        # Keep the position from the highest-scored wallet
        if market_id not in seen_markets or wallet_score > seen_markets[market_id].get("score", 0):
            seen_markets[market_id] = pos

    logger.info(f"Smart money: found {len(seen_markets)} candidate positions after filtering")

    for market_id, pos in seen_markets.items():
        wallet_addr = pos.get("address", "")
        wallet_score = pos.get("score", 0)
        win_rate = pos.get("win_rate", 0)
        current_price = pos.get("current_price", 0)
        avg_entry = pos.get("avg_entry_price", 0)
        outcome = pos.get("outcome", "")
        market_title = pos.get("market_title", "")
        category = pos.get("category", "")

        # Estimate edge: use the wallet's historical win rate as our model probability
        # for the outcome they're holding. This is a rough but reasonable heuristic —
        # if a wallet wins 85% of the time and they bought YES, we estimate P(YES) ≈ 0.85
        model_prob = max(0.5, win_rate)  # floor at 50% (no signal below that)
        edge = model_prob - current_price

        if edge <= 0:
            continue

        # Calculate fees
        fee = poly_fees.fee_per_contract(current_price, category=category, is_taker=True)
        expected_profit = edge - fee
        if expected_profit <= 0:
            continue

        # Position sizing: mirror a fraction of the wallet's position
        wallet_size = int(pos.get("size", 0))
        max_contracts = max(1, min(wallet_size, 100))

        # Build the opportunity
        opp = Opportunity(
            strategy="smart_money",
            legs=[
                OrderLeg(
                    platform="polymarket",
                    market_id=market_id,
                    outcome_id=outcome,
                    side="buy",
                    price=current_price,
                    size=max_contracts,
                )
            ],
            fees=[
                FeeEstimate(
                    platform="polymarket",
                    fee_per_contract=fee,
                    total_fee=fee * max_contracts,
                    fee_type="taker",
                )
            ],
            gross_profit_per_contract=edge,
            total_fees_per_contract=fee,
            net_profit_per_contract=expected_profit,
            max_contracts=max_contracts,
            total_net_profit=expected_profit * max_contracts,
            matched_market=None,
            detected_at=datetime.now(timezone.utc),
            description=f"Copy {wallet_addr[:8]}... on: {market_title}",
            model_probability=model_prob,
            market_price=current_price,
            edge=edge,
            signal_confidence=wallet_score,
            signal_source=(
                f"wallet:{wallet_addr[:10]}... "
                f"(score={wallet_score:.2f}, win_rate={win_rate:.0%}, "
                f"entry={avg_entry:.2f}, category={category})"
            ),
        )
        opportunities.append(opp)

    # Sort by signal confidence (wallet score) * edge
    opportunities.sort(
        key=lambda o: (o.signal_confidence or 0) * (o.edge or 0), reverse=True
    )

    return opportunities
