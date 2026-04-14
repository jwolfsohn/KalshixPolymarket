"""Wallet scoring heuristics — identify insider-like wallets vs bots."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from src.signals.smart_money.polygon_client import PolygonClient, WalletTrade
from src.signals.smart_money.wallet_db import WalletDB, WalletProfile

logger = logging.getLogger(__name__)

# Categories that correlate with insider knowledge on real-world events
REAL_EVENT_CATEGORIES = {
    "politics", "economics", "finance", "energy", "geopolitics",
    "science", "technology", "business", "crypto",
}

# Categories that are less likely to involve insider trading
NOISE_CATEGORIES = {"culture", "mentions", "entertainment", "memes", "sports"}


def score_wallet(trades: list[WalletTrade]) -> WalletProfile | None:
    """Score a wallet based on its trading history.

    Heuristics:
    - High win rate with few trades = likely informed
    - Many trades with low win rate = retail
    - Very many trades with tight spreads = likely bot
    - Large positions on real-world events = stronger signal
    """
    if not trades:
        return None

    address = trades[0].address

    # Separate resolved and unresolved trades
    resolved = [t for t in trades if t.resolved and t.resolution is not None]
    buy_trades = [t for t in trades if t.side == "buy"]

    total_trades = len(trades)
    total_volume = sum(t.size * t.price for t in trades if t.price > 0)
    avg_position = total_volume / total_trades if total_trades > 0 else 0

    # Count wins: a winning trade is one where you bought the correct outcome
    winning = 0
    for t in resolved:
        if t.side == "buy" and t.resolution:
            if t.outcome.lower() == t.resolution.lower():
                winning += 1
            elif t.outcome.lower() in ("yes", "no"):
                if t.outcome.lower() == t.resolution.lower():
                    winning += 1

    win_rate = winning / len(resolved) if resolved else 0.0

    # Category breakdown
    categories: dict[str, int] = Counter()
    for t in trades:
        if t.category:
            categories[t.category.lower()] += 1

    last_active = max(t.timestamp for t in trades)

    # --- Bot Detection ---
    bot_score = _compute_bot_score(trades)

    # --- Insider Score ---
    insider_score = _compute_insider_score(
        total_trades=total_trades,
        win_rate=win_rate,
        avg_position=avg_position,
        total_volume=total_volume,
        categories=categories,
        resolved_count=len(resolved),
        buy_trades=buy_trades,
    )

    # Final composite score: insider likelihood penalized by bot probability
    final_score = max(0.0, insider_score * (1.0 - bot_score))

    # Assign label
    if bot_score >= 0.7:
        label = "bot"
    elif final_score >= 0.7:
        label = "likely_insider"
    elif final_score >= 0.5:
        label = "smart_money"
    else:
        label = "retail"

    return WalletProfile(
        address=address,
        total_trades=total_trades,
        winning_trades=winning,
        win_rate=win_rate,
        total_volume_usd=total_volume,
        avg_position_size_usd=avg_position,
        total_pnl_usd=0.0,  # computed separately from resolved trades
        categories=dict(categories),
        last_active=last_active,
        score=round(final_score, 3),
        label=label,
    )


def _compute_bot_score(trades: list[WalletTrade]) -> float:
    """Estimate likelihood that this wallet is a bot (0-1).

    Bot indicators:
    - Very high trade frequency (> 1000 trades in 30 days)
    - Short time between trades (< 30 seconds mean)
    - Trading both sides of same market within minutes
    - Active across 50+ markets simultaneously
    """
    if len(trades) < 10:
        return 0.0

    score = 0.0

    # Factor 1: Trade frequency
    now = datetime.now(timezone.utc)
    recent = [t for t in trades if (now - t.timestamp).days <= 30]
    if len(recent) > 1000:
        score += 0.35
    elif len(recent) > 500:
        score += 0.20
    elif len(recent) > 200:
        score += 0.10

    # Factor 2: Mean time between trades
    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    if len(sorted_trades) >= 2:
        deltas = [
            (sorted_trades[i + 1].timestamp - sorted_trades[i].timestamp).total_seconds()
            for i in range(min(100, len(sorted_trades) - 1))
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else 9999
        if mean_delta < 30:
            score += 0.30
        elif mean_delta < 120:
            score += 0.15

    # Factor 3: Both-sides trading (market making pattern)
    market_sides: dict[str, set[str]] = {}
    for t in recent:
        market_sides.setdefault(t.market_id, set()).add(t.side)
    both_sides_count = sum(1 for sides in market_sides.values() if len(sides) > 1)
    both_sides_ratio = both_sides_count / len(market_sides) if market_sides else 0
    if both_sides_ratio > 0.5:
        score += 0.20
    elif both_sides_ratio > 0.3:
        score += 0.10

    # Factor 4: Market breadth (too many simultaneous markets)
    unique_markets = len(set(t.market_id for t in recent))
    if unique_markets > 50:
        score += 0.15
    elif unique_markets > 30:
        score += 0.05

    return min(1.0, score)


def _compute_insider_score(
    total_trades: int,
    win_rate: float,
    avg_position: float,
    total_volume: float,
    categories: dict[str, int],
    resolved_count: int,
    buy_trades: list[WalletTrade],
) -> float:
    """Estimate likelihood of insider-like trading behavior (0-1).

    Insider indicators:
    - High win rate (> 70%) with few total trades (< 50)
    - Large average position size (> $5,000)
    - Concentrated in real-world event categories
    - Bought at low prices that resolved YES (high payout ratio)
    - Trades well before market resolution (not last-minute)
    """
    if resolved_count < 3:
        return 0.3  # insufficient data, neutral score

    score = 0.0

    # Factor 1: Win rate (most important signal)
    if win_rate >= 0.90 and resolved_count >= 5:
        score += 0.35
    elif win_rate >= 0.80 and resolved_count >= 5:
        score += 0.28
    elif win_rate >= 0.70 and resolved_count >= 5:
        score += 0.20
    elif win_rate >= 0.60:
        score += 0.10

    # Factor 2: Trade selectivity (few, high-conviction trades)
    if total_trades <= 20:
        score += 0.20
    elif total_trades <= 50:
        score += 0.15
    elif total_trades <= 100:
        score += 0.08
    # Many trades reduces insider likelihood (more like a regular trader)

    # Factor 3: Position size (insiders bet big on what they know)
    if avg_position >= 10000:
        score += 0.15
    elif avg_position >= 5000:
        score += 0.12
    elif avg_position >= 1000:
        score += 0.08
    elif avg_position >= 500:
        score += 0.04

    # Factor 4: Category focus (real-world events, not memes)
    total_cat_trades = sum(categories.values())
    if total_cat_trades > 0:
        real_event_trades = sum(
            count for cat, count in categories.items()
            if cat in REAL_EVENT_CATEGORIES
        )
        real_ratio = real_event_trades / total_cat_trades
        if real_ratio >= 0.8:
            score += 0.15
        elif real_ratio >= 0.5:
            score += 0.08

    # Factor 5: Payout ratio (bought cheap, resolved in their favor)
    low_price_wins = sum(
        1 for t in buy_trades
        if t.resolved and t.price <= 0.35
        and t.resolution and t.outcome.lower() == t.resolution.lower()
    )
    if buy_trades and low_price_wins >= 3:
        score += 0.15
    elif low_price_wins >= 1:
        score += 0.08

    return min(1.0, score)


def refresh_tracked_wallets(
    client: PolygonClient,
    db: WalletDB,
    max_wallets: int = 50,
    min_score_to_keep: float = 0.4,
):
    """Discover and rescore wallets.

    1. Fetch leaderboard for initial wallet discovery
    2. Look for wallets making large recent trades
    3. Score each wallet and persist to DB
    """
    # Discover wallets from leaderboard
    leaderboard = client.get_leaderboard(limit=max_wallets * 2)
    addresses: set[str] = set()
    for entry in leaderboard:
        addr = entry.get("address", entry.get("wallet", ""))
        if addr:
            addresses.add(addr)

    # Discover wallets from recent large trades
    large_trades = client.get_recent_large_trades(min_size_usd=1000, limit=200)
    for trade in large_trades:
        if trade.address:
            addresses.add(trade.address)

    logger.info(f"Discovered {len(addresses)} wallet addresses to evaluate")

    scored = 0
    for addr in list(addresses)[:max_wallets * 2]:
        trades = client.get_wallet_history(addr, limit=200)
        if not trades:
            continue

        profile = score_wallet(trades)
        if profile is None:
            continue

        if profile.score >= min_score_to_keep:
            db.upsert_wallet(profile)
            scored += 1
            logger.debug(
                f"Wallet {addr[:10]}...: score={profile.score:.2f}, "
                f"label={profile.label}, win_rate={profile.win_rate:.0%}, "
                f"trades={profile.total_trades}"
            )

        # Record individual trades for history
        for trade in trades:
            db.record_trade(addr, {
                "market_id": trade.market_id,
                "market_title": trade.market_title,
                "outcome": trade.outcome,
                "side": trade.side,
                "price": trade.price,
                "size": trade.size,
                "timestamp": trade.timestamp.isoformat(),
                "category": trade.category,
                "resolved": trade.resolved,
                "resolution": trade.resolution,
            })

    logger.info(f"Scored {scored} wallets above threshold {min_score_to_keep}")

    # Update positions for high-scoring wallets
    top_wallets = db.get_tracked_wallets(min_score=0.5, limit=max_wallets)
    for wallet in top_wallets:
        positions = client.get_wallet_positions(wallet.address)
        for pos in positions:
            db.upsert_position(wallet.address, {
                "market_id": pos.market_id,
                "market_title": pos.market_title,
                "outcome": pos.outcome,
                "size": pos.size,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "category": pos.category,
            })

    logger.info(f"Updated positions for {len(top_wallets)} top wallets")
