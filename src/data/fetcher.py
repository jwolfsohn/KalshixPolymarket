"""Market data fetching via PMXT unified API."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pmxt

from src.arb.models import Market, Outcome
from src.config import BotConfig

logger = logging.getLogger(__name__)


def _outcome_from_pmxt(
    platform: str,
    market_id: str,
    outcome: pmxt.MarketOutcome,
    category: str,
    side: str,
) -> Outcome:
    """Convert a PMXT MarketOutcome to our Outcome model."""
    return Outcome(
        platform=platform,
        market_id=market_id,
        outcome_id=outcome.outcome_id,
        label=outcome.label,
        side=side,
        best_bid=0.0,  # populated later from order book
        best_ask=outcome.price if outcome.price else 0.0,
        bid_depth=0,
        ask_depth=0,
        category=category,
    )


def _market_from_pmxt(platform: str, m: pmxt.UnifiedMarket) -> Market:
    """Convert a PMXT UnifiedMarket to our Market model."""
    category = m.category or ""
    outcomes = []

    # Binary markets have .yes and .no
    if m.yes:
        outcomes.append(
            _outcome_from_pmxt(platform, m.market_id, m.yes, category, "yes")
        )
    if m.no:
        outcomes.append(
            _outcome_from_pmxt(platform, m.market_id, m.no, category, "no")
        )

    # Multi-outcome markets list outcomes directly
    if m.outcomes and not outcomes:
        for o in m.outcomes:
            side = o.label.lower() if o.label else "unknown"
            outcomes.append(
                _outcome_from_pmxt(platform, m.market_id, o, category, side)
            )

    return Market(
        platform=platform,
        market_id=m.market_id,
        title=m.title,
        category=category,
        settlement_date=m.resolution_date,
        outcome_count=len(outcomes),
        outcomes=outcomes,
        url=m.url,
    )


class MarketFetcher:
    """Fetch markets from Kalshi and Polymarket via PMXT."""

    def __init__(self, config: BotConfig):
        self.config = config
        self._kalshi: pmxt.Kalshi | None = None
        self._polymarket: pmxt.Polymarket | None = None

    def _get_kalshi(self) -> pmxt.Kalshi:
        if self._kalshi is None:
            kwargs: dict[str, Any] = {}
            if self.config.kalshi_api_key:
                kwargs["api_key"] = self.config.kalshi_api_key
            if self.config.kalshi_private_key_path:
                kwargs["private_key"] = self.config.kalshi_private_key_path
            self._kalshi = pmxt.Kalshi(**kwargs)
        return self._kalshi

    def _get_polymarket(self) -> pmxt.Polymarket:
        if self._polymarket is None:
            kwargs: dict[str, Any] = {}
            if self.config.polymarket_private_key:
                kwargs["private_key"] = self.config.polymarket_private_key
            if self.config.polymarket_proxy_address:
                kwargs["proxy_address"] = self.config.polymarket_proxy_address
            self._polymarket = pmxt.Polymarket(**kwargs)
        return self._polymarket

    def fetch_kalshi_markets(
        self, queries: list[str] | None = None
    ) -> list[Market]:
        """Fetch active markets from Kalshi.

        Args:
            queries: Optional list of search terms. If provided, fetches events
                matching each query. If None, fetches all open markets (slow).
        """
        try:
            kalshi = self._get_kalshi()
            if queries:
                raw_markets = []
                for q in queries:
                    events = kalshi.fetch_events(query=q)
                    for event in events:
                        if hasattr(event, "markets") and event.markets:
                            raw_markets.extend(event.markets)
                # Deduplicate by market_id
                seen = set()
                deduped = []
                for m in raw_markets:
                    if m.market_id not in seen:
                        seen.add(m.market_id)
                        deduped.append(m)
                raw_markets = deduped
            else:
                raw_markets = kalshi.fetch_markets(status="open")

            markets = [_market_from_pmxt("kalshi", m) for m in raw_markets]
            logger.info(f"Fetched {len(markets)} markets from Kalshi")
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi markets: {e}")
            return []

    def fetch_polymarket_markets(
        self, queries: list[str] | None = None
    ) -> list[Market]:
        """Fetch active markets from Polymarket.

        Args:
            queries: List of search terms (required for Polymarket -- fetching
                all markets at once times out). Each query searches events by keyword.
        """
        try:
            poly = self._get_polymarket()
            if queries:
                raw_markets = []
                for q in queries:
                    events = poly.fetch_events(query=q)
                    for event in events:
                        if hasattr(event, "markets") and event.markets:
                            raw_markets.extend(event.markets)
                # Deduplicate by market_id
                seen = set()
                deduped = []
                for m in raw_markets:
                    if m.market_id not in seen:
                        seen.add(m.market_id)
                        deduped.append(m)
                raw_markets = deduped
            else:
                raw_markets = poly.fetch_markets(status="open")

            markets = [_market_from_pmxt("polymarket", m) for m in raw_markets]
            logger.info(f"Fetched {len(markets)} markets from Polymarket")
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch Polymarket markets: {e}")
            return []

    def fetch_all_markets(
        self, queries: list[str] | None = None
    ) -> tuple[list[Market], list[Market]]:
        """Fetch markets from both platforms.

        Args:
            queries: Search terms to focus the scan. Recommended for Polymarket
                which times out on full fetch. Example:
                ["Fed rate", "inflation", "Bitcoin", "Trump", "election"]
        """
        kalshi = self.fetch_kalshi_markets(queries)
        poly = self.fetch_polymarket_markets(queries)
        return kalshi, poly

    def fetch_orderbook(
        self, platform: str, outcome_id: str
    ) -> pmxt.OrderBook | None:
        """Fetch order book for a specific outcome."""
        try:
            exchange = (
                self._get_kalshi()
                if platform == "kalshi"
                else self._get_polymarket()
            )
            return exchange.fetch_order_book(outcome_id)
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {platform}/{outcome_id}: {e}")
            return None

    def enrich_with_orderbook(self, outcome: Outcome) -> Outcome:
        """Fill in bid/ask/depth from live order book data."""
        ob = self.fetch_orderbook(outcome.platform, outcome.outcome_id)
        if ob is None:
            return outcome

        if ob.bids:
            outcome.best_bid = ob.bids[0].price
            outcome.bid_depth = int(ob.bids[0].size) if hasattr(ob.bids[0], 'size') else 0
        if ob.asks:
            outcome.best_ask = ob.asks[0].price
            outcome.ask_depth = int(ob.asks[0].size) if hasattr(ob.asks[0], 'size') else 0

        return outcome

    def close(self):
        """Clean up exchange connections."""
        if self._kalshi:
            try:
                self._kalshi.close()
            except Exception:
                pass
        if self._polymarket:
            try:
                self._polymarket.close()
            except Exception:
                pass
