#!/usr/bin/env python3
"""Scan Kalshi and Polymarket for arbitrage opportunities.

Usage:
    python scripts/run_scanner.py                    # single scan
    python scripts/run_scanner.py --loop              # continuous scanning
    python scripts/run_scanner.py --loop --interval 60 # scan every 60s
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arb.scanner import ArbitrageScanner
from src.config import load_config
from src.data.fetcher import MarketFetcher
from src.matching.cache import MatchCache
from src.matching.matcher import MarketMatcher


def setup_logging(level: str = "INFO", fmt: str = "human"):
    log_format = (
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
        if fmt == "human"
        else "%(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%H:%M:%S",
    )


def run_scan(config):
    """Execute a single scan cycle."""
    logger = logging.getLogger("scanner")

    # Initialize components
    fetcher = MarketFetcher(config)
    cache = MatchCache(ttl_hours=config.matching.cache_ttl_hours)
    matcher = MarketMatcher(
        similarity_threshold=config.matching.similarity_threshold,
        cache=cache,
        structural_prefilter=config.matching.structural_prefilter,
    )
    scanner = ArbitrageScanner(config)

    try:
        # Fetch markets using configured search queries
        queries = config.scan.queries
        logger.info(f"Fetching markets using {len(queries)} search queries...")
        kalshi_markets, poly_markets = fetcher.fetch_all_markets(queries=queries)

        if not kalshi_markets and not poly_markets:
            logger.warning("No markets fetched from either platform")
            return []

        logger.info(
            f"Fetched {len(kalshi_markets)} Kalshi + "
            f"{len(poly_markets)} Polymarket markets"
        )

        # Match markets
        logger.info("Matching markets across platforms...")
        matched = matcher.match_markets(kalshi_markets, poly_markets)
        logger.info(f"Found {len(matched)} matched market pairs")

        # Print top matches
        if matched:
            logger.info("\n=== Top Matched Markets ===")
            for m in sorted(matched, key=lambda x: x.similarity_score, reverse=True)[
                :10
            ]:
                logger.info(
                    f"  [{m.similarity_score:.2f}] {m.kalshi.title}\n"
                    f"         <-> {m.polymarket.title}\n"
                    f"         Settlement risk: {m.settlement_risk}"
                )

        # Scan for opportunities
        logger.info("\nScanning for arbitrage opportunities...")
        opportunities = scanner.scan(matched, kalshi_markets, poly_markets)

        # Print results
        if opportunities:
            logger.info(f"\n{'='*60}")
            logger.info(f"  FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES")
            logger.info(f"{'='*60}\n")

            for i, opp in enumerate(opportunities, 1):
                logger.info(f"--- Opportunity #{i} ---")
                logger.info(opp.summary())
                logger.info("")
        else:
            logger.info("\nNo arbitrage opportunities found above threshold.")
            logger.info(
                f"  (min edge: {config.arbitrage.min_edge_pct:.1%}, "
                f"min profit: {config.arbitrage.min_profit_after_fees_cents}c)"
            )

        return opportunities

    finally:
        fetcher.close()
        cache.close()


def main():
    parser = argparse.ArgumentParser(description="Prediction Market Arbitrage Scanner")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--loop", action="store_true", help="Run continuously"
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Scan interval in seconds (overrides config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.format)

    interval = args.interval or config.scan.interval_seconds

    if args.loop:
        logging.info(f"Starting continuous scan (every {interval}s). Ctrl+C to stop.")
        while True:
            try:
                run_scan(config)
                logging.info(f"\nSleeping {interval}s until next scan...\n")
                time.sleep(interval)
            except KeyboardInterrupt:
                logging.info("\nStopped by user.")
                break
    else:
        run_scan(config)


if __name__ == "__main__":
    main()
