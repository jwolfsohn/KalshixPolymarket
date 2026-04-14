#!/usr/bin/env python3
"""Run backtests on prediction market arbitrage strategies.

Usage:
    python scripts/run_backtest.py                      # synthetic data
    python scripts/run_backtest.py --data ./archive/     # real archive data
    python scripts/run_backtest.py --sweep               # parameter sweep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestConfig, parameter_sweep, run_backtest


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Arbitrage Backtester")
    parser.add_argument(
        "--data", default=None,
        help="Path to archive data directory (Parquet files)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run parameter sweep instead of single backtest"
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Initial capital in dollars"
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.02,
        help="Minimum edge percentage (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--slippage", type=float, default=5.0,
        help="Slippage in basis points (default: 5)"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("backtest")

    if args.sweep:
        logger.info("Running parameter sweep on synthetic data...")
        results = parameter_sweep()
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(results.to_string(index=False))
        print(f"\nBest config: net profit = ${results.iloc[0]['net_profit']:.2f}")
        print(f"  Params: {dict(results.iloc[0])}")
    else:
        config = BacktestConfig(
            initial_capital=args.capital,
            min_edge_pct=args.min_edge,
            slippage_bps=args.slippage,
        )

        logger.info(f"Running backtest (capital=${config.initial_capital:.0f}, "
                     f"min_edge={config.min_edge_pct:.1%}, "
                     f"slippage={config.slippage_bps}bps)...")

        metrics = run_backtest(data=None, config=config)
        print("\n" + metrics.summary())

        if metrics.total_trades > 0:
            df = metrics.to_dataframe()
            print(f"\nTrade log ({len(df)} trades):")
            cols = ["market_title", "contracts", "total_cost", "net_profit", "fee_pct"]
            display_cols = [c for c in cols if c in df.columns]
            print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
