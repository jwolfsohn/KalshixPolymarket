"""Walk-forward backtesting engine for arbitrage strategies.

Simulates the scan → detect → execute → hold → settle cycle
using either real archive data or synthetic data.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.backtest.data_loader import generate_synthetic_data
from src.backtest.metrics import BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)

kalshi_fees = KalshiFeeCalculator()
poly_fees = PolymarketFeeCalculator()


@dataclass
class BacktestConfig:
    """Parameters for a backtest run."""
    initial_capital: float = 1000.0
    min_edge_pct: float = 0.02
    min_profit_after_fees_cents: float = 0.5
    max_position_per_market: int = 50
    max_contracts_per_trade: int = 20
    slippage_bps: float = 5.0  # assumed slippage in basis points
    hold_to_settlement: bool = True  # if False, exit when spread closes


def simulate_trade(
    kalshi_yes_ask: float,
    poly_no_ask: float,
    contracts: int,
    category: str,
    slippage_bps: float,
) -> dict:
    """Simulate a single cross-platform arb trade with fees and slippage."""
    # Apply slippage (prices move against us)
    slip = slippage_bps / 10000
    effective_yes = kalshi_yes_ask * (1 + slip)
    effective_no = poly_no_ask * (1 + slip)

    total_cost = effective_yes + effective_no
    gross_per = 1.0 - total_cost

    # Fees
    k_fee = kalshi_fees.fee_per_contract(effective_yes, is_taker=True)
    p_fee = poly_fees.fee_per_contract(effective_no, category=category, is_taker=True)
    fees_per = k_fee + p_fee

    net_per = gross_per - fees_per

    return {
        "kalshi_yes": effective_yes,
        "poly_no": effective_no,
        "total_cost": total_cost,
        "contracts": contracts,
        "gross_profit": gross_per * contracts,
        "fees": fees_per * contracts,
        "net_profit": net_per * contracts,
        "capital_used": total_cost * contracts,
        "fee_pct": fees_per / total_cost if total_cost > 0 else 0,
        "net_per_contract": net_per,
    }


def run_backtest(
    data: list[dict] | None = None,
    config: BacktestConfig | None = None,
) -> BacktestMetrics:
    """Run a walk-forward backtest on market data.

    Args:
        data: List of dicts with keys: timestamp, market_id, market_title,
              kalshi_yes_ask, poly_no_ask, kalshi_yes_depth, poly_no_depth, category.
              If None, generates synthetic data.
        config: Backtest parameters.
    """
    if config is None:
        config = BacktestConfig()

    if data is None:
        logger.info("No data provided, generating synthetic data...")
        data = generate_synthetic_data()

    df = pd.DataFrame(data)
    timestamps = sorted(df["timestamp"].unique())
    logger.info(
        f"Backtesting over {len(timestamps)} timestamps, "
        f"{df['market_id'].nunique()} markets"
    )

    trade_log: list[dict] = []
    capital = config.initial_capital
    positions: dict[str, dict] = {}  # market_id -> position info

    for t in timestamps:
        snapshot = df[df["timestamp"] == t]

        for _, row in snapshot.iterrows():
            mid = row["market_id"]
            k_yes = row["kalshi_yes_ask"]
            p_no = row["poly_no_ask"]
            category = row.get("category", "politics")

            total_cost = k_yes + p_no
            gross = 1.0 - total_cost

            if gross <= 0:
                continue

            edge_pct = gross / total_cost if total_cost > 0 else 0
            if edge_pct < config.min_edge_pct:
                continue

            # Check fees
            k_fee = kalshi_fees.fee_per_contract(k_yes, is_taker=True)
            p_fee = poly_fees.fee_per_contract(p_no, category=category, is_taker=True)
            net_per = gross - k_fee - p_fee

            if net_per * 100 < config.min_profit_after_fees_cents:
                continue

            # Skip if already in this market
            if mid in positions:
                continue

            # Size the trade
            max_by_depth = min(
                row.get("kalshi_yes_depth", 100),
                row.get("poly_no_depth", 100),
            )
            max_by_capital = int(capital / total_cost) if total_cost > 0 else 0
            contracts = min(
                config.max_contracts_per_trade,
                config.max_position_per_market,
                max_by_depth,
                max_by_capital,
            )

            if contracts <= 0:
                continue

            # Execute
            trade = simulate_trade(k_yes, p_no, contracts, category, config.slippage_bps)
            trade["timestamp"] = t
            trade["market_id"] = mid
            trade["market_title"] = row.get("market_title", mid)
            trade["hold_days"] = 1  # simplified: assume settlement next period

            capital -= trade["capital_used"]
            capital += trade["capital_used"] + trade["net_profit"]  # settlement

            trade_log.append(trade)

    total_days = len(timestamps)
    metrics = compute_metrics(trade_log, config.initial_capital, total_days)
    return metrics


def parameter_sweep(
    data: list[dict] | None = None,
    param_grid: dict | None = None,
) -> pd.DataFrame:
    """Sweep over parameter combinations and return results DataFrame.

    Args:
        data: Market data (or None for synthetic).
        param_grid: Dict of param_name -> list of values. Example:
            {"min_edge_pct": [0.01, 0.02, 0.03, 0.05],
             "slippage_bps": [0, 5, 10, 20]}
    """
    if param_grid is None:
        param_grid = {
            "min_edge_pct": [0.01, 0.02, 0.03, 0.05],
            "min_profit_after_fees_cents": [0.0, 0.5, 1.0],
            "slippage_bps": [0, 5, 10],
        }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info(f"Running parameter sweep: {len(combos)} combinations")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        config = BacktestConfig(**params)

        metrics = run_backtest(data=data, config=config)

        row = {**params}
        row["total_trades"] = metrics.total_trades
        row["net_profit"] = metrics.total_net_profit
        row["win_rate"] = metrics.win_rate
        row["sharpe"] = metrics.sharpe_ratio
        row["max_drawdown"] = metrics.max_drawdown
        row["return_pct"] = metrics.total_return_pct
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("net_profit", ascending=False)
    return results_df
