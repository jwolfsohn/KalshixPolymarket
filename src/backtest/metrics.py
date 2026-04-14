"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Summary statistics for a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_gross_profit: float = 0.0
    total_fees: float = 0.0
    total_net_profit: float = 0.0
    max_drawdown: float = 0.0
    avg_profit_per_trade: float = 0.0
    win_rate: float = 0.0
    avg_hold_days: float = 0.0
    capital_deployed: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"=== Backtest Results ===\n"
            f"Trades: {self.total_trades} "
            f"(W: {self.winning_trades} / L: {self.losing_trades})\n"
            f"Win rate: {self.win_rate:.1%}\n"
            f"Gross profit: ${self.total_gross_profit:.2f}\n"
            f"Total fees: ${self.total_fees:.2f}\n"
            f"Net profit: ${self.total_net_profit:.2f}\n"
            f"Return: {self.total_return_pct:.2%}\n"
            f"Annualized: {self.annualized_return_pct:.2%}\n"
            f"Sharpe: {self.sharpe_ratio:.2f}\n"
            f"Max drawdown: {self.max_drawdown:.2%}\n"
            f"Avg profit/trade: ${self.avg_profit_per_trade:.4f}\n"
            f"Avg hold: {self.avg_hold_days:.1f} days"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade log to DataFrame for analysis."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)


def compute_metrics(
    trade_log: list[dict],
    initial_capital: float,
    total_days: int,
) -> BacktestMetrics:
    """Compute metrics from a list of trade records.

    Each trade dict should have:
    - gross_profit: float
    - fees: float
    - net_profit: float
    - hold_days: int (optional)
    - contracts: int
    - capital_used: float
    """
    if not trade_log:
        return BacktestMetrics()

    net_profits = [t["net_profit"] for t in trade_log]
    gross_profits = [t["gross_profit"] for t in trade_log]
    fees = [t["fees"] for t in trade_log]

    total_net = sum(net_profits)
    total_gross = sum(gross_profits)
    total_fees_paid = sum(fees)

    winners = [p for p in net_profits if p > 0]
    losers = [p for p in net_profits if p <= 0]

    # Cumulative P&L for drawdown
    cumulative = np.cumsum(net_profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = abs(min(drawdowns)) / initial_capital if len(drawdowns) > 0 else 0

    # Sharpe: annualized ratio of mean daily return to std
    if len(net_profits) > 1:
        daily_returns = np.array(net_profits) / initial_capital
        sharpe = (
            np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)
            if np.std(daily_returns) > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    hold_days = [t.get("hold_days", 1) for t in trade_log]
    total_return = total_net / initial_capital if initial_capital > 0 else 0
    ann_return = (
        total_return * (365 / total_days) if total_days > 0 else 0
    )

    return BacktestMetrics(
        total_trades=len(trade_log),
        winning_trades=len(winners),
        losing_trades=len(losers),
        total_gross_profit=total_gross,
        total_fees=total_fees_paid,
        total_net_profit=total_net,
        max_drawdown=max_dd,
        avg_profit_per_trade=total_net / len(trade_log),
        win_rate=len(winners) / len(trade_log),
        avg_hold_days=sum(hold_days) / len(hold_days),
        capital_deployed=initial_capital,
        total_return_pct=total_return,
        annualized_return_pct=ann_return,
        sharpe_ratio=sharpe,
        trade_log=trade_log,
    )
