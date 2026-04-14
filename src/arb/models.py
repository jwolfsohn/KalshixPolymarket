"""Core data models for the arbitrage system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Outcome:
    """A single tradeable outcome on a platform."""
    platform: str  # "kalshi" or "polymarket"
    market_id: str
    outcome_id: str
    label: str  # e.g. "Yes", "No", "Kevin Warsh"
    side: str  # "yes" or "no"
    best_bid: float  # best bid price (0-1)
    best_ask: float  # best ask price (0-1)
    bid_depth: int  # number of contracts at best bid
    ask_depth: int  # number of contracts at best ask
    category: str = ""


@dataclass
class Market:
    """A market on a single platform."""
    platform: str
    market_id: str
    title: str
    category: str
    settlement_date: datetime | None
    outcome_count: int
    outcomes: list[Outcome] = field(default_factory=list)
    event_id: str = ""
    event_title: str = ""
    url: str = ""


@dataclass
class MatchedMarket:
    """A pair of markets across platforms representing the same event."""
    kalshi: Market
    polymarket: Market
    similarity_score: float
    match_method: str  # "cache", "structural+fuzzy", "manual"
    settlement_risk: str  # "identical", "similar", "different_criteria"
    outcome_mapping: dict[str, str] = field(default_factory=dict)
    # Maps kalshi outcome_id -> polymarket outcome_id


@dataclass
class OrderLeg:
    """A single order to be placed."""
    platform: str
    market_id: str
    outcome_id: str
    side: Literal["buy", "sell"]
    price: float
    size: int  # contracts


@dataclass
class FeeEstimate:
    """Fee breakdown for a single leg."""
    platform: str
    fee_per_contract: float
    total_fee: float
    fee_type: str  # "taker" or "maker"


@dataclass
class Opportunity:
    """A detected arbitrage opportunity."""
    strategy: Literal["cross_platform", "bundle", "multi_outcome", "smart_money", "weather_ml"]
    legs: list[OrderLeg]
    fees: list[FeeEstimate]
    gross_profit_per_contract: float
    total_fees_per_contract: float
    net_profit_per_contract: float
    max_contracts: int  # limited by order book depth
    total_net_profit: float
    matched_market: MatchedMarket | None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_date: datetime | None = None
    days_to_settlement: int = 0
    annualized_return: float = 0.0
    description: str = ""

    # Signal-based strategy metadata (None for riskless arb strategies)
    model_probability: float | None = None
    market_price: float | None = None
    edge: float | None = None
    signal_confidence: float | None = None
    signal_source: str = ""

    def summary(self) -> str:
        legs_str = " + ".join(
            f"{l.side.upper()} {l.platform} @{l.price:.2f}" for l in self.legs
        )
        lines = [
            f"[{self.strategy}] {self.description}",
            f"  Legs: {legs_str}",
            f"  Gross: {self.gross_profit_per_contract:.4f}/contract | "
            f"Fees: {self.total_fees_per_contract:.4f} | "
            f"Net: {self.net_profit_per_contract:.4f}",
            f"  Max size: {self.max_contracts} contracts | "
            f"Total profit: ${self.total_net_profit:.2f}",
            f"  Settlement: {self.days_to_settlement}d | "
            f"Ann. return: {self.annualized_return:.1%}",
        ]
        if self.model_probability is not None:
            lines.append(
                f"  Model: {self.model_probability:.1%} vs Market: {self.market_price:.1%} | "
                f"Edge: {self.edge:+.1%} | Confidence: {self.signal_confidence:.2f}"
            )
        if self.signal_source:
            lines.append(f"  Source: {self.signal_source}")
        return "\n".join(lines)
