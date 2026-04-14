"""Tests for the Smart Money wallet tracker signal strategy."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.smart_money.polygon_client import WalletTrade
from src.signals.smart_money.tracker import score_wallet
from src.signals.smart_money.wallet_db import WalletDB, WalletProfile


def _make_trade(
    address: str = "0xinsider",
    market_id: str = "mkt-1",
    outcome: str = "Yes",
    side: str = "buy",
    price: float = 0.25,
    size: float = 2000.0,
    days_ago: int = 5,
    category: str = "economics",
    resolved: bool = True,
    resolution: str | None = "Yes",
) -> WalletTrade:
    return WalletTrade(
        address=address,
        market_id=market_id,
        market_title=f"Test Market {market_id}",
        outcome=outcome,
        side=side,
        price=price,
        size=size,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
        category=category,
        resolved=resolved,
        resolution=resolution,
    )


# ========== Wallet Scoring Tests ==========


class TestWalletScoring:
    def test_high_winrate_low_trades_scores_high(self):
        """A wallet with 90% win rate on 10 resolved trades = likely insider."""
        trades = []
        for i in range(10):
            won = i < 9  # 9 out of 10 win
            trades.append(_make_trade(
                market_id=f"mkt-{i}",
                price=0.20,
                size=5000.0,
                resolved=True,
                resolution="Yes" if won else "No",
                category="economics",
            ))

        profile = score_wallet(trades)
        assert profile is not None
        assert profile.win_rate >= 0.8
        assert profile.score >= 0.5
        assert profile.label in ("likely_insider", "smart_money")

    def test_bot_pattern_scores_low(self):
        """A wallet with 2000 trades in 30 days, tight timing, both sides = bot."""
        now = datetime.now(timezone.utc)
        trades = []
        for i in range(2000):
            trades.append(WalletTrade(
                address="0xbot",
                market_id=f"mkt-{i % 50}",  # trades across 50+ markets
                market_title=f"Test Market {i % 50}",
                outcome="Yes" if i % 2 == 0 else "No",
                side="buy" if i % 2 == 0 else "sell",  # both sides
                price=0.50,
                size=10.0,
                # 10-second intervals = clear bot behavior
                timestamp=now - timedelta(seconds=i * 10),
                category="crypto",
                resolved=False,
                resolution=None,
            ))

        profile = score_wallet(trades)
        assert profile is not None
        assert profile.label == "bot"
        assert profile.score < 0.3

    def test_retail_average_profile(self):
        """A wallet with 50% win rate = retail."""
        trades = []
        for i in range(20):
            won = i % 2 == 0  # 50% win rate
            trades.append(_make_trade(
                market_id=f"mkt-{i}",
                price=0.50,
                size=100.0,
                resolved=True,
                resolution="Yes" if won else "No",
                category="politics",
            ))

        profile = score_wallet(trades)
        assert profile is not None
        assert profile.win_rate == pytest.approx(0.5, abs=0.05)
        assert profile.label == "retail"

    def test_empty_trades_returns_none(self):
        assert score_wallet([]) is None

    def test_few_resolved_neutral_score(self):
        """With only 2 resolved trades, score should be neutral."""
        trades = [
            _make_trade(market_id="mkt-1", resolved=True, resolution="Yes"),
            _make_trade(market_id="mkt-2", resolved=True, resolution="Yes"),
        ]
        profile = score_wallet(trades)
        assert profile is not None
        # With only 2 resolved trades, insufficient data → moderate score
        assert profile.score <= 0.6

    def test_real_event_categories_boost_score(self):
        """Trades in economics/politics score higher than culture/memes."""
        # Economics-focused wallet
        econ_trades = [
            _make_trade(
                market_id=f"mkt-{i}", price=0.25, size=3000.0,
                resolved=True, resolution="Yes", category="economics",
            )
            for i in range(10)
        ]
        econ_profile = score_wallet(econ_trades)

        # Culture-focused wallet (same trades, different category)
        culture_trades = [
            _make_trade(
                market_id=f"mkt-{i}", price=0.25, size=3000.0,
                resolved=True, resolution="Yes", category="culture",
            )
            for i in range(10)
        ]
        culture_profile = score_wallet(culture_trades)

        assert econ_profile is not None and culture_profile is not None
        assert econ_profile.score >= culture_profile.score

    def test_large_positions_boost_score(self):
        """Wallets with large average positions score higher."""
        small_trades = [
            _make_trade(market_id=f"mkt-{i}", size=50.0, resolved=True, resolution="Yes")
            for i in range(10)
        ]
        large_trades = [
            _make_trade(market_id=f"mkt-{i}", size=10000.0, resolved=True, resolution="Yes")
            for i in range(10)
        ]

        small = score_wallet(small_trades)
        large = score_wallet(large_trades)
        assert small is not None and large is not None
        assert large.score >= small.score

    def test_low_price_wins_boost_score(self):
        """Buying at low prices that resolve correctly = high payout ratio."""
        trades = [
            _make_trade(
                market_id=f"mkt-{i}",
                price=0.15,  # bought cheap
                size=5000.0,
                resolved=True,
                resolution="Yes",
            )
            for i in range(5)
        ]
        profile = score_wallet(trades)
        assert profile is not None
        assert profile.score >= 0.5

    def test_profile_fields_populated(self):
        trades = [
            _make_trade(market_id="mkt-1", category="economics", resolved=True, resolution="Yes"),
            _make_trade(market_id="mkt-2", category="politics", resolved=True, resolution="No"),
        ]
        profile = score_wallet(trades)
        assert profile is not None
        assert profile.address == "0xinsider"
        assert profile.total_trades == 2
        assert profile.total_volume_usd > 0
        assert "economics" in profile.categories
        assert "politics" in profile.categories


# ========== Wallet DB Tests ==========


class TestWalletDB:
    def test_upsert_and_retrieve(self, tmp_path):
        db = WalletDB(db_path=str(tmp_path / "test.db"))
        try:
            profile = WalletProfile(
                address="0xtest123",
                total_trades=15,
                winning_trades=12,
                win_rate=0.80,
                total_volume_usd=50000.0,
                avg_position_size_usd=3333.33,
                total_pnl_usd=10000.0,
                categories={"economics": 10, "politics": 5},
                last_active=datetime.now(timezone.utc),
                score=0.85,
                label="likely_insider",
            )
            db.upsert_wallet(profile)

            wallets = db.get_tracked_wallets(min_score=0.7)
            assert len(wallets) == 1
            assert wallets[0].address == "0xtest123"
            assert wallets[0].win_rate == 0.80
            assert wallets[0].label == "likely_insider"
            assert wallets[0].categories == {"economics": 10, "politics": 5}
        finally:
            db.close()

    def test_bot_excluded_from_tracked(self, tmp_path):
        db = WalletDB(db_path=str(tmp_path / "test.db"))
        try:
            bot = WalletProfile(
                address="0xbot",
                total_trades=5000,
                winning_trades=2500,
                win_rate=0.50,
                total_volume_usd=100000.0,
                avg_position_size_usd=20.0,
                total_pnl_usd=0.0,
                categories={"crypto": 5000},
                last_active=datetime.now(timezone.utc),
                score=0.8,  # even with high score
                label="bot",  # labeled as bot
            )
            db.upsert_wallet(bot)

            wallets = db.get_tracked_wallets(min_score=0.5)
            assert len(wallets) == 0  # bots are excluded
        finally:
            db.close()

    def test_min_score_filter(self, tmp_path):
        db = WalletDB(db_path=str(tmp_path / "test.db"))
        try:
            for addr, score in [("0xa", 0.9), ("0xb", 0.6), ("0xc", 0.3)]:
                db.upsert_wallet(WalletProfile(
                    address=addr,
                    total_trades=10,
                    winning_trades=8,
                    win_rate=0.8,
                    total_volume_usd=10000.0,
                    avg_position_size_usd=1000.0,
                    total_pnl_usd=5000.0,
                    categories={"economics": 10},
                    last_active=datetime.now(timezone.utc),
                    score=score,
                    label="smart_money",
                ))

            high = db.get_tracked_wallets(min_score=0.7)
            assert len(high) == 1
            assert high[0].address == "0xa"

            medium = db.get_tracked_wallets(min_score=0.5)
            assert len(medium) == 2
        finally:
            db.close()


# ========== Opportunity Model Backward Compatibility ==========


class TestOpportunityModel:
    def test_existing_strategies_still_work(self):
        """Existing arb strategies should work with new optional signal fields."""
        from src.arb.models import FeeEstimate, MatchedMarket, Opportunity, OrderLeg

        opp = Opportunity(
            strategy="cross_platform",
            legs=[
                OrderLeg("kalshi", "m1", "o1", "buy", 0.40, 10),
                OrderLeg("polymarket", "m2", "o2", "buy", 0.55, 10),
            ],
            fees=[
                FeeEstimate("kalshi", 0.02, 0.20, "taker"),
                FeeEstimate("polymarket", 0.01, 0.10, "taker"),
            ],
            gross_profit_per_contract=0.05,
            total_fees_per_contract=0.03,
            net_profit_per_contract=0.02,
            max_contracts=10,
            total_net_profit=0.20,
            matched_market=None,
            days_to_settlement=30,
            annualized_return=0.24,
            description="Test cross-platform arb",
        )

        # Signal fields should be None by default
        assert opp.model_probability is None
        assert opp.market_price is None
        assert opp.edge is None
        assert opp.signal_confidence is None
        assert opp.signal_source == ""

        # Summary should NOT include signal info
        summary = opp.summary()
        assert "Model:" not in summary
        assert "Source:" not in summary

    def test_signal_fields_in_summary(self):
        from src.arb.models import Opportunity, OrderLeg, FeeEstimate

        opp = Opportunity(
            strategy="weather_ml",
            legs=[OrderLeg("kalshi", "m1", "o1", "buy", 0.30, 50)],
            fees=[FeeEstimate("kalshi", 0.02, 1.0, "taker")],
            gross_profit_per_contract=0.55,
            total_fees_per_contract=0.02,
            net_profit_per_contract=0.53,
            max_contracts=50,
            total_net_profit=26.50,
            matched_market=None,
            description="SF high temp above 70",
            model_probability=0.85,
            market_price=0.30,
            edge=0.55,
            signal_confidence=0.72,
            signal_source="weather_ensemble(3 sources)",
        )

        summary = opp.summary()
        assert "Model: 85.0%" in summary
        assert "Market: 30.0%" in summary
        assert "Edge: +55.0%" in summary
        assert "weather_ensemble" in summary
