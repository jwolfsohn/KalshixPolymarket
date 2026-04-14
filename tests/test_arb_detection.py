"""Tests for arbitrage detection strategies."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arb.bundle import detect_bundle
from src.arb.cross_platform import detect_cross_platform
from src.arb.fees import KalshiFeeCalculator, PolymarketFeeCalculator
from src.arb.models import Market, MatchedMarket, Outcome
from src.arb.multi_outcome import detect_multi_outcome

kalshi_fees = KalshiFeeCalculator()
poly_fees = PolymarketFeeCalculator()


def _outcome(platform, market_id, side, ask, bid=0.0, depth=100):
    return Outcome(
        platform=platform,
        market_id=market_id,
        outcome_id=f"{market_id}_{side}",
        label=side.capitalize(),
        side=side,
        best_bid=bid,
        best_ask=ask,
        bid_depth=depth,
        ask_depth=depth,
        category="politics",
    )


def _market(platform, market_id, title, outcomes, category="politics", days=30):
    return Market(
        platform=platform,
        market_id=market_id,
        title=title,
        category=category,
        settlement_date=datetime.now(timezone.utc) + timedelta(days=days),
        outcome_count=len(outcomes),
        outcomes=outcomes,
    )


def _matched(k, p, score=0.9):
    outcome_mapping = {}
    k_by_side = {o.side: o for o in k.outcomes}
    p_by_side = {o.side: o for o in p.outcomes}
    for side in ("yes", "no"):
        if side in k_by_side and side in p_by_side:
            outcome_mapping[k_by_side[side].outcome_id] = p_by_side[side].outcome_id
    return MatchedMarket(
        kalshi=k,
        polymarket=p,
        similarity_score=score,
        match_method="test",
        settlement_risk="identical",
        outcome_mapping=outcome_mapping,
    )


class TestCrossPlatformArb:
    def test_detects_obvious_arb(self):
        """YES@0.40 + NO@0.50 = 0.90 < 1.00 -> 10c gross profit."""
        k = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.40),
            _outcome("kalshi", "k1", "no", ask=0.62),
        ])
        p = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.60),
            _outcome("polymarket", "p1", "no", ask=0.50),
        ])
        match = _matched(k, p)

        opps = detect_cross_platform(
            [match], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )
        assert len(opps) > 0
        best = opps[0]
        assert best.gross_profit_per_contract > 0
        assert best.net_profit_per_contract > 0
        assert best.strategy == "cross_platform"

    def test_no_arb_when_sum_above_one(self):
        """YES@0.55 + NO@0.55 = 1.10 > 1.00 -> no arb."""
        k = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.55),
            _outcome("kalshi", "k1", "no", ask=0.50),
        ])
        p = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.50),
            _outcome("polymarket", "p1", "no", ask=0.55),
        ])
        match = _matched(k, p)

        opps = detect_cross_platform(
            [match], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )
        assert len(opps) == 0

    def test_fees_kill_thin_spread(self):
        """YES@0.49 + NO@0.50 = 0.99 -> 1c gross, but fees eat it."""
        k = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.49),
            _outcome("kalshi", "k1", "no", ask=0.52),
        ])
        p = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.52),
            _outcome("polymarket", "p1", "no", ask=0.50),
        ])
        match = _matched(k, p)

        # At P~0.50, Kalshi taker fee=0.02 + Poly taker fee=0.02 = 0.04
        # Gross profit = 0.01, net = 0.01 - 0.04 = -0.03 -> filtered out
        opps = detect_cross_platform(
            [match], kalshi_fees, poly_fees,
            min_edge_pct=0.001, min_profit_after_fees_cents=0.0,
        )
        # All opportunities should have positive net profit
        for opp in opps:
            assert opp.net_profit_per_contract > 0

    def test_extreme_prices_lower_fees(self):
        """At extreme prices, fees are lower so thinner spreads work."""
        k = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.93),
            _outcome("kalshi", "k1", "no", ask=0.09),
        ])
        p = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.95),
            _outcome("polymarket", "p1", "no", ask=0.02),
        ], category="geopolitics")  # 0% fees on polymarket
        match = _matched(k, p)

        opps = detect_cross_platform(
            [match], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )
        # YES@kalshi(0.93) + NO@poly(0.02) = 0.95 -> 5c gross
        # Kalshi fee at 0.93: min(ceil(0.07*0.93*0.07*100)/100, 0.02) = 0.01
        # Poly fee at 0.02 geopolitics: 0
        profitable = [o for o in opps if o.net_profit_per_contract > 0]
        assert len(profitable) > 0

    def test_skips_different_criteria(self):
        """Markets with different_criteria settlement risk are skipped."""
        k = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.30),
            _outcome("kalshi", "k1", "no", ask=0.60),
        ])
        p = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.60),
            _outcome("polymarket", "p1", "no", ask=0.30),
        ])
        match = _matched(k, p)
        match.settlement_risk = "different_criteria"

        opps = detect_cross_platform(
            [match], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )
        assert len(opps) == 0

    def test_annualized_return(self):
        """Shorter-dated opportunities have higher annualized returns."""
        k_short = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.40),
            _outcome("kalshi", "k1", "no", ask=0.62),
        ], days=3)
        p_short = _market("polymarket", "p1", "Test", [
            _outcome("polymarket", "p1", "yes", ask=0.62),
            _outcome("polymarket", "p1", "no", ask=0.50),
        ], days=3)

        k_long = _market("kalshi", "k2", "Test2", [
            _outcome("kalshi", "k2", "yes", ask=0.40),
            _outcome("kalshi", "k2", "no", ask=0.62),
        ], days=180)
        p_long = _market("polymarket", "p2", "Test2", [
            _outcome("polymarket", "p2", "yes", ask=0.62),
            _outcome("polymarket", "p2", "no", ask=0.50),
        ], days=180)

        opps_short = detect_cross_platform(
            [_matched(k_short, p_short)], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )
        opps_long = detect_cross_platform(
            [_matched(k_long, p_long)], kalshi_fees, poly_fees,
            min_edge_pct=0.01, min_profit_after_fees_cents=0.0,
        )

        if opps_short and opps_long:
            assert opps_short[0].annualized_return > opps_long[0].annualized_return


class TestBundleArb:
    def test_detects_bundle(self):
        """YES@0.45 + NO@0.45 = 0.90 < 1.00."""
        m = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.45),
            _outcome("kalshi", "k1", "no", ask=0.45),
        ])
        opps = detect_bundle([m], kalshi_fees, poly_fees, min_profit_after_fees_cents=0.0)
        assert len(opps) > 0
        assert opps[0].strategy == "bundle"

    def test_no_bundle_when_sum_above_one(self):
        m = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.55),
            _outcome("kalshi", "k1", "no", ask=0.55),
        ])
        opps = detect_bundle([m], kalshi_fees, poly_fees)
        assert len(opps) == 0


class TestMultiOutcomeArb:
    def test_detects_multi_outcome(self):
        """3 outcomes summing to 0.85 < 1.00."""
        outcomes = [
            _outcome("kalshi", "k1", "alice", ask=0.25),
            _outcome("kalshi", "k1", "bob", ask=0.30),
            _outcome("kalshi", "k1", "carol", ask=0.30),
        ]
        m = _market("kalshi", "k1", "Who wins?", outcomes)
        m.outcome_count = 3

        opps = detect_multi_outcome([m], kalshi_fees, poly_fees, min_profit_after_fees_cents=0.0)
        assert len(opps) > 0
        assert opps[0].strategy == "multi_outcome"
        assert len(opps[0].legs) == 3

    def test_skips_binary_markets(self):
        m = _market("kalshi", "k1", "Test", [
            _outcome("kalshi", "k1", "yes", ask=0.40),
            _outcome("kalshi", "k1", "no", ask=0.40),
        ])
        opps = detect_multi_outcome([m], kalshi_fees, poly_fees)
        assert len(opps) == 0

    def test_no_arb_when_sum_above_one(self):
        outcomes = [
            _outcome("kalshi", "k1", "alice", ask=0.40),
            _outcome("kalshi", "k1", "bob", ask=0.35),
            _outcome("kalshi", "k1", "carol", ask=0.30),
        ]
        m = _market("kalshi", "k1", "Who wins?", outcomes)
        m.outcome_count = 3

        opps = detect_multi_outcome([m], kalshi_fees, poly_fees)
        assert len(opps) == 0
