"""Tests for fuzzy matching and market matcher."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arb.models import Market, Outcome
from src.matching.fuzzy import (
    combined_similarity,
    jaccard_similarity,
    levenshtein_normalized,
    token_sort_ratio,
)
from src.matching.matcher import MarketMatcher


def _make_market(
    platform: str,
    market_id: str,
    title: str,
    settlement_date: datetime | None = None,
    category: str = "politics",
) -> Market:
    return Market(
        platform=platform,
        market_id=market_id,
        title=title,
        category=category,
        settlement_date=settlement_date,
        outcome_count=2,
        outcomes=[
            Outcome(
                platform=platform,
                market_id=market_id,
                outcome_id=f"{market_id}_yes",
                label="Yes",
                side="yes",
                best_bid=0.0,
                best_ask=0.50,
                bid_depth=100,
                ask_depth=100,
                category=category,
            ),
            Outcome(
                platform=platform,
                market_id=market_id,
                outcome_id=f"{market_id}_no",
                label="No",
                side="no",
                best_bid=0.0,
                best_ask=0.50,
                bid_depth=100,
                ask_depth=100,
                category=category,
            ),
        ],
    )


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("apple banana", "cat dog fish") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity(
            "Will the Fed cut rates in June",
            "Fed rate cut June 2026",
        )
        assert sim > 0.3

    def test_stop_words_ignored(self):
        # "the" and "will" are stop words
        sim = jaccard_similarity(
            "the Fed will cut rates",
            "Fed cut rates",
        )
        assert sim == 1.0


class TestLevenshteinNormalized:
    def test_identical(self):
        assert levenshtein_normalized("hello", "hello") == 1.0

    def test_one_char_diff(self):
        sim = levenshtein_normalized("hello", "hallo")
        assert 0.7 < sim < 1.0

    def test_completely_different(self):
        sim = levenshtein_normalized("abc", "xyz")
        assert sim == 0.0


class TestTokenSortRatio:
    def test_reordered_words(self):
        sim = token_sort_ratio(
            "Fed rate cut June",
            "June Fed cut rate",
        )
        assert sim > 0.95

    def test_different_text(self):
        sim = token_sort_ratio("apple pie", "car engine")
        assert sim < 0.5


class TestCombinedSimilarity:
    def test_identical_markets(self):
        sim = combined_similarity(
            "Will the Fed cut rates by 25bps in June 2026?",
            "Will the Fed cut rates by 25bps in June 2026?",
        )
        assert sim > 0.95

    def test_similar_markets(self):
        sim = combined_similarity(
            "Will the Fed cut rates in June 2026?",
            "Federal Reserve June 2026 rate cut",
        )
        assert sim > 0.4

    def test_different_markets(self):
        sim = combined_similarity(
            "Will Bitcoin reach $100k?",
            "Who will win the presidential election?",
        )
        assert sim < 0.25


class TestMarketMatcher:
    def test_matches_identical_titles(self):
        k = [_make_market("kalshi", "k1", "Will the Fed cut rates in June 2026?")]
        p = [_make_market("polymarket", "p1", "Will the Fed cut rates in June 2026?")]
        matcher = MarketMatcher(similarity_threshold=0.5)
        matches = matcher.match_markets(k, p)
        assert len(matches) == 1
        assert matches[0].kalshi.market_id == "k1"
        assert matches[0].polymarket.market_id == "p1"
        assert matches[0].similarity_score > 0.9

    def test_matches_similar_titles(self):
        k = [_make_market("kalshi", "k1", "Will the Fed cut interest rates in June 2026?")]
        p = [_make_market("polymarket", "p1", "Federal Reserve rate cut June 2026")]
        matcher = MarketMatcher(similarity_threshold=0.4)
        matches = matcher.match_markets(k, p)
        assert len(matches) == 1

    def test_rejects_different_markets(self):
        k = [_make_market("kalshi", "k1", "Bitcoin price above 100k by December")]
        p = [_make_market("polymarket", "p1", "Who will win the NBA Finals?")]
        matcher = MarketMatcher(similarity_threshold=0.5)
        matches = matcher.match_markets(k, p)
        assert len(matches) == 0

    def test_structural_filter_rejects_mismatched_dates(self):
        now = datetime.utcnow()
        k = [_make_market("kalshi", "k1", "Same title", settlement_date=now)]
        p = [
            _make_market(
                "polymarket",
                "p1",
                "Same title",
                settlement_date=now + timedelta(days=30),
            )
        ]
        matcher = MarketMatcher(similarity_threshold=0.5, structural_prefilter=True)
        matches = matcher.match_markets(k, p)
        assert len(matches) == 0

    def test_one_to_one_matching(self):
        """Each Polymarket market should be matched at most once."""
        k = [
            _make_market("kalshi", "k1", "Fed rate cut June"),
            _make_market("kalshi", "k2", "Fed rate cut June"),  # duplicate
        ]
        p = [_make_market("polymarket", "p1", "Fed rate cut June")]
        matcher = MarketMatcher(similarity_threshold=0.5)
        matches = matcher.match_markets(k, p)
        # Only one match should exist (p1 matched to whichever k comes first)
        assert len(matches) == 1

    def test_outcome_mapping_binary(self):
        k = [_make_market("kalshi", "k1", "Same event")]
        p = [_make_market("polymarket", "p1", "Same event")]
        matcher = MarketMatcher(similarity_threshold=0.5)
        matches = matcher.match_markets(k, p)
        assert len(matches) == 1
        m = matches[0]
        assert "k1_yes" in m.outcome_mapping
        assert m.outcome_mapping["k1_yes"] == "p1_yes"
