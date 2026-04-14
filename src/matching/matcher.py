"""Three-stage market matching pipeline.

Stage 1: Cache lookup (skip re-matching known pairs)
Stage 2: Structural pre-filter (reject obviously different markets)
Stage 3: Fuzzy text matching (Jaccard + Levenshtein + token sort)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.arb.models import Market, MatchedMarket
from src.matching.cache import MatchCache
from src.matching.fuzzy import combined_similarity

logger = logging.getLogger(__name__)


def _assess_settlement_risk(k: Market, p: Market) -> str:
    """Estimate whether two markets have the same settlement criteria.

    Returns "identical", "similar", or "different_criteria".
    """
    if k.settlement_date and p.settlement_date:
        delta = abs((k.settlement_date - p.settlement_date).total_seconds())
        if delta < 3600:  # within 1 hour
            return "identical"
        elif delta < 86400:  # within 1 day
            return "similar"
        else:
            return "different_criteria"
    # If one or both are missing settlement dates, can't confirm
    return "similar"


def _match_outcomes(k: Market, p: Market) -> dict[str, str]:
    """Map Kalshi outcome IDs to Polymarket outcome IDs.

    For binary markets: match yes↔yes and no↔no.
    For multi-outcome: use fuzzy matching on outcome labels.
    """
    mapping = {}

    if k.outcome_count == 2 and p.outcome_count == 2:
        # Binary: match by side
        k_by_side = {o.side: o for o in k.outcomes}
        p_by_side = {o.side: o for o in p.outcomes}
        for side in ("yes", "no"):
            if side in k_by_side and side in p_by_side:
                mapping[k_by_side[side].outcome_id] = p_by_side[side].outcome_id
        return mapping

    # Multi-outcome: match by label similarity
    p_unmatched = list(p.outcomes)
    for k_out in k.outcomes:
        best_score = 0.0
        best_match = None
        for p_out in p_unmatched:
            score = combined_similarity(k_out.label, p_out.label)
            if score > best_score:
                best_score = score
                best_match = p_out
        if best_match and best_score > 0.5:
            mapping[k_out.outcome_id] = best_match.outcome_id
            p_unmatched.remove(best_match)

    return mapping


class MarketMatcher:
    """Match identical events across Kalshi and Polymarket."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        cache: MatchCache | None = None,
        structural_prefilter: bool = True,
    ):
        self.threshold = similarity_threshold
        self.cache = cache
        self.structural_prefilter = structural_prefilter

    def match_markets(
        self,
        kalshi_markets: list[Market],
        poly_markets: list[Market],
    ) -> list[MatchedMarket]:
        """Run the three-stage matching pipeline."""
        # Pre-index: build keyword buckets to avoid O(n*m) full scan
        # Each Polymarket market gets indexed by its significant words
        poly_by_word: dict[str, list[Market]] = {}
        for p in poly_markets:
            words = set(w.lower() for w in p.title.split() if len(w) > 3)
            for w in words:
                poly_by_word.setdefault(w, []).append(p)

        matches: list[MatchedMarket] = []
        poly_matched_ids: set[str] = set()

        for k in kalshi_markets:
            best_match: MatchedMarket | None = None
            best_score = 0.0

            # Find candidate Polymarket markets that share at least one word
            k_words = set(w.lower() for w in k.title.split() if len(w) > 3)
            candidates: set[str] = set()
            for w in k_words:
                for p in poly_by_word.get(w, []):
                    candidates.add(p.market_id)

            # Build lookup for candidates
            poly_lookup = {p.market_id: p for p in poly_markets}

            for p_id in candidates:
                if p_id in poly_matched_ids:
                    continue
                p = poly_lookup[p_id]

                # Stage 1: Cache lookup
                if self.cache:
                    cached = self.cache.get(k.market_id, p.market_id)
                    if cached and cached["similarity"] >= self.threshold:
                        match = MatchedMarket(
                            kalshi=k,
                            polymarket=p,
                            similarity_score=cached["similarity"],
                            match_method="cache",
                            settlement_risk=cached["settlement_risk"],
                            outcome_mapping=cached["outcome_mapping"],
                        )
                        if cached["similarity"] > best_score:
                            best_score = cached["similarity"]
                            best_match = match
                        continue

                # Stage 2: Structural pre-filter
                if self.structural_prefilter:
                    if not self._passes_structural_filter(k, p):
                        continue

                # Stage 3: Fuzzy text matching
                title_sim = combined_similarity(k.title, p.title)

                if title_sim >= self.threshold and title_sim > best_score:
                    outcome_mapping = _match_outcomes(k, p)
                    settlement_risk = _assess_settlement_risk(k, p)

                    match = MatchedMarket(
                        kalshi=k,
                        polymarket=p,
                        similarity_score=title_sim,
                        match_method="structural+fuzzy",
                        settlement_risk=settlement_risk,
                        outcome_mapping=outcome_mapping,
                    )
                    best_score = title_sim
                    best_match = match

                    # Cache the result
                    if self.cache:
                        self.cache.put(
                            k.market_id,
                            p.market_id,
                            title_sim,
                            outcome_mapping,
                            settlement_risk,
                        )

            if best_match:
                matches.append(best_match)
                poly_matched_ids.add(best_match.polymarket.market_id)

        logger.info(
            f"Matched {len(matches)} markets from "
            f"{len(kalshi_markets)} Kalshi x {len(poly_markets)} Polymarket"
        )
        return matches

    def _passes_structural_filter(self, k: Market, p: Market) -> bool:
        """Quick rejection: different structure means different market."""
        # Outcome count mismatch (binary vs multi-outcome)
        if k.outcome_count > 0 and p.outcome_count > 0:
            if k.outcome_count != p.outcome_count:
                return False

        # Settlement date too far apart (> 7 days)
        if k.settlement_date and p.settlement_date:
            delta = abs((k.settlement_date - p.settlement_date).days)
            if delta > 7:
                return False

        return True
