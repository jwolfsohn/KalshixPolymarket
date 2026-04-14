"""Fuzzy text matching for pairing markets across platforms.

Uses Jaccard similarity on word sets + normalized Levenshtein distance.
The rapidfuzz library provides optimized C implementations.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# Words that don't help distinguish markets
STOP_WORDS = frozenset({
    "the", "a", "an", "will", "be", "is", "are", "was", "were",
    "to", "of", "in", "for", "on", "by", "at", "or", "and",
    "this", "that", "it", "its", "with", "as", "from",
    "market", "contract", "event", "prediction",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard index on tokenized word sets."""
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def levenshtein_normalized(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (1 = identical, 0 = completely different)."""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if not a_lower or not b_lower:
        return 0.0
    return 1.0 - Levenshtein.normalized_distance(a_lower, b_lower)


def token_sort_ratio(a: str, b: str) -> float:
    """Token sort ratio via rapidfuzz (order-insensitive similarity)."""
    return fuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0


def combined_similarity(
    a: str,
    b: str,
    jaccard_weight: float = 0.4,
    levenshtein_weight: float = 0.3,
    token_sort_weight: float = 0.3,
) -> float:
    """Weighted combination of similarity metrics.

    Jaccard handles synonym/word overlap well.
    Levenshtein catches character-level similarity.
    Token sort ratio handles reordered words.
    """
    j = jaccard_similarity(a, b)
    l = levenshtein_normalized(a, b)
    t = token_sort_ratio(a, b)
    return jaccard_weight * j + levenshtein_weight * l + token_sort_weight * t
