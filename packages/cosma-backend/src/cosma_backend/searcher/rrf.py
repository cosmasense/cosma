"""
Reciprocal Rank Fusion (RRF) implementation for combining ranked search results.

RRF is a method for combining multiple ranked lists that doesn't require score
normalization. It works by assigning each item a score based on its rank in each
list: score = 1 / (k + rank), then summing across all lists.

Reference: Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher.
"Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
SIGIR 2009.
"""

from typing import TypeVar, Hashable

T = TypeVar("T")


def compute_rrf_scores(
    ranked_lists: list[list[tuple[Hashable, T]]],
    k: int = 60
) -> dict[Hashable, float]:
    """
    Compute RRF scores for items across multiple ranked lists.

    RRF score = sum(1 / (k + rank)) for each ranking system where the item appears.
    Items appearing in multiple lists get higher scores.

    Args:
        ranked_lists: List of ranked lists, where each list contains (id, data) tuples
                     ordered by rank (best first, 0-indexed)
        k: Smoothing constant (default 60, commonly used value)

    Returns:
        Dictionary mapping item IDs to their RRF scores
    """
    scores: dict[Hashable, float] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, _data) in enumerate(ranked_list):
            # RRF formula: 1 / (k + rank)
            # rank is 0-indexed, so first item gets 1/(k+0) = 1/k
            rrf_contribution = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_contribution

    return scores


def merge_with_rrf(
    semantic_results: list[tuple[Hashable, T, float]],
    keyword_results: list[tuple[Hashable, T, float]],
    k: int = 60
) -> list[tuple[Hashable, T, float, str, int | None, int | None]]:
    """
    Merge semantic and keyword search results using Reciprocal Rank Fusion.

    Args:
        semantic_results: List of (id, data, score) tuples from semantic search,
                         ordered by relevance (best first)
        keyword_results: List of (id, data, score) tuples from keyword search,
                        ordered by relevance (best first)
        k: RRF smoothing constant (default 60)

    Returns:
        List of (id, data, rrf_score, match_type, semantic_rank, keyword_rank) tuples,
        sorted by RRF score descending.

        match_type is one of: "hybrid", "semantic", "keyword"
    """
    # Build ranked lists for RRF computation
    semantic_ranked = [(item_id, data) for item_id, data, _score in semantic_results]
    keyword_ranked = [(item_id, data) for item_id, data, _score in keyword_results]

    # Compute RRF scores
    rrf_scores = compute_rrf_scores([semantic_ranked, keyword_ranked], k=k)

    # Build lookup maps for data and ranks
    semantic_data: dict[Hashable, T] = {}
    semantic_ranks: dict[Hashable, int] = {}
    for rank, (item_id, data, _score) in enumerate(semantic_results):
        semantic_data[item_id] = data
        semantic_ranks[item_id] = rank

    keyword_data: dict[Hashable, T] = {}
    keyword_ranks: dict[Hashable, int] = {}
    for rank, (item_id, data, _score) in enumerate(keyword_results):
        keyword_data[item_id] = data
        keyword_ranks[item_id] = rank

    # Build merged results
    merged: list[tuple[Hashable, T, float, str, int | None, int | None]] = []

    for item_id, rrf_score in rrf_scores.items():
        # Get data from whichever source has it (prefer semantic for completeness)
        data = semantic_data.get(item_id) or keyword_data.get(item_id)

        # Determine match type
        in_semantic = item_id in semantic_data
        in_keyword = item_id in keyword_data

        if in_semantic and in_keyword:
            match_type = "hybrid"
        elif in_semantic:
            match_type = "semantic"
        else:
            match_type = "keyword"

        # Get ranks (None if not in that list)
        sem_rank = semantic_ranks.get(item_id)
        kw_rank = keyword_ranks.get(item_id)

        merged.append((item_id, data, rrf_score, match_type, sem_rank, kw_rank))

    # Sort by RRF score descending
    merged.sort(key=lambda x: x[2], reverse=True)

    return merged
