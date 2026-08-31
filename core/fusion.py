"""
Reciprocal Rank Fusion — merges multiple independently-ranked id lists into
one ranking, instead of trying to add heterogeneous scores (cosine
similarity, BM25 score, day-distance) on the same numeric scale, which
isn't principled since they live in different ranges/units.

score(id) = sum_over_rankings( weight * 1 / (k + rank_in_that_ranking + 1) )

Follows the formula used by Cognis (arXiv:2604.19771):
    score_fused = 0.70 * RRF_vector + 0.30 * RRF_BM25
"""
from typing import Dict, List, Optional


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    weights: Optional[List[float]] = None,
    k: int = 10,
) -> List[str]:
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings)

    scores: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, entry_id in enumerate(ranking):
            scores[entry_id] = scores.get(entry_id, 0.0) + w * (1.0 / (k + rank + 1))
    return sorted(scores.keys(), key=lambda eid: -scores[eid])
