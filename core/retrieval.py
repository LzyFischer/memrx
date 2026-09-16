"""View-aware retrieval: retrieve(store, query, condition, top_k).

  baseline / summary      top-k cosine
  augmentation=keywords   dense + BM25(keywords), fused with weighted RRF (0.7 / 0.3)
  graph=entity            top_k//2 cosine seeds, then 1-hop neighbours
"""
from typing import Dict, List, Optional

from core.conditions import Condition
from core.entry import MemoryEntry
from core.store import MemoryStore


def retrieve(store: MemoryStore, query: str, condition: Condition, top_k: int = 5) -> List[MemoryEntry]:
    if condition.dimension == "augmentation":
        return _retrieve_keywords_hybrid(store, query, top_k)
    if condition.dimension == "graph":
        return _retrieve_graph(store, query, top_k, hops=1)
    return store.semantic_search(query, top_k=top_k)


def reciprocal_rank_fusion(rankings: List[List[str]], weights: Optional[List[float]] = None,
                           k: int = 10) -> List[str]:
    """score(id) = sum_r w_r / (k + rank_r(id) + 1). Rank-based, so cosine and
    BM25 scores never have to share a scale."""
    weights = weights or [1.0] * len(rankings)
    scores: Dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, eid in enumerate(ranking):
            scores[eid] = scores.get(eid, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=lambda eid: -scores[eid])


def _retrieve_keywords_hybrid(store: MemoryStore, query: str, top_k: int) -> List[MemoryEntry]:
    pool = min(len(store), max(top_k * 4, 10))
    sem_ids = [e.entry_id for e in store.semantic_search(query, top_k=pool)]
    bm25_ids = store.bm25_ranked_ids(query, top_k=pool)
    fused = reciprocal_rank_fusion([sem_ids, bm25_ids], weights=[0.7, 0.3])
    return [e for e in (store.get(i) for i in fused[:top_k]) if e is not None]


def _retrieve_graph(store: MemoryStore, query: str, top_k: int, hops: int) -> List[MemoryEntry]:
    seeds = store.semantic_search(query, top_k=max(1, top_k // 2))
    out, seen = list(seeds), {e.entry_id for e in seeds}
    for s in seeds:
        for nid in store.neighbors(s.entry_id, hops=hops):
            if len(out) >= top_k:
                break
            ent = store.get(nid)
            if nid not in seen and ent is not None:
                out.append(ent)
                seen.add(nid)
        if len(out) >= top_k:
            break
    return out[:top_k]
