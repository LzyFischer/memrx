"""View-aware retrieval: retrieve(store, query, condition, top_k).

  baseline / summary      top-k cosine
  augmentation            dense + BM25 over raw+attributes text, weighted RRF (0.7 / 0.3)
  graph                   dense + relevance passed from the top dense chunks to their
                          entity / semantic neighbours (core/graph.py)
"""
from typing import Dict, List, Optional

import numpy as np

import config

from core.conditions import Condition
from core.entry import MemoryEntry
from core.store import MemoryStore


def retrieve(store: MemoryStore, query: str, condition: Condition, top_k: int = 5) -> List[MemoryEntry]:
    if condition.dimension == "augmentation":
        return _retrieve_keywords_hybrid(store, query, top_k)
    if condition.dimension == "graph":
        return _retrieve_graph(store, query, top_k)
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


def _retrieve_graph(store: MemoryStore, query: str, top_k: int, seeds: int = None,
                    lam: float = None) -> List[MemoryEntry]:
    """score(v) = cos(q, v) + lam * max_{i in seeds, i ~ v} cos(q, i)

    The top `seeds` chunks by cosine pass their own similarity to their graph
    neighbours (seeds can boost each other). A chunk that is not near the query
    but is linked to a strong match moves up. lam = 0 is exactly baseline.
    """
    seeds = config.GRAPH_SEEDS if seeds is None else seeds
    lam = config.GRAPH_LAMBDA if lam is None else lam
    if lam <= 0 or seeds <= 0:
        return store.semantic_search(query, top_k=top_k)
    ids, E = store.embedding_matrix()
    if not ids:
        return []
    cos = E @ store.embedding_model.encode_single(query, is_query=True)
    graph = store.chunk_graph()
    passed = np.zeros(len(ids))
    for i in np.argsort(-cos, kind="stable")[:seeds]:
        for j in graph.neighbors[i]:
            passed[j] = max(passed[j], cos[i])
    score = cos.astype(np.float64) + lam * passed
    return [store.entries[ids[i]] for i in np.argsort(-score, kind="stable")[:top_k]]
