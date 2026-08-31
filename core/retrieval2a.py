"""
Dimension-aware retrieval for the 2a ablation.

retrieve(store, query, condition, llm, top_k) -> List[MemoryEntry]

  - baseline / summary(session_level, fine_grained): plain top-k cosine.
  - augmentation(keywords): semantic (dense) + BM25 (sparse) run as two
    INDEPENDENT channels, fused via Reciprocal Rank Fusion (core/fusion.py).
    The BM25 channel indexes each entry's LLM-extracted
    metadata["keywords"] (core/augmentation_builder.py::augment_keywords),
    not the raw chunk text — see core/memory_store.py::_ensure_bm25.
    Matches Mem0's production design (semantic + BM25 + entity scored in
    parallel, fused) and Cognis's RRF formula
    (score_fused = 0.70*RRF_vector + 0.30*RRF_BM25).
  - augmentation(note): no special retrieval path — the LLM-extracted
    keywords+context note was appended onto the raw chunk text at
    construction time (core/augmentation_builder.py::augment_note), so it's
    already part of what got embedded. Plain semantic search, same as
    baseline/summary.
  - graph(semantic/entity): embedding search picks SEED nodes, then the
    store's graph is traversed 1 hop to pull in connected nodes.

augmentation="temporal" and augmentation="causal" have both been removed
from this dimension — see core/augmentation_builder.py for the current
"keywords" / "note" design and why temporal's independent-signal-plus-RRF
machinery was dropped. graph="causal" has also been removed — see
core/graph_builder.py.
"""
from typing import List, Optional

from config_2a import Condition
from core.fusion import reciprocal_rank_fusion
from core.memory_store import MemoryStore
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient


def retrieve(
    store: MemoryStore, query: str, condition: Condition,
    llm: Optional[LLMClient] = None, top_k: int = 5,
) -> List[MemoryEntry]:
    if condition.dimension in ("baseline", "summary"):
        return store.semantic_search(query, top_k=top_k)

    if condition.dimension == "augmentation":
        if condition.augmentation == "keywords":
            return _retrieve_keywords_hybrid(store, query, top_k)
        if condition.augmentation == "note":
            return store.semantic_search(query, top_k=top_k)

    if condition.dimension == "graph":
        return _retrieve_graph(store, query, top_k, hops=1)

    return store.semantic_search(query, top_k=top_k)


# ---------------------------------------------------------------------- #
def _retrieve_keywords_hybrid(store: MemoryStore, query: str, top_k: int) -> List[MemoryEntry]:
    pool = min(len(store), max(top_k * 4, 10))
    sem_entries, _ = store.semantic_search_scored(query, top_k=pool)
    sem_ids = [e.entry_id for e in sem_entries]
    bm25_ids = store.bm25_ranked_ids(query, top_k=pool)

    fused_ids = reciprocal_rank_fusion([sem_ids, bm25_ids], weights=[0.7, 0.3])
    out = [store.get(eid) for eid in fused_ids[:top_k]]
    return [e for e in out if e is not None]


# ---------------------------------------------------------------------- #
def _retrieve_graph(store: MemoryStore, query: str, top_k: int, hops: int) -> List[MemoryEntry]:
    seed_k = max(1, top_k // 2)
    seeds = store.semantic_search(query, top_k=seed_k)
    out, seen = list(seeds), {e.entry_id for e in seeds}
    for s in seeds:
        for nid in store.neighbors(s.entry_id, hops=hops):
            if nid not in seen:
                ent = store.get(nid)
                if ent is not None:
                    out.append(ent)
                    seen.add(nid)
            if len(out) >= top_k:
                break
        if len(out) >= top_k:
            break
    return out[:top_k]
