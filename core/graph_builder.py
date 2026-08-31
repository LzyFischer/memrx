"""
Graph Builder — semantic / entity edges over raw chunks.

Both write edges into MemoryStore.graph (see core/memory_store.py).
Node TEXT is never touched (same raw chunks as baseline) — only the edge
set differs between the two graph variants, and vs. no edges at all
(baseline / augmentation, where store.graph stays all-empty sets).

causal (build_causal_graph) has been removed: on the LoCoMo scale used here
it needed 2-hop traversal to be useful (a fixed lookback=5 backward-only
candidate window, one LLM call per chunk) and consistently underperformed
semantic/entity in the 2a runs without a clear story for why the graph
dimension specifically should carry causal reasoning rather than the QA
prompt itself. Kept out for now rather than reworked; see git history if
it needs to come back.
"""
from itertools import combinations
from typing import Dict, List

import numpy as np

from core.memory_store import MemoryStore
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list


# ---------------------------------------------------------------------- #
# semantic: no LLM call needed — connect each chunk to its top-N nearest
# neighbors by embedding cosine similarity (excluding itself). This is the
# "GraphRAG-lite" / kNN-graph baseline referenced in the literature (e.g.
# semantic graphs built directly over embedding space rather than LLM-
# extracted relations).
# ---------------------------------------------------------------------- #
def build_semantic_graph(store: MemoryStore, top_n: int = 3, min_sim: float = 0.3) -> None:
    ids = list(store.entries.keys())
    if len(ids) < 2:
        return
    mat = np.stack([store._embeddings[i] for i in ids])  # already normalized
    sims = mat @ mat.T
    for row, entry_id in enumerate(ids):
        order = np.argsort(-sims[row])
        added = 0
        for col in order:
            if col == row:
                continue
            if sims[row, col] < min_sim:
                break
            store.add_edge(entry_id, ids[col])
            added += 1
            if added >= top_n:
                break


# ---------------------------------------------------------------------- #
# entity: LLM extracts named entities per chunk; two chunks get an edge if
# they share at least one entity (classic entity co-occurrence graph, the
# simplified single-hop version of HippoRAG / GraphRAG-style entity graphs).
#
# High-frequency filtering: without this, a protagonist's name (e.g. the
# speaker themselves) can appear in most chunks once enough sessions
# accumulate, and connect(a, b) for every pair sharing that entity turns
# combinations(n, 2) into tens of thousands of edges — the graph collapses
# toward "everything connects to everything", which defeats the point of
# having a graph at all and makes 1-hop expansion at retrieval time return
# almost the whole memory store. Same problem BM25's IDF term solves for
# keyword search (a word in every document carries no discriminative
# power) — here we hard-exclude instead of just downweighting, since edges
# are binary (exist or not), there's no continuous score to downweight.
# ---------------------------------------------------------------------- #
def build_entity_graph(
    store: MemoryStore, chunks: List[MemoryEntry], llm: LLMClient,
    top_frequency_percentile: float = 0.01, min_entities_for_filtering: int = 20,
) -> None:
    entity_index: Dict[str, List[str]] = {}  # entity_name -> [entry_id, ...]
    for chunk in chunks:
        prompt = f"""Extract all named entities (people, places, organizations, specific objects/events) from this dialogue excerpt.

Dialogue:
{chunk.lossless_restatement}

Return a JSON array of entity name strings (deduplicated, singular canonical form). Return ONLY the JSON array."""
        try:
            resp = llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
            entities = coerce_json_list(llm.extract_json(resp), context="entity extraction")
        except Exception:
            entities = []
        entities = [str(e).strip().lower() for e in entities if str(e).strip()]
        chunk.metadata["entities"] = entities
        for e in entities:
            entity_index.setdefault(e, []).append(chunk.entry_id)

    excluded = _high_frequency_entities(
        entity_index, top_frequency_percentile, min_entities_for_filtering
    )
    if excluded:
        print(f"  [graph=entity] excluding {len(excluded)} high-frequency entities "
              f"(top {top_frequency_percentile:.0%} by chunk count) from edge-building: "
              f"{sorted(excluded, key=lambda e: -len(entity_index[e]))[:5]}...")

    for name, entry_ids in entity_index.items():
        if name in excluded or len(entry_ids) < 2:
            continue
        for a, b in combinations(set(entry_ids), 2):
            store.add_edge(a, b)


def _high_frequency_entities(
    entity_index: Dict[str, List[str]], top_frequency_percentile: float,
    min_entities_for_filtering: int,
) -> set:
    """Returns the set of entity names whose document frequency (number of
    distinct chunks they appear in) puts them in the top
    `top_frequency_percentile` of all extracted entities.

    Skipped entirely (returns empty set) when there are too few distinct
    entities for "top 1%" to be a meaningful cut — on a short conversation
    with only a handful of entities, forcibly dropping the single most
    frequent one (e.g. a speaker's own name) does more harm than good.
    `min_entities_for_filtering` guards against that; raise it if you find
    filtering kicking in too eagerly on small conversations."""
    import math

    n = len(entity_index)
    if n < min_entities_for_filtering:
        return set()

    n_exclude = max(1, math.ceil(n * top_frequency_percentile))
    ranked = sorted(entity_index.items(), key=lambda kv: -len(kv[1]))
    return {name for name, _ in ranked[:n_exclude]}
