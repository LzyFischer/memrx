"""graph=entity: an LLM extracts named entities per raw chunk; two chunks get
an edge if they share an entity. Node text is the same raw chunk as baseline.

The most frequent entities (e.g. a speaker's own name) are excluded from
edge-building: otherwise they connect almost every pair of chunks and 1-hop
expansion returns most of the store. Edges are binary, so this hard-excludes
rather than down-weights (the graph analogue of BM25's IDF).
"""
import math
from itertools import combinations
from typing import Dict, List

from core.entry import MemoryEntry
from core.store import MemoryStore
from utils.llm_client import LLMClient, coerce_json_list

_PROMPT = """Extract all named entities (people, places, organizations, specific objects/events) from this dialogue excerpt.

Dialogue:
{text}

Return a JSON array of entity name strings (deduplicated, singular canonical form). Return ONLY the JSON array."""


def _extract_entities(llm: LLMClient, text: str) -> List[str]:
    try:
        resp = llm.chat_completion([{"role": "user", "content": _PROMPT.format(text=text)}],
                                   temperature=0.0)
        entities = coerce_json_list(llm.extract_json(resp), context="entity extraction")
    except Exception:
        entities = []
    return [str(e).strip().lower() for e in entities if str(e).strip()]


def _high_frequency_entities(index: Dict[str, List[str]], top_frac: float, min_entities: int) -> set:
    """Top `top_frac` of entities by chunk count; skipped when there are too
    few distinct entities for that cut to mean anything."""
    if len(index) < min_entities:
        return set()
    n_exclude = max(1, math.ceil(len(index) * top_frac))
    ranked = sorted(index.items(), key=lambda kv: -len(kv[1]))
    return {name for name, _ in ranked[:n_exclude]}


def build_entity_graph(store: MemoryStore, chunks: List[MemoryEntry], llm: LLMClient,
                       top_frequency_percentile: float = 0.01,
                       min_entities_for_filtering: int = 20) -> None:
    index: Dict[str, List[str]] = {}
    for chunk in chunks:
        entities = _extract_entities(llm, chunk.lossless_restatement)
        chunk.metadata["entities"] = entities
        for e in entities:
            index.setdefault(e, []).append(chunk.entry_id)

    excluded = _high_frequency_entities(index, top_frequency_percentile, min_entities_for_filtering)
    if excluded:
        top = sorted(excluded, key=lambda e: -len(index[e]))[:5]
        print(f"  [graph=entity] excluding {len(excluded)} high-frequency entities: {top}")

    for name, ids in index.items():
        if name in excluded or len(ids) < 2:
            continue
        for a, b in combinations(set(ids), 2):
            store.add_edge(a, b)
