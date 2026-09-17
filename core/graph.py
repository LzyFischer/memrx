"""graph: chunk graph with two edge types.

Nodes     raw chunks (same text and embeddings as baseline).
Entity    chunks sharing an LLM-extracted entity. Entities found in more than
          MAX_ENTITY_DF of the chunks (the two speakers) make no edges;
          MAX_ENTITY_DF = 0 turns entity edges off.
Semantic  each chunk links to its KNN most similar chunks by embedding cosine
          (KNN = 0 turns semantic edges off).

The only LLM cost is entity extraction at build time. Edges are derived from
metadata and embeddings when the store is first queried, so edge settings can
change without rebuilding stores. Retrieval: core/retrieval.py::_retrieve_graph.
"""
import re
from typing import Dict, List, Set

import numpy as np

import config
from core.chunking import map_chunks
from core.entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list

# Unchanged from the previous graph view, so stores cached from it still load.
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
    return sorted({str(e).strip().lower() for e in entities if str(e).strip()})


def extract_chunk_entities(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    for chunk, ents in zip(chunks, map_chunks(lambda c: _extract_entities(llm, c.lossless_restatement),
                                              chunks, config.LLM_WORKERS)):
        chunk.metadata["entities"] = ents
    n_ent = len({e for c in chunks for e in c.metadata["entities"]})
    print(f"  [graph] {len(chunks)} chunks, {n_ent} distinct entities")


def normalize_entity(e: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z ]+", " ", str(e).lower()).split())


class ChunkGraph:
    """neighbors[k] = set of chunk positions linked to chunk k (store order)."""

    def __init__(self, ids: List[str], entries: Dict[str, MemoryEntry], embeddings: np.ndarray,
                 max_entity_df: float = None, knn: int = None):
        max_entity_df = config.GRAPH_MAX_ENTITY_DF if max_entity_df is None else max_entity_df
        knn = config.GRAPH_KNN if knn is None else knn
        n = len(ids)
        self.neighbors: List[Set[int]] = [set() for _ in range(n)]

        # entity edges
        chunks_of: Dict[str, Set[int]] = {}
        for k, eid in enumerate(ids):
            for raw in entries[eid].metadata.get("entities") or []:
                e = normalize_entity(raw)
                if e:
                    chunks_of.setdefault(e, set()).add(k)
        max_df = max(2, int(max_entity_df * n)) if max_entity_df > 0 else 0   # 0: no entity edges
        self.dropped = sorted((e for e, s in chunks_of.items() if len(s) > max_df),
                              key=lambda e: -len(chunks_of[e])) if max_df else []
        for e, members in chunks_of.items():
            if 1 < len(members) <= max_df:
                for k in members:
                    self.neighbors[k] |= members
        for k in range(n):
            self.neighbors[k].discard(k)
        entity_pairs = sum(len(s) for s in self.neighbors) // 2

        # semantic kNN edges (symmetric)
        if knn > 0 and n > 1:
            sims = embeddings @ embeddings.T
            np.fill_diagonal(sims, -np.inf)
            for k, row in enumerate(np.argsort(-sims, axis=1)[:, :min(knn, n - 1)]):
                for j in row:
                    self.neighbors[k].add(int(j))
                    self.neighbors[int(j)].add(k)

        self.n_entity_edges = entity_pairs
        self.n_edges = sum(len(s) for s in self.neighbors) // 2

    def stats(self) -> str:
        return (f"{self.n_edges} edges ({self.n_entity_edges} from entities), "
                f"{len(self.dropped)} frequent entities dropped {self.dropped[:3]}")
