"""MemoryStore: one in-memory retrieval backend shared by every view.

All views use the same backend (numpy cosine similarity, a BM25 index, and an
entity graph) so that the only difference between views is
the treatment applied at construction time, not the retrieval implementation.
LoCoMo conversations have a few hundred units at most, so no vector DB is
needed.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from core.bm25 import SimpleBM25, tokenize
from core.entry import MemoryEntry
from utils.embedding import EmbeddingModel


class MemoryStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.entries: Dict[str, MemoryEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._bm25: Optional[SimpleBM25] = None    # built lazily
        self._bm25_ids: List[str] = []
        self._graph = None                         # graph view, built lazily

    # ------------------------------------------------------------------ build
    def add_batch(self, entries: List[MemoryEntry]) -> None:
        if not entries:
            return
        vecs = self.embedding_model.encode_documents([e.lossless_restatement for e in entries])
        for e, v in zip(entries, vecs):
            self.entries[e.entry_id] = e
            self._embeddings[e.entry_id] = v
        self._bm25 = None
        self._graph = None

    # ------------------------------------------------------------------ dense
    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        return self.semantic_search_scored(query, top_k=top_k)[0]

    def semantic_search_scored(self, query: str, top_k: int = 5):
        if not self.entries:
            return [], []
        q = self.embedding_model.encode_single(query, is_query=True)
        ids = list(self.entries.keys())
        sims = np.stack([self._embeddings[i] for i in ids]) @ q   # embeddings are L2-normalised
        order = np.argsort(-sims)[:top_k]
        return [self.entries[ids[i]] for i in order], [float(sims[i]) for i in order]

    # ------------------------------------------------------------------ sparse
    def bm25_ranked_ids(self, query: str, top_k: int) -> List[str]:
        """BM25 over each entry's text (for augmentation this includes its attributes)."""
        if self._bm25 is None:
            self._bm25_ids = list(self.entries.keys())
            self._bm25 = SimpleBM25([tokenize(self.entries[i].lossless_restatement)
                                     for i in self._bm25_ids])
        if not self._bm25_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self._bm25_ids[i] for i in order if scores[i] > 0]

    # ------------------------------------------------------------------ graph
    def embedding_matrix(self):
        """(ids in store order, (N, d) L2-normalised embeddings)."""
        ids = list(self.entries.keys())
        return ids, (np.stack([self._embeddings[i] for i in ids]) if ids else np.zeros((0, 1)))

    def chunk_graph(self):
        """Entity + semantic kNN chunk graph (graph view), built on first use."""
        if self._graph is None:
            from core.graph import ChunkGraph
            ids, E = self.embedding_matrix()
            self._graph = ChunkGraph(ids, self.entries, E)
        return self._graph

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ cache
    def save(self, path: str) -> None:
        ids = list(self.entries.keys())
        blob = {
            "entries": [self.entries[i].model_dump() for i in ids],
            "embs": np.stack([self._embeddings[i] for i in ids]) if ids else np.zeros((0, 1)),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(blob, f)

    @classmethod
    def load(cls, path: str, embedding_model: EmbeddingModel) -> "MemoryStore":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        store = cls(embedding_model)
        for k, d in enumerate(blob["entries"]):
            e = MemoryEntry(**d)
            store.entries[e.entry_id] = e
            store._embeddings[e.entry_id] = blob["embs"][k]
        # older cache files also carry a "graph" edge dict, which is no longer used
        return store
