"""
MemoryStore — unified in-memory backend for the 2a ablation.

Design note (important for the validity of the 2a comparison):
-----------------------------------------------------------------
The original repo uses LanceDB (database/vector_store.py) as the retrieval
backend ONLY for the summary-granularity pipeline. For the 2a experiment we
need FOUR conditions (baseline, summary, augmentation, graph) that are
comparable to each other — i.e. the only thing that should differ between
them is the treatment applied at construction time, not the retrieval
backend itself. Swapping in a different DB/index per condition would
confound "treatment effect" with "backend implementation effect".

So this module replaces LanceDB with a small, dependency-light in-memory
store (numpy cosine similarity + a plain adjacency dict for graph edges)
that ALL FOUR conditions use identically. LoCoMo conversations are small
(a few hundred memory units per conversation at most), so there's no
performance reason to use a real vector DB here — plain numpy is more than
enough for the ablation and easier to reason about.

If you want the results directly comparable to a LanceDB-backed run
elsewhere in the repo, be aware this is a deliberate substitution.

Also hosts the sparse (BM25) index used by augmentation="keywords" and the
graph adjacency used by graph=* — see core/bm25.py and core/fusion.py for
how these get combined with the dense (embedding) channel at retrieval
time (Reciprocal Rank Fusion, not raw score addition — see core/fusion.py
for why). augmentation="note" needs none of this — its LLM-extracted note
is appended directly onto the entry's embedded text at construction time
(core/augmentation_builder.py::augment_note), so it's retrieved with plain
semantic_search like baseline/summary.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set

from core.bm25 import SimpleBM25, tokenize
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel


class MemoryStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.entries: Dict[str, MemoryEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        # graph=* dimension only: entry_id -> set of neighbor entry_ids.
        # Populated by core/graph_builder.py. Empty / unused for the other
        # three dimensions.
        self.graph: Dict[str, Set[str]] = {}
        # augmentation="keywords" only: lazily built BM25 index over each
        # entry's LLM-extracted metadata["keywords"] (falls back to the raw
        # text for entries without that metadata — see _ensure_bm25). Built
        # on first use, not at construction time.
        self._bm25: Optional[SimpleBM25] = None
        self._bm25_ids: List[str] = []

    # ------------------------------------------------------------------
    def add(self, entry: MemoryEntry) -> None:
        self.entries[entry.entry_id] = entry
        self._embeddings[entry.entry_id] = self.embedding_model.encode_single(
            entry.lossless_restatement, is_query=False
        )
        self.graph.setdefault(entry.entry_id, set())
        self._bm25 = None  # invalidate cached index

    def add_batch(self, entries: List[MemoryEntry]) -> None:
        if not entries:
            return
        vecs = self.embedding_model.encode_documents(
            [e.lossless_restatement for e in entries]
        )
        for e, v in zip(entries, vecs):
            self.entries[e.entry_id] = e
            self._embeddings[e.entry_id] = v
            self.graph.setdefault(e.entry_id, set())
        self._bm25 = None  # invalidate cached index

    def add_edge(self, a: str, b: str, bidirectional: bool = True) -> None:
        if a not in self.entries or b not in self.entries:
            return
        self.graph.setdefault(a, set()).add(b)
        if bidirectional:
            self.graph.setdefault(b, set()).add(a)

    # ------------------------------------------------------------------
    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Plain cosine-similarity top-k. Used directly by baseline/summary,
        and as one of the fused channels for augmentation/graph retrieval."""
        return self.semantic_search_scored(query, top_k=top_k)[0]

    def semantic_search_scored(self, query: str, top_k: int = 5):
        if not self.entries:
            return [], []
        q = self.embedding_model.encode_single(query, is_query=True)
        ids = list(self.entries.keys())
        mat = np.stack([self._embeddings[i] for i in ids])
        # embeddings are already L2-normalized by EmbeddingModel.encode()
        sims = mat @ q
        order = np.argsort(-sims)[:top_k]
        top_ids = [ids[i] for i in order]
        top_scores = [float(sims[i]) for i in order]
        return [self.entries[i] for i in top_ids], top_scores

    # ------------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        ids = list(self.entries.keys())
        tokenized = []
        for i in ids:
            entry = self.entries[i]
            keywords = entry.metadata.get("keywords")
            # augmentation="keywords" sets metadata["keywords"] via an LLM
            # extraction pass (core/augmentation_builder.py::augment_keywords)
            # — BM25 indexes those extracted terms, not the raw chunk text.
            # Other dimensions never set this key, so they fall back to the
            # raw text (keeps bm25_ranked_ids usable/sane even if called
            # against a store that wasn't built with augmentation="keywords").
            text = " ".join(keywords) if keywords else entry.lossless_restatement
            tokenized.append(tokenize(text))
        self._bm25 = SimpleBM25(tokenized)
        self._bm25_ids = ids

    def bm25_ranked_ids(self, query: str, top_k: int) -> List[str]:
        """Sparse-retrieval channel (augmentation="keywords"). Indexes each
        entry's LLM-extracted metadata["keywords"] — see _ensure_bm25."""
        self._ensure_bm25()
        if not self._bm25_ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self._bm25_ids[i] for i in order if scores[i] > 0]

    # ------------------------------------------------------------------
    def neighbors(self, entry_id: str, hops: int = 1) -> List[str]:
        """BFS out to `hops` hops in the graph dimension."""
        frontier = {entry_id}
        visited = {entry_id}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                nxt |= self.graph.get(n, set())
            nxt -= visited
            visited |= nxt
            frontier = nxt
        visited.discard(entry_id)
        return list(visited)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def all_entries(self) -> List[MemoryEntry]:
        return list(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)
