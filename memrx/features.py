"""Curated JSONL -> router input X (N, d) and metric matrix Y (N, K)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from memrx.probe import PROBE_FEATURE_NAMES, probe_vector


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_matrix(recs: List[dict], views: List[str], metric: str = "f1") -> np.ndarray:
    """(N, K) scores, NaN where a view is missing."""
    return np.array([[np.nan if r[metric].get(v) is None else float(r[metric][v]) for v in views]
                     for r in recs])


class FeatureBuilder:
    """x = [proj(q_emb) ; proj(probe_ctx_emb) ; probe features].

    The projections are fixed seeded Gaussian matrices (fitted to nothing), so
    the 384-d embeddings do not drown out the 7 probe numbers by width alone.
    use_probe=False zeros the context and probe parts (query-only router).
    """

    def __init__(self, d_emb: int, d_q: int = 64, d_ctx: int = 64, seed: int = 0,
                 use_probe: bool = True):
        rng_q, rng_c = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        self.Pq = rng_q.normal(0, 1 / np.sqrt(d_q), size=(d_emb, d_q))
        self.Pc = rng_c.normal(0, 1 / np.sqrt(d_ctx), size=(d_emb, d_ctx))
        self.use_probe = use_probe

    def __call__(self, recs: List[dict]) -> np.ndarray:
        q = np.array([r["q_emb"] for r in recs]) @ self.Pq
        if not self.use_probe:
            return np.hstack([q, np.zeros((len(recs), self.Pc.shape[1] + len(PROBE_FEATURE_NAMES)))])
        c = np.array([r["probe_ctx_emb"] for r in recs]) @ self.Pc
        pr = np.array([probe_vector(r["probe"]) for r in recs])
        return np.hstack([q, c, pr])


def view_embeddings(views: List[str], cache: Optional[str] = None) -> np.ndarray:
    """Embed each view's one-line description (core/conditions.py). Cached to .npz."""
    if cache and Path(cache).exists():
        blob = np.load(cache, allow_pickle=True)
        if list(blob["views"]) == list(views):
            return blob["embs"]
    from core.conditions import describe
    from utils.embedding import EmbeddingModel

    embs = np.asarray(EmbeddingModel().encode([describe(v) for v in views]), dtype=np.float32)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, views=np.array(views), embs=embs)
    return embs
