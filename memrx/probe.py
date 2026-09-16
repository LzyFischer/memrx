"""
MemRx component 1 — probe-then-route.

The router has to commit to a processing view before any real retrieval has
happened, so the only thing it can condition on for free is one cheap
retrieval pass against the raw view. This module turns that pass into a small
feature vector.

The point is not to hand the router more content. It is to hand it the
*shape* of the retrieval distribution: how concentrated the evidence is, how
redundant the top hits are, and how much of the query any single chunk can
account for. Those are properties of the (query, corpus) pair rather than of
the query's topic, which is what lets a router fitted on one conversation set
be applied to another.

Everything is computed from artefacts the probe retrieval already produced —
the scores, and the embeddings that were matched against — so the cost on top
of one embedding search is a few numpy ops on an n x d matrix with n ~ 10.

Features, grouped by what they measure:
  conf : margin_norm, decay_norm        how peaked the score distribution is
  disp : n_eff_frac, redundancy         how spread out / duplicated the hits are
  sem  : cos_q_ctx                      semantic distance query <-> retrieved mass
  cov  : q_cov_max, q_cov_gap           can ONE chunk cover the query, or does
                                        it take the union of several
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROBE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "conf": ["margin_norm", "decay_norm"],
    "disp": ["n_eff_frac", "redundancy"],
    "sem": ["cos_q_ctx"],
    "cov": ["q_cov_max", "q_cov_gap"],
}
PROBE_FEATURE_NAMES: List[str] = [n for g in PROBE_FEATURE_GROUPS.values() for n in g]

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "what",
    "when", "where", "who", "whom", "which", "how", "why", "has", "have", "had",
    "that", "this", "these", "those", "it", "its", "as", "by", "from", "about",
}
_WORD = re.compile(r"[a-z0-9']+")


def _tok(text: str) -> set:
    return {w for w in _WORD.findall(str(text).lower()) if w not in _STOP and len(w) > 1}


def compute_probe_features(
    query: str,
    scores: Sequence[float],
    chunk_texts: Sequence[str],
    chunk_embs: np.ndarray,
    query_emb: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Return (named features, weighted-pooled context embedding).

    `chunk_embs` must be L2-normalised and row-aligned with `scores` /
    `chunk_texts` in descending score order — exactly what a top-k semantic
    search returns. `query_emb` must be L2-normalised too.
    """
    n = len(scores)
    if n == 0:
        return {k: 0.0 for k in PROBE_FEATURE_NAMES}, np.zeros_like(query_emb)

    r = np.asarray(scores, dtype=np.float64)
    sd = float(r.std()) + 1e-9

    # conf. Normalised by the within-query spread: without that these mostly
    # encode how lexically common the query is, which is a property of the
    # query rather than of the evidence, and it leaks into the router as a
    # difficulty signal we do not want it using.
    margin_norm = float((r[0] - r[1]) / sd) if n > 1 else 0.0
    decay_norm = float((r[0] - r.mean()) / sd)

    # disp. Temperature tied to the query's own spread so n_eff does not
    # depend on the embedding model's score scale.
    z = r / sd
    z = z - z.max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    n_eff_frac = float(math.exp(ent) / n)  # 1/n = one chunk carries it, 1.0 = flat

    E = np.asarray(chunk_embs, dtype=np.float64)
    if n > 1:
        S = E @ E.T
        iu = np.triu_indices(n, k=1)
        redundancy = float(S[iu].mean())
    else:
        redundancy = 0.0

    # sem
    pooled = (p[:, None] * E).sum(axis=0)
    nrm = np.linalg.norm(pooled)
    pooled = pooled / nrm if nrm > 1e-9 else pooled
    cos_q_ctx = float(np.asarray(query_emb, dtype=np.float64) @ pooled)

    # cov. q_cov_gap > 0 means no single retrieved chunk accounts for the
    # query on its own but the union of them does.
    qt = _tok(query)
    if qt:
        per = [len(qt & _tok(t)) / len(qt) for t in chunk_texts]
        union = set()
        for t in chunk_texts:
            union |= _tok(t)
        q_cov_max = float(max(per)) if per else 0.0
        q_cov_gap = float(len(qt & union) / len(qt) - q_cov_max)
    else:
        q_cov_max, q_cov_gap = 0.0, 0.0

    feats = {
        "margin_norm": margin_norm, "decay_norm": decay_norm,
        "n_eff_frac": n_eff_frac, "redundancy": redundancy,
        "cos_q_ctx": cos_q_ctx, "q_cov_max": q_cov_max, "q_cov_gap": q_cov_gap,
    }
    return feats, pooled.astype(np.float32)


def probe_vector(feats: Dict[str, float]) -> np.ndarray:
    """Dict -> vector in PROBE_FEATURE_NAMES order."""
    return np.array([float(feats.get(n, 0.0)) for n in PROBE_FEATURE_NAMES], dtype=np.float32)


def probe_retrieve(store, query: str, top_n: int = 10):
    """One cheap semantic pass, returning everything the features need.

    Reads MemoryStore._embeddings directly: those vectors were computed at
    build time, and re-encoding the retrieved chunks to get them back would
    turn the probe from a lookup into a second embedding pass.
    """
    entries, scores = store.semantic_search_scored(query, top_k=top_n)
    if not entries:
        d = store.embedding_model.dimension or 384
        return [], [], np.zeros((0, d), dtype=np.float32)
    embs = np.stack([store._embeddings[e.entry_id] for e in entries])
    return entries, scores, embs
