"""
No vLLM or embedding model needed.

    python tests/smoke_test.py

  1. targets: tau=0 one-hot argmax with ties shared, tau>0 softmax, missing views masked
  2. writes a synthetic curated dataset with a planted "best view" and checks
     the router finds it; then you can run the real training script on it
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from memrx.features import FeatureBuilder, metric_matrix
from memrx.probe import PROBE_FEATURE_NAMES
from memrx.router import Router, make_target

VIEWS = ["baseline", "summary__session_level", "augmentation__keywords", "graph__entity"]
D = 384
OUT = Path("results_synth")


def test_target():
    f1 = torch.tensor([[0.9, 0.2, 0.9, float("nan")]])
    assert torch.allclose(make_target(f1, tau=0), torch.tensor([[0.5, 0, 0.5, 0]])), make_target(f1, 0)
    t = make_target(f1, tau=0.1)
    assert t[0, 3] == 0 and abs(t.sum() - 1) < 1e-6 and t[0, 0] > t[0, 1]
    f1 = torch.tensor([[0.50, 0.48, 0.30, float("nan")]])
    assert torch.allclose(make_target(f1, tau=0, margin=0.05), torch.tensor([[0.5, 0.5, 0, 0]]))
    assert torch.allclose(make_target(f1, tau=0, margin=0.0), torch.tensor([[1.0, 0, 0, 0]]))
    t = make_target(f1, tau=0.1, margin=0.05)
    assert torch.isclose(t[0, 0], t[0, 1]) and t[0, 2] < t[0, 0]
    print("[tgt ] tau=0 one-hot with shared ties, tau>0 softmax, NaN masked, margin merges near-ties")


def make_synthetic(n, sids, rng):
    """Latent type z decides the best view. The probe features carry z clearly,
    the query embedding only weakly; 60% of questions are all-correct ties."""
    recs = []
    for t in range(n):
        z = int(rng.integers(0, 3))
        q = rng.normal(size=D); q[z] += 1.0; q /= np.linalg.norm(q)
        probe = {name: float(rng.normal(0, 0.35) + (1.6 if k % 3 == z else 0.0))
                 for k, name in enumerate(PROBE_FEATURE_NAMES)}
        f1 = np.ones(4) if rng.random() < 0.6 else np.full(4, 0.2)
        f1[z + 1] = 1.0
        recs.append({
            "sample_id": sids[t % len(sids)], "question": f"q{t} z={z}", "category": 1, "views": VIEWS,
            "q_emb": q.round(5).tolist(), "probe": probe,
            "probe_ctx_emb": rng.normal(size=D).round(5).tolist(),
            "f1": dict(zip(VIEWS, f1.tolist())), "em": dict(zip(VIEWS, f1.tolist())),
        })
    return recs


def test_synthetic():
    rng = np.random.default_rng(0)
    train, val = make_synthetic(400, ["A", "B"], rng), make_synthetic(200, ["C"], rng)
    OUT.mkdir(exist_ok=True)
    for name, recs in [("memrx_train.jsonl", train), ("memrx_val.jsonl", val)]:
        with open(OUT / name, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    view_embs = rng.normal(size=(4, D))
    np.savez(OUT / "view_embs.npz", views=np.array(VIEWS), embs=view_embs)

    fb = FeatureBuilder(D)
    router = Router(view_embs, epochs=200).fit(fb(train), metric_matrix(train, VIEWS))
    Y = metric_matrix(val, VIEWS)
    got = Y[np.arange(len(val)), router.predict(fb(val))].mean()
    best_fixed = np.nanmean(Y, 0).max()
    print(f"[synth] val router {got:.3f}  best fixed {best_fixed:.3f}  oracle {Y.max(1).mean():.3f}")
    assert got > best_fixed + 0.1


if __name__ == "__main__":
    test_target()
    test_synthetic()
    print(f"\nall checks passed. Now try:\n  python scripts/train_memrx.py --train {OUT}/memrx_train.jsonl "
          f"--val {OUT}/memrx_val.jsonl --view-emb-cache {OUT}/view_embs.npz --out-dir {OUT}")
