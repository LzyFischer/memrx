"""
Checks that need neither a vLLM server nor a downloaded embedding model.

  1. finite-difference gradient check on PLRouter's hand-written backward
  2. mixed_target: alpha behaviour, NaN fallback, tau=0 degenerate case
  3. an end-to-end fit on a synthetic curated file whose structure is known,
     so the training/eval path can be checked against a ground truth

    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from memrx.pl_router import PLRouter, mixed_target, _segment_softmax
from memrx.probe import PROBE_FEATURE_NAMES

VIEWS = ["baseline", "summary__session_level", "augmentation__keywords", "graph__entity"]
D = 384


# ---------------------------------------------------------------------- #
def test_gradients():
    rng = np.random.default_rng(0)
    K, d_e, d_x = 4, 12, 9
    ve = rng.normal(size=(K, d_e)).astype(np.float32)
    groups = []
    for _ in range(6):
        m = int(rng.integers(2, K + 1))
        c = list(rng.choice(K, size=m, replace=False))
        groups.append({"x": rng.normal(size=d_x).astype(np.float32), "cands": c,
                       "s_f1": list(rng.random(m)), "s_ll": list(-rng.random(m))})

    r = PLRouter(ve, hidden=7, tau=0.2)
    X, E, G, F, L = r._flatten(groups)
    X = r._standardise(X, fit=True)
    r._init_params(X.shape[1])
    # float64 for the check only: the model runs in float32, where a small
    # perturbation of a ~1.0 loss is barely above the representable step and
    # the finite difference is dominated by rounding rather than by any bug.
    r.params = {k: v.astype(np.float64) for k, v in r.params.items()}
    X, E = X.astype(np.float64), E.astype(np.float64)
    n = len(groups)
    pt = mixed_target(F, L, G, n, tau=r.tau, tau_ll=r.tau_ll, delta=r.delta)

    def loss():
        s, _ = r._forward(X, E)
        return -float((pt * np.log(_segment_softmax(s, G, n) + 1e-12)).sum() / n)

    s, cache = r._forward(X, E)
    grads = r._backward((_segment_softmax(s, G, n) - pt) / n, cache)

    worst, skipped = 0.0, 0
    for name in ["Wx", "bx", "We", "be", "W1", "b1", "W2", "b2"]:
        flat = r.params[name].ravel()
        for _ in range(12):
            i = int(rng.integers(0, flat.size))
            old, eps = flat[i], 1e-6
            flat[i] = old + eps; lp = loss()
            flat[i] = old - eps; lm = loss()
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), grads[name].ravel()[i]
            # Some parameters have a gradient of exactly zero — b2, because a
            # constant shift cancels inside the softmax, and any weight
            # feeding a dead ReLU. Against a loss of order 1, both sides are
            # then at machine-noise level and a ratio between them is
            # meaningless, so those probes are counted rather than scored.
            if max(abs(num), abs(ana)) < 1e-8:
                skipped += 1
                continue
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana)))
    print(f"[grad] worst relative error over {96 - skipped} probes: {worst:.2e}  "
          f"({skipped} zero-gradient probes skipped)  {'OK' if worst < 1e-5 else 'FAIL'}")
    assert worst < 1e-5


def test_mixed_target():
    G = np.array([0, 0, 0, 0], np.int64)

    # F1 separates the views -> alpha = 1, likelihood ignored entirely
    f1 = np.array([0.9, 0.1, 0.1, 0.1])
    ll = np.array([-3.0, -0.1, -0.1, -0.1])  # would prefer a different view
    t = mixed_target(f1, ll, G, 1, tau=0.1, tau_ll=0.1, delta=0.1)
    assert int(np.argmax(t)) == 0, t

    # F1 fully tied -> alpha = 0, the ordering comes from likelihood
    f1 = np.array([1.0, 1.0, 1.0, 1.0])
    ll = np.array([-2.0, -0.5, -1.5, -3.0])
    t = mixed_target(f1, ll, G, 1, tau=0.1, tau_ll=0.1, delta=0.1)
    assert int(np.argmax(t)) == 1, t
    print("[mix ] alpha: F1 when it separates, likelihood when it ties  OK")

    # missing likelihood -> fall back to F1 rather than dropping the group
    t = mixed_target(np.array([1.0, 1.0, 1.0, 0.2]),
                     np.array([np.nan, -0.5, -1.5, -3.0]), G, 1, delta=0.1)
    assert np.isfinite(t).all() and abs(t.sum() - 1.0) < 1e-6
    assert t[3] < t[0], t
    print("[mix ] NaN likelihood falls back to F1-only  OK")

    # tau -> 0 is the argmax classifier; ties are shared, not broken by noise
    t = mixed_target(np.array([0.9, 0.2, 0.9, 0.1]), np.full(4, np.nan), G, 1, tau=0.0)
    assert np.allclose(t, [0.5, 0.0, 0.5, 0.0]), t
    print("[tau ] tau=0 reduces to one-hot argmax (ties shared)  OK")


# ---------------------------------------------------------------------- #
def make_synthetic(path_dir, n_train=400, n_val=200, seed=0):
    """Planted structure: a latent type z that the PROBE carries cleanly and
    the query embedding carries only weakly, and a heavy tie rate in F1 whose
    ordering survives only in the likelihood. Both components should show up
    in the ablations if the plumbing is right."""
    rng = np.random.default_rng(seed)
    K = len(VIEWS)
    best_view = [1, 2, 3]  # one per latent type

    def gen(n, sids):
        recs = []
        for t in range(n):
            z = int(rng.integers(0, 3))
            q = rng.normal(size=D); q[:8] += 0.4 * np.eye(3)[z].repeat(3)[:8]
            q /= np.linalg.norm(q)
            ctx = rng.normal(size=D); ctx[:8] += 0.3 * np.eye(3)[z].repeat(3)[:8]
            ctx /= np.linalg.norm(ctx)
            probe = {name: float(rng.normal(0, 0.35) + (1.6 if k % 3 == z else 0.0))
                     for k, name in enumerate(PROBE_FEATURE_NAMES)}

            quality = np.full(K, 0.35)
            quality[best_view[z]] = 0.85
            quality = np.clip(quality + rng.normal(0, 0.05, K), 0, 1)
            # F1 is the censored view of quality: everything above the reader's
            # threshold looks identical. 60% of questions are easy enough that
            # every pipeline clears it.
            easy = rng.random() < 0.6
            f1 = np.ones(K) if easy else (quality > 0.6).astype(float) * np.clip(
                quality + rng.normal(0, 0.03, K), 0, 1)
            ll = -1.5 + 1.2 * quality + rng.normal(0, 0.05, K)   # never saturates

            recs.append({
                "sample_id": sids[t % len(sids)], "question": f"q{t} (z={z})",
                "category": 1 + (t % 4), "gold": "-", "n_evidence": 2, "views": VIEWS,
                "q_emb": [round(float(x), 5) for x in q],
                "probe": {k: round(v, 5) for k, v in probe.items()},
                "probe_ctx_emb": [round(float(x), 5) for x in ctx],
                "f1": {v: round(float(f1[i]), 4) for i, v in enumerate(VIEWS)},
                "em": {v: round(float(f1[i]), 4) for i, v in enumerate(VIEWS)},
                "ll": {v: round(float(ll[i]), 5) for i, v in enumerate(VIEWS)},
                "n_retrieved": {v: 10 for v in VIEWS},
            })
        return recs

    os.makedirs(path_dir, exist_ok=True)
    for name, n, sids in [("memrx_train.jsonl", n_train, ["conv_A", "conv_B"]),
                          ("memrx_val.jsonl", n_val, ["conv_C"])]:
        with open(os.path.join(path_dir, name), "w") as f:
            for r in gen(n, sids):
                f.write(json.dumps(r) + "\n")
    embs = rng.normal(size=(K, D)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    np.savez(os.path.join(path_dir, "view_embs.npz"), views=np.array(VIEWS), embs=embs)
    print(f"[synth] wrote {n_train}/{n_val} records to {path_dir}")


if __name__ == "__main__":
    test_gradients()
    test_mixed_target()
    make_synthetic("results_synth")
    print("\nall checks passed — now run:\n"
          "  python scripts/train_memrx.py --train results_synth/memrx_train.jsonl \\\n"
          "      --val results_synth/memrx_val.jsonl "
          "--view-emb-cache results_synth/view_embs.npz")
