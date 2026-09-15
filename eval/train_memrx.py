"""
Stage 2 — fit the MemRx router on train, score it on val (and test).

    python eval/train_memrx.py --train results/memrx_train.jsonl \
                               --val   results/memrx_val.jsonl \
                               --test  results/memrx_test.jsonl

Reported:

  fixed <view>        each processing pipeline used on its own — the incumbents
  best fixed          the best of those, CHOSEN ON TRAIN (choosing it on the
                      split being reported would be a second oracle)
  random              uniform over views, averaged over seeds
  MemRx               probe input + mixed listwise supervision
  oracle              per-question max

plus the two ablations that isolate the two components: dropping the probe
from the input (query-only), and dropping the likelihood from the target
(alpha forced to 1, i.e. F1-only). tau=0 with F1-only is the argmax
classifier, which the same code path produces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.pl_router import PLRouter, RandomProjector
from core.probe import PROBE_FEATURE_GROUPS, probe_vector


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def view_embeddings(views: List[str], cache: Optional[str] = None):
    """Encode each view's one-line behavioural description.

    Conditioning on the description rather than on a per-view free parameter
    is what lets a view that was not in training still be scored.
    """
    if cache and Path(cache).exists():
        blob = np.load(cache, allow_pickle=True)
        if list(blob["views"]) == list(views):
            return blob["embs"]
    from core.router import _CONDITION_DESCRIPTIONS
    from utils.embedding import EmbeddingModel

    texts = [_CONDITION_DESCRIPTIONS.get(v, v.replace("__", " ")) for v in views]
    embs = np.asarray(EmbeddingModel().encode(texts, is_query=False), dtype=np.float32)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, views=np.array(views), embs=embs)
    return embs


class FeatureSpace:
    """Fixed seeded projections, shared by train and val."""

    def __init__(self, d_emb, d_q=64, d_ctx=64, seed=0):
        self.pq = RandomProjector(d_emb, d_q, seed=seed)
        self.pc = RandomProjector(d_emb, d_ctx, seed=seed + 1)
        self.n_probe = len(probe_vector({}))

    def x(self, rec, use_probe=True, drop_groups=()):
        q = self.pq(np.asarray(rec["q_emb"], np.float32))
        if not use_probe:
            # Query only: the ceiling on any router that decides without
            # looking at the corpus.
            return np.concatenate([q, np.zeros(self.pc.d_out, np.float32),
                                   np.zeros(self.n_probe, np.float32)])
        return np.concatenate([q,
                               self.pc(np.asarray(rec["probe_ctx_emb"], np.float32)),
                               probe_vector(rec["probe"], drop_groups=drop_groups)])


def make_groups(recs, fs, views, use_probe=True, drop_groups=(), use_ll=True):
    out = []
    for r in recs:
        cands, f1s, lls = [], [], []
        for k, v in enumerate(views):
            if r["f1"].get(v) is None:
                continue
            cands.append(k)
            f1s.append(float(r["f1"][v]))
            ll = r.get("ll", {}).get(v)
            lls.append(float(ll) if (use_ll and ll is not None) else np.nan)
        if len(cands) >= 2:
            out.append({"x": fs.x(r, use_probe=use_probe, drop_groups=drop_groups),
                        "cands": cands, "s_f1": f1s, "s_ll": lls, "rec": r})
    return out


def eval_router(router, recs, fs, views, use_probe=True, metric="f1"):
    vals = []
    for r in recs:
        cands = [k for k, v in enumerate(views) if r[metric].get(v) is not None]
        if not cands:
            continue
        x = fs.x(r, use_probe=use_probe)
        k = cands[int(np.argmax(router.score(x, cands)))]
        vals.append(float(r[metric][views[k]]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--metric", default="f1", choices=["f1", "em"])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--tau-ll", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-seeds", type=int, default=5)
    p.add_argument("--view-emb-cache", default="results/view_embs.npz")
    p.add_argument("--tau-sweep", nargs="*", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    train, val = load_jsonl(args.train), load_jsonl(args.val)
    test = load_jsonl(args.test) if args.test else None
    views = train[0]["views"]
    M = args.metric

    print(f"[data] train={len(train)} val={len(val)}"
          + (f" test={len(test)}" if test else ""))

    n_ll = sum(1 for r in train for v in views if r.get("ll", {}).get(v) is not None)
    print(f"[data] likelihood present on {n_ll}/{len(train)*len(views)} train cells")

    ties = sum(1 for r in train
               if max(r[M][v] for v in views) - min(r[M][v] for v in views) < 1e-9)
    all_hi = sum(1 for r in train if min(r[M][v] for v in views) > 0.99)
    all_lo = sum(1 for r in train if max(r[M][v] for v in views) < 1e-9)
    print(f"[data] train {M} ties: {100*ties/len(train):.1f}%  "
          f"(all-correct {100*all_hi/len(train):.1f}%, all-wrong {100*all_lo/len(train):.1f}%)")

    view_embs = view_embeddings(views, cache=args.view_emb_cache)
    fs = FeatureSpace(len(train[0]["q_emb"]), seed=args.seed)

    def make(tau=None, **kw):
        return PLRouter(view_embs, hidden=args.hidden,
                        tau=args.tau if tau is None else tau,
                        tau_ll=args.tau_ll, delta=args.delta, lr=args.lr,
                        weight_decay=args.weight_decay, epochs=args.epochs,
                        seed=args.seed, **kw)

    memrx = make().fit(make_groups(train, fs, views))
    no_probe = make().fit(make_groups(train, fs, views, use_probe=False))
    no_ll = make().fit(make_groups(train, fs, views, use_ll=False))
    cls = make(tau=0.0).fit(make_groups(train, fs, views, use_ll=False))

    tr_means = {v: np.mean([r[M][v] for r in train if r[M].get(v) is not None]) for v in views}
    bf = max(tr_means, key=tr_means.get)

    drop_routers = {g: make().fit(make_groups(train, fs, views, drop_groups=(g,)))
                    for g in PROBE_FEATURE_GROUPS}
    sweep_routers = ([(t, make(tau=t).fit(make_groups(train, fs, views)))
                      for t in args.tau_sweep] if args.tau_sweep else [])

    def report(recs, tag):
        rows = []
        for v in views:
            vals = [r[M][v] for r in recs if r[M].get(v) is not None]
            rows.append((f"fixed  {v}", float(np.mean(vals))))
        rows.append((f"best fixed (train-picked: {bf})",
                     float(np.mean([r[M][bf] for r in recs if r[M].get(bf) is not None]))))

        rnd = []
        for s in range(args.random_seeds):
            rg = np.random.default_rng(s)
            rnd.append(float(np.mean([r[M][str(rg.choice(views))] for r in recs])))
        rows.append((f"random (+/- {np.std(rnd):.4f})", float(np.mean(rnd))))

        rows.append(("classification (tau=0, F1-only)", eval_router(cls, recs, fs, views, metric=M)))
        rows.append(("MemRx  - probe (query only)",
                     eval_router(no_probe, recs, fs, views, use_probe=False, metric=M)))
        rows.append(("MemRx  - likelihood (F1-only target)",
                     eval_router(no_ll, recs, fs, views, metric=M)))
        rows.append(("MemRx", eval_router(memrx, recs, fs, views, metric=M)))
        rows.append(("oracle", float(np.mean([max(r[M][v] for v in views) for r in recs]))))

        print(f"\n=== {tag}: mean {M} ===")
        for name, v in rows:
            print(f"  {name:<40s} {v:.4f}")

        print(f"\n=== probe feature groups ({tag}, dropped one at a time) ===")
        for g, rr in drop_routers.items():
            vals = []
            for r in recs:
                cands = [k for k, v in enumerate(views) if r[M].get(v) is not None]
                x = fs.x(r, drop_groups=(g,))
                vals.append(float(r[M][views[cands[int(np.argmax(rr.score(x, cands)))]]]))
            print(f"  drop {g:<6s} {np.mean(vals):.4f}")

        if sweep_routers:
            print(f"\n=== tau sweep ({tag}; tau=0 is the classification baseline) ===")
            for t, rr in sweep_routers:
                print(f"  tau={t:<6.3f} {eval_router(rr, recs, fs, views, metric=M):.4f}")
        return rows

    out_rows = {"val": report(val, "val")}
    if test:
        out_rows["test"] = report(test, "test")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"metric": M, **{k: dict(v) for k, v in out_rows.items()}}, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()