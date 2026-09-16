"""
Stage 2 — fit the router on train, report on train / val / test.

    python scripts/train_memrx.py --train results/memrx_train.jsonl \
                                  --val   results/memrx_val.jsonl

For each split prints:
  fixed <view>     every question uses that one view
  best fixed       the view with the best TRAIN mean, applied to this split
  random           expected score of picking a view uniformly
  router           the learned router
  oracle           per-question max (upper bound)
then the same rows broken down by the four LoCoMo query types (multi_hop,
temporal, open_domain, single_hop) with both F1 and BLEU-1, plus how often
the router picks each view. Writes one CSV row per question (router pick +
every view's F1 and BLEU-1) to --out-dir for inspection.

Ablations are flags rather than extra built-in rows: --no-probe for a
query-only router, --tau 0 for a hard argmax classifier.

--margin eps treats views within eps F1 of the question's best as equally good.
Several values run a sweep and print one compact table instead:

    python scripts/train_memrx.py ... --tau 0 --margin 0 0.02 0.05 0.1
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from memrx.features import FeatureBuilder, load_jsonl, metric_matrix, view_embeddings
from memrx.router import Router
from utils.locomo import print_type_table


def tie_stats(Y: np.ndarray) -> str:
    spread = np.nanmax(Y, 1) - np.nanmin(Y, 1)
    tie = spread < 1e-9
    all_hi = tie & (np.nanmin(Y, 1) > 0.99)
    all_lo = tie & (np.nanmax(Y, 1) < 1e-9)
    return (f"ties {tie.mean():.1%} (all-correct {all_hi.mean():.1%}, all-wrong {all_lo.mean():.1%}, "
            f"other {(tie & ~all_hi & ~all_lo).mean():.1%}), informative {(~tie).mean():.1%}")


def report(tag, recs, X, Y, views, router, best_fixed, out_dir, metric):
    N = len(recs)
    mask = np.isfinite(Y)
    pick = router.predict(X, mask)
    got = Y[np.arange(N), pick]

    print(f"\n=== {tag}  (n={N}, mean {metric}) ===")
    print(f"  {tie_stats(Y)}")
    rows = [(f"fixed {v}", np.nanmean(Y[:, k])) for k, v in enumerate(views)]
    rows += [(f"best fixed on train ({views[best_fixed]})", np.nanmean(Y[:, best_fixed])),
             ("random", np.nanmean(np.nanmean(Y, 1))),
             ("router", got.mean()),
             ("oracle", np.nanmax(Y, 1).mean())]
    for name, val in rows:
        print(f"  {name:<48s} {val:.4f}")

    counts = Counter(pick.tolist())
    print("  router picks: " + ", ".join(f"{views[k]} {counts.get(k, 0)}" for k in range(len(views))))

    # ---- F1 and BLEU-1 by query type -----------------------------------
    scores = {m: metric_matrix(recs, views, m) for m in ("f1", "bleu1")}
    idx = np.arange(N)

    def per_q(fn):
        return {m: fn(S) for m, S in scores.items()}

    type_rows = [(f"fixed {v}", per_q(lambda S, k=k: S[:, k])) for k, v in enumerate(views)]
    type_rows += [
        (f"best fixed ({views[best_fixed]})", per_q(lambda S: S[:, best_fixed])),
        ("random", per_q(lambda S: np.nanmean(S, 1))),
        ("router", per_q(lambda S: S[idx, pick])),
        ("oracle (per metric)", per_q(lambda S: np.nanmax(S, 1))),
    ]
    cats = [int(r.get("category", 0)) for r in recs]
    print_type_table(f"  --- {tag}: F1 / BLEU-1 (x100) by query type, router trained on {metric} ---",
                     [(name, cats, vals) for name, vals in type_rows])

    path = Path(out_dir) / f"router_preds_{tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "question", "category", "pick", "router_f1", "router_bleu1",
                    "oracle_f1"] + [f"f1:{v}" for v in views] + [f"bleu1:{v}" for v in views])
        for i, r in enumerate(recs):
            w.writerow([r["sample_id"], r["question"], r.get("category"), views[pick[i]],
                        scores["f1"][i, pick[i]], scores["bleu1"][i, pick[i]], np.nanmax(scores["f1"][i])]
                       + list(scores["f1"][i]) + list(scores["bleu1"][i]))
    print(f"  -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--metric", default="f1", choices=["f1", "em", "bleu1"],
                   help="what the router is trained on; the report always shows F1 and BLEU-1")
    p.add_argument("--tau", type=float, default=0.1, help="target temperature; 0 = hard argmax")
    p.add_argument("--margin", type=float, nargs="+", default=[0.0],
                   help="F1 gap below which two views count as a tie; several values = sweep")
    p.add_argument("--no-probe", action="store_true", help="query embedding only")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--view-emb-cache", default="results/view_embs.npz")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    splits = {"train": load_jsonl(args.train), "val": load_jsonl(args.val)}
    if args.test:
        splits["test"] = load_jsonl(args.test)
    views = splits["train"][0]["views"]
    print(f"[data] views={views}  " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()))

    featurize = FeatureBuilder(len(splits["train"][0]["q_emb"]), seed=args.seed,
                               use_probe=not args.no_probe)
    data = {k: (recs, featurize(recs), metric_matrix(recs, views, args.metric))
            for k, recs in splits.items()}

    _, X_tr, Y_tr = data["train"]
    view_embs = view_embeddings(views, cache=args.view_emb_cache)
    best_fixed = int(np.nanargmax(np.nanmean(Y_tr, 0)))
    sweep = len(args.margin) > 1

    def fit(margin):
        router = Router(view_embs, hidden=args.hidden, tau=args.tau, margin=margin, lr=args.lr,
                        weight_decay=args.weight_decay, epochs=args.epochs, seed=args.seed,
                        log_every=0 if sweep else args.log_every)
        print(f"[fit] X={X_tr.shape} tau={args.tau} margin={margin} probe={not args.no_probe}")
        return router.fit(X_tr, Y_tr)

    if not sweep:
        router = fit(args.margin[0])
        for tag, (recs, X, Y) in data.items():
            report(tag, recs, X, Y, views, router, best_fixed, args.out_dir, args.metric)
        return

    # ---- margin sweep: one row per margin --------------------------------
    rows = []
    for m in args.margin:
        router = fit(m)
        spread = np.nanmax(Y_tr, 1) - np.nanmin(Y_tr, 1)
        row = {"margin": m, "tied": (spread <= m + 1e-9).mean(), "floor": router.floor,
               "loss": router.history[-1]}
        for tag, (_, X, Y) in data.items():
            pick = router.predict(X, np.isfinite(Y))
            row[tag] = Y[np.arange(len(Y)), pick].mean()
        rows.append(row)

    tags = list(data)
    print(f"\n=== margin sweep (tau={args.tau}, mean {args.metric}) ===")
    print("  tied = share of train questions whose views are all within margin (target uniform)")
    print(f"  {'margin':>7s} {'tied':>6s} {'floor':>7s} {'loss':>7s}" + "".join(f"{t:>8s}" for t in tags))
    for r in rows:
        print(f"  {r['margin']:>7.3f} {r['tied']:>6.1%} {r['floor']:>7.4f} {r['loss']:>7.4f}"
              + "".join(f"{r[t]:>8.4f}" for t in tags))
    for tag, (_, _, Y) in data.items():
        print(f"  {tag:<6s} best fixed on train {np.nanmean(Y[:, best_fixed]):.4f}   "
              f"random {np.nanmean(np.nanmean(Y, 1)):.4f}   oracle {np.nanmax(Y, 1).mean():.4f}")

if __name__ == "__main__":
    main()
