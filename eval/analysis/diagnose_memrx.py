"""
Diagnostics for a MemRx run. Reads the curated JSONL only — no vLLM, no
embedding model, no generations.

    python eval/analysis/diagnose_memrx.py \
        --train results/memrx_train.jsonl --val results/memrx_val.jsonl

Five questions, in the order they should be asked:

  1. SELECTION DISTRIBUTION. Does the router actually route, or has it
     collapsed onto one view? A constant router and a working one look
     identical in a mean-F1 table, so this has to be checked separately.

  2. IS THE HEADROOM REACHABLE. The per-question max over K noisy scores
     overstates the value of the oracle policy, because argmax selects partly
     on noise. Two corrections: score the argmax with an INDEPENDENT second
     measurement (--val2), which removes the curse outright; and check whether
     routing on the gold question type -- an upper bound no deployable router
     can exceed at that granularity -- beats a fixed view at all.

  3. VIEW CORRELATION. If the views agree question by question, they are
     producing near-identical evidence and there is nothing to route between.

  4. TIE DECOMPOSITION. All-correct ties are a reader ceiling, all-wrong ties
     a floor; neither can be fixed by routing. What is left is the only part
     of the data where the choice of view matters at all.

  5. LIKELIHOOD SANITY. Does the gold-answer likelihood order the views the
     same way F1 does where F1 has an opinion, and is it just tracking
     context size? If it fails both, the mixed target is injecting noise.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from eval.train_memrx import FeatureSpace, load_jsonl, make_groups, view_embeddings
from core.pl_router import PLRouter


def entropy(counts, K):
    p = np.asarray([counts.get(k, 0) for k in range(K)], float)
    p = p / max(p.sum(), 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()), float(-(nz * np.log(nz)).sum() / np.log(K))


def report_selection(name, picks, views, M, recs):
    K = len(views)
    c = Counter(picks)
    H, Hn = entropy(c, K)
    print(f"\n  {name}")
    print(f"    entropy {H:.3f} / {np.log(K):.3f}  (normalised {Hn:.3f})")
    for k in range(K):
        n = c.get(k, 0)
        if n == 0:
            continue
        sel = [float(r[M][views[k]]) for r, p in zip(recs, picks) if p == k]
        print(f"      {views[k]:<26s} {n:4d} ({100*n/len(picks):5.1f}%)  "
              f"mean {M} when chosen {np.mean(sel):.4f}")
    if Hn < 0.15:
        print("    -> collapsed: this is a constant router wearing a router's clothes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--metric", default="f1", choices=["f1", "em"])
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--tau-ll", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val2", default=None,
                    help="second curated run over the same questions "
                         "(different decoding seed) for the honest oracle")
    ap.add_argument("--view-emb-cache", default="results/view_embs.npz")
    args = ap.parse_args()

    train, val = load_jsonl(args.train), load_jsonl(args.val)
    views = train[0]["views"]
    K, M = len(views), args.metric
    print(f"[data] train={len(train)} val={len(val)} | K={K} | metric={M}")

    view_embs = view_embeddings(views, cache=args.view_emb_cache)
    fs = FeatureSpace(len(train[0]["q_emb"]), seed=args.seed)

    def fit(tau=None, **kw):
        r = PLRouter(view_embs, hidden=args.hidden, tau=args.tau if tau is None else tau,
                     tau_ll=args.tau_ll, delta=args.delta, epochs=args.epochs, seed=args.seed)
        return r.fit(make_groups(train, fs, views, **kw))

    variants = {
        "MemRx": (fit(), True),
        "MemRx - probe (query only)": (fit(use_probe=False), False),
        "MemRx - likelihood (F1-only)": (fit(use_ll=False), True),
        "classification (tau=0, F1-only)": (fit(tau=0.0, use_ll=False), True),
    }

    # ---- 1. selection distribution --------------------------------
    print("\n=== 1. selection distribution on val ===")
    for name, (router, use_probe) in variants.items():
        picks = []
        for r in val:
            cands = list(range(K))
            x = fs.x(r, use_probe=use_probe)
            picks.append(cands[int(np.argmax(router.score(x, cands)))])
        report_selection(name, picks, views, M, val)

    oracle_pick = [int(np.argmax([r[M][v] for v in views])) for r in val]
    report_selection("oracle (per-question argmax)", oracle_pick, views, M, val)

    # ---- 2. is the oracle headroom reachable? ---------------------
    # The per-question max over K noisy measurements sits above the value of
    # the oracle POLICY, because argmax partly selects on noise. Two things
    # are reported here; neither is the naive oracle on its own.
    print("\n=== 2. is the oracle headroom reachable? ===")
    S = np.array([[float(r[M][v]) for v in views] for r in val])
    best_fixed = float(S.mean(0).max())
    naive = float(S.max(1).mean())
    print(f"  best fixed                {best_fixed:.4f}")
    print(f"  oracle (naive, inflated)  {naive:.4f}   apparent headroom {naive-best_fixed:+.4f}")

    # (a) honest oracle: pick with one measurement, score with an independent
    # one. This is the only way to remove the winner's curse outright, and it
    # costs one extra generation pass over the same contexts.
    if args.val2:
        val2 = {(r["sample_id"], r["question"]): r for r in load_jsonl(args.val2)}
        pairs = [(r, val2[(r["sample_id"], r["question"])]) for r in val
                 if (r["sample_id"], r["question"]) in val2]
        if pairs:
            honest = float(np.mean([float(b[M][views[int(np.argmax([a[M][v] for v in views]))]])
                                    for a, b in pairs]))
            print(f"  oracle (honest, run1 picks / run2 scores, n={len(pairs)})  {honest:.4f}")
            print(f"  -> winner's curse {naive-honest:+.4f}; REACHABLE headroom "
                  f"{honest-best_fixed:+.4f}")
    else:
        print("  [no --val2] re-run curation with a second decoding seed to the same")
        print("  contexts, then pass it here: picking on run 1 and scoring on run 2")
        print("  removes the winner's curse and gives the headroom a router can reach.")

    # (b) transferable structure, gold metadata allowed. Fit argmax-per-category
    # on train, apply to val. This uses a label a deployed router would not
    # have, so it is an UPPER BOUND on routing at question-type granularity --
    # if it does not beat best fixed, the signal is not there to be learned.
    cats = sorted({r["category"] for r in train})
    by_cat = {}
    for c in cats:
        rows = [r for r in train if r["category"] == c]
        if rows:
            by_cat[c] = views[int(np.argmax([np.mean([float(r[M][v]) for r in rows])
                                             for v in views]))]
    fallback = views[int(np.argmax([np.mean([float(r[M][v]) for r in train]) for v in views]))]
    cat_route = float(np.mean([float(r[M][by_cat.get(r["category"], fallback)]) for r in val]))
    print(f"  category-argmax router (uses gold labels, NOT deployable)  {cat_route:.4f}")
    print("    train-fitted mapping: " + ", ".join(f"cat{c}->{v.split('__')[-1]}"
                                                   for c, v in sorted(by_cat.items())))
    if cat_route <= best_fixed + 1e-9:
        print("    -> even with the gold question type, routing does not beat a fixed")
        print("       view. Whatever separates the views is not question type.")

    # ---- 3. view correlation --------------------------------------
    print(f"\n=== 3. do the views disagree? (pairwise {M} correlation on val) ===")
    C = np.corrcoef(S.T)
    print(f"    {'':<26s}" + "".join(f"{v.split('__')[-1][:9]:>11s}" for v in views))
    for i, v in enumerate(views):
        print(f"    {v:<26s}" + "".join(f"{C[i,j]:>10.2f} " for j in range(K)))
    iu = np.triu_indices(K, 1)
    print(f"    mean off-diagonal correlation: {C[iu].mean():.3f} "
          f"(high = the views are producing the same evidence)")

    # ---- 4. tie decomposition -------------------------------------
    print("\n=== 4. where does the signal live? (train) ===")
    T = np.array([[float(r[M][v]) for v in views] for r in train])
    spread = T.max(1) - T.min(1)
    tie = spread < 1e-9
    hi_tie = tie & (T.min(1) > 0.99)
    lo_tie = tie & (T.max(1) < 1e-9)
    mid_tie = tie & ~hi_tie & ~lo_tie
    print(f"  ties                     {100*tie.mean():5.1f}%")
    print(f"    all-correct (ceiling)  {100*hi_tie.mean():5.1f}%")
    print(f"    all-wrong   (floor)    {100*lo_tie.mean():5.1f}%")
    print(f"    tied in between        {100*mid_tie.mean():5.1f}%")
    print(f"  informative              {100*(~tie).mean():5.1f}%  "
          f"(mean spread {spread[~tie].mean():.3f})")
    print(f"  -> only the informative rows can ever move the mean. Routing them")
    print(f"     perfectly would be worth {(T.max(1)-T.mean(1))[~tie].mean()*(~tie).mean():+.4f} "
          f"over picking at\n     random -- inflated by the same winner's curse as the "
          f"naive oracle above.")

    # ---- 5. likelihood sanity -------------------------------------
    has_ll = all(r.get("ll", {}).get(v) is not None for r in train for v in views)
    print(f"\n=== 5. likelihood ===")
    if not has_ll:
        n = sum(1 for r in train for v in views if r.get("ll", {}).get(v) is not None)
        print(f"  only {n}/{len(train)*K} train cells have a likelihood — the mixed")
        print("  target is falling back to F1 on the rest. Check that the server")
        print("  supports /v1/completions with echo=True.")
        return
    L = np.array([[float(r["ll"][v]) for v in views] for r in train])
    inf = spread > 1e-9
    agree = (L[inf].argmax(1) == T[inf].argmax(1)).mean()
    print(f"  on informative rows, argmax(ll) == argmax({M}): {100*agree:.1f}% "
          f"(chance {100/K:.1f}%)")
    if agree < 1.5 / K:
        print("  -> likelihood is not ordering the views the way the task metric does.")
        print("     Using it to break ties injects a different preference, not a finer one.")
    N = np.array([[float(r["n_retrieved"][v]) for v in views] for r in train])
    dl = L - L.mean(1, keepdims=True)
    dn = N - N.mean(1, keepdims=True)
    if dn.std() > 1e-9:
        print(f"  within-question corr(ll, n_retrieved): "
              f"{np.corrcoef(dl.ravel(), dn.ravel())[0,1]:+.3f} "
              f"(large = it is measuring context size, not evidence quality)")


if __name__ == "__main__":
    main()
