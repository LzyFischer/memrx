"""
Query-level routing, evaluated as a SELECTION over an already-computed full
condition matrix — NOT by rebuilding memory on demand per query.

Why: a question is only known after a conversation's memory has already
been built (turns accumulate first, questions get asked later against
whatever memory already exists) — a router can't decide "which condition
to build" per query, because there's no query yet at build time. So the
memory-build step still has to build EVERY condition for a conversation,
exactly like run_2a_locomo.py already does. Routing only happens at
retrieval/answer time: for each question, pick which of the already-built
conditions' stores to read the answer from.

That means router evaluation needs zero new LLM/embedding calls to the
memory pipeline — it's a pure lookup over the full-matrix CSV
run_2a_locomo.py already produced (one row per sample_id x condition_id x
question). (`--routers judge` and `--routers naive` do call the LLM once per
question to make the routing DECISION, but never rebuild a store.)

This script routes with one or more strategies and prints:
  1. the mean-F1 pivot (all 7 fixed conditions + one row per router), and
  2. per-question win / tie / lose for every router against every fixed
     condition and against the other routers.

Routers available (see core/router.py for what each one is testing):
    random   lower-bound control, uniform pick; run over several seeds
    naive    open-ended LLM prompt, fills in the 3 dimensions
    judge    LLM judge, closed-set pick from the described candidate menu
    learned  logistic regression on train-split feedback
    oracle   per-question argmax = upper bound, not a real router

Workflow:
    # 1) full matrix on train — the feedback LearnedRouter fits on
    python eval/run_2a_locomo.py --data data/locomo10.json --split train --out-dir results

    # 2) full matrix on val — what the routers will SELECT from
    python eval/run_2a_locomo.py --data data/locomo10.json --split val --out-dir results

    # 3) route val's questions with everything at once: F1 table + W/T/L grid
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_val.csv \
        --routers random judge learned oracle \
        --train-csv results/2a_locomo_results_train.csv \
        --model Qwen/Qwen3-0.6B --base-url http://localhost:8000/v1 \
        --out-dir results

    # cheap pass, no LLM and no train CSV needed (random + oracle only):
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_val.csv \
        --routers random oracle --out-dir results

    # 4) once a router is picked on val, get the final table on test
    python eval/run_2a_locomo.py --data data/locomo10.json --split test --out-dir results
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_test.csv \
        --routers random naive judge learned oracle \
        --train-csv results/2a_locomo_results_train.csv \
        --model Qwen/Qwen3-0.6B --base-url http://localhost:8000/v1 \
        --out-dir results --wtl-by-category
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import numpy as np
import pandas as pd

from config_2a import build_condition_matrix
from core.router import ROUTER_NAMES, build_router
from eval.analysis.win_tie_lose import (
    build_wtl_table, print_wtl_table, selection_frame, wtl_by_category,
)
from eval.run_2a_locomo import print_summary


def route_and_select(df: pd.DataFrame, router, router_label: str,
                     seed=None, verbose: bool = True) -> pd.DataFrame:
    """For every (sample_id, question) in the full-matrix `df`, ask `router`
    which condition_id to read the answer from, then pull that exact
    already-scored row out of `df`.

    Returns a DataFrame with the same columns as `df` (so it concatenates
    cleanly with `df` for print_summary): `condition_id` is relabeled to
    `router_label` so the router shows up as its own row in the summary
    table, and the real underlying pick is kept in a new
    `predicted_condition_id` column (plus `seed` for stochastic routers).
    """
    # Dict index instead of a boolean mask per question: with 5 routers x
    # several seeds this runs dozens of times over the same frame, and the
    # mask version is O(rows) per question.
    lookup = {}
    for row in df.to_dict("records"):
        lookup.setdefault((str(row["sample_id"]), row["question"], row["condition_id"]), row)

    unique_q = df.drop_duplicates(["sample_id", "question"])[["sample_id", "question", "category"]]
    rows, missing = [], 0
    for q in unique_q.to_dict("records"):
        sample_id = str(q["sample_id"])
        predicted = router.predict(q["question"], int(q["category"]), sample_id=sample_id).condition_id
        row = lookup.get((sample_id, q["question"], predicted))
        if row is None:
            # predicted a condition_id not present in this CSV — shouldn't
            # happen since the matrix is the fixed 7 from
            # config_2a.build_condition_matrix(), but don't silently drop
            # the question either; fall back to baseline's row.
            missing += 1
            row = lookup.get((sample_id, q["question"], "baseline"))
            if row is None:
                continue
        row = dict(row)
        row["predicted_condition_id"] = predicted
        row["condition_id"] = router_label
        if seed is not None:
            row["seed"] = seed
        rows.append(row)

    if verbose and missing:
        print(f"  [warn] {missing} predictions had no matching row in the CSV -> fell back to baseline")
    return pd.DataFrame(rows)


def _mean_metric(frame: pd.DataFrame, metric: str) -> float:
    return pd.to_numeric(frame[metric], errors="coerce").mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True,
                   help="full-matrix CSV from run_2a_locomo.py, e.g. results/2a_locomo_results_val.csv")
    p.add_argument("--routers", nargs="+", choices=ROUTER_NAMES, default=None,
                   help=f"one or more of {ROUTER_NAMES} (default: random judge learned oracle)")
    p.add_argument("--router", choices=ROUTER_NAMES, default=None,
                   help="deprecated single-router form, kept so old commands keep working")
    p.add_argument("--train-csv", default=None,
                   help="2a_locomo_results_train.csv from `run_2a_locomo.py --split train` "
                        "(required for the learned router)")
    p.add_argument("--metric", default="f1", choices=["f1", "em"],
                   help="metric routers are scored and compared on")
    p.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="seeds for the random router; results are averaged and the spread reported")
    p.add_argument("--judge-use-category", action="store_true",
                   help="also show the LLM judge the gold LoCoMo category name. Diagnostic only — "
                        "it is gold metadata a deployed router would not have.")
    p.add_argument("--wtl-refs", default="all", choices=["all", "conditions", "baseline", "best"],
                   help="what to compare each router against: every fixed condition AND every other "
                        "router (all), the 7 fixed conditions only, baseline only, or the "
                        "best-mean fixed condition only")
    p.add_argument("--wtl-tol", type=float, default=1e-9,
                   help="per-question margin below which a difference counts as a tie. "
                        "Default = exact equality; try 0.05 for 'meaningfully different' ties.")
    p.add_argument("--wtl-by-category", action="store_true",
                   help="additionally print per-category W/T/L for each router vs baseline and "
                        "vs the best fixed condition")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="only used by the naive/judge routers")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="only used by the naive/judge routers")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    router_names = args.routers or ([args.router] if args.router else
                                    ["random", "judge", "learned", "oracle"])
    router_names = list(dict.fromkeys(router_names))  # de-dup, keep order
    metric = args.metric

    if "learned" in router_names and not args.train_csv:
        p.error("the learned router requires --train-csv (see file docstring for the workflow)")

    df = pd.read_csv(args.results_csv)
    df["sample_id"] = df["sample_id"].astype(str)
    for m in ("f1", "em"):
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    condition_ids = [c.condition_id for c in build_condition_matrix()
                     if c.condition_id in set(df["condition_id"])]
    n_questions = df.drop_duplicates(["sample_id", "question"]).shape[0]
    print(f"[data] {args.results_csv}: {len(df)} rows, {n_questions} unique questions, "
          f"{len(condition_ids)} conditions")

    # --- shared dependencies, built once and only if actually needed ----
    llm = embedding_model = train_df = None
    if {"naive", "judge"} & set(router_names):
        from utils.llm_client import LLMClient
        llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url,
                        use_streaming=False)
    if "learned" in router_names:
        from utils.embedding import EmbeddingModel
        embedding_model = EmbeddingModel()
        train_df = pd.read_csv(args.train_csv)
        train_df["sample_id"] = train_df["sample_id"].astype(str)
        print(f"[learned router] fitting on {len(train_df)} rows from {args.train_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- run every router --------------------------------------------- #
    runs: dict = {}   # label -> list of selection DataFrames (>1 only for seeded routers)
    for name in router_names:
        label = f"router({name})"
        seeds = args.random_seeds if name == "random" else [None]
        frames = []
        for seed in seeds:
            router = build_router(
                name, llm=llm, embedding_model=embedding_model, train_df=train_df,
                results_df=df, seed=seed or 0, metric=metric,
                judge_use_category=args.judge_use_category,
            )
            print(f"\n[{label}{'' if seed is None else f' seed={seed}'}] routing "
                  f"{n_questions} questions...")
            sel = route_and_select(df, router, label, seed=seed)
            frames.append(sel)
            usage = dict(Counter(sel["predicted_condition_id"]))
            print(f"  picked: {dict(sorted(usage.items(), key=lambda kv: -kv[1]))}")
            print(f"  mean {metric}: {_mean_metric(sel, metric)*100:.2f}%")
            if hasattr(router, "stats"):
                print(f"  judge stats: {router.stats}")
        runs[label] = frames

        out_path = os.path.join(args.out_dir, f"router_{name}_selection.csv")
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        print(f"  -> {out_path}")

        if len(frames) > 1:
            means = [_mean_metric(f, metric) for f in frames]
            print(f"  across {len(frames)} seeds: mean {np.mean(means)*100:.2f}% "
                  f"+/- {np.std(means)*100:.2f} (min {min(means)*100:.2f}, max {max(means)*100:.2f})")

    # --- table 1: mean metric by condition x category ------------------ #
    # Router rows are the pooled selections; for a seeded router that pooled
    # mean equals the mean over seeds, since every seed routes the same
    # question set.
    router_rows = pd.concat([f for frames in runs.values() for f in frames], ignore_index=True)
    combined = pd.concat([df, router_rows], ignore_index=True)
    print_summary(combined.to_dict("records"))

    summary = (combined.groupby("condition_id")[metric].agg(["mean", "count"])
               .reset_index().rename(columns={"mean": f"mean_{metric}"}))
    summary_path = os.path.join(args.out_dir, f"router_summary_{metric}.csv")
    summary.to_csv(summary_path, index=False)

    # --- table 2: win / tie / lose ------------------------------------- #
    systems = {label: [selection_frame(f, metric) for f in frames]
               for label, frames in runs.items()}

    cond_means = {cid: _mean_metric(df[df["condition_id"] == cid], metric) for cid in condition_ids}
    best_condition = max(cond_means, key=cond_means.get) if cond_means else None

    if args.wtl_refs == "baseline":
        ref_ids = [c for c in ["baseline"] if c in condition_ids]
    elif args.wtl_refs == "best":
        ref_ids = [best_condition] if best_condition else []
    else:
        ref_ids = condition_ids
    references = {cid: selection_frame(df, metric, condition_id=cid) for cid in ref_ids}
    if args.wtl_refs == "all":
        # Router-vs-router too: "learned beats random" is the claim that
        # actually shows routing learned something, and it is a different
        # question from "learned beats baseline". Seeded routers contribute
        # their first seed as the reference so pairing stays a clean 1:1.
        for label, frames in systems.items():
            references[label] = frames[0]

    table = build_wtl_table(systems, references, metric=metric, tol=args.wtl_tol)
    table = table.sort_values(["system", "reference"])
    print_wtl_table(table, metric=metric)
    wtl_path = os.path.join(args.out_dir, f"router_win_tie_lose_{metric}.csv")
    table.to_csv(wtl_path, index=False)

    if best_condition:
        print(f"\nbest fixed condition on this split: {best_condition} "
              f"({cond_means[best_condition]*100:.2f}% mean {metric})")

    # --- table 3 (optional): per-category W/T/L ------------------------ #
    if args.wtl_by_category:
        focus = [c for c in dict.fromkeys(["baseline", best_condition]) if c in references]
        cat_rows = []
        for label, frames in systems.items():
            for ref_label in focus:
                sub = wtl_by_category(frames[0], references[ref_label], metric=metric,
                                      tol=args.wtl_tol)
                if sub.empty:
                    continue
                sub.insert(0, "reference", ref_label)
                sub.insert(0, "system", label)
                cat_rows.append(sub)
        if cat_rows:
            cat_table = pd.concat(cat_rows, ignore_index=True)
            print(f"\n=== per-category win/tie/lose ({metric}, first seed for seeded routers) ===")
            head = (f"{'system':<24s}{'vs':<26s}{'category':<14s}{'W':>5s}{'T':>5s}{'L':>5s}"
                    f"{'net':>9s}{'mean_d':>9s}")
            print(head)
            print("-" * len(head))
            for _, r in cat_table.iterrows():
                print(f"{r['system']:<24s}{r['reference']:<26s}{r['category']:<14s}"
                      f"{r['win']:>5d}{r['tie']:>5d}{r['lose']:>5d}"
                      f"{r['net_win_rate']*100:>8.1f}%{r['mean_delta']*100:>9.2f}")
            cat_path = os.path.join(args.out_dir, f"router_wtl_by_category_{metric}.csv")
            cat_table.to_csv(cat_path, index=False)
            print(f"-> {cat_path}")

    print(f"\nsaved: {summary_path}\n       {wtl_path}")


if __name__ == "__main__":
    main()
