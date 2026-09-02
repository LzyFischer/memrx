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
question). This script does that lookup, then prints ALL conditions' mean
F1 (the full comparison table, not just the router's single number) with
the router's selected-only performance appended as one more row, so you
read the router's number in context.

Workflow:
    # 1) full matrix on train — the feedback LearnedRouter fits on
    python eval/run_2a_locomo.py --data data/locomo10.json --split train --out-dir results

    # 2) full matrix on val — what the router will SELECT from
    python eval/run_2a_locomo.py --data data/locomo10.json --split val --out-dir results

    # 3) route val's questions, print full condition x category table + router row
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_val.csv \
        --router learned --train-csv results/2a_locomo_results_train.csv \
        --out-dir results

    # naive router (no train-csv needed, but does call the LLM once per
    # question to decide — still zero rebuilds of memory stores):
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_val.csv \
        --router naive --model Qwen/Qwen3-0.6B --base-url http://localhost:8000/v1 \
        --out-dir results

    # 4) once a router is picked on val, get the final table on test
    python eval/run_2a_locomo.py --data data/locomo10.json --split test --out-dir results
    python eval/run_router_locomo.py \
        --results-csv results/2a_locomo_results_test.csv \
        --router learned --train-csv results/2a_locomo_results_train.csv \
        --out-dir results
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import pandas as pd

from core.router import LearnedRouter, PromptRouter
from eval.run_2a_locomo import print_summary
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


def route_and_select(df: pd.DataFrame, router, router_label: str) -> pd.DataFrame:
    """For every (sample_id, question) in the full-matrix `df`, ask `router`
    which condition_id to read the answer from, then pull that exact
    already-scored row out of `df`.

    Returns a DataFrame with the same columns as `df` (so it concatenates
    cleanly with `df` for print_summary): `condition_id` is relabeled to
    `router_label` so the router shows up as its own row in the summary
    table, and the real underlying pick is kept in a new
    `predicted_condition_id` column.
    """
    unique_q = df.drop_duplicates(["sample_id", "question"])[["sample_id", "question", "category"]]
    rows = []
    for _, q in unique_q.iterrows():
        predicted = router.predict(q["question"], int(q["category"])).condition_id
        match = df[(df["sample_id"] == q["sample_id"]) &
                    (df["question"] == q["question"]) &
                    (df["condition_id"] == predicted)]
        if match.empty:
            # predicted a condition_id not present in this CSV — shouldn't
            # happen since the matrix is the fixed 7 from
            # config_2a.build_condition_matrix(), but don't silently drop
            # the question either; fall back to baseline's row.
            match = df[(df["sample_id"] == q["sample_id"]) &
                       (df["question"] == q["question"]) &
                       (df["condition_id"] == "baseline")]
            if match.empty:
                continue
        row = match.iloc[0].to_dict()
        row["predicted_condition_id"] = predicted
        row["condition_id"] = router_label
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True,
                    help="full-matrix CSV from run_2a_locomo.py, e.g. results/2a_locomo_results_val.csv")
    p.add_argument("--router", choices=["naive", "learned"], default="learned")
    p.add_argument("--train-csv", default=None,
                    help="2a_locomo_results_train.csv from `run_2a_locomo.py --split train` "
                         "(required for --router learned)")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="only used by --router naive")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="only used by --router naive")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    if args.router == "learned" and not args.train_csv:
        p.error("--router learned requires --train-csv (see file docstring for the workflow)")

    df = pd.read_csv(args.results_csv)
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    router_label = f"router({args.router})"
    if args.router == "naive":
        llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url, use_streaming=False)
        router = PromptRouter(llm)
    else:
        embedding_model = EmbeddingModel()
        train_df = pd.read_csv(args.train_csv)
        router = LearnedRouter(embedding_model).fit(train_df)
        print(f"[learned router] fit on {len(train_df)} rows from {args.train_csv}")

    router_df = route_and_select(df, router, router_label)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"router_{args.router}_selection.csv")
    router_df.to_csv(out_path, index=False)
    print(f"\n[router={args.router}] {len(router_df)} / {df['question'].nunique()} questions routed -> {out_path}")
    print("condition usage:", dict(Counter(router_df["predicted_condition_id"])))

    # Full comparison table: all 7 original conditions (from the untouched
    # `df`) PLUS the router's selected-only row appended at the bottom —
    # read the router's number in context, not alone.
    combined = pd.concat([df, router_df], ignore_index=True)
    print_summary(combined.to_dict("records"))


if __name__ == "__main__":
    main()