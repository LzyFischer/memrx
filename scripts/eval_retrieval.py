"""
Retrieval-only evaluation: gold-evidence recall@k per view and query type, no reader.

Recall of a question = share of its gold evidence turns covered by the source
turn ranges of the top-k entries. Questions without evidence are skipped.
Stores come from --cache-dir (built with the LLM only if missing).

    python scripts/eval_retrieval.py --split val --by-category

    # graph sweep: every combination is one row
    python scripts/eval_retrieval.py --split val --views baseline graph__entity --by-category \
        --graph-lambda 0.3 0.5 1.0 --graph-knn 0 3 5 --graph-max-entity-df 0 0.1

    --graph-knn 0               entity edges only
    --graph-max-entity-df 0     no entity edges (semantic edges only)
    --graph-lambda 0            identical to baseline
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from core.conditions import build_condition_matrix
from core.graph import ChunkGraph
from core.retrieval import _retrieve_graph, retrieve
from core.views import get_store
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient
from utils.locomo import (CATEGORY_NAMES, QUERY_TYPES, build_dia_id_index, evidence_flat_ids,
                          load_locomo, sample_to_dialogues, split_locomo)


def recall(entries, evidence):
    covered = sum(any(e.metadata.get("dia_id_start", -1) <= d <= e.metadata.get("dia_id_end", -2)
                      for e in entries) for d in evidence)
    return covered / len(evidence)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=config.DATA_PATH)
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--n-train", type=int, default=2)
    p.add_argument("--n-val", type=int, default=1)
    p.add_argument("--views", nargs="*", default=None)
    p.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--by-category", action="store_true")
    p.add_argument("--graph-max-entity-df", type=float, nargs="+", default=[config.GRAPH_MAX_ENTITY_DF])
    p.add_argument("--graph-knn", type=int, nargs="+", default=[config.GRAPH_KNN])
    p.add_argument("--graph-seeds", type=int, nargs="+", default=[config.GRAPH_SEEDS])
    p.add_argument("--graph-lambda", type=float, nargs="+", default=[config.GRAPH_LAMBDA])
    p.add_argument("--cache-dir", default="results/store_cache")
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--overlap", type=int, default=config.OVERLAP_SIZE)
    p.add_argument("--model", default=config.LLM_MODEL)
    p.add_argument("--base-url", default=config.OPENAI_BASE_URL)
    args = p.parse_args()

    data = load_locomo(args.data)
    if args.split != "all":
        data = split_locomo(data, args.n_train, args.n_val)[args.split]
    conditions = [c for c in build_condition_matrix() if not args.views or c.condition_id in args.views]
    llm = LLMClient(api_key=config.OPENAI_API_KEY, model=args.model, base_url=args.base_url,
                    enable_thinking=False, use_streaming=False)
    emb = EmbeddingModel()
    K = max(args.top_k)

    edge_settings = list(itertools.product(args.graph_max_entity_df, args.graph_knn))
    systems = []   # (row name, condition, (edge setting, seeds, lambda) or None)
    for c in conditions:
        if c.dimension != "graph":
            systems.append((c.condition_id, c, None))
            continue
        for (df, knn), seeds, lam in itertools.product(edge_settings, args.graph_seeds, args.graph_lambda):
            if lam == 0 and ((df, knn), seeds) != (edge_settings[0], args.graph_seeds[0]):
                continue   # lambda 0 ignores the graph: one row is enough
            systems.append((f"graph df={df:g} knn={knn} seeds={seeds} lam={lam:g}", c, ((df, knn), seeds, lam)))

    rec = defaultdict(list)   # (system, k, group) -> recalls
    for sample in data:
        sid = sample["sample_id"]
        dialogues, dia_index = sample_to_dialogues(sample), build_dia_id_index(sample)
        stores = {c.condition_id: get_store(dialogues, c, llm, emb, sid, args.cache_dir,
                                            window_size=args.window_size, overlap=args.overlap)[0]
                  for c in conditions}
        qas = [(qa, evidence_flat_ids(qa, dia_index)) for qa in sample["qa"]]
        qas = [(qa, ev) for qa, ev in qas if ev]
        print(f"[{sid}] {len(qas)} questions with evidence")

        graphs = {}
        for name, c, g in systems:
            st = stores[c.condition_id]
            if g is not None:
                (df, knn), seeds, lam = g
                if (df, knn) not in graphs:
                    ids, E = st.embedding_matrix()
                    graphs[(df, knn)] = ChunkGraph(ids, st.entries, E, max_entity_df=df, knn=knn)
                    print(f"  df={df:g} knn={knn}: {graphs[(df, knn)].stats()}")
                st._graph = graphs[(df, knn)]
            for qa, ev in qas:
                if g is None:
                    got = retrieve(st, qa["question"], c, top_k=K)
                else:
                    got = _retrieve_graph(st, qa["question"], K, seeds=g[1], lam=g[2])
                cat = int(qa.get("category", 0))
                for k in args.top_k:
                    r = recall(got[:k], ev)
                    rec[(name, k, "all")].append(r)
                    if cat in QUERY_TYPES:
                        rec[(name, k, CATEGORY_NAMES[cat])].append(r)

    groups = ["all"] + ([CATEGORY_NAMES[c] for c in QUERY_TYPES] if args.by_category else [])
    width = max(len(s[0]) for s in systems) + 2
    for grp in groups:
        n = len(rec[(systems[0][0], args.top_k[0], grp)])
        if not n:
            continue
        print(f"\n=== evidence recall ({grp}, n={n}) ===")
        print(" " * width + "".join(f"{'R@' + str(k):>8s}" for k in args.top_k))
        for name, _, _ in systems:
            print(f"{name:<{width}s}" + "".join(f"{np.mean(rec[(name, k, grp)]):>8.4f}" for k in args.top_k))


if __name__ == "__main__":
    main()
