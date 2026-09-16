"""
Stage 1 — curate MemRx router data.

One JSONL record per question holding everything the router needs, so every
router, ablation and oracle downstream is a table lookup instead of another
pass over the LLM:

  q_emb, probe, probe_ctx_emb   router input
  f1[view], em[view]            router supervision

Each view is retrieved from and read from on its own; the only addition on
top of plain per-view QA is the probe pass on the raw view.

    vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384

    python scripts/curate_memrx.py --split train --out results/memrx_train.jsonl
    python scripts/curate_memrx.py --split val   --out results/memrx_val.jsonl
    python scripts/curate_memrx.py --split test  --out results/memrx_test.jsonl

Cost per question: K generations.
Resumable by (sample_id, question); memory stores are cached under
--cache-dir so a resumed run does not re-pay the LLM extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from core.conditions import build_condition_matrix
from core.qa import answer_question, format_context, gold_answer
from core.retrieval import retrieve
from core.views import get_store
from memrx.probe import compute_probe_features, probe_retrieve
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient
from utils.locomo import (
    bleu1_score, build_dia_id_index, evidence_flat_ids, exact_match, f1_score,
    load_locomo, sample_to_dialogues, split_locomo,
)

PROBE_VIEW = "baseline"


def _r(v, nd=5):
    if isinstance(v, np.ndarray):
        return [round(float(x), nd) for x in v]
    return round(float(v), nd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=config.DATA_PATH)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--n-train", type=int, default=2)
    p.add_argument("--n-val", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default="results/store_cache")
    p.add_argument("--model", default=config.LLM_MODEL)
    p.add_argument("--base-url", default=config.OPENAI_BASE_URL)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--top-k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--probe-n", type=int, default=20)
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--overlap", type=int, default=config.OVERLAP_SIZE)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()

    conditions = build_condition_matrix()
    views = [c.condition_id for c in conditions]
    cond_by_id = {c.condition_id: c for c in conditions}

    llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url,
                    enable_thinking=args.thinking, use_streaming=False)
    emb = EmbeddingModel()

    data = load_locomo(args.data)
    if args.split != "all":
        data = split_locomo(data, n_train=args.n_train, n_val=args.n_val)[args.split]
    print(f"[curate] split={args.split}  {len(data)} conversations  views={views}")

    done = set()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["question"]))
                except Exception:
                    pass
        print(f"[curate] resume: {len(done)} records already present")

    out_f = open(args.out, "a", encoding="utf-8")
    n_gen = 0
    t0 = time.time()

    for si, sample in enumerate(data):
        sample_id = str(sample.get("sample_id", si))
        dialogues = sample_to_dialogues(sample)
        dia_index = build_dia_id_index(sample)
        qas = sample.get("qa", [])
        if args.max_questions:
            qas = qas[: args.max_questions]
        qas = [qa for qa in qas if (sample_id, qa["question"]) not in done]
        if not qas:
            continue

        print(f"\n[conv {si+1}/{len(data)}] {sample_id}: {len(dialogues)} turns, "
              f"{len(qas)} pending QAs")
        stores = {}
        for c in conditions:
            t = time.time()
            stores[c.condition_id], cached = get_store(
                dialogues, c, llm, emb, sample_id, args.cache_dir,
                window_size=args.window_size, overlap=args.overlap)
            tag = "cached" if cached else f"built in {time.time()-t:.0f}s"
            print(f"  [{c.condition_id:<24s}] {len(stores[c.condition_id]):4d} units ({tag})")

        for qi, qa in enumerate(qas):
            question = qa["question"]
            category = int(qa.get("category", 1))
            gold = gold_answer(qa)

            q_emb = emb.encode_single(question, is_query=True)

            # ---- probe: one cheap pass on the raw view -----------------
            p_ents, p_scores, p_embs = probe_retrieve(stores[PROBE_VIEW], question, args.probe_n)
            feats, ctx_emb = compute_probe_features(
                question, p_scores, [e.lossless_restatement for e in p_ents], p_embs, q_emb)

            # ---- per-view retrieval + context ------------------------
            contexts, n_retrieved = {}, {}
            for v in views:
                ents = retrieve(stores[v], question, cond_by_id[v], top_k=args.top_k)
                contexts[v] = format_context(ents)
                n_retrieved[v] = len(ents)

            # ---- generation (F1/EM) ----------------------------------
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                preds = dict(zip(views, ex.map(
                    lambda v: answer_question(llm, contexts[v], question, category), views)))
            n_gen += len(views)

            rec = {
                "sample_id": sample_id, "question": question, "category": category,
                "gold": gold, "n_evidence": len(evidence_flat_ids(qa, dia_index)),
                "views": views,
                "q_emb": _r(q_emb),
                "probe": {k: _r(v) for k, v in feats.items()},
                "probe_ctx_emb": _r(ctx_emb),
                "pred": preds,
                "f1": {v: _r(f1_score(preds[v], gold), 4) for v in views},
                "em": {v: _r(exact_match(preds[v], gold), 4) for v in views},
                "bleu1": {v: _r(bleu1_score(preds[v], gold), 4) for v in views},
                "n_retrieved": n_retrieved,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if (qi + 1) % 20 == 0:
                el = time.time() - t0
                print(f"    {qi+1}/{len(qas)} questions | {n_gen} generations | {el/60:.1f} min")

    out_f.close()
    print(f"\n[curate] done -> {args.out}  ({n_gen} generations, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
