"""
Stage 1 — curate MemRx router data.

One JSONL record per question holding everything the router needs, so every
router, ablation and oracle downstream is a table lookup instead of another
pass over the LLM:

  probe features + query embedding   router input
  f1[view], em[view]                 the censored end-to-end signal
  ll[view]                           mean log P(gold | q, ctx_view), the
                                     uncensored one

Single stage: each view is retrieved from and read from on its own, exactly
as in run_2a_locomo.py. The only additions are the probe pass and the
likelihood pass.

    vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384

    python eval/curate_memrx.py --split train --out results/memrx_train.jsonl
    python eval/curate_memrx.py --split val   --out results/memrx_val.jsonl

Cost per question: K generations (K=4) plus K scoring calls that generate
nothing. Resumable by (sample_id, question); memory stores are cached under
--cache-dir so a resumed run does not re-pay the LLM extraction that built
them.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config_2a import build_condition_matrix, RETRIEVAL_TOP_K, WINDOW_SIZE, OVERLAP_SIZE
from core.memory_store import MemoryStore
from core.probe import compute_probe_features, probe_retrieve
from core.retrieval2a import retrieve
from core.treatments import build_memory_store
from eval.locomo_loader import (
    build_dia_id_index, build_qa_prompt, evidence_flat_ids, exact_match, f1_score,
    load_locomo, sample_to_dialogues, split_locomo,
)
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient

MAX_CHARS = 6000
PROBE_VIEW = "baseline"


def _r(v, nd=5):
    if isinstance(v, np.ndarray):
        return [round(float(x), nd) for x in v]
    return round(float(v), nd)


def format_context(entries, max_chars: int = MAX_CHARS) -> str:
    parts, total = [], 0
    for i, e in enumerate(entries, 1):
        line = f"[{i}] {e.lossless_restatement}"
        total += len(line)
        if total > max_chars:
            break
        parts.append(line)
    return "\n".join(parts)


def scoring_prefix(context: str, question: str) -> str:
    """Plain-text prefix for the likelihood pass.

    Deliberately not the JSON {reasoning, answer} prompt used for generation:
    we are scoring the gold answer string itself, and wrapping it in JSON
    would put most of the measured tokens on punctuation. Only compared
    within a query, so the format offset cancels.
    """
    return (f"Context:\n{context}\n\n"
            f"Answer the question using only the context above.\n"
            f"Question: {question}\nAnswer: ")


# ---------------------------------------------------------------------- #
def _cache_path(cache_dir, sample_id, cid, w, o):
    return os.path.join(cache_dir, f"{sample_id}__{cid}__w{w}o{o}.pkl")


def save_store(store: MemoryStore, path: str) -> None:
    ids = list(store.entries.keys())
    blob = {
        "entries": [store.entries[i].model_dump() for i in ids],
        "embs": np.stack([store._embeddings[i] for i in ids]) if ids else np.zeros((0, 1)),
        "graph": {k: list(v) for k, v in store.graph.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_store(path: str, embedding_model) -> MemoryStore:
    from models.memory_entry import MemoryEntry

    with open(path, "rb") as f:
        blob = pickle.load(f)
    store = MemoryStore(embedding_model)
    for k, d in enumerate(blob["entries"]):
        e = MemoryEntry(**d)
        store.entries[e.entry_id] = e
        store._embeddings[e.entry_id] = blob["embs"][k]
    store.graph = {k: set(v) for k, v in blob["graph"].items()}
    return store


def get_store(dialogues, condition, llm, emb, sample_id, cache_dir, window, overlap):
    path = _cache_path(cache_dir, sample_id, condition.condition_id, window, overlap)
    if os.path.exists(path):
        return load_store(path, emb), True
    store = build_memory_store(dialogues, condition, llm, emb,
                               window_size=window, overlap=overlap)
    save_store(store, path)
    return store, False


# ---------------------------------------------------------------------- #
def answer_one(llm, context, question, category) -> str:
    prompt = build_qa_prompt(context, question, category)
    try:
        raw = llm.chat_completion([{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=1024).strip()
    except Exception as e:
        return f"[error] {e}"
    try:
        parsed = llm.extract_json(raw)
        if isinstance(parsed, dict):
            a = parsed.get("answer")
            if isinstance(a, str) and a.strip():
                return a.strip()
    except Exception:
        pass
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/locomo10.json")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--n-train", type=int, default=2)
    p.add_argument("--n-val", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default="results/store_cache")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    p.add_argument("--probe-n", type=int, default=20)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--overlap", type=int, default=OVERLAP_SIZE)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-likelihood", action="store_true",
                   help="skip the scoring pass (F1-only supervision)")
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
                args.window_size, args.overlap)
            tag = "cached" if cached else f"built in {time.time()-t:.0f}s"
            print(f"  [{c.condition_id:<24s}] {len(stores[c.condition_id]):4d} units ({tag})")

        for qi, qa in enumerate(qas):
            question = qa["question"]
            category = int(qa.get("category", 1))
            gold = qa.get("answer", "")
            if category == 5 and not gold:
                gold = "Not mentioned in the conversation"
            gold = str(gold)

            q_emb = emb.encode_single(question, is_query=True)

            # ---- probe: one cheap pass on the raw view -----------------
            p_ents, p_scores, p_embs = probe_retrieve(stores[PROBE_VIEW], question, args.probe_n)
            feats, ctx_emb = compute_probe_features(
                question, p_scores, [e.lossless_restatement for e in p_ents], p_embs, q_emb)

            # ---- per-view retrieval + context ------------------------
            contexts, n_retrieved = {}, {}
            for v in views:
                ents = retrieve(stores[v], question, cond_by_id[v], llm=llm, top_k=args.top_k)
                contexts[v] = format_context(ents)
                n_retrieved[v] = len(ents)

            # ---- generation (F1/EM) ----------------------------------
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                preds = dict(zip(views, ex.map(
                    lambda v: answer_one(llm, contexts[v], question, category), views)))
            n_gen += len(views)

            # ---- likelihood ------------------------------------------
            if args.no_likelihood:
                lls = {v: None for v in views}
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    lls = dict(zip(views, ex.map(
                        lambda v: llm.gold_answer_logprob(
                            scoring_prefix(contexts[v], question), gold), views)))

            rec = {
                "sample_id": sample_id, "question": question, "category": category,
                "gold": gold, "n_evidence": len(evidence_flat_ids(qa, dia_index)),
                "views": views,
                "q_emb": _r(q_emb),
                "probe": {k: _r(v) for k, v in feats.items()},
                "probe_ctx_emb": _r(ctx_emb),
                "f1": {v: _r(f1_score(preds[v], gold), 4) for v in views},
                "em": {v: _r(exact_match(preds[v], gold), 4) for v in views},
                "ll": {v: (None if lls[v] is None else _r(lls[v], 5)) for v in views},
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
