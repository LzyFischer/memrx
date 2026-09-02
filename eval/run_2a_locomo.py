"""
2a 实验主脚本：Does heterogeneous treatment effect exist?

对每个 LoCoMo 对话 × 每个 condition（baseline + 3 summary + 3 augmentation + 3 graph）
× 每个 query，记录一条 performance 记录，长表存到 results/2a_locomo_results.csv。

用法（先起 vLLM）：
    vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --max-model-len 16384

    python eval/run_2a_locomo.py \
        --data data/locomo10.json \
        --model Qwen/Qwen3-0.6B \
        --base-url http://localhost:8000/v1 \
        --out-dir results \
        --max-conversations 2          # 先用小样本跑通，再去掉这个参数跑全量
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import config
from config_2a import build_condition_matrix, RETRIEVAL_TOP_K, WINDOW_SIZE, OVERLAP_SIZE
from core.retrieval2a import retrieve
from core.treatments import build_memory_store
from eval.locomo_loader import (
    CATEGORY_NAMES, build_dia_id_index, build_qa_prompt, evidence_flat_ids, exact_match,
    f1_score, load_locomo, sample_to_dialogues, split_locomo,
)
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


def format_context(entries, max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for i, e in enumerate(entries, 1):
        line = f"[{i}] {e.lossless_restatement}"
        total += len(line)
        if total > max_chars:
            break
        parts.append(line)
    return "\n".join(parts)


def _entry_covers_dialogue(entry, dia_id: int) -> bool:
    """True if `entry` was built from a raw-chunk / summary window whose
    source dialogue-turn range ([dia_id_start, dia_id_end], inclusive)
    contains `dia_id`. Every condition stamps this metadata identically —
    baseline/augmentation/graph via core/chunking.py::build_raw_chunks,
    summary via core/memory_builder.py::_stamp_source_range — so retrieval
    recall is comparable across all 4 dimensions."""
    start = entry.metadata.get("dia_id_start")
    end = entry.metadata.get("dia_id_end")
    if start is None or end is None:
        return False
    return start <= dia_id <= end


def run_one_qa(llm: LLMClient, store, condition, qa, top_k: int, evidence_ids=None):
    category = int(qa.get("category", 1))
    question = qa["question"]
    gold = qa.get("answer", "")
    if category == 5 and not gold:
        gold = "Not mentioned in the conversation"

    t0 = time.time()
    retrieved = retrieve(store, question, condition, llm=llm, top_k=top_k)
    context = format_context(retrieved)

    # Retrieval-phase recall against LoCoMo's gold evidence turns (2a
    # preliminary experiment 3, "retrieval phase单独算recall"). Undefined
    # (empty string, not 0) when the QA has no evidence annotation — e.g.
    # category 5 / adversarial — so it doesn't silently drag down averages.
    evidence_ids = evidence_ids or []
    evidence_total = len(evidence_ids)
    evidence_covered = sum(
        1 for d in evidence_ids if any(_entry_covers_dialogue(e, d) for e in retrieved)
    )
    retrieval_recall = (evidence_covered / evidence_total) if evidence_total else None
    prompt = build_qa_prompt(context, question, category)

    try:
        raw = llm.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,  # generous headroom for Qwen3 thinking-mode trace + short answer
        ).strip()
    except Exception as e:
        raw = f"[error] {e}"

    # build_qa_prompt now asks for {"reasoning": ..., "answer": ...} JSON —
    # pull out just "answer" for scoring. Falls back to the raw text if the
    # model didn't produce valid JSON (rare, but small models occasionally
    # ignore the format instruction), so a parse miss never turns into an
    # empty prediction.
    pred = raw
    try:
        parsed = llm.extract_json(raw)
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip():
                pred = answer.strip()
    except Exception:
        pass
    latency = time.time() - t0

    return {
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "?"),
        "question": question,
        "gold": gold,
        "prediction": pred,
        "f1": f1_score(pred, str(gold)),
        "em": exact_match(pred, str(gold)),
        "n_retrieved": len(retrieved),
        "latency_sec": round(latency, 3),
        "evidence_total": evidence_total,
        "evidence_covered": evidence_covered,
        "retrieval_recall": "" if retrieval_recall is None else round(retrieval_recall, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=config_2a_default_data())
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--overlap", type=int, default=OVERLAP_SIZE)
    p.add_argument("--max-conversations", type=int, default=None,
                    help="仅跑前 N 个对话（调试用，先设成 1-2 跑通再去掉）")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                    help="train=前 n_train 个对话（训练/获取 feedback），"
                         "val=接下来 n_val 个（validation），"
                         "test=其余（held-out test），all=不切分（旧行为）")
    p.add_argument("--n-train", type=int, default=2,
                    help="--split 用：train 集对话数")
    p.add_argument("--n-val", type=int, default=1,
                    help="--split 用：val 集对话数")
    p.add_argument("--conditions", nargs="*", default=None,
                    help="仅跑指定 condition_id（如 summary__hierarchical），默认全部10个")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--thinking", action="store_true",
                    help="关闭 Qwen3 thinking mode（chat_template_kwargs enable_thinking=false）。"
                         "augmentation/graph 维度每个 chunk 都要单独调一次 LLM，关掉思考链能显著提速；"
                         "代价是 0.6B 这种小模型的抽取/QA 准确率可能下降，建议先分别跑一次对比。")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.split}" if args.split != "all" else ""
    out_csv = os.path.join(args.out_dir, f"2a_locomo_results{suffix}.csv")
    out_json = os.path.join(args.out_dir, f"2a_locomo_results{suffix}.json")

    llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url,
                     enable_thinking=args.thinking, use_streaming=False)
    embedding_model = EmbeddingModel()

    data = load_locomo(args.data)
    if args.max_conversations:
        data = data[: args.max_conversations]
    if args.split != "all":
        data = split_locomo(data, n_train=args.n_train, n_val=args.n_val)[args.split]
    print(f"[split={args.split}] {len(data)} conversations")

    conditions = build_condition_matrix()
    if args.conditions:
        conditions = [c for c in conditions if c.condition_id in args.conditions]

    print(f"[2a] {len(data)} conversations x {len(conditions)} conditions")

    fieldnames = [
        "sample_id", "condition_id", "dimension", "summary", "augmentation", "graph",
        "category", "category_name", "question", "gold", "prediction",
        "f1", "em", "n_retrieved", "latency_sec", "n_memory_units",
        "evidence_total", "evidence_covered", "retrieval_recall",
    ]
    results = []
    # Resume support: skip (sample_id, condition_id, question) triples already done.
    done_keys = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(row)
                done_keys.add((row["sample_id"], row["condition_id"], row["question"]))
        print(f"[resume] {len(results)} existing rows found, skipping those.")

    def flush():
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    for sample_idx, sample in enumerate(data):
        sample_id = str(sample.get("sample_id", sample_idx))
        dialogues = sample_to_dialogues(sample)
        dia_id_index = build_dia_id_index(sample)
        qas = sample.get("qa", [])
        print(f"\n[conv {sample_idx+1}/{len(data)}] {sample_id}: {len(dialogues)} turns, {len(qas)} QAs")

        for condition in conditions:
            pending_qas = [qa for qa in qas
                           if (sample_id, condition.condition_id, qa["question"]) not in done_keys]
            if not pending_qas:
                continue

            print(f"  [{condition.condition_id}] building memory store...")
            t0 = time.time()
            store = build_memory_store(
                dialogues, condition, llm, embedding_model,
                window_size=args.window_size, overlap=args.overlap,
            )
            build_time = time.time() - t0
            print(f"  [{condition.condition_id}] {len(store)} memory units built in {build_time:.1f}s")

            for qa in pending_qas:
                evidence_ids = evidence_flat_ids(qa, dia_id_index)
                rec = run_one_qa(llm, store, condition, qa, args.top_k, evidence_ids)
                rec.update({
                    "sample_id": sample_id,
                    "condition_id": condition.condition_id,
                    "dimension": condition.dimension,
                    "summary": condition.summary,
                    "augmentation": condition.augmentation,
                    "graph": condition.graph,
                    "n_memory_units": len(store),
                })
                results.append(rec)
                if len(results) % args.save_every == 0:
                    flush()

    flush()
    print(f"\n[2a] Saved {len(results)} rows -> {out_csv}")
    print_summary(results)


def config_2a_default_data() -> str:
    return "data/locomo10.json"


def print_summary(results):
    """Pivot table: condition_id x category -> mean F1. This is the table
    you eyeball for heterogeneous treatment effect (does the best condition
    change across categories?)."""
    from collections import defaultdict
    grid = defaultdict(list)
    conditions_seen, categories_seen = [], []
    for r in results:
        cid, cat = r["condition_id"], r.get("category_name", "?")
        try:
            f1 = float(r["f1"])
        except (TypeError, ValueError):
            continue
        grid[(cid, cat)].append(f1)
        if cid not in conditions_seen:
            conditions_seen.append(cid)
        if cat not in categories_seen:
            categories_seen.append(cat)

    print("\n=== 2a: mean F1 by condition x category ===")
    header = f"{'condition':<28s}" + "".join(f"{c:>14s}" for c in categories_seen) + f"{'overall':>14s}"
    print(header)
    for cid in conditions_seen:
        row_vals = []
        all_f1 = []
        for cat in categories_seen:
            xs = grid.get((cid, cat), [])
            row_vals.append(sum(xs) / len(xs) if xs else float("nan"))
            all_f1.extend(xs)
        overall = sum(all_f1) / len(all_f1) if all_f1 else float("nan")
        row_str = f"{cid:<28s}" + "".join(f"{v*100:>13.1f}%" for v in row_vals) + f"{overall*100:>13.1f}%"
        print(row_str)


if __name__ == "__main__":
    main()
