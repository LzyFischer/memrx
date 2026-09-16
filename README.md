# MemRx

Query-level routing over memory processing pipelines. Each conversation is stored under several views (raw / summary / keyword augmentation / entity graph), and a router picks one view per question.

Two components (details in [`docs/method.md`](docs/method.md)):

1. **probe-then-route**: one cheap retrieval on the raw view gives the router the shape of the evidence distribution
2. **mixed-likelihood listwise supervision**: end-to-end F1 ties on most questions, so gold-answer likelihood supplies the ordering where F1 has none

## Layout

```
config.py                 LLM / embedding / window defaults
core/                     memory views (shared by everything)
  conditions.py           the 3+1 candidate menu + one-line view descriptions
  entry.py                Dialogue, MemoryEntry
  chunking.py             raw chunks (baseline / augmentation / graph)
  summary.py              summary=session_level
  augmentation.py         augmentation=keywords
  graph.py                graph=entity
  store.py                MemoryStore: dense + BM25 + graph, pickle cache
  bm25.py
  views.py                build_memory_store / get_store (cached)
  retrieval.py            view-aware retrieval (+ RRF)
  qa.py                   context formatting, answer generation
memrx/                    the method
  probe.py                probe features
  pl_router.py            Plackett-Luce router + mixed target
scripts/                  MemRx pipeline
  curate_memrx.py         stage 1: per-question JSONL (probe, F1/EM, likelihood per view)
  train_memrx.py          stage 2: fit router, report baselines + ablations
  diagnose_memrx.py       routing collapse / headroom / tie decomposition / LL sanity
prelim/                   motivation experiments, see docs/prelim.md
  run_2a_locomo.py        full view x question matrix -> CSV
  analysis.py             win/tie/loss, per-type heatmap, retrieval vs answer phase
  routers.py              random / LLM-judge / oracle
  run_router_baselines.py
  win_tie_lose.py
utils/
  llm_client.py           OpenAI-compatible client, Qwen3 thinking strip, JSON repair, gold logprob
  embedding.py
  locomo.py               LoCoMo loading, QA prompt, F1/EM, split
tests/smoke_test.py       no GPU / no server needed
docs/                     method.md, prelim.md, notes.md
```

## Run

```bash
pip install -r requirements.txt
# data/locomo10.json, see data/README.md

# 0. sanity check, no vLLM or embedding download needed
python tests/smoke_test.py
python scripts/train_memrx.py --train results_synth/memrx_train.jsonl \
    --val results_synth/memrx_val.jsonl --view-emb-cache results_synth/view_embs.npz

# 1. curate (train = first 2 conversations, val = 3rd, test = rest)
vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384
python scripts/curate_memrx.py --split train --out results/memrx_train.jsonl
python scripts/curate_memrx.py --split val   --out results/memrx_val.jsonl
python scripts/curate_memrx.py --split test  --out results/memrx_test.jsonl

# 2. train + report
python scripts/train_memrx.py --train results/memrx_train.jsonl \
    --val results/memrx_val.jsonl --test results/memrx_test.jsonl \
    --tau-sweep 0.0 0.1 1.0 --out results/memrx_report.json

# 3. diagnostics
python scripts/diagnose_memrx.py --train results/memrx_train.jsonl --val results/memrx_val.jsonl
```

Memory stores are cached in `results/store_cache/`, keyed by `(sample_id, view, window, overlap)` and shared between `scripts/curate_memrx.py` and `prelim/run_2a_locomo.py`, so the LLM extraction is paid once per conversation.
