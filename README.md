# MemRx

Query-level routing over memory processing pipelines. Each conversation is stored under several views (raw / summary / keyword augmentation / entity graph), and a router picks one view per question.

The router conditions on the question plus a cheap probe retrieval on the raw view, and is trained listwise on per-view F1. Details: [`docs/method.md`](docs/method.md).

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
  features.py             JSONL -> X (N, d), F1 matrix (N, K), view embeddings
  router.py               listwise router (PyTorch MLP, softmax(F1/tau) target)
scripts/                  MemRx pipeline
  curate_memrx.py         stage 1: per-question JSONL (probe, prediction + F1/EM per view)
  train_memrx.py          stage 2: fit router, report vs fixed/random/oracle, per-question CSV
prelim/                   motivation experiments, see docs/prelim.md
  run_2a_locomo.py        full view x question matrix -> CSV
  analysis.py             win/tie/loss, per-type heatmap, retrieval vs answer phase
  routers.py              random / LLM-judge / oracle
  run_router_baselines.py
  win_tie_lose.py
utils/
  llm_client.py           OpenAI-compatible client, Qwen3 thinking strip, JSON repair
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
    --val results_synth/memrx_val.jsonl --view-emb-cache results_synth/view_embs.npz --out-dir results_synth

# 1. curate (train = first 2 conversations, val = 3rd, test = rest)
vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384
python scripts/curate_memrx.py --split train --out results/memrx_train.jsonl
python scripts/curate_memrx.py --split val   --out results/memrx_val.jsonl

# 2. train + report (also writes results/router_preds_{train,val}.csv)
python scripts/train_memrx.py --train results/memrx_train.jsonl --val results/memrx_val.jsonl

# ablations are flags
python scripts/train_memrx.py ... --no-probe      # query embedding only
python scripts/train_memrx.py ... --tau 0         # hard argmax classifier
```

Memory stores are cached in `results/store_cache/`, keyed by `(sample_id, view, window, overlap)` and shared between `scripts/curate_memrx.py` and `prelim/run_2a_locomo.py`, so the LLM extraction is paid once per conversation.
