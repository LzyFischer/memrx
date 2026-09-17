"""Global defaults. Every script can override these from the command line."""

# ── LLM (OpenAI-compatible server, e.g. vLLM) ──────────────────────────
OPENAI_API_KEY = "EMPTY"                  # vLLM does not check the key
OPENAI_BASE_URL = "http://localhost:8123/v1"
LLM_MODEL = "Qwen/Qwen3-1.7B"
# Qwen3 thinks by default. False asks vLLM to skip it via chat_template_kwargs;
# utils/llm_client.strip_thinking() cleans the output either way.
ENABLE_THINKING = False
USE_STREAMING = False

# ── Embedding ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Memory construction / retrieval ────────────────────────────────────
WINDOW_SIZE = 10          # turns per raw chunk / summary window
OVERLAP_SIZE = 1
RETRIEVAL_TOP_K = 20
LLM_WORKERS = 8          # concurrent per-chunk LLM calls when building a view

# graph view (core/graph.py): chunk graph with entity + semantic kNN edges
GRAPH_MAX_ENTITY_DF = 0.1   # entities in more than this share of chunks make no edges (speakers)
GRAPH_KNN = 3               # semantic edges: each chunk links to its KNN most similar chunks
GRAPH_SEEDS = 10             # top dense chunks that pass relevance to their neighbours
GRAPH_LAMBDA = 0.2          # weight of the relevance a neighbour receives; 0 = baseline

# ── Data ───────────────────────────────────────────────────────────────
DATA_PATH = "data/locomo10.json"
RESULTS_DIR = "results"
