"""Global defaults. Every script can override these from the command line."""

# ── LLM (OpenAI-compatible server, e.g. vLLM) ──────────────────────────
OPENAI_API_KEY = "EMPTY"                  # vLLM does not check the key
OPENAI_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3-1.7B"
# Qwen3 thinks by default. False asks vLLM to skip it via chat_template_kwargs;
# utils/llm_client.strip_thinking() cleans the output either way.
ENABLE_THINKING = False
USE_STREAMING = False

# ── Embedding ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Memory construction / retrieval ────────────────────────────────────
WINDOW_SIZE = 5          # turns per raw chunk / summary window
OVERLAP_SIZE = 1
RETRIEVAL_TOP_K = 20

# ── Data ───────────────────────────────────────────────────────────────
DATA_PATH = "data/locomo10.json"
RESULTS_DIR = "results"
