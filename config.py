# config.py

# ── LLM 配置 ──────────────────────────────────────────────
OPENAI_API_KEY = "EMPTY"                          # vLLM 默认不校验,填 EMPTY 或任意字符串
OPENAI_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "Qwen/Qwen3-1.7B"

# Qwen3 系列默认开启 thinking mode（chat template 会自动生成 <think>...</think>）。
# 不需要在这里手动开关生成行为——vLLM serve 时思考与否由模型自身的 chat template
# 决定；这个 flag 只影响 utils/llm_client.py 里对 Qwen 官方 dashscope API 的
# enable_thinking 参数传递（本地 vLLM 不吃这个参数，会被忽略）。
# 真正兜底 thinking 输出的是 llm_client.strip_thinking()，对本地 vLLM 无条件生效。
ENABLE_THINKING = True

# ── Embedding 配置 ─────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_CONTEXT_LENGTH = 512

# ── vLLM 对 JSON / Streaming 支持都比 Ollama 好 ─────────────
USE_JSON_FORMAT = False         # vLLM 支持 guided_json / response_format,可以打开
USE_STREAMING = False          # 调试期保持关闭,稳定后可开
ENABLE_THINKING = False

# ── 并发可以适当调高,vLLM 的 batching 比 Ollama 强很多 ──────
WINDOW_SIZE = 5               # 可以恢复到原值
OVERLAP_SIZE = 2
MAX_PARALLEL_WORKERS = 8       # vLLM 内部有 continuous batching,并发收益明显
MAX_RETRIEVAL_WORKERS = 8
ENABLE_PARALLEL_PROCESSING = True
ENABLE_PARALLEL_RETRIEVAL = True

# ── 检索参数 ───────────────────────────────────────────────
SEMANTIC_TOP_K = 15
KEYWORD_TOP_K = 3
STRUCTURED_TOP_K = 3

ENABLE_PLANNING = True
ENABLE_REFLECTION = False       # vLLM 快,可以打开 reflection
MAX_REFLECTION_ROUNDS = 1

# ── 数据库 ─────────────────────────────────────────────────
LANCEDB_PATH = "./lancedb_data"
MEMORY_TABLE_NAME = "memory_entries"

# ── Judge 配置 ─────────────────────────────────────────────
# 如果 judge 用同一个 vLLM 实例,model 名要和启动时一致
# 如果想用不同模型,需要起第二个 vLLM 实例(不同端口)
JUDGE_API_KEY = "EMPTY"
JUDGE_BASE_URL  = "http://localhost:11434/v1"
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_ENABLE_THINKING = False
JUDGE_USE_STREAMING = False
JUDGE_TEMPERATURE = 0.3