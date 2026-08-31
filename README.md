# 2a 实验：memory 处理方式的异质效应

精简自你原来的 MemSuit 代码库，只保留这次 2a 实验（`summary` / `augmentation` / `graph`
三个维度、每次固定另外两个维度为 none）实际用到的文件。8 个 condition：1 baseline + 2 summary
（session_level / fine_grained，hierarchical 已移除）+ 2 augmentation（keywords / note，
temporal 和 causal 均已移除）+ 3 graph。

**完整说明、设计决策、运行步骤见 [`eval/README_2a.md`](eval/README_2a.md)。**

## 目录结构

```
config.py                    # LLM / embedding 基础配置（Qwen3-0.6B 默认）
config_2a.py                 # 8 个 condition 矩阵定义
core/
  memory_builder.py          # 复用自原库，summary 维度的 LLM 抽取 prompt
  chunking.py                # baseline/augmentation/graph 共用的 raw chunk
  augmentation_builder.py    # keywords / note 两个变体的逐 chunk 关键词(+上下文)抽取
  graph_builder.py           # semantic / entity / causal 图构建
  memory_store.py            # 四个维度共用的检索后端（embedding + BM25 + 图邻接表）
  bm25.py                    # 轻量 BM25 实现
  fusion.py                  # Reciprocal Rank Fusion 融合工具
  treatments.py               # 按 condition 分发构建逻辑
  retrieval2a.py              # 按 condition 分发检索逻辑
database/vector_store.py     # 保留（memory_builder.py 的类型引用需要），2a 本身不用 LanceDB
models/memory_entry.py       # MemoryEntry，加了 metadata 字段
utils/llm_client.py          # Qwen3 thinking-mode 处理 + 小模型 JSON 容错
utils/embedding.py           # embedding 模型封装，原样保留
eval/
  locomo_loader.py           # LoCoMo 数据加载 + F1/EM 评分
  run_2a_locomo.py           # 主驱动脚本
  README_2a.md               # 完整实验说明（设计决策 + 运行步骤）
data/                        # 把 locomo10.json 放这里
results/                     # 实验输出 CSV/JSON
```

## 快速开始

```bash
pip install -r requirements.txt
# 把 locomo10.json 放进 data/ (见 data/README.md)

vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384

python eval/run_2a_locomo.py --data data/locomo10.json \
    --model Qwen/Qwen3-1.7B --base-url http://localhost:8000/v1 \
    --out-dir results --max-conversations 1   # 先小样本跑通
```
