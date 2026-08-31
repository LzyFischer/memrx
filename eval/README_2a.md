# 2a 实验：Does heterogeneous treatment effect exist?

## 文件结构（相对你原来的 MemSuit repo）

**新增文件**
- `config_2a.py` —— 7 个 condition 的矩阵定义（1 baseline + 2 summary + 2 augmentation + 2 graph）
- `core/chunking.py` —— baseline/augmentation/graph 三者共用的 raw chunk（不经 LLM 改写）
- `core/augmentation_builder.py` —— keywords / note 两个变体的逐 chunk LLM 抽取（见下）
- `core/graph_builder.py` —— semantic / entity 两种图构建
- `core/memory_store.py` —— 四个维度共用的内存态检索后端，附带 BM25 索引和图邻接表
- `core/bm25.py` —— 轻量 Okapi BM25 实现，无外部依赖
- `core/fusion.py` —— Reciprocal Rank Fusion 工具函数
- `core/treatments.py` —— 根据 Condition 分发到对应的构建逻辑
- `core/retrieval2a.py` —— 根据维度分发到对应的检索逻辑
- `eval/locomo_loader.py` —— LoCoMo 数据加载 + F1/EM 评分
- `eval/run_2a_locomo.py` —— 主驱动脚本

**改了的文件**
- `utils/llm_client.py` —— 加了 `strip_thinking()`（Qwen3 thinking mode 处理）和 `coerce_json_list()`（容错小模型把数组包成对象的畸形输出）
- `config.py` —— `LLM_MODEL` 换成 `Qwen/Qwen3-0.6B`，`ENABLE_THINKING = True`
- `models/memory_entry.py` —— `MemoryEntry` 加了 `metadata: dict` 字段

---

## 三个维度的当前设计（几轮迭代后的版本）

### summary：两种粒度，见 `core/memory_builder.py` + `core/treatments.py::_build_summary_entries`
- `session_level`：每个 window 强制压成 1 条 entry（`single_entry_mode`）
- `fine_grained`：LLM 自己决定切几条（`adaptive_split_mode`）

（原本还有 `hierarchical`——粗细两层都建、检索走 RAPTOR 式 coarse-to-fine——已移除。每个 window 要跑两遍 LLM（一次粗一次细），构建成本接近前两者之和，先砍掉降低复杂度和成本，之后如果发现两种粒度都覆盖不了某类 query，可以再考虑加回来单独对比。）

### augmentation：keywords 和 note，temporal / causal 均已移除

**为什么去掉 causal**：查了 MAGMA、根因分析类记忆系统等几篇论文，因果关系在所有做过的工作里都是当**图**处理的（节点+有向边+多跳遍历），没有一篇是"在单条记忆内部写一句因果摘要文字"。这本身也好理解——因果关系天然是"指向另一条记忆的指针"，单 chunk 内自说自话很容易写成对原文的空泛复述，起不到真正的增强作用。`graph="causal"`（`core/graph_builder.py::build_causal_graph`）已经把这个概念用真正的图结构实现了，`augmentation="causal"` 是它的一个更弱、冗余的版本，直接删掉。

**为什么去掉 temporal**：旧版本是"query 时间意图推断 + 独立日期匹配打分 + RRF 融合"，跟 keywords/note 这两个变体放在一起讲不清楚——temporal 需要额外解释"为什么日期字符串不能直接折进 embedding""为什么要在检索期而不是构建期推断意图"，机制上自成一路，跟另外两个变体不是同一套故事。砍掉之后 augmentation 维度统一成"逐 chunk 抽取 + 用于检索"这一个叙事，两个变体只在"抽取的东西怎么用"上分叉，更容易讲清楚、也更容易在论文里并排对比。

**keywords：LLM 逐 chunk 抽关键词，喂给 BM25，再跟语义检索 RRF 融合**

现在的版本（`core/augmentation_builder.py::augment_keywords` + `core/retrieval2a.py::_retrieve_keywords_hybrid`）：
1. **构建期**：LLM 给每个 raw chunk 抽 3-8 个关键词/短语（人名、地名、专有名词、数字、日期这类），存进 `metadata["keywords"]`——逐 chunk 跑，跟 summary 的每 window 一次调用同量级
2. **检索期**：语义检索拿一批候选（`MemoryStore.semantic_search_scored`），`MemoryStore.bm25_ranked_ids` 用 `core/bm25.py` 里的 Okapi BM25 对同一批 entry 的 `metadata["keywords"]`（不是原文）做稀疏检索，两路排名按 `core/fusion.py::reciprocal_rank_fusion` 的 `0.7 * RRF_语义 + 0.3 * RRF_BM25` 融合（参考 Cognis 论文的融合公式）

这是相对更早一版设计（BM25 直接跑在 raw chunk 全文上，不需要任何构建期 LLM 调用）的改动：那一版更省成本，但 BM25 做的事跟"augmentation 处理"没有关系（不管有没有这个 condition，BM25 都能对任意文本跑）。先让 LLM 抽一遍关键词，稀疏通道匹配的就是模型判断"重要"的词而不是单纯词频高的词，这才是这个 condition 真正想测的东西——具体是否真的比原始全文 BM25 更好，是这轮 ablation 要看的结果，不是先验假定的。

**note：LLM 逐 chunk 抽关键词+上下文，直接拼进原文做 embedding 检索**

现在的版本（`core/augmentation_builder.py::augment_note`）：
1. **构建期**：LLM 给每个 raw chunk 抽 3-8 个关键词 + 一两句"这段在讲什么、为什么可能有用"的上下文注解，两者格式化成一段 `[Note] Keywords: ... | Context: ...`，**直接追加在原文后面**，成为 `entry.lossless_restatement` 的一部分——原文本身不改写，只是在末尾加了这一段
2. **检索期**：不需要单独的检索路径，就是普通的语义检索（`store.semantic_search`），因为 note 已经在构建期被编码进 embedding 了

这跟 keywords 变体是同一件事的两条不同实现路径：keywords 让抽出来的东西走稀疏检索（BM25），note 让抽出来的东西（外加一点解释性上下文）走稠密检索（embedding）。两者构建期都是每个 chunk 恰好一次 LLM 调用，成本上跟 summary 同量级，方便做"同样的抽取成本，走哪条检索路径更有效"这个层面的对比。

### graph：semantic / entity，见 `core/graph_builder.py`

**为什么去掉 causal**：`build_causal_graph` 固定用 backward-only、lookback=5 的候选窗口逐 chunk 调 LLM 判因果，是三个图变体里成本最高的一个（entity 也是逐 chunk 调用，但只需 1 hop 就能用；causal 的因果链要 2 hop 才能体现出跟 semantic/entity 的差异，检索延迟和构建成本都明显更高），在 2a 的初步跑分里也没有看到它相对 semantic/entity 有稳定优势，先砍掉降低复杂度，专注把 semantic/entity 这两个更便宜、更容易讲清楚故事的变体做扎实。

**entity 图加了高频实体过滤**（`build_entity_graph` 里的 `_high_frequency_entities`）：session 数一多，主角名字这种实体会出现在几乎所有 chunk 里，"共享实体就连边"的规则会让 `combinations(n, 2)` 炸出成千上万条边，图直接塌陷成近乎全连接——检索时沿边扩展基本等于把整个记忆库都捞出来，graph 维度名存实亡。现在按实体的文档频率（出现在多少个不同 chunk 里）排序，**频率排进 top 1% 的实体直接不参与建边**（但仍然保留在 `chunk.metadata["entities"]` 里，只是不用来连边）。类比 BM25 用 IDF 压低高频词权重的思路，只是这里的边是二元的（连/不连），没有连续分数可压，所以是硬过滤而不是降权。

小样本保护：如果一个对话总共抽出来的不同实体数少于 20 个（`min_entities_for_filtering`），直接跳过过滤——短对话里"1%"这种统计量没什么意义，硬砍掉出现频率最高的那个实体（很可能就是某个说话人自己的名字）弊大于利。

（semantic 图的 O(N²) 相似度矩阵在 LoCoMo 规模下不是瓶颈，没有改。）

## 已知待办（session 数量大时）

- **entity 图的 LLM 调用次数是 O(chunk 数)**：目前是每个 chunk 单独调一次 LLM 抽实体。session 一多、chunk 数上去之后调用次数会线性增长，是主要的成本瓶颈，还没做批量化（把连续几个 chunk 打包进同一次请求）。

---

## Qwen3 thinking mode 处理

见 `utils/llm_client.py::strip_thinking()`：处理本地 vLLM 默认情况（只露出 `</think>` 闭合标签）和 `--reasoning-parser` 情况（思考内容单独放 `reasoning_content` 字段）。`config.ENABLE_THINKING = False` 时会带上 `chat_template_kwargs: {"enable_thinking": false}` 直接让 vLLM 不生成思考链。

## 小模型 JSON 输出容错

见 `utils/llm_client.py::coerce_json_list()`：Qwen3-0.6B 经常忽略"返回 JSON 数组"的要求、直接吐一个裸对象（尤其是 `single_entry_mode` 下"只有一条"很容易被理解成"一个对象"而不是"长度为1的数组"）。`memory_builder.py` 和 `augmentation_builder.py` 里所有解析 LLM 输出的地方都统一走这个函数，容忍常见的畸形形状，避免白白重试 3 次导致整段数据丢失。

**给 0.6B 模型的成本提示**：`summary`（每个 window 1-2 次）、`augmentation="keywords"`（每个 chunk 1 次）、`augmentation="note"`（每个 chunk 1 次）都是构建期一次过、检索期不再调 LLM，成本量级相当；`graph="semantic"` 完全不需要 LLM。`graph="entity"` 仍然是逐 chunk 调用，成本最高。建议先跑 `--max-conversations 1` 摸一下各 condition 的耗时差异。

---

## 怎么跑（你本地环境，不是这个沙盒）

```bash
# 1. 下载 LoCoMo 数据
mkdir -p data
wget https://github.com/snap-research/locomo/raw/main/data/locomo10.json -O data/locomo10.json

# 2. 起 vLLM
pip install "vllm>=0.6.0"
vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --max-model-len 16384

# 3. 先小样本跑通
python eval/run_2a_locomo.py \
    --data data/locomo10.json \
    --model Qwen/Qwen3-0.6B \
    --base-url http://localhost:8000/v1 \
    --out-dir results \
    --max-conversations 1

# 4. 全量
python eval/run_2a_locomo.py \
    --data data/locomo10.json \
    --model Qwen/Qwen3-0.6B \
    --base-url http://localhost:8000/v1 \
    --out-dir results

# 中途中断可以直接重跑同样的命令，会跳过已经跑过的 (sample_id, condition_id, question)。
# 只想重跑某几个 condition：
python eval/run_2a_locomo.py --data data/locomo10.json --out-dir results \
    --conditions summary__fine_grained augmentation__keywords augmentation__note
```

## 输出

`results/2a_locomo_results.csv`（长表，每行 = 一个 query 在一个 condition 下的结果）：

| 字段 | 含义 |
|---|---|
| sample_id | 对话 id |
| condition_id | 如 `augmentation__keywords` |
| dimension / summary / augmentation / graph | condition 的三个维度取值 |
| category / category_name | LoCoMo 官方类别 |
| question / gold / prediction | 原文 |
| f1 / em | 逐条打分 |
| n_retrieved | 实际检索到的 memory 单元数 |
| latency_sec | 单条 QA 的检索+生成耗时 |
| n_memory_units | 该 condition 下这个对话构建了多少个 memory 单元 |
| evidence_total | 这条 QA 的 gold evidence 轮次数（LoCoMo `qa[].evidence`），adversarial 等无标注问题为 0 |
| evidence_covered | 有多少条 gold evidence 轮次落在某个被检索到的 memory 单元的 `dia_id_start..dia_id_end` 范围内 |
| retrieval_recall | `evidence_covered / evidence_total`，`evidence_total==0` 时留空（不是 0）——见 `eval/analysis/retrieval_recall.py` |

脚本跑完会打印 `condition × category` 的 F1 透视表，用来肉眼判断异质效应存在与否。
