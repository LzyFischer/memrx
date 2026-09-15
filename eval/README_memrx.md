# MemRx 实现说明

单阶段路由，4 个候选，两个组件：

1. **probe-then-route** — 路由器在真正检索之前决策，所以用一次廉价检索的分布形状补足输入
2. **mixed-likelihood listwise supervision** — end-to-end 指标是被截断的观测，在大多数 query 上打平；用 gold answer 的 likelihood 在打平处恢复序关系

## 候选集：3+1

`config_2a.py::build_condition_matrix` 收到 4 个：

| view | 构建期 LLM 调用 |
|---|---|
| `baseline` | 0 |
| `summary__session_level` | 每窗口 1 次 |
| `augmentation__keywords` | 每 chunk 1 次 |
| `graph__entity` | 每 chunk 1 次 |

每个维度只留一个代表。原来两个变体/维度是为了回答"哪个变体更好"，和路由问的不是同一个问题，而且两个 summary 变体之间的竞争远强于它们和 graph 的竞争——大部分路由信号会花在维度**内部**的区分上。

`core/router.py` 里那些旧 router（PromptRouter / LLMJudgeRouter / OracleRouter）和 `run_2a_locomo.py` 都跟着新矩阵走，不用改。

## 新增/改动文件

| 文件 | 内容 |
|---|---|
| `config_2a.py` | 候选集收到 4 个 |
| `core/probe.py` | probe 特征（7 维）+ 加权池化 context embedding |
| `core/pl_router.py` | PL listwise 路由器 + `mixed_target` |
| `utils/llm_client.py` | 新增 `gold_answer_logprob` |
| `eval/curate_memrx.py` | Stage 1：curation |
| `eval/train_memrx.py` | Stage 2：训练 + val 评估 |
| `tests/smoke_test.py` | 梯度检验 + `mixed_target` 语义 + 合成数据 |

`core/__init__.py` 改成惰性导入，这样 `core.probe` / `core.pl_router` 不再顺带拉进 lancedb。`from core import MemoryBuilder` 行为不变。

没有 early stopping，没有 dev split，固定 epoch 数训到底。也没有加成本项、no-retrieval 候选、两阶段——只有上面两个组件。

## 组件 1：probe

在 `baseline` 视图上做一次 top-10 检索，从**分数分布**和**片段 embedding** 里抽 7 个量，全部是已有产物上的 numpy 运算：

| 组 | 特征 | 含义 |
|---|---|---|
| conf | `margin_norm`, `decay_norm` | 分数分布有多陡（除以组内 std，否则带进 query 难度） |
| disp | `n_eff_frac`, `redundancy` | 证据集中还是摊平、top 命中之间有多冗余 |
| sem | `cos_q_ctx` | query 和检索到的质心有多远 |
| cov | `q_cov_max`, `q_cov_gap` | 单个 chunk 能否覆盖 query，还是必须取并集 |

`q_cov_gap > 0` 直接量化"必须合并多条才能覆盖 query"。

路由器输入 = `[proj(q_emb, 64) ; proj(ctx_emb, 32) ; probe(7)]`。两个投影是固定随机投影（seeded，不拟合任何数据，因此不泄漏）——384 维原样喂进去会靠维度数压过 7 个统计量。

## 组件 2：混合监督

```
p_F1(k) = softmax_k( s_k / tau )
p_LL(k) = softmax_k( l_k / tau_ll )
alpha_q = clip( (max_k s_k - min_k s_k) / delta , 0, 1 )
p~      = alpha_q * p_F1 + (1 - alpha_q) * p_LL

L = KL( p~ || p_theta ),   p_theta(k|x) = softmax_k f_theta(x, e_k)
```

**alpha 是按组算的，不是全局超参。** F1 能区分开时完全信 F1；F1 打平时 alpha→0，这一组改由 likelihood 供能，而不是被丢掉——这正是"有效训练数据变少"的对策。

两个信号不能在原始分数层相加：F1 是 [0,1]，likelihood 是负值且尺度随 query 变。但**组内**可比——同一组里 `a*` 相同、长度归一化的分母相同，所以 `l_k` 之间的差只反映上下文差异。PL 只吃组内序，这个性质正好让两者能共用一个损失。

两个极限：`alpha≡1` 是 F1-only 监督；`alpha≡1 且 tau→0` 是 argmax 分类器。所以分类 baseline 是这一族里的一个点，代码里 `tau=0.0` 就是走这条路径。

likelihood 缺失（服务器不支持 echo/logprobs）的格子记 `None`，那一组自动退回 F1-only，不会中断 curation。

打分函数吃**视图描述的 embedding** `e_k` 而不是 K 个输出头，所以训练时没见过的视图仍能打分。

## likelihood 怎么拿

`/v1/completions` 加 `echo=True, max_tokens=0, logprobs=0`，不生成任何 token，直接读服务器给 prompt 打的 logprob，按 `text_offset` 切出 gold answer 那一段，除以 token 数。比一次生成便宜大约一个量级。

打分用的是纯文本 prefix，不是生成时的 JSON `{reasoning, answer}` 模板——把 gold answer 包进 JSON 会让大部分被测 token 落在标点上。因为只在组内比较（同一个 `a*`，只有上下文不同），格式带来的偏移会抵消。

## 跑的顺序

```bash
# 0. 不需要 vLLM，不需要下载 embedding 模型
python tests/smoke_test.py
python eval/train_memrx.py --train results_synth/memrx_train.jsonl \
    --val results_synth/memrx_val.jsonl --view-emb-cache results_synth/view_embs.npz
```

```bash
vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000 --max-model-len 16384

# 1. curation（train = 前 2 个对话，val = 第 3 个，和原 split 一致）
python eval/curate_memrx.py --split train --out results/memrx_train.jsonl
python eval/curate_memrx.py --split val   --out results/memrx_val.jsonl

# 2. 训练 + val 报表
python eval/train_memrx.py --train results/memrx_train.jsonl \
    --val results/memrx_val.jsonl --tau-sweep 0.0 0.05 0.1 0.3 1.0 \
    --out results/memrx_report.json
```

每题 4 次生成 + 4 次打分（不生成）。train 集 199 QA × 2 conv ≈ 1.6k 次生成。可断点续跑，memory store 缓存在 `--cache-dir`。

## 训练脚本会先打出来的一个数

```
[data] train f1 ties: XX.X%  (all-correct XX.X%, all-wrong XX.X%)
```

这是组件 2 的全部动机。三个区要分开看：

- **all-correct 打平** = ceiling，阅读器不需要帮助
- **all-wrong 打平** = floor，任何 processing 都救不了
- 剩下的才是 F1 有区分度的有效样本

如果打平比例很高，同时 `MemRx − likelihood` 那一行明显低于 `MemRx`，组件 2 就立住了。

## 主要风险

`gold_answer_logprob` 可能主要在测**上下文长度**而不是证据质量——各视图的上下文长度差异很大，logprob 很容易跟着长度走。这是我认为最大的隐患，建议 curation 跑完先做两件事：

1. 只在 F1 有区分度的组上，比 `argmax_k l_k` 和 `argmax_k s_k` 的一致率。如果接近随机，说明 likelihood 测的不是同一个东西，拿它打破平局就是在引入偏差——**这条不过，组件 2 作废**。
2. 算 `l_k` 与上下文 token 数的偏相关。相关性主要来自长度的话，需要先做长度校正。

记录里存了 `n_retrieved`，上下文长度可以从 curation 再补，这两个检查都不需要重跑 LLM。

## 已验证 / 未验证

这边跑过的：手写反向传播的有限差分检验（78 个非零梯度点，最大相对误差 1.5e-07）；`mixed_target` 的 alpha 行为、NaN 回退、`tau=0` 退化；合成数据上整条训练/评估链路能把埋进去的结构 recover 出来。

没跑过的：**真实数据一次都没跑**——这个环境没有 vLLM，也下载不了 sentence-transformers 权重。合成数据只验证代码通路，不验证想法。


python eval/analysis/diagnose_memrx.py \
    --train results/memrx_train.jsonl --val results/memrx_val.jsonl