# MemRx 实现说明

每个对话预先按 4 个 view 存好，router 对每个问题挑一个 view 去检索并回答。

## 候选集：3+1

`core/conditions.py::build_condition_matrix`。所有 view 都从同一批 raw chunk（`--window-size` 轮一段）出发，每个 chunk 一次 LLM 调用，view 的单元数与 baseline 相同。

| view | 处理 | 检索 | 参考 |
|---|---|---|---|
| `baseline` | 无 | dense top-k | — |
| `summary__structured` | chunk → C(chunk)：LLM 把 chunk 压缩成**一条**稠密的 lossless_restatement（全名、绝对时间），不保留原文 | dense top-k | — |
| `augmentation__attributes` | chunk → chunk + A(chunk)：原文不动，追加 Entities/Events/Time/Keywords；embedding 和 BM25 用原文+属性，**reader 只看原文**（`metadata["display"]`） | dense 与 BM25 做 RRF，0.7/0.3 | MemInsight |
| `graph__entity` | chunk → G(chunk)：抽 entity，chunk 通过共享 entity 相连 | `0.7·minmax(dense) + 0.3·S_g/max S_g`，`S_g = Σ_{e∈E_q∩E_m} idf(e)`，统一排序取 top-k | Mem0 graph memory |

graph 的 query entity 用 store 里已有的 entity 词表在 query 中做整词匹配，不额外调 LLM；query 不含任何已知 entity 时退化为 dense。summary 的 LLM 输出解析失败 3 次时回退为原文，数量记在构建日志和 `metadata["summary_fallback"]`。

## 数据流

```
scripts/curate_memrx.py   每个问题：probe 一次 + 4 个 view 各回答一次  ->  JSONL
scripts/train_memrx.py    JSONL -> 特征 X (N,d) 和 F1 矩阵 Y (N,K) -> 训练 -> 报表 + 逐题 CSV
```

JSONL 每行字段：`q_emb`, `probe`, `probe_ctx_emb`（router 输入），`pred`, `f1`, `em`（每个 view 的预测和分数），`sample_id`, `question`, `category`, `gold`, `n_retrieved`。

## 输入特征：probe（`memrx/probe.py`）

router 在检索之前决策，所以先在 `baseline` 上做一次 top-`--probe-n`（默认 20）检索，从分数分布和片段 embedding 里算 7 个数：

| 组 | 特征 | 含义 |
|---|---|---|
| conf | `margin_norm`, `decay_norm` | 分数分布有多陡 |
| disp | `n_eff_frac`, `redundancy` | 证据集中还是摊平、top 命中之间多冗余 |
| sem | `cos_q_ctx` | query 和检索质心的距离 |
| cov | `q_cov_max`, `q_cov_gap` | 单个 chunk 能否覆盖 query，还是要取并集 |

`X = [proj(q_emb, 64) ; proj(probe_ctx_emb, 64) ; probe(7)]`，两个投影是固定随机矩阵（`memrx/features.py`）。`--no-probe` 只保留 query 部分。

## Router（`memrx/router.py`）

```
score(x, e_k) = MLP([h_x ; h_e ; h_x * h_e])     e_k = view k 的描述文本 embedding
F1'_k         = best F1   if F1_k >= best - margin，否则 F1_k
target_k      = softmax_k(F1'_k / tau)            tau=0 时是 argmax（差距 ≤ margin 的 view 平分）
loss          = CE(target, softmax_k score)
```

PyTorch 实现，autograd 求梯度，全 batch Adam（weight decay 只加在权重上），固定 epoch，没有 early stopping。网络结构在 `Scorer` 里，直接改 `forward` 即可。

`--margin eps`：同一题里 F1 与最好 view 相差不超过 eps 的 view 视为一样好，避免 0.51 vs 0.50 这种噪声差异变成标签。传多个值就是 sweep，只打印一张汇总表：

```bash
python scripts/train_memrx.py --train ... --val ... --tau 0 --margin 0 0.02 0.05 0.1
```

表里 `tied` = train 上所有 view 都在 margin 内（target 均匀、不提供信号）的题占比；`floor` = target 熵，也就是 loss 能降到的最低值。

## 调试顺序

1. `python tests/smoke_test.py`：target 构造检查 + 合成数据上 router 必须明显超过 best fixed。
2. 看 `train_memrx.py` 每个 split 开头那行 `ties ...`：all-correct 和 all-wrong 的题怎么选都一样，`informative` 才是 router 能拉开差距的部分。
3. 对比 `router` 和 `random` / `best fixed` / `oracle` 四行；再看 `router picks`，全塌到一个 view 说明没学到东西。
4. **train 上 router 远高于 val**：过拟合，调小 `--hidden` / `--epochs`，调大 `--weight-decay`。**train 上也不高**：特征里没信号，对比 `--no-probe`。
5. 打开 `results/router_preds_val.csv`，逐题看 router 选了什么、每个 view 的分数。
