# MemRx 实现说明

每个对话预先按 4 个 view 存好，router 对每个问题挑一个 view 去检索并回答。

## 候选集：3+1

`core/conditions.py::build_condition_matrix`

| view | 构建期 LLM 调用 |
|---|---|
| `baseline` | 0 |
| `summary__session_level` | 每窗口 1 次 |
| `augmentation__keywords` | 每 chunk 1 次 |
| `graph__entity` | 每 chunk 1 次 |

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
