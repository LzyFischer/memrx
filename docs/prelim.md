# Preliminary experiments（motivation 部分）

`prelim/` 里的代码回答 MemRx 之前的两个问题：**异质效应是否存在**，以及**不训练的 router 能拿到多少**。和 MemRx 主线共享 `core/`（建 memory、检索、QA），但不依赖 `memrx/`。

## 1. 全矩阵：`prelim/run_2a_locomo.py`

对每个 LoCoMo 对话 × 每个 view × 每个问题跑一遍 QA，写长表 `results/2a_locomo_results[_split].csv`。可断点续跑（按 `(sample_id, condition_id, question)` 跳过）。

```bash
python prelim/run_2a_locomo.py --split val --out-dir results --max-conversations 1   # 先跑通
python prelim/run_2a_locomo.py --split val --out-dir results
```

主要字段：`condition_id`, `dimension`, `category_name`, `question/gold/prediction`, `f1/em`, `n_retrieved`, `n_memory_units`, `evidence_total`, `evidence_covered`, `retrieval_recall`（`evidence_total==0` 时留空而不是 0，比如 category 5）。

Recall 的算法：检索到的 entry 的 `metadata["dia_id_start".."dia_id_end"]` 是否覆盖 gold evidence 轮次。所有 view 都打这个范围（raw chunk 在 `core/chunking.py`，summary 在 `core/summary.py`），所以跨 view 可比。

## 2. 分析图表：`prelim/analysis.py`

```bash
python -m prelim.analysis --csv results/2a_locomo_results_val.csv --out-dir results/figures
# 旧 CSV 没有 recall 列时加 --skip-retrieval
```

1. **win/tie/loss vs baseline**，每个维度一张堆叠条形图 ——"processing 不是 consistently 有效"
2. **按 query 类型的 win rate 热力图**（平局不进分母）——"不同 query 适用不同 processing"
3. **retrieval phase vs answer phase**：recall 的 delta / 配对 W/T/L；answer 阶段只看两边都检索到 gold evidence 的题

分析函数完全由 CSV 驱动，旧的 7-condition CSV 也能直接画。

## 3. 不训练的 router：`prelim/run_router_baselines.py`

在全矩阵 CSV 上做**选择**，不重建任何 store（query 出现时 memory 早就建好了，路由只能决定读哪个）。

| router | 作用 | 要 LLM |
|---|---|---|
| `random` | 下界：享受"混用多个 view"的收益但没有 query 条件化。按 `(seed, sample_id, question)` 播种，默认 5 个 seed 报 mean ± std | 否 |
| `judge` | closed-set LLM judge：给出每个 view 的行为描述（`core/conditions.py::VIEW_DESCRIPTIONS`），只返回一个 id。非法输出重试，仍失败落回 baseline，次数记在 `stats` 里——**fallback 率高时 F1 不能当 judge 能力读** | 是 |
| `oracle` | 逐题 argmax，headroom 上界，不是 router | 否 |

```bash
python prelim/run_router_baselines.py --results-csv results/2a_locomo_results_val.csv \
    --routers random judge oracle --out-dir results --wtl-by-category
```

`--judge-use-category` 会把 gold 类别喂给 judge，只能当诊断。学习型 router（包括 argmax 分类器，即 `tau=0` + F1-only）在 `scripts/train_memrx.py` 里评估。

### W/T/L 表怎么读

| 列 | 含义 |
|---|---|
| `W / T / L` | 逐题比较高/平/低的题数（多 seed router 取跨 seed 平均） |
| `net` | `(W-L)/n` |
| `mean_d` | 逐题指标差均值 ×100 |
| `p` | 去掉平局后的双边符号检验，`*` = p<0.05 |

LoCoMo 上逐题 F1 是双峰的，同样 +1.5 mean F1 可能是"多数题小赢"也可能是"几道题 0→1"，W/T/L 能区分。平局数要一起报：符号检验只用非平局样本。`--wtl-tol 0.05` 可以只算"实质改变答案"的差异。
