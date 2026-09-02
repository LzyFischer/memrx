# Router 实验：query 级别选择 condition

在 2a（`eval/README_2a.md`）确认异质效应存在之后，这一层问的是**能不能在 query 级别把最合适的
condition 挑出来**。全部都是在 `2a_locomo_results_*.csv` 上做选择，不重建任何 memory store。

## 新增/改动的文件

- `core/router.py` —— 新增 `RandomRouter`、`LLMJudgeRouter`、`OracleRouter`、`build_router()` 工厂；
  原有 `PromptRouter` / `LearnedRouter` 只改了 `predict()` 签名（多接一个 `sample_id`）
- `eval/analysis/win_tie_lose.py` —— 新增，逐 query 的 win/tie/lose + 符号检验
- `eval/run_router_locomo.py` —— 改成一次跑多个 router（`--routers`），并输出 W/T/L 表。
  旧的 `--router learned` 单个形式仍然可用

## 五个 router

| 名字 | 是什么 | 要 LLM | 要 train CSV | 在论文里的角色 |
|---|---|---|---|---|
| `random` | 从 7 个 condition 里均匀随机抽 | 否 | 否 | **下界对照** |
| `naive` | 让 LLM 自由填 3 个维度的取值（原 `PromptRouter`） | 是 | 否 | 无训练数据的 prompt baseline |
| `judge` | 把 7 个候选连同各自的行为描述列给 LLM judge，只让它返回一个 `condition_id` | 是 | 否 | **无训练数据的 closed-set baseline** |
| `learned` | question embedding (+category one-hot) → logistic regression | 否（要 embedding） | 是 | 你的方法 |
| `oracle` | 逐 query 取 CSV 里 argmax | 否 | 否 | **上界 / headroom** |

### 为什么要加 random

这是整套 routing 叙事里最容易被 reviewer 抓的一刀：如果 learned router 只比 `baseline` 高，
那这个提升可能**完全来自"有时候不用 baseline"**，而不是来自"按 query 选"。random 就是把这个
混淆量化出来的对照——它享受同样的"混用多个 condition"的收益，但没有任何 query 条件化。
learned 必须显著高于 random，"routing 学到了东西"这句话才站得住。

实现上是按 `(seed, sample_id, question)` 播种，而不是一条全局 RNG 流，所以：同一个 seed 下某条
query 的抽取结果跟遍历顺序无关，CSV 被过滤、重排、断点续跑都不影响复现。默认跑 5 个 seed
(`--random-seeds 0 1 2 3 4`)，脚本会打 mean ± std——75~200 条 query 上单次随机抽的标准误
足够大，单 seed 的 random 偶尔能"打赢"真 router，只报一个数会很难看。

### 为什么 judge 和 naive 要分开算两个实验

`naive`（原 `PromptRouter`）让模型自由生成三个维度的取值，然后代码再做合法性修补：非法变体名
降级成 `none`、同时开多个维度时按 `summary > augmentation > graph` 砍掉其余。这两条修补规则本身
就在悄悄注入偏置——小模型输出越乱，结果越向 `baseline` 和 `summary` 塌陷（你在测试里也能看到
0.6B 很容易全塌到 `baseline`）。所以 naive 的分数混了两件事：**judge 能力**和**schema 猜测能力**。

`judge` 把后者去掉：候选表直接列给它，每个候选配一句"这个处理对记忆做了什么、什么时候适合用"
的行为描述（`core/router.py::_CONDITION_DESCRIPTIONS`，故意不用 `augmentation` 这类只有本 repo
内部才有定义的维度名），模型只需要返回一个 id。非法输出会重试（`max_retries=2`），仍然非法才
落回 baseline，落回次数记在 `router.stats` 里跟 F1 一起打出来——**fallback 率高的时候那个 F1 不
能当 judge 的能力读**，要在论文里一起报。

默认只给 question。`--judge-use-category` 可以额外把 LoCoMo 的 category 名喂进去，但那是 gold
标注，真实部署拿不到，只能当诊断实验（"如果 query 类型已知，judge 上限是多少"），不能跟
learned 放在同一列比。

### oracle

不是 router，它读的就是被评测这一 split 自己的分数。放进表里只为了给 headroom：learned 0.31
对 oracle 0.33，和 learned 0.31 对 oracle 0.52，是完全不同的两个故事——前者说明这批 condition
之间本来就没多少可选空间，后者说明选择器还有很大改进余地。

## 怎么跑

```bash
# 前置：train / val 两个 split 的全矩阵（这一步才是花钱的部分）
python eval/run_2a_locomo.py --data data/locomo10.json --split train --out-dir results
python eval/run_2a_locomo.py --data data/locomo10.json --split val   --out-dir results

# 一次把所有 router 跑完 + W/T/L
python eval/run_router_locomo.py \
    --results-csv results/2a_locomo_results_val.csv \
    --routers random naive judge learned oracle \
    --train-csv results/2a_locomo_results_train.csv \
    --model Qwen/Qwen3-0.6B --base-url http://localhost:8000/v1 \
    --out-dir results --wtl-by-category

# 不要 LLM 的快速版（random + oracle，秒出，先确认 headroom 值不值得做）
python eval/run_router_locomo.py \
    --results-csv results/2a_locomo_results_val.csv \
    --routers random oracle --out-dir results
```

只有 `naive` / `judge` 会调 LLM（每条 query 一次，做路由决策），且 `judge` 对同一条 question 文本
有缓存，重复评测不重复花钱。其余 router 全程零 LLM 调用。

常用参数：

- `--metric f1|em` —— 路由标签和 W/T/L 都按这个指标算（oracle 的 argmax 也跟着变）
- `--random-seeds 0 1 2 3 4` —— random 的 seed 列表
- `--wtl-tol 0.05` —— 差距小于这个值算平局。默认是精确相等（只挡浮点噪声）；F1 上
  差一个 token 会让分数微动但答案对错没变，跑一版 `0.05` 的可以回答"有没有**实质**改变答案"
- `--wtl-refs all|conditions|baseline|best` —— 对照谁。`all` = 7 个固定 condition + 其他所有 router
- `--wtl-by-category` —— 再按 LoCoMo category 拆一层

## 输出

```
results/router_<name>_selection.csv     每个 router 选了什么（含 predicted_condition_id、seed）
results/router_summary_f1.csv           每个系统的 mean f1 + 样本数
results/router_win_tie_lose_f1.csv      W/T/L 主表
results/router_wtl_by_category_f1.csv   （加 --wtl-by-category 才有）
```

终端会打两张表：`condition × category` 的 mean F1 透视表（7 个固定 condition + 每个 router 一行），
以及 W/T/L 表。

## W/T/L 表怎么读

| 列 | 含义 |
|---|---|
| `W / T / L` | 逐 query 比较，system 比 reference 高 / 平 / 低的题数（多 seed 的 router 是跨 seed 的**平均**，不是求和，所以跟确定性 router 在同一量纲上） |
| `net` | `(W-L)/n`。routing 效应的方向和大小；跟 `win_rate` 不同，一个赢一半输一半的策略在这里是 0 |
| `mean_d` | 逐 query 指标差的均值 ×100，也就是 mean F1 的差 |
| `p` | 去掉平局后的**双边符号检验**精确 p 值，`*` = p<0.05 |

**为什么均值之外还要这张表**：LoCoMo 上逐 query 的 F1 分布是双峰的（大部分题接近 0 或接近 1），
+1.5 的 mean F1 可能来自"在大多数题上小赢一点"，也可能来自"把 5 道题从 0 翻成 1、其余不变"，
这是两个完全不同的 claim，均值区分不了，W/T/L 能。

**平局数要一起报**：30 胜 5 负这个比例，在 400 道题里（365 平）和在 40 道题里（5 平）意义完全
不同，符号检验只用非平局样本，所以 tie 那一列是解读 p 值的必要背景。

多 seed 的 random 那一行的 p 值只当描述性数字看——几个 seed 共享同一批 query，不是独立样本。

论文里建议摆的顺序：先 mean F1 表（含 random 和 oracle 两行界），再 W/T/L 表只保留几个关键对照
（learned vs baseline、learned vs best fixed condition、**learned vs random**、learned vs judge），
剩下的完整网格放 appendix。

## 已知边界 / 可能的下一步

- **judge 目前只看 query**。CSV 里其实还存着每个 condition 各自的 `prediction`，所以还有一个更强的
  变体：把候选**答案**（而不是候选处理方式）给 judge 选。那测的是"事后挑答案"而不是"事前挑处理
  方式"，成本也是 7 倍的上下文，跟当前这条 routing 叙事不是一件事，所以没有默认实现——如果要做，
  它的位置是介于 judge 和 oracle 之间的另一个上界。
- **learned router 的标签有 ties**：`fit()` 里同分取 condition 矩阵里靠前的那个（baseline 优先），
  oracle 用了同一条规则，这样 oracle 不会因为"平局时偏好贵的 condition"而被额外加分。
- `--routers` 里如果只有 `random`/`oracle`，脚本不会加载 embedding 模型也不会连 vLLM，可以在没有
  GPU 的机器上直接跑分析。
