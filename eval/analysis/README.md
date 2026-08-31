# 2a preliminary experiments — analysis code

Implements the 3 exploratory analyses from `docs/task.md` (8.25), computed
directly on `results/2a_locomo_results.csv` (the CSV `eval/run_2a_locomo.py`
already writes).

```
pip install pandas matplotlib
```

## Experiments 1 & 2 — no pipeline changes needed

These only use columns `eval/run_2a_locomo.py` has always written
(`sample_id`, `question`, `condition_id`, `dimension`, `category_name`,
`f1`), so they work on any existing `2a_locomo_results.csv` you already have,
old or new.

- **Experiment 1** (`plot_win_tie_loss` / `win_tie_loss_table`): "processing
  不是 consistently 有效的" — for every non-baseline condition, pair each
  question against the baseline row for the same `(sample_id, question)`,
  classify win/tie/loss on the delta in `f1`, one 100%-stacked bar figure
  per dimension. Matches Image 1.
- **Experiment 2** (`plot_win_rate_heatmap` / `win_rate_by_query_type_table`):
  "不同种类的 query 适用什么样的 processing" — same pairing, but win rate
  (ties excluded from the denominator) broken out by `category_name`, one
  heatmap cell per (condition, query type). Matches Image 2.

## Experiment 3 — needs the evidence-tracking changes run first

"选择 processing 在不同的阶段有不同的效果" needs to know whether the
*correct* material was retrieved, which the original CSV schema had no way
to represent (`n_retrieved` is just a count). This repo's
`eval/run_2a_locomo.py` / `eval/locomo_loader.py` / `core/memory_builder.py`
now also write `evidence_total`, `evidence_covered`, `retrieval_recall` per
row — **re-run `eval/run_2a_locomo.py` (or resume an existing run) to
populate these** before calling `retrieval_recall_table` /
`conditional_answer_accuracy_table` / `plot_retrieval_vs_answer_phase`, or
they'll raise (missing column) or silently exclude every old row (NaN
`retrieval_recall`).

- **Retrieval phase** (`retrieval_recall_table`, `retrieval_recall_win_tie_loss_table`): mean
  recall — share of gold LoCoMo `evidence` turns whose source dialogue range
  was covered by some retrieved memory unit — per condition, **including a
  baseline comparison**: `retrieval_recall_table` adds `mean_recall_baseline`
  / `mean_recall_delta` columns (group-mean delta vs baseline), and
  `retrieval_recall_win_tie_loss_table` gives the paired per-question
  win/tie/loss (same question, did this condition retrieve the gold turns
  more/less often than baseline did on that exact question) — the
  retrieval-phase analogue of experiment 1.
- **Answer phase** (`conditional_answer_accuracy_table`,
  `answer_accuracy_win_tie_loss_given_good_retrieval`): mean `f1` restricted
  to questions where `retrieval_recall >= recall_threshold` (default 1.0,
  i.e. every gold turn was actually retrieved) — "given the answerer had the
  right material, how good is the answer" — again **with a baseline
  comparison**: `conditional_answer_accuracy_table` adds baseline
  reference/delta columns (note its two deltas are each computed over a
  different, condition-specific "good retrieval" question subset, so use
  them to see retrieval-problem vs answering-problem at a glance, not for an
  exact paired count); `answer_accuracy_win_tie_loss_given_good_retrieval`
  gives the fairer paired win/tie/loss restricted to questions where BOTH
  the condition and baseline had good retrieval on that exact question, so
  every win/loss there is purely about answer quality.
- `plot_retrieval_vs_answer_phase` draws both phases as absolute-value bars
  side by side with a dashed baseline reference line in each panel;
  `plot_retrieval_recall_win_tie_loss` /
  `plot_answer_accuracy_win_tie_loss_given_good_retrieval` draw the paired
  win/tie/loss versions in the same per-dimension stacked-bar style as
  experiment 1 (saved as `win_tie_loss_retrieval_recall_<dim>.png` /
  `win_tie_loss_f1_given_good_retrieval_<dim>.png` — experiment 1's own
  figures are now `win_tie_loss_f1_<dim>.png`, so all three don't collide
  in the same `--out-dir`).

Recall is computed by checking each retrieved `MemoryEntry`'s
`metadata["dia_id_start"/"dia_id_end"]` (the raw-dialogue-turn range it was
built from) against the gold evidence turns' flat ids — this is populated
identically for all 4 dimensions (`core/chunking.py` for
baseline/augmentation/graph, `core/memory_builder.py::_stamp_source_range`
for summary), so recall is directly comparable across conditions.
`category == 5` (adversarial) questions have no LoCoMo evidence annotation
and are excluded (not scored as recall = 0).

## Usage

```python
from eval.analysis.preliminary_experiments import (
    load_results, plot_win_tie_loss, plot_win_rate_heatmap,
    retrieval_recall_table, retrieval_recall_win_tie_loss_table,
    conditional_answer_accuracy_table, answer_accuracy_win_tie_loss_given_good_retrieval,
    plot_retrieval_recall_win_tie_loss, plot_answer_accuracy_win_tie_loss_given_good_retrieval,
    plot_retrieval_vs_answer_phase,
)

df = load_results("results/2a_locomo_results.csv")
plot_win_tie_loss(df, save_dir="figures/")            # exp 1
plot_win_rate_heatmap(df, save_path="figures/heatmap.png")  # exp 2

# exp 3, all vs baseline:
retrieval_recall_table(df)                             # mean recall + delta vs baseline
retrieval_recall_win_tie_loss_table(df)                # paired win/tie/loss on recall
conditional_answer_accuracy_table(df)                  # mean accuracy + delta vs baseline
answer_accuracy_win_tie_loss_given_good_retrieval(df)  # paired win/tie/loss, both sides retrieval OK
plot_retrieval_recall_win_tie_loss(df, save_dir="figures/")
plot_answer_accuracy_win_tie_loss_given_good_retrieval(df, save_dir="figures/")
plot_retrieval_vs_answer_phase(df, save_path="figures/retrieval_vs_answer.png")
```

Or the CLI, which prints every table and saves every figure in one go:

```bash
python -m eval.analysis.preliminary_experiments \
    --csv results/2a_locomo_results.csv --out-dir results/figures

# on an older CSV without evidence_total/retrieval_recall:
python -m eval.analysis.preliminary_experiments \
    --csv results/2a_locomo_results.csv --out-dir results/figures --skip-retrieval
```
