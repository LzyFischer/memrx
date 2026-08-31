"""
2a preliminary experiments — computed directly from `results/2a_locomo_results.csv`
(the long-format output of eval/run_2a_locomo.py).

Implements the three exploratory analyses from docs/task.md (8.25):

  1. win_tie_loss    — "processing 并不是 consistently 有效的": for every
                        non-baseline condition, paired win/tie/loss vs
                        baseline on the same (sample_id, question), one
                        stacked-bar figure per dimension (summary /
                        augmentation / graph). See Image 1 in the task.

  2. query_type_heatmap — "不同种类的query适用什么样的processing": win rate
                        vs baseline (ties excluded) broken out by LoCoMo
                        category, condition x category heatmap. See Image 2.

  3. retrieval_vs_answer — "选择processing在不同的阶段有不同的效果": retrieval
                        phase measured as recall against gold evidence turns,
                        answer phase measured as accuracy restricted to
                        questions where retrieval succeeded. Both phases are
                        reported vs baseline: group-mean deltas
                        (retrieval_recall_table / conditional_answer_accuracy_table),
                        paired per-question win/tie/loss
                        (retrieval_recall_win_tie_loss_table /
                        answer_accuracy_win_tie_loss_given_good_retrieval),
                        and a combined absolute-value chart with a baseline
                        reference line (plot_retrieval_vs_answer_phase).

Experiments 1 and 2 only need columns that eval/run_2a_locomo.py has always
written (sample_id, question, condition_id, dimension, category_name, f1).
Experiment 3 needs the evidence_total / evidence_covered / retrieval_recall
columns added alongside this file — re-run eval/run_2a_locomo.py (or resume
an existing run; new columns backfill as empty/0 for old rows, which
correctly means "no recall signal for that row", not "recall = 0" — see
_only_scored_rows below) to populate them.

All functions take a plain pandas DataFrame (load with `load_results`) and
return either a table (DataFrame/Series, for eyeballing or a paper table)
or a matplotlib Figure (for the visual). No function reads or writes CSVs
by itself except `load_results`, so these compose freely — e.g. run
win_tie_loss_table on a filtered subset of `df` (a single sample_id, a
single category) for a slice you're debugging.

Usage as a library:
    from eval.analysis.preliminary_experiments import load_results, plot_win_tie_loss
    df = load_results("results/2a_locomo_results.csv")
    figs = plot_win_tie_loss(df)  # {"summary": Figure, "augmentation": Figure, "graph": Figure}

Usage as a CLI (generates every figure + prints every table):
    python -m eval.analysis.preliminary_experiments --csv results/2a_locomo_results.csv --out-dir figures/
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_ID = "baseline"

# Consistent colors across figures 1 and 2 so the two are visually linked.
WIN_COLOR = "#1f6fd6"
TIE_COLOR = "#c9c5b8"
LOSS_COLOR = "#d64545"


# --------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------- #
def load_results(csv_path: str) -> pd.DataFrame:
    """Load a 2a_locomo_results.csv into a typed DataFrame.

    Numeric columns are coerced explicitly rather than left to pandas'
    dtype sniffing, because resumed/partial runs can mix a header written
    before evidence_total/evidence_covered/retrieval_recall existed with
    rows written after — those older rows read back as empty strings for
    the new columns, which must become NaN (not 0) so experiment 3 doesn't
    silently treat "no recall signal" as "recall = 0".
    """
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "f1", "em", "n_retrieved", "latency_sec", "n_memory_units",
        "evidence_total", "evidence_covered", "retrieval_recall",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _dimension_conditions(df: pd.DataFrame, dimension: str) -> List[str]:
    """condition_ids belonging to one dimension, in the order they first
    appear in the CSV (i.e. the order config_2a.py's build_condition_matrix
    produced them in) — keeps figure row order stable and matching the
    condition matrix rather than alphabetical."""
    sub = df[df["dimension"] == dimension]
    return list(dict.fromkeys(sub["condition_id"]))


def _variant_label(condition_id: str) -> str:
    """'augmentation__keywords' -> 'keywords' (drop the dimension prefix
    for the win_tie_loss per-dimension figures, which already say the
    dimension in the title)."""
    return condition_id.split("__", 1)[-1] if "__" in condition_id else condition_id


# --------------------------------------------------------------------- #
# Experiment 1 — win / tie / loss vs baseline, one figure per dimension
# --------------------------------------------------------------------- #
def _pair_with_baseline(df: pd.DataFrame, metric: str, baseline_id: str = BASELINE_ID) -> pd.DataFrame:
    """Inner-join every non-baseline row to its baseline row on
    (sample_id, question), so each pair is a same-question, same-dialogue
    comparison. Rows whose (sample_id, question) has no baseline row (e.g.
    baseline hasn't finished running yet, or a resumed run skipped it) are
    dropped rather than silently scored as ties.
    """
    base = df[df["condition_id"] == baseline_id][["sample_id", "question", metric]]
    base = base.rename(columns={metric: f"{metric}_baseline"})
    treat = df[df["condition_id"] != baseline_id]
    paired = treat.merge(base, on=["sample_id", "question"], how="inner")
    paired["delta"] = paired[metric] - paired[f"{metric}_baseline"]
    return paired


def win_tie_loss_table(
    df: pd.DataFrame, metric: str = "f1", tie_epsilon: float = 1e-9,
    baseline_id: str = BASELINE_ID,
) -> pd.DataFrame:
    """One row per non-baseline condition_id: n paired questions, win/tie/loss
    counts and shares vs baseline on `metric`.

    tie_epsilon is an absolute tolerance on the delta (treatment metric -
    baseline metric): |delta| <= tie_epsilon counts as a tie. Default is a
    near-zero float tolerance (i.e. "tie" means literally the same score,
    which happens whenever both conditions produce the same prediction) —
    widen it (e.g. 0.05) if you want "materially the same" rather than
    "identical" to count as a tie.
    """
    paired = _pair_with_baseline(df, metric, baseline_id)

    def _classify(delta: float) -> str:
        if delta > tie_epsilon:
            return "win"
        if delta < -tie_epsilon:
            return "loss"
        return "tie"

    paired["outcome"] = paired["delta"].apply(_classify)

    rows = []
    for condition_id, g in paired.groupby("condition_id", sort=False):
        counts = g["outcome"].value_counts()
        n = len(g)
        win, tie, loss = counts.get("win", 0), counts.get("tie", 0), counts.get("loss", 0)
        rows.append({
            "condition_id": condition_id,
            "dimension": g["dimension"].iloc[0],
            "n": n,
            "win": win, "tie": tie, "loss": loss,
            "win_pct": win / n if n else float("nan"),
            "tie_pct": tie / n if n else float("nan"),
            "loss_pct": loss / n if n else float("nan"),
        })
    out = pd.DataFrame(rows).set_index("condition_id")
    # Preserve condition_matrix order (see _dimension_conditions) rather
    # than groupby's arbitrary-ish ordering.
    order = list(dict.fromkeys(df.loc[df["condition_id"] != baseline_id, "condition_id"]))
    return out.reindex([c for c in order if c in out.index])


def plot_win_tie_loss(
    df: pd.DataFrame, metric: str = "f1", tie_epsilon: float = 1e-9,
    baseline_id: str = BASELINE_ID, dimensions: Optional[Sequence[str]] = None,
    save_dir: Optional[str] = None, file_tag: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """One 100%-stacked horizontal bar figure per dimension (Image 1 style):
    each row is a condition (variant) within that dimension, segments are
    win / tie / loss share vs baseline.

    Returns {dimension_name: Figure}. If save_dir is given, also saves each
    as "<save_dir>/win_tie_loss_<file_tag>_<dimension>.png" (file_tag
    defaults to `metric`, so experiment-1's f1 figures and experiment-3's
    retrieval_recall figures don't collide when saved to the same dir).
    """
    table = win_tie_loss_table(df, metric=metric, tie_epsilon=tie_epsilon, baseline_id=baseline_id)
    dims = list(dimensions) if dimensions else [
        d for d in dict.fromkeys(df["dimension"]) if d != baseline_id and d != "baseline"
    ]
    tag = file_tag or metric

    figs = {}
    for dimension in dims:
        cond_ids = _dimension_conditions(df, dimension)
        sub = table.reindex([c for c in cond_ids if c in table.index])
        if sub.empty:
            continue

        labels = [_variant_label(c) for c in sub.index]
        fig, ax = plt.subplots(figsize=(8, 0.9 * len(sub) + 1.2))
        y = np.arange(len(sub))

        ax.barh(y, sub["win_pct"], color=WIN_COLOR, label="win")
        ax.barh(y, sub["tie_pct"], left=sub["win_pct"], color=TIE_COLOR, label="tie")
        ax.barh(y, sub["loss_pct"], left=sub["win_pct"] + sub["tie_pct"], color=LOSS_COLOR, label="loss")

        for i, (_, row) in enumerate(sub.iterrows()):
            for pct, offset, color in [
                (row["win_pct"], 0.0, "white"),
                (row["tie_pct"], row["win_pct"], "#3a362c"),
                (row["loss_pct"], row["win_pct"] + row["tie_pct"], "white"),
            ]:
                if pct > 0.06:  # skip labels on slivers too thin to read
                    ax.text(offset + pct / 2, i, f"{pct*100:.0f}%",
                             ha="center", va="center", color=color, fontsize=10)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Share of questions ({metric} win/tie/loss vs baseline)")
        ax.set_title(f"{dimension.capitalize()} dimension — win / tie / loss vs baseline")
        ax.invert_yaxis()
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
        fig.tight_layout()

        figs[dimension] = fig
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"win_tie_loss_{tag}_{dimension}.png"), dpi=150, bbox_inches="tight")

    return figs


# --------------------------------------------------------------------- #
# Experiment 2 — win rate by condition x query type (ties excluded)
# --------------------------------------------------------------------- #
def win_rate_by_query_type_table(
    df: pd.DataFrame, metric: str = "f1", tie_epsilon: float = 1e-9,
    baseline_id: str = BASELINE_ID, category_col: str = "category_name",
) -> pd.DataFrame:
    """condition_id x category_name pivot table of win rate vs baseline,
    ties EXCLUDED from the denominator (win / (win + loss), not win / n) —
    matches Image 2. Cells with zero decided (non-tie) comparisons are NaN,
    not 0, so they render blank rather than misleadingly "worst".

    condition_id is relabeled "dimension: variant" (e.g.
    "augmentation: keywords") to match Image 2's row labels.
    """
    paired = _pair_with_baseline(df, metric, baseline_id)

    def _classify(delta: float) -> str:
        if delta > tie_epsilon:
            return "win"
        if delta < -tie_epsilon:
            return "loss"
        return "tie"

    paired["outcome"] = paired["delta"].apply(_classify)
    decided = paired[paired["outcome"] != "tie"]

    pivot = (
        decided.groupby(["condition_id", category_col])["outcome"]
        .apply(lambda s: (s == "win").mean())
        .unstack(category_col)
    )
    # Row order = condition matrix order; column order = first-seen category order.
    row_order = list(dict.fromkeys(df.loc[df["condition_id"] != baseline_id, "condition_id"]))
    col_order = list(dict.fromkeys(df[category_col]))
    pivot = pivot.reindex(index=[c for c in row_order if c in pivot.index],
                           columns=[c for c in col_order if c in pivot.columns])

    pivot.index = [
        f"{cid.split('__', 1)[0]}: {_variant_label(cid)}" if "__" in cid else cid
        for cid in pivot.index
    ]
    return pivot


def plot_win_rate_heatmap(
    df: pd.DataFrame, metric: str = "f1", tie_epsilon: float = 1e-9,
    baseline_id: str = BASELINE_ID, category_col: str = "category_name",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Diverging heatmap of win rate vs baseline (ties excluded), condition x
    category — Image 2. Centered at 50% (below = red, above = blue) since
    50% is "no signal either way" for a win/loss-only rate.
    """
    pivot = win_rate_by_query_type_table(df, metric=metric, tie_epsilon=tie_epsilon,
                                          baseline_id=baseline_id, category_col=category_col)

    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 3, 0.6 * pivot.shape[0] + 1.5))
    masked = np.ma.masked_invalid(pivot.values)
    im = ax.imshow(masked, cmap="RdBu", vmin=0.2, vmax=0.8, aspect="auto")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            color = "white" if (v < 0.32 or v > 0.68) else "#3a362c"
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Win rate vs baseline by query type ({metric}, ties excluded)")
    fig.colorbar(im, ax=ax, label="win rate (%)", fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------- #
# Experiment 3 — retrieval-phase recall vs answer-phase (conditional) accuracy
# --------------------------------------------------------------------- #
def _only_scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a defined retrieval_recall — i.e. the QA had gold evidence
    turns AND was run after the evidence_total/retrieval_recall columns
    existed. Excludes both category-5/adversarial questions (no evidence by
    design) and any pre-instrumentation rows (NaN, not 0)."""
    if "retrieval_recall" not in df.columns:
        raise KeyError(
            "This results CSV has no retrieval_recall column — re-run "
            "eval/run_2a_locomo.py (this repo's version, after the "
            "evidence-tracking changes) to populate it before calling "
            "the experiment-3 functions."
        )
    return df[df["evidence_total"].fillna(0) > 0].dropna(subset=["retrieval_recall"])


def _add_baseline_delta(table: pd.DataFrame, value_col: str, baseline_id: str = BASELINE_ID) -> pd.DataFrame:
    """Add `<value_col>_baseline` (baseline's group value, broadcast to every
    row) and `<value_col>_delta` (row value minus baseline's) columns to a
    group-mean table indexed/keyed by condition_id. Group-level counterpart
    to the paired per-question win/tie/loss functions below — use this for
    "how much higher/lower is the mean", use the win/tie/loss tables for
    "on how many individual questions".
    """
    baseline_rows = table[table["condition_id"] == baseline_id]
    if baseline_rows.empty:
        table[f"{value_col}_baseline"] = float("nan")
        table[f"{value_col}_delta"] = float("nan")
        return table
    bv = baseline_rows[value_col].iloc[0]
    table = table.copy()
    table[f"{value_col}_baseline"] = bv
    table[f"{value_col}_delta"] = table[value_col] - bv
    return table


def retrieval_recall_table(df: pd.DataFrame, group_cols: Sequence[str] = ("dimension", "condition_id")) -> pd.DataFrame:
    """Mean retrieval-phase recall (share of gold evidence turns actually
    retrieved) per group, plus `mean_recall_baseline` / `mean_recall_delta`
    vs baseline. Includes baseline's own row (delta = 0 for it) so it's
    visible as the reference point, not just implicitly subtracted out."""
    scored = _only_scored_rows(df)
    out = (
        scored.groupby(list(group_cols))
        .agg(n_questions=("retrieval_recall", "size"), mean_recall=("retrieval_recall", "mean"))
        .reset_index()
    )
    return _add_baseline_delta(out, "mean_recall")


def retrieval_recall_win_tie_loss_table(
    df: pd.DataFrame, tie_epsilon: float = 1e-9, baseline_id: str = BASELINE_ID,
) -> pd.DataFrame:
    """Paired win/tie/loss vs baseline on retrieval_recall itself — same
    question, does this condition retrieve the gold evidence turns more
    often than baseline did for that exact question. This is the
    retrieval-phase analogue of experiment 1's win_tie_loss_table (which
    compares f1); reuses the same pairing logic (just on a different
    metric column), restricted to questions with a defined recall (see
    _only_scored_rows).
    """
    return win_tie_loss_table(_only_scored_rows(df), metric="retrieval_recall",
                               tie_epsilon=tie_epsilon, baseline_id=baseline_id)


def plot_retrieval_recall_win_tie_loss(
    df: pd.DataFrame, tie_epsilon: float = 1e-9, baseline_id: str = BASELINE_ID,
    dimensions: Optional[Sequence[str]] = None, save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """plot_win_tie_loss, but on retrieval_recall instead of f1/em — same
    Image-1-style stacked bars, one figure per dimension, now answering
    "does this condition retrieve the right evidence more often than
    baseline" rather than "does this condition answer better than baseline".
    """
    return plot_win_tie_loss(_only_scored_rows(df), metric="retrieval_recall", tie_epsilon=tie_epsilon,
                              baseline_id=baseline_id, dimensions=dimensions, save_dir=save_dir,
                              file_tag="retrieval_recall")


def conditional_answer_accuracy_table(
    df: pd.DataFrame, metric: str = "f1", recall_threshold: float = 1.0,
    group_cols: Sequence[str] = ("dimension", "condition_id"),
) -> pd.DataFrame:
    """Answer-phase accuracy (mean `metric`), restricted to questions where
    retrieval succeeded (retrieval_recall >= recall_threshold, default 1.0
    = every gold evidence turn was actually retrieved). This isolates "given
    the answerer actually had the right material, how good is the answer" —
    separate from cases where a bad answer is really a retrieval miss.

    Returns, per group: `accuracy_given_good_retrieval` (mean metric on that
    group's own good-retrieval questions) with baseline reference/delta
    columns; `accuracy_unconditional` (mean metric on the SAME question set,
    ignoring the recall filter) with its own baseline reference/delta; and
    `retrieval_success_rate`. Comparing accuracy_given_good_retrieval's delta
    to accuracy_unconditional's delta tells you whether a condition's overall
    accuracy gap vs baseline is a retrieval problem or an answering problem —
    but note the two deltas are computed over DIFFERENT, condition-specific
    question subsets (each condition's own good-retrieval questions), so for
    a directly paired, same-question-set comparison use
    answer_accuracy_win_tie_loss_given_good_retrieval instead.
    """
    scored = _only_scored_rows(df)
    hit = scored[scored["retrieval_recall"] >= recall_threshold]

    conditional = (
        hit.groupby(list(group_cols))
        .agg(n_questions=(metric, "size"), accuracy_given_good_retrieval=(metric, "mean"))
        .reset_index()
    )
    conditional = _add_baseline_delta(conditional, "accuracy_given_good_retrieval")

    unconditional = (
        scored.groupby(list(group_cols))
        .agg(accuracy_unconditional=(metric, "mean"), retrieval_success_rate=("retrieval_recall", lambda s: (s >= recall_threshold).mean()))
        .reset_index()
    )
    unconditional = _add_baseline_delta(unconditional, "accuracy_unconditional")

    out = conditional.merge(unconditional, on=list(group_cols), how="outer")
    return out


def _both_good_retrieval_subset(df: pd.DataFrame, recall_threshold: float, baseline_id: str) -> pd.DataFrame:
    """Rows where retrieval succeeded for that row's own condition AND for
    baseline on that same (sample_id, question) — the fair-comparison
    subset used by both answer_accuracy_win_tie_loss_given_good_retrieval
    and its plotting counterpart."""
    scored = _only_scored_rows(df)
    good = scored[scored["retrieval_recall"] >= recall_threshold]
    baseline_good_qs = good.loc[good["condition_id"] == baseline_id, ["sample_id", "question"]].drop_duplicates()
    return good.merge(baseline_good_qs, on=["sample_id", "question"], how="inner")


def answer_accuracy_win_tie_loss_given_good_retrieval(
    df: pd.DataFrame, metric: str = "f1", recall_threshold: float = 1.0,
    tie_epsilon: float = 1e-9, baseline_id: str = BASELINE_ID,
) -> pd.DataFrame:
    """Paired win/tie/loss on `metric` vs baseline, restricted to
    (sample_id, question) pairs where BOTH the condition AND baseline
    achieved retrieval_recall >= recall_threshold on that exact question.

    This is the fairest answer-phase-only comparison to baseline: every row
    counted here is a question where both sides had the gold evidence in
    hand, so a win/loss reflects answering quality, not a retrieval
    difference (unlike conditional_answer_accuracy_table's deltas, which
    compare two different condition-specific question subsets). The
    trade-off is a smaller, condition-dependent n — a condition that
    retrieves much worse than baseline will have few or no qualifying
    questions here, which shows up as small `n`, not as losses.
    """
    both_good = _both_good_retrieval_subset(df, recall_threshold, baseline_id)
    return win_tie_loss_table(both_good, metric=metric, tie_epsilon=tie_epsilon, baseline_id=baseline_id)


def plot_answer_accuracy_win_tie_loss_given_good_retrieval(
    df: pd.DataFrame, metric: str = "f1", recall_threshold: float = 1.0,
    tie_epsilon: float = 1e-9, baseline_id: str = BASELINE_ID,
    dimensions: Optional[Sequence[str]] = None, save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """plot_win_tie_loss on the answer-phase-only comparison from
    answer_accuracy_win_tie_loss_given_good_retrieval — same Image-1-style
    stacked bars, but restricted to questions where both the condition and
    baseline actually retrieved the gold evidence, so the win/tie/loss is
    purely about answer quality."""
    both_good = _both_good_retrieval_subset(df, recall_threshold, baseline_id)
    return plot_win_tie_loss(both_good, metric=metric, tie_epsilon=tie_epsilon, baseline_id=baseline_id,
                              dimensions=dimensions, save_dir=save_dir,
                              file_tag=f"{metric}_given_good_retrieval")





def plot_retrieval_vs_answer_phase(
    df: pd.DataFrame, metric: str = "f1", recall_threshold: float = 1.0,
    baseline_id: str = BASELINE_ID, save_path: Optional[str] = None,
) -> plt.Figure:
    """Two side-by-side bar panels, one row per non-baseline condition:
    left = retrieval-phase recall, right = answer-phase accuracy conditioned
    on retrieval having succeeded (recall_threshold). A dashed vertical line
    in each panel marks baseline's own value, so both panels read as
    "vs baseline" rather than as absolute numbers in isolation — for the
    per-question paired version of this comparison (win/tie/loss, with a
    definite n), see retrieval_recall_win_tie_loss_table and
    answer_accuracy_win_tie_loss_given_good_retrieval instead.
    """
    recall = retrieval_recall_table(df, group_cols=("dimension", "condition_id"))
    acc = conditional_answer_accuracy_table(df, metric=metric, recall_threshold=recall_threshold,
                                             group_cols=("dimension", "condition_id"))
    merged = recall.merge(acc, on=["dimension", "condition_id"], how="outer")

    baseline_recall = merged.loc[merged["condition_id"] == baseline_id, "mean_recall"]
    baseline_recall = baseline_recall.iloc[0] if not baseline_recall.empty else None
    baseline_acc = merged.loc[merged["condition_id"] == baseline_id, "accuracy_given_good_retrieval"]
    baseline_acc = baseline_acc.iloc[0] if not baseline_acc.empty else None

    merged = merged[merged["condition_id"] != baseline_id]

    order = list(dict.fromkeys(df.loc[df["condition_id"] != baseline_id, "condition_id"]))
    merged["_order"] = merged["condition_id"].apply(lambda c: order.index(c) if c in order else len(order))
    merged = merged.sort_values("_order")

    labels = [
        f"{cid.split('__', 1)[0]}: {_variant_label(cid)}" if "__" in cid else cid
        for cid in merged["condition_id"]
    ]
    y = np.arange(len(merged))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 0.6 * len(merged) + 1.5), sharey=True)

    ax_l.barh(y, merged["mean_recall"], color=WIN_COLOR)
    if baseline_recall is not None:
        ax_l.axvline(baseline_recall, color="black", linestyle="--", linewidth=1,
                      label=f"baseline ({baseline_recall*100:.0f}%)")
        ax_l.legend(loc="lower right", fontsize=8, frameon=False)
    ax_l.set_xlim(0, 1)
    ax_l.set_xlabel("Retrieval-phase recall")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels)
    ax_l.invert_yaxis()

    ax_r.barh(y, merged["accuracy_given_good_retrieval"], color="#2f9e44", label=f"{metric}, retrieval OK")
    ax_r.barh(y, merged["accuracy_unconditional"], color="#adb5bd", height=0.35,
              label=f"{metric}, unconditional", alpha=0.9)
    if baseline_acc is not None:
        ax_r.axvline(baseline_acc, color="black", linestyle="--", linewidth=1,
                      label=f"baseline ({baseline_acc*100:.0f}%)")
    ax_r.set_xlim(0, 1)
    ax_r.set_xlabel(f"Answer-phase accuracy ({metric})")
    ax_r.legend(loc="lower right", fontsize=8, frameon=False)

    fig.suptitle(f"Retrieval phase vs answer phase (retrieval OK = recall >= {recall_threshold:g}; "
                 f"dashed line = baseline)")
    for ax in (ax_l, ax_r):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/2a_locomo_results.csv")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--metric", default="f1", choices=["f1", "em"])
    p.add_argument("--tie-epsilon", type=float, default=1e-9)
    p.add_argument("--recall-threshold", type=float, default=1.0)
    p.add_argument("--skip-retrieval", action="store_true",
                    help="Skip experiment 3 (use for a CSV produced before the "
                         "evidence-tracking changes, which has no retrieval_recall column)")
    args = p.parse_args()

    df = load_results(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== Experiment 1: win / tie / loss vs baseline ===")
    print(win_tie_loss_table(df, metric=args.metric, tie_epsilon=args.tie_epsilon).to_string())
    plot_win_tie_loss(df, metric=args.metric, tie_epsilon=args.tie_epsilon, save_dir=args.out_dir)

    print("\n=== Experiment 2: win rate by condition x query type (ties excluded) ===")
    heatmap_table = win_rate_by_query_type_table(df, metric=args.metric, tie_epsilon=args.tie_epsilon)
    print(heatmap_table.to_string(float_format=lambda v: f"{v*100:.0f}%" if pd.notna(v) else "  -"))
    plot_win_rate_heatmap(df, metric=args.metric, tie_epsilon=args.tie_epsilon,
                           save_path=os.path.join(args.out_dir, "win_rate_heatmap.png"))

    if not args.skip_retrieval:
        print("\n=== Experiment 3a: retrieval-phase recall (mean, vs baseline) ===")
        print(retrieval_recall_table(df).to_string(index=False))
        print("\n=== Experiment 3a (paired): retrieval recall win/tie/loss vs baseline ===")
        print(retrieval_recall_win_tie_loss_table(df, tie_epsilon=args.tie_epsilon).to_string())
        plot_retrieval_recall_win_tie_loss(df, tie_epsilon=args.tie_epsilon, save_dir=args.out_dir)

        print(f"\n=== Experiment 3b: answer-phase accuracy | retrieval recall >= {args.recall_threshold:g} (vs baseline) ===")
        print(conditional_answer_accuracy_table(df, metric=args.metric,
                                                 recall_threshold=args.recall_threshold).to_string(index=False))
        print(f"\n=== Experiment 3b (paired): answer accuracy win/tie/loss vs baseline, "
              f"both sides retrieval OK ===")
        print(answer_accuracy_win_tie_loss_given_good_retrieval(
            df, metric=args.metric, recall_threshold=args.recall_threshold, tie_epsilon=args.tie_epsilon
        ).to_string())
        plot_answer_accuracy_win_tie_loss_given_good_retrieval(
            df, metric=args.metric, recall_threshold=args.recall_threshold,
            tie_epsilon=args.tie_epsilon, save_dir=args.out_dir,
        )
        plot_retrieval_vs_answer_phase(df, metric=args.metric, recall_threshold=args.recall_threshold,
                                        save_path=os.path.join(args.out_dir, "retrieval_vs_answer_phase.png"))

    print(f"\nFigures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
