"""
Per-question win / tie / lose between two selection strategies.

Why this on top of mean F1: on LoCoMo the per-question F1 distribution is
bimodal (most questions score near 0 or near 1), so a mean gap of +1.5 F1
can come either from a router that helps a little on most questions or from
one that flips a handful of questions from 0 to 1 and hurts nothing —
qualitatively different claims that the mean cannot separate. W/T/L reports
the actual per-question direction, and the sign test says whether the
win-lose imbalance survives the number of questions you have.

Everything here is a pure lookup over `2a_locomo_results_*.csv` plus the
router selection frames produced by eval/run_router_locomo.py — no LLM
calls, no memory rebuilds.

Pairing key is (sample_id, question). A "system" here is any per-question
selection: a fixed condition (`baseline`, `summary__fine_grained`, ...) or a
router's selected rows.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List, Optional, Sequence

import pandas as pd

KEY = ["sample_id", "question"]


def selection_frame(df: pd.DataFrame, metric: str = "f1",
                    condition_id: Optional[str] = None) -> pd.DataFrame:
    """Normalize either a full-matrix slice or a router selection into
    (sample_id, question, category_name, <metric>), one row per question."""
    out = df.copy()
    if condition_id is not None:
        out = out[out["condition_id"] == condition_id]
    out["sample_id"] = out["sample_id"].astype(str)
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=[metric])
    cols = KEY + [metric]
    if "category_name" in out.columns:
        cols.append("category_name")
    # Resume runs can leave duplicate (sample_id, condition_id, question)
    # rows in the CSV; keep the first so pairing stays 1:1 instead of
    # exploding into a cross join on merge.
    return out[cols].drop_duplicates(subset=KEY, keep="first")


def sign_test_p(win: int, lose: int) -> float:
    """Exact two-sided binomial sign test on the non-tied questions.

    Ties are dropped (standard for a sign test) — which is why the tie count
    is reported alongside: 30 wins / 5 losses out of 400 questions is a much
    weaker claim than 30/5 out of 40, and only the tie column shows that.
    """
    n = win + lose
    if n == 0:
        return 1.0
    k = min(win, lose)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wtl(system: pd.DataFrame, reference: pd.DataFrame, metric: str = "f1",
        tol: float = 1e-9) -> Dict[str, float]:
    """Compare `system` against `reference` question by question.

    `tol` is the margin below which a difference counts as a tie. Default is
    float-noise only (exact equality); set it to e.g. 0.05 to ask the
    stricter question "did it change the answer meaningfully", which matters
    on F1 where a one-token difference moves the score slightly without
    changing whether the answer is right.
    """
    merged = system.merge(reference, on=KEY, suffixes=("_sys", "_ref"))
    if merged.empty:
        return {"n": 0, "win": 0, "tie": 0, "lose": 0, "win_rate": float("nan"),
                "net_win_rate": float("nan"), "mean_delta": float("nan"), "p_sign": 1.0}

    delta = merged[f"{metric}_sys"] - merged[f"{metric}_ref"]
    win = int((delta > tol).sum())
    lose = int((delta < -tol).sum())
    tie = int(len(delta) - win - lose)
    n = len(delta)
    return {
        "n": n,
        "win": win,
        "tie": tie,
        "lose": lose,
        "win_rate": win / n,
        # net_win_rate = (win - lose) / n. Sign and magnitude of the routing
        # effect in one number, and unlike win_rate it cannot be inflated by
        # a strategy that wins and loses equally often.
        "net_win_rate": (win - lose) / n,
        "mean_delta": float(delta.mean()),
        "p_sign": sign_test_p(win, lose),
    }


def wtl_by_category(system: pd.DataFrame, reference: pd.DataFrame, metric: str = "f1",
                    tol: float = 1e-9) -> pd.DataFrame:
    """Same comparison, split by LoCoMo category — this is where a router is
    supposed to earn its keep (winning on multi_hop while not losing on
    single_hop), and where a flat overall W/T/L can hide that it wins on one
    category and loses on another."""
    merged = system.merge(reference, on=KEY, suffixes=("_sys", "_ref"))
    cat_col = "category_name_sys" if "category_name_sys" in merged.columns else "category_name"
    if cat_col not in merged.columns or merged.empty:
        return pd.DataFrame()

    rows = []
    for cat, grp in merged.groupby(cat_col):
        delta = grp[f"{metric}_sys"] - grp[f"{metric}_ref"]
        win = int((delta > tol).sum())
        lose = int((delta < -tol).sum())
        rows.append({
            "category": cat, "n": len(delta), "win": win,
            "tie": int(len(delta) - win - lose), "lose": lose,
            "net_win_rate": (win - lose) / len(delta),
            "mean_delta": float(delta.mean()),
            "p_sign": sign_test_p(win, lose),
        })
    return pd.DataFrame(rows).sort_values("category")


def _average_runs(runs: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average W/T/L over repeated runs of a stochastic system (RandomRouter
    across seeds). Counts are averaged, not summed, so the numbers stay on
    the same per-question scale as a deterministic router's row and the two
    can be read off the same table. The sign test is recomputed on the
    rounded average counts — treat its p-value on a multi-seed random row as
    descriptive only, since the seeds share the same questions and are not
    independent samples."""
    if len(runs) == 1:
        return dict(runs[0])
    keys = ["n", "win", "tie", "lose", "win_rate", "net_win_rate", "mean_delta"]
    avg = {k: sum(r[k] for r in runs) / len(runs) for k in keys}
    avg["p_sign"] = sign_test_p(round(avg["win"]), round(avg["lose"]))
    return avg


def build_wtl_table(systems: Dict[str, List[pd.DataFrame]],
                    references: Dict[str, pd.DataFrame],
                    metric: str = "f1", tol: float = 1e-9) -> pd.DataFrame:
    """Cross every system against every reference.

    `systems` maps a label to a LIST of selection frames — one entry for a
    deterministic system, several for a seeded one (results are averaged
    across the list, see _average_runs).
    """
    rows = []
    for sys_label, sys_runs in systems.items():
        for ref_label, ref_df in references.items():
            if sys_label == ref_label:
                continue
            per_run = [wtl(s, ref_df, metric=metric, tol=tol) for s in sys_runs]
            rec = {"system": sys_label, "reference": ref_label, "n_runs": len(sys_runs)}
            rec.update(_average_runs(per_run))
            rows.append(rec)
    return pd.DataFrame(rows)


def print_wtl_table(table: pd.DataFrame, metric: str = "f1", title: str = "") -> None:
    if table.empty:
        print("(no win/tie/lose rows)")
        return
    print(f"\n=== win / tie / lose ({metric}){' — ' + title if title else ''} ===")
    head = (f"{'system':<24s}{'vs reference':<26s}{'W':>7s}{'T':>7s}{'L':>7s}"
            f"{'net':>9s}{'mean_d':>9s}{'p':>9s}")
    print(head)
    print("-" * len(head))
    for _, r in table.iterrows():
        star = "*" if r["p_sign"] < 0.05 else " "
        print(f"{r['system']:<24s}{r['reference']:<26s}"
              f"{r['win']:>7.1f}{r['tie']:>7.1f}{r['lose']:>7.1f}"
              f"{r['net_win_rate']*100:>8.1f}%{r['mean_delta']*100:>9.2f}"
              f"{r['p_sign']:>8.3f}{star}")
    print("net = (W-L)/n; mean_d = mean per-question metric delta x100; "
          "p = two-sided sign test on non-ties; * = p<0.05")
