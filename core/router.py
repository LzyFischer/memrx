"""
Query-level routing for the 2a ablation.

Unlike run_2a_locomo.py (one Condition applied to an entire conversation),
a router here picks a Condition PER QUESTION — i.e. two questions about the
same conversation can end up retrieving from two different memory stores
(summary__fine_grained for one, baseline for the other).

Two implementations, same interface — `.predict(question, category) ->
Condition` — so eval/run_router_locomo.py can swap one for the other:

  - PromptRouter  : naive baseline, zero training data. Asks the LLM once
                    per question to pick a Condition directly.
  - LearnedRouter : supervised. Fit on a train-split `2a_locomo_results_*.csv`
                    (produced by `run_2a_locomo.py --split train`, which
                    still runs the full condition matrix per conversation —
                    that's the "feedback" this router learns from). The
                    label for each (sample_id, question) is whichever
                    condition_id scored the highest f1 in that train run.
                    Features = question embedding (+ one-hot category).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config_2a import Condition, build_condition_matrix
from eval.locomo_loader import CATEGORY_NAMES
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient

_CONDITION_BY_ID: Dict[str, Condition] = {c.condition_id: c for c in build_condition_matrix()}

_VALID_VARIANTS = {
    "summary": {"none", "session_level", "fine_grained"},
    "augmentation": {"none", "keywords", "note"},
    "graph": {"none", "semantic", "entity"},
}


def condition_from_id(condition_id: str) -> Condition:
    """Look up a Condition by its id (e.g. "summary__fine_grained"), falling
    back to baseline for anything unrecognized (e.g. a stale/typo'd id from
    an old CSV) rather than raising mid-run."""
    return _CONDITION_BY_ID.get(condition_id, _CONDITION_BY_ID["baseline"])


# --------------------------------------------------------------------- #
# 1. Naive prompt router — no train data needed
# --------------------------------------------------------------------- #
_ROUTER_PROMPT = """You are deciding how to pre-process conversation memory before answering a question.

There are 3 independent processing dimensions. Turn EXACTLY ONE of them on (pick a variant),
the other two must stay "none" — you are picking the single best treatment for this question,
not stacking several:
- summary: none | session_level | fine_grained
- augmentation: none | keywords | note
- graph: none | semantic | entity

Guidance:
- exact wording / specific fact lookup -> augmentation (keywords or note)
- needs a compressed/aggregated timeline or gist -> summary
- needs multi-hop relations between people, events, or entities -> graph
- simple direct single-fact question -> leave all three "none" (baseline)

Question: {question}

Return ONLY JSON, no other text:
{{"summary": "...", "augmentation": "...", "graph": "..."}}
"""


class PromptRouter:
    """Zero-shot: ask the LLM to pick a Condition per query."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def predict(self, question: str, category: Optional[int] = None) -> Condition:
        prompt = _ROUTER_PROMPT.format(question=question)
        summary = augmentation = graph = "none"
        try:
            raw = self.llm.chat_completion(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200,
            )
            parsed = self.llm.extract_json(raw)
            if isinstance(parsed, dict):
                summary = parsed.get("summary", "none")
                augmentation = parsed.get("augmentation", "none")
                graph = parsed.get("graph", "none")
        except Exception:
            pass  # falls through to baseline below

        # Guard against the LLM hallucinating a variant name or turning on
        # more than one dimension — config_2a.build_condition_matrix() only
        # ever varies one dimension at a time and core/treatments.py assumes
        # that invariant, so an invalid combo silently degrades to baseline
        # rather than crashing build_memory_store downstream.
        if summary not in _VALID_VARIANTS["summary"]:
            summary = "none"
        if augmentation not in _VALID_VARIANTS["augmentation"]:
            augmentation = "none"
        if graph not in _VALID_VARIANTS["graph"]:
            graph = "none"
        non_none = [d for d in (summary, augmentation, graph) if d != "none"]
        if len(non_none) > 1:
            # fixed priority when the model picks more than one: summary >
            # augmentation > graph. Arbitrary but deterministic.
            if summary != "none":
                augmentation, graph = "none", "none"
            elif augmentation != "none":
                graph = "none"

        if summary != "none":
            return Condition(condition_id=f"summary__{summary}", dimension="summary", summary=summary)
        if augmentation != "none":
            return Condition(condition_id=f"augmentation__{augmentation}", dimension="augmentation",
                              augmentation=augmentation)
        if graph != "none":
            return Condition(condition_id=f"graph__{graph}", dimension="graph", graph=graph)
        return Condition(condition_id="baseline", dimension="baseline")


# --------------------------------------------------------------------- #
# 2. Learned router — fit on train-split feedback
# --------------------------------------------------------------------- #
class LearnedRouter:
    """Supervised router fit on a train-split `2a_locomo_results_*.csv`.

    Call `.fit(train_df)` once (train_df = pd.read_csv(...) of the CSV
    `run_2a_locomo.py --split train` produced — the long-format table with
    one row per (sample_id, condition_id, question)), then `.predict(...)`
    per question at eval time.
    """

    def __init__(self, embedding_model: EmbeddingModel, use_category: bool = True):
        self.embedding_model = embedding_model
        self.use_category = use_category
        self.clf = None
        self._majority = "baseline"
        self._category_ids: List[int] = sorted(CATEGORY_NAMES.keys())

    def _features(self, questions: List[str], categories: List[int]) -> np.ndarray:
        emb = self.embedding_model.encode(questions, is_query=True)
        if not self.use_category:
            return emb
        cat_onehot = np.zeros((len(categories), len(self._category_ids)))
        for i, c in enumerate(categories):
            if c in self._category_ids:
                cat_onehot[i, self._category_ids.index(c)] = 1.0
        return np.hstack([emb, cat_onehot])

    def fit(self, train_df: pd.DataFrame) -> "LearnedRouter":
        from sklearn.linear_model import LogisticRegression

        df = train_df.copy()
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        df = df.dropna(subset=["f1"])

        # Best condition per (sample_id, question) = the supervision label.
        # Ties keep the first row in file order, i.e. condition order from
        # config_2a.build_condition_matrix() (baseline first) — a mild bias
        # toward the cheaper condition on ties, which is a reasonable
        # default for a router whose whole point is to avoid paying for
        # expensive processing when it doesn't help.
        best = (df.sort_values("f1", ascending=False)
                  .drop_duplicates(["sample_id", "question"], keep="first"))

        if best.empty:
            self.clf = None
            self._majority = "baseline"
            return self

        X = self._features(best["question"].tolist(), best["category"].astype(int).tolist())
        y = best["condition_id"].values
        self._majority = pd.Series(y).mode().iloc[0]

        if len(set(y)) < 2:
            # Degenerate train fold (e.g. n_train=2 conversations -> one
            # condition wins every single question) -> no decision boundary
            # to learn; predict() falls back to the majority label.
            self.clf = None
        else:
            self.clf = LogisticRegression(max_iter=2000).fit(X, y)
        return self

    def predict(self, question: str, category: Optional[int] = None) -> Condition:
        if self.clf is None:
            condition_id = self._majority
        else:
            x = self._features([question], [category if category is not None else -1])
            condition_id = self.clf.predict(x)[0]
        return condition_from_id(condition_id)
