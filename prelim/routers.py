"""
Training-free query-level routers, evaluated as a selection over an
already-scored full condition matrix (prelim/run_2a_locomo.py CSV).

All share `.predict(question, category=None, sample_id=None) -> Condition`:

  RandomRouter    lower-bound control. Enjoys the same "mix several views"
                  benefit as any router but has no query conditioning, so a
                  router must beat it before routing can be said to learn
                  anything. Seeded per (seed, sample_id, question), so a draw
                  does not depend on question order.
  LLMJudgeRouter  closed-set LLM judge: shown every candidate with its
                  behavioural description, returns one condition_id. Only the
                  question is given by default; the LoCoMo category is gold
                  metadata (use_category=True is diagnostic only).
  OracleRouter    per-question argmax over the CSV it is scored on. Not a
                  router — the headroom ceiling.

The learned router is memrx/router.py, evaluated by scripts/train_memrx.py.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import pandas as pd

from core.conditions import Condition, build_condition_matrix, describe
from utils.llm_client import LLMClient
from utils.locomo import CATEGORY_NAMES

_CONDITION_BY_ID: Dict[str, Condition] = {c.condition_id: c for c in build_condition_matrix()}


def condition_from_id(condition_id: str) -> Condition:
    """Unknown ids (e.g. from an old CSV) fall back to baseline instead of raising."""
    return _CONDITION_BY_ID.get(condition_id, _CONDITION_BY_ID["baseline"])


# --------------------------------------------------------------------- #
class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 0, conditions: Optional[Sequence[Condition]] = None):
        self.seed = seed
        self.conditions = list(conditions) if conditions is not None else build_condition_matrix()

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        return random.Random(f"{self.seed}||{sample_id}||{question}").choice(self.conditions)


# --------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are routing a question to ONE memory-processing setup.

A long conversation has already been stored under several different processing setups.
Your job: given only the question, judge which single setup is most likely to let a
downstream model answer it correctly.

Candidate setups:
{menu}
{category_line}
Question: {question}

Think about what kind of evidence this question needs, then commit to one setup.
Return ONLY JSON, no other text:
{{"choice": "<exact condition_id from the list above>", "reason": "<one short sentence>"}}
"""


class LLMJudgeRouter:
    """Answers are cached per question text, so re-scoring costs no LLM calls."""

    name = "judge"

    def __init__(self, llm: LLMClient, use_category: bool = False, max_retries: int = 2,
                 conditions: Optional[Sequence[Condition]] = None):
        self.llm = llm
        self.use_category = use_category
        self.max_retries = max_retries
        conditions = list(conditions) if conditions is not None else build_condition_matrix()
        self.valid_ids = [c.condition_id for c in conditions]
        self.menu = "\n".join(f"- {cid}: {describe(cid)}" for cid in self.valid_ids)
        self._cache: Dict[str, str] = {}
        self.reasons: Dict[str, str] = {}
        # a judge that falls back to baseline on 40% of questions is not really being evaluated
        self.stats = {"calls": 0, "cache_hits": 0, "invalid": 0, "fallback": 0}

    def _ask(self, question: str, category: Optional[int]) -> Optional[str]:
        category_line = ""
        if self.use_category and category is not None:
            category_line = f"\nQuestion type: {CATEGORY_NAMES.get(int(category), 'unknown')}\n"
        prompt = _JUDGE_PROMPT.format(menu=self.menu, category_line=category_line, question=question)

        for _ in range(self.max_retries):
            self.stats["calls"] += 1
            try:
                raw = self.llm.chat_completion([{"role": "user", "content": prompt}],
                                               temperature=0.0, max_tokens=256)
                parsed = self.llm.extract_json(raw)
            except Exception:
                self.stats["invalid"] += 1
                continue
            choice = (parsed.get("choice") or parsed.get("condition_id")) if isinstance(parsed, dict) else None
            if isinstance(choice, str):
                choice = choice.strip().strip('"')
                if choice in self.valid_ids:
                    reason = parsed.get("reason")
                    if isinstance(reason, str):
                        self.reasons[question] = reason[:200]
                    return choice
                # small models often return the bare variant ("entity") instead of the full id
                matches = [cid for cid in self.valid_ids
                           if cid.endswith(f"__{choice}") or cid == choice.replace(".", "")]
                if len(matches) == 1:
                    return matches[0]
            self.stats["invalid"] += 1
        return None

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        if question in self._cache:
            self.stats["cache_hits"] += 1
        else:
            choice = self._ask(question, category)
            if choice is None:
                self.stats["fallback"] += 1
                choice = "baseline"
            self._cache[question] = choice
        return condition_from_id(self._cache[question])


# --------------------------------------------------------------------- #
class OracleRouter:
    name = "oracle"

    def __init__(self, results_df: pd.DataFrame, metric: str = "f1"):
        df = results_df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        # ties -> first row in file order = condition-matrix order (baseline first)
        best = (df.dropna(subset=[metric])
                  .sort_values(metric, ascending=False, kind="mergesort")
                  .drop_duplicates(["sample_id", "question"], keep="first"))
        self._lookup = {(str(r.sample_id), r.question): r.condition_id for r in best.itertuples()}

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        return condition_from_id(self._lookup.get((str(sample_id), question), "baseline"))


# --------------------------------------------------------------------- #
ROUTER_NAMES = ["random", "judge", "oracle"]


def build_router(name: str, *, llm: Optional[LLMClient] = None,
                 results_df: Optional[pd.DataFrame] = None, seed: int = 0,
                 metric: str = "f1", judge_use_category: bool = False):
    if name == "random":
        return RandomRouter(seed=seed)
    if name == "judge":
        if llm is None:
            raise ValueError("router 'judge' needs an LLMClient")
        return LLMJudgeRouter(llm, use_category=judge_use_category)
    if name == "oracle":
        if results_df is None:
            raise ValueError("router 'oracle' needs the results_df it will be scored on")
        return OracleRouter(results_df, metric=metric)
    raise ValueError(f"Unknown router: {name} (expected one of {ROUTER_NAMES})")
