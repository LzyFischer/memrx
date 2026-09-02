"""
Query-level routing for the 2a ablation.

Unlike run_2a_locomo.py (one Condition applied to an entire conversation),
a router here picks a Condition PER QUESTION — i.e. two questions about the
same conversation can end up retrieving from two different memory stores
(summary__fine_grained for one, baseline for the other).

All implementations share one interface — `.predict(question, category=None,
sample_id=None) -> Condition` — so eval/run_router_locomo.py can swap one
for another (or run all of them in the same pass):

  - RandomRouter    : lower-bound control. Picks uniformly at random from the
                      condition matrix. This is the number every other router
                      has to beat before "routing" can be claimed to do
                      anything at all: if a learned router only matches
                      random, its apparent gain over `baseline` is just the
                      gain from *sometimes not using baseline*, not from
                      query-conditional selection. Seeded per question so a
                      given (seed, question) always draws the same condition
                      — reproducible and order-independent, and safe to run
                      over several seeds and average.
  - PromptRouter    : naive baseline, zero training data. Asks the LLM once
                      per question to fill in the 3 dimensions freely
                      (open-ended generation, then validated down to a legal
                      Condition).
  - LLMJudgeRouter  : closed-set LLM judge. Same "no training data" regime as
                      PromptRouter, but instead of generating dimension
                      values it is shown the full candidate list with a
                      one-line description of what each condition does to
                      memory, and must return exactly one `condition_id`.
                      Isolates "can an LLM pick from a menu given only the
                      query" from PromptRouter's confound of also having to
                      guess the schema. Only the question is given by
                      default (`use_category=False`) — the LoCoMo category
                      label is gold metadata, so feeding it in for free makes
                      the judge non-comparable to a deployable router; pass
                      `use_category=True` only for the diagnostic variant.
  - LearnedRouter   : supervised. Fit on a train-split `2a_locomo_results_*.csv`
                      (produced by `run_2a_locomo.py --split train`, which
                      still runs the full condition matrix per conversation —
                      that's the "feedback" this router learns from). The
                      label for each (sample_id, question) is whichever
                      condition_id scored the highest f1 in that train run.
                      Features = question embedding (+ one-hot category).
  - OracleRouter    : upper bound, not a real router. Reads the eval CSV it
                      is being scored on and returns the per-question argmax
                      condition. Reports the headroom the whole routing idea
                      has on this data — a learned router at 0.31 F1 reads
                      very differently against an oracle at 0.33 than
                      against an oracle at 0.52.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

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

    name = "naive"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
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

    name = "learned"

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

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        if self.clf is None:
            condition_id = self._majority
        else:
            x = self._features([question], [category if category is not None else -1])
            condition_id = self.clf.predict(x)[0]
        return condition_from_id(condition_id)


# --------------------------------------------------------------------- #
# 3. Random router — the control every other router must beat
# --------------------------------------------------------------------- #
class RandomRouter:
    """Uniform random selection over the condition matrix.

    Seeding is per (seed, sample_id, question) rather than one global RNG
    stream, so the draw for a given question does not depend on how many
    questions were routed before it. That keeps a run reproducible even if
    the eval CSV is filtered, re-ordered, or resumed halfway — and makes
    "same seed, same pick" hold across the win/tie/lose pass and the mean-F1
    pass without having to cache selections.

    Run it over several seeds (eval/run_router_locomo.py --random-seeds)
    and report mean +/- std: a single random draw over ~200 questions has a
    standard error large enough to accidentally beat a real router.
    """

    name = "random"

    def __init__(self, seed: int = 0, conditions: Optional[Sequence[Condition]] = None):
        self.seed = seed
        self.conditions = list(conditions) if conditions is not None else build_condition_matrix()

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        rng = random.Random(f"{self.seed}||{sample_id}||{question}")
        return rng.choice(self.conditions)


# --------------------------------------------------------------------- #
# 4. LLM-judge router — closed-set choice over the candidate menu
# --------------------------------------------------------------------- #
# One line per condition, describing what it DOES to memory rather than
# naming the dimension, so the judge is choosing between behaviours it can
# reason about from the query alone. Kept next to the condition matrix it
# mirrors — if config_2a.build_condition_matrix() gains a variant, add its
# description here too (build_menu() below will otherwise fall back to a
# generic auto-generated line for it).
_CONDITION_DESCRIPTIONS: Dict[str, str] = {
    "baseline":
        "Raw conversation chunks, verbatim, no extra processing. Best when the answer is "
        "stated literally somewhere and any rewriting risks losing the exact wording.",
    "summary__session_level":
        "Each window of turns is compressed by an LLM into ONE summary entry. Coarse: good "
        "for gist, overall state, or 'what happened over time' questions; loses fine detail.",
    "summary__fine_grained":
        "Each window is rewritten by an LLM into SEVERAL small standalone facts. Good when "
        "the answer is one specific fact buried in chatter that needs isolating.",
    "augmentation__keywords":
        "Raw chunks plus LLM-extracted keywords (names, places, numbers, dates) indexed for "
        "sparse keyword matching alongside semantic search. Good for exact-term lookups where "
        "the query and the source share rare words.",
    "augmentation__note":
        "Raw chunks with an LLM-written note (keywords plus a sentence of context) appended "
        "before embedding. Good when the query is phrased differently from the source and "
        "needs the semantic gap bridged.",
    "graph__semantic":
        "Raw chunks linked to their most similar chunks; retrieval walks one hop out from the "
        "matched chunks. Good when the answer needs several related passages together.",
    "graph__entity":
        "Raw chunks linked when they mention the same entities; retrieval walks one hop out. "
        "Good for multi-hop questions that chain across people, places, or objects.",
}

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
    """Ask an LLM judge to pick one condition_id out of the explicit menu.

    Differences from PromptRouter, which matter for what the ablation shows:
      - closed set: the judge sees every candidate and returns an id, so it
        can never produce an illegal combination that has to be silently
        repaired into baseline (PromptRouter's repair rule quietly biases it
        toward `summary` and toward `baseline`).
      - the candidates are described by behaviour, not by dimension name, so
        a small model does not have to already know what "augmentation" means
        in this codebase.

    Answers are cached per question: the same question text always gets the
    same route within a run, and re-scoring (mean-F1 pass, then win/tie/lose
    pass) costs zero extra LLM calls.
    """

    name = "judge"

    def __init__(self, llm: LLMClient, use_category: bool = False, max_retries: int = 2,
                 conditions: Optional[Sequence[Condition]] = None):
        self.llm = llm
        self.use_category = use_category
        self.max_retries = max_retries
        self.conditions = list(conditions) if conditions is not None else build_condition_matrix()
        self.valid_ids = [c.condition_id for c in self.conditions]
        self.menu = self.build_menu(self.conditions)
        self._cache: Dict[str, str] = {}
        # Diagnostics worth printing next to the F1 number: a judge that
        # falls back to baseline on 40% of questions is not really being
        # evaluated as a judge.
        self.stats = {"calls": 0, "cache_hits": 0, "invalid": 0, "fallback": 0}
        self.reasons: Dict[str, str] = {}

    @staticmethod
    def build_menu(conditions: Sequence[Condition]) -> str:
        lines = []
        for c in conditions:
            desc = _CONDITION_DESCRIPTIONS.get(
                c.condition_id,
                f"{c.dimension} treatment (summary={c.summary}, augmentation={c.augmentation}, "
                f"graph={c.graph}).",
            )
            lines.append(f"- {c.condition_id}: {desc}")
        return "\n".join(lines)

    def _ask(self, question: str, category: Optional[int]) -> Optional[str]:
        category_line = ""
        if self.use_category and category is not None:
            category_line = f"\nQuestion type: {CATEGORY_NAMES.get(int(category), 'unknown')}\n"
        prompt = _JUDGE_PROMPT.format(menu=self.menu, category_line=category_line, question=question)

        for attempt in range(self.max_retries):
            try:
                self.stats["calls"] += 1
                raw = self.llm.chat_completion(
                    [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256,
                )
                parsed = self.llm.extract_json(raw)
                choice = None
                if isinstance(parsed, dict):
                    choice = parsed.get("choice") or parsed.get("condition_id")
                if isinstance(choice, str):
                    choice = choice.strip().strip('"')
                    if choice in self.valid_ids:
                        reason = parsed.get("reason") if isinstance(parsed, dict) else None
                        if isinstance(reason, str):
                            self.reasons[question] = reason[:200]
                        return choice
                    # Small models often answer with the bare variant name
                    # ("fine_grained") or the dimension ("graph") instead of
                    # the full id — accept an unambiguous suffix/prefix match
                    # rather than burning a retry on a choice that is
                    # actually well-formed intent.
                    matches = [cid for cid in self.valid_ids
                               if cid.endswith(f"__{choice}") or cid == choice.replace(".", "")]
                    if len(matches) == 1:
                        return matches[0]
                self.stats["invalid"] += 1
            except Exception:
                self.stats["invalid"] += 1
        return None

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        if question in self._cache:
            self.stats["cache_hits"] += 1
            return condition_from_id(self._cache[question])

        choice = self._ask(question, category)
        if choice is None:
            self.stats["fallback"] += 1
            choice = "baseline"
        self._cache[question] = choice
        return condition_from_id(choice)


# --------------------------------------------------------------------- #
# 5. Oracle — upper bound on what any router could achieve here
# --------------------------------------------------------------------- #
class OracleRouter:
    """Per-question argmax over the eval CSV's own scores.

    NOT a router — it reads the labels of the split it is scored on. Include
    it in the table purely as the ceiling: router gains are only meaningful
    relative to the gap between the best fixed condition and this row.
    """

    name = "oracle"

    def __init__(self, results_df: pd.DataFrame, metric: str = "f1"):
        df = results_df.copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=[metric])
        # Ties -> first row in file order = condition matrix order (baseline
        # first), matching LearnedRouter.fit()'s tie rule so the oracle is not
        # quietly credited for preferring an expensive condition on a tie.
        best = (df.sort_values(metric, ascending=False, kind="mergesort")
                  .drop_duplicates(["sample_id", "question"], keep="first"))
        self._lookup: Dict[tuple, str] = {
            (str(r.sample_id), r.question): r.condition_id for r in best.itertuples()
        }

    def predict(self, question: str, category: Optional[int] = None,
                sample_id: Optional[str] = None) -> Condition:
        return condition_from_id(self._lookup.get((str(sample_id), question), "baseline"))


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #
ROUTER_NAMES = ["random", "naive", "judge", "learned", "oracle"]


def build_router(name: str, *, llm: Optional[LLMClient] = None,
                 embedding_model: Optional[EmbeddingModel] = None,
                 train_df: Optional[pd.DataFrame] = None,
                 results_df: Optional[pd.DataFrame] = None,
                 seed: int = 0, metric: str = "f1",
                 judge_use_category: bool = False):
    """Build one router by name. Raises early (before any LLM call) if a
    required dependency for that router is missing, so a multi-router run
    fails at argument-parse time rather than 40 minutes in."""
    if name == "random":
        return RandomRouter(seed=seed)
    if name == "naive":
        if llm is None:
            raise ValueError("router 'naive' needs an LLMClient")
        return PromptRouter(llm)
    if name == "judge":
        if llm is None:
            raise ValueError("router 'judge' needs an LLMClient")
        return LLMJudgeRouter(llm, use_category=judge_use_category)
    if name == "learned":
        if embedding_model is None or train_df is None:
            raise ValueError("router 'learned' needs an EmbeddingModel and a train_df")
        return LearnedRouter(embedding_model).fit(train_df)
    if name == "oracle":
        if results_df is None:
            raise ValueError("router 'oracle' needs the results_df it will be scored on")
        return OracleRouter(results_df, metric=metric)
    raise ValueError(f"Unknown router: {name} (expected one of {ROUTER_NAMES})")
