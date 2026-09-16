"""The candidate menu the router chooses from: raw text plus one processing
variant per dimension (3+1).

Each view is described by one line of behaviour. The MemRx router encodes
these descriptions instead of learning a per-view output head, so a view not
seen in training can still be scored; the LLM-judge baseline shows the judge
the same text.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Condition:
    condition_id: str
    dimension: str                 # "baseline" | "summary" | "augmentation" | "graph"
    summary: str = "none"          # "none" | "session_level"
    augmentation: str = "none"     # "none" | "keywords"
    graph: str = "none"            # "none" | "entity"


def build_condition_matrix() -> List[Condition]:
    return [
        Condition("baseline", "baseline"),
        Condition("summary__session_level", "summary", summary="session_level"),
        Condition("augmentation__keywords", "augmentation", augmentation="keywords"),
        Condition("graph__entity", "graph", graph="entity"),
    ]


VIEW_DESCRIPTIONS: Dict[str, str] = {
    "baseline":
        "Raw conversation chunks, verbatim, no extra processing. Best when the answer is "
        "stated literally somewhere and any rewriting risks losing the exact wording.",
    "summary__session_level":
        "Each window of turns is compressed by an LLM into ONE summary entry. Coarse: good "
        "for gist, overall state, or 'what happened over time' questions; loses fine detail.",
    "augmentation__keywords":
        "Raw chunks plus LLM-extracted keywords (names, places, numbers, dates) indexed for "
        "sparse keyword matching alongside semantic search. Good for exact-term lookups where "
        "the query and the source share rare words.",
    "graph__entity":
        "Raw chunks linked when they mention the same entities; retrieval walks one hop out. "
        "Good for multi-hop questions that chain across people, places, or objects.",
}


def describe(condition_id: str) -> str:
    return VIEW_DESCRIPTIONS.get(condition_id, condition_id.replace("__", " "))
