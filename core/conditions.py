"""The candidate menu the router chooses from: raw text plus one processing
variant per dimension (3+1), all applied per raw chunk.

    summary__structured       chunk -> C(chunk)          one dense LLM restatement per chunk
    augmentation__attributes  chunk -> chunk + A(chunk)  MemInsight-lite attributes
    graph__entity             chunk -> G(chunk)          Mem0-style entity links as a ranking signal

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
    summary: str = "none"          # "none" | "structured"
    augmentation: str = "none"     # "none" | "attributes"
    graph: str = "none"            # "none" | "entity"


def build_condition_matrix() -> List[Condition]:
    return [
        Condition("baseline", "baseline"),
        Condition("summary__structured", "summary", summary="structured"),
        Condition("augmentation__attributes", "augmentation", augmentation="attributes"),
        Condition("graph__entity", "graph", graph="entity"),
    ]


VIEW_DESCRIPTIONS: Dict[str, str] = {
    "baseline":
        "Raw conversation chunks, verbatim, no extra processing. Best when the answer is "
        "stated literally somewhere and any rewriting risks losing the exact wording.",
    "summary__structured":
        "Each chunk is compressed by an LLM into one dense, self-contained restatement with full "
        "names and absolute timestamps instead of pronouns and relative time. Compact; drops the "
        "original wording.",
    "augmentation__attributes":
        "Raw chunks kept verbatim, with LLM-generated entities, events, absolute dates and topic "
        "keywords appended for dense and keyword retrieval. Good when the question uses different "
        "words than the conversation or asks about a date.",
    "graph__entity":
        "Raw chunks linked through the entities they mention; chunks sharing entities named in "
        "the question are ranked higher on top of semantic search. Good for questions about "
        "specific people, places, or objects that are spread across the conversation.",
}


def describe(condition_id: str) -> str:
    return VIEW_DESCRIPTIONS.get(condition_id, condition_id.replace("__", " "))
