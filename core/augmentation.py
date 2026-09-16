"""augmentation=attributes (MemInsight-lite), chunk level.

Each raw chunk is KEPT and extended with LLM-generated attributes:

    M_aug = M_raw + Entities / Events / Time / Keywords

The augmented text is what gets embedded and BM25-indexed, so a query can
match an attribute (a resolved date, a paraphrased event, a topic keyword)
that the raw wording does not contain. The reader is shown only the raw text
(metadata["display"]), so this view spends the same context budget per chunk
as baseline and the difference is purely in retrieval.
"""
from typing import List

from core.entry import MemoryEntry
from core.extract import extract_fields, render
from utils.llm_client import LLMClient

FIELDS = ["entities", "events", "time", "keywords"]
SECTIONS = [("entities", "Entities", False), ("events", "Events", False),
            ("time", "Time", False), ("keywords", "Keywords", False)]

_PROMPT = """Annotate this conversation excerpt with attributes that would help find it again later.

Conversation date: {date}

Excerpt:
{text}

Return a JSON object with exactly these fields (use [] when nothing applies):
{{"entities": [], "events": [], "time": [], "keywords": []}}

- entities: people, places, organizations, objects, titles mentioned.
- events: short phrases for what happened, with who (e.g. "Alice adopted a dog").
- time: time references rewritten as absolute dates using the conversation date (e.g. "last Saturday" -> "the Saturday before 9 June 2023").
- keywords: 3-8 topic words or synonyms someone might use to ask about this excerpt, including words that do not appear in it.

Do not add facts that are not stated. Return ONLY the JSON object."""


def augment_attributes(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    failed = 0
    for c in chunks:
        raw = c.lossless_restatement
        fields = extract_fields(llm, _PROMPT.format(date=c.metadata.get("date") or "unknown",
                                                    text=raw), FIELDS)
        c.metadata["display"] = raw
        if fields is None:
            failed += 1
            continue
        c.metadata["attributes"] = fields
        c.lossless_restatement = raw + "\n" + render(fields, SECTIONS)
    print(f"  [augmentation] {len(chunks)} chunks, {failed} without attributes")
