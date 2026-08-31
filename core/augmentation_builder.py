"""
Augmentation Builder — per-chunk LLM extraction for the two augmentation
variants.

"temporal" has been removed from this dimension. The mechanism (query-time
intent inference -> independent day-distance ranking -> RRF fusion with
semantic search) needed too much machinery to explain cleanly against the
other two variants, so it's dropped in favor of two variants that are both
"extract something from the chunk, then use it at retrieval time" — easier
to present side by side:

  - "keywords": LLM extracts a short list of salient keywords/phrases per
    chunk (metadata["keywords"]). These keywords — NOT the raw chunk text —
    are what MemoryStore's BM25 index is built over (see
    core/memory_store.py::_ensure_bm25), fused with the semantic channel at
    retrieval time via RRF (core/retrieval2a.py::_retrieve_keywords_hybrid).
    This is a deliberate change from an earlier version of this file, where
    "keywords" needed no LLM call at all and BM25 ran directly over the raw
    text (term frequency stats over the full chunk WERE the "keyword
    signal"). That's still a defensible design, but it means BM25 is doing
    exactly what BM25 always does — nothing about it is specific to an
    "augmentation" treatment. Extracting keywords first makes the sparse
    channel reflect what the LLM judged salient, not just what's frequent,
    which is the more interesting comparison for the ablation.

  - "note": LLM extracts a handful of SHORT, CONCRETE fields per chunk —
    keywords, timestamp, location, persons, entities, topic (metadata["keywords"],
    metadata["persons"], etc.) — and a note built by templating those fields
    together (metadata["note"]) is appended directly after the raw chunk
    text, becoming part of entry.lossless_restatement — the single field
    MemoryStore embeds (see core/memory_store.py::add_batch). So this
    variant needs no special retrieval path at all; it's plain semantic
    search over raw-text+note, same as baseline/summary (core/retrieval2a.py).
    The raw text itself is never rewritten, only appended to — chunking.py's
    isolation argument (augmentation only ADDS on top of the same raw chunk)
    still holds.

    This is a deliberate change from an earlier version of this file, where
    the model was asked to freely write a 1-2 sentence "context" summary
    (a mini compression task) rather than fill in short, concrete fields.
    That free-text summary turned out to be the weak point — a 0.6B model
    writing open-ended prose is exactly the failure mode that hurts the
    summary dimension too (vague, occasionally hallucinated). Every field
    here is either a short list (keywords/persons/entities) or a single
    short phrase (location/timestamp/topic) — nothing longer than a few
    words per field — which is the same kind of easy, low-error extraction
    that already made "keywords" work about as well as baseline. The note
    is then composed by TEMPLATING these fields together in code, not by
    asking the LLM to compose the sentence itself.

Both variants make exactly ONE LLM call per chunk, the same order of
construction-time cost as summary's one-call-per-window (see
core/memory_builder.py) — this was a deliberate constraint, not an
afterthought, so augmentation stays comparable in cost to summary rather
than ballooning into a graph-dimension-style O(chunks) with multi-field
extraction.

"causal" was removed earlier and stays removed — a causal link is inherently
a pointer to another chunk, not a fact about one chunk in isolation, so it
belongs to the graph dimension's edge structure rather than here. (graph
also no longer has a "causal" variant — see core/graph_builder.py — but the
reasoning for why this augmentation-level version doesn't belong here is
unaffected by that.)
"""
from typing import List

from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list


def _extract_keywords_list(llm: LLMClient, text: str) -> List[str]:
    prompt = f"""Extract the 3-8 most retrieval-salient keywords or short phrases from this dialogue excerpt: names, places, specific nouns, technical terms, numbers, dates — the kind of terms someone would type into a search box to find this excerpt again. Do not include common/filler words.

Dialogue:
{text}

Return a JSON array of short strings. If nothing salient stands out, return []. Return ONLY the JSON array."""
    try:
        resp = llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
        keywords = coerce_json_list(llm.extract_json(resp), context="keyword extraction")
    except Exception:
        keywords = []
    return [k.strip() for k in keywords if isinstance(k, str) and k.strip()]


def augment_keywords(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    """augmentation="keywords": one LLM call per chunk, extracted terms
    stored in metadata["keywords"] and consumed by MemoryStore's BM25 index
    (core/memory_store.py::_ensure_bm25). The raw chunk text itself is left
    untouched — only metadata is added, per chunking.py's isolation note."""
    for chunk in chunks:
        chunk.metadata["keywords"] = _extract_keywords_list(llm, chunk.lossless_restatement)


def augment_note(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    """augmentation="note": one LLM call per chunk extracting a handful of
    SHORT, CONCRETE fields — keywords, timestamp, location, persons,
    entities, topic — never a free-text summary sentence (that was the
    earlier, weaker design; see the module docstring). The fields are then
    templated together in code into a note appended after the raw chunk
    text, so it becomes part of what gets embedded — this is the variant's
    whole point (retrieval over raw text + note, not raw text alone).
    metadata["keywords"] / metadata["persons"] / etc. keep the pieces
    available separately too, for inspection/debugging."""
    for chunk in chunks:
        prompt = f"""Extract structured information from this dialogue excerpt. This will be appended after the excerpt itself to help retrieval later, so every field must be SHORT and CONCRETE — a list of terms or a short phrase, never a summary sentence or a restatement of the dialogue.

Dialogue:
{chunk.lossless_restatement}

[Requirements — Precise Extraction, not summarization]
- keywords: 3-8 core keywords/short phrases (names, places, specific nouns, technical terms, numbers, dates)
- topic: ONE short phrase (a few words, not a sentence) naming what this excerpt is about
- context: one sentence summary for this chunk

Return ONLY a JSON object: {{"keywords": [...], "topic": "...", "context": "..."}}"""
        try:
            resp = llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
            data = llm.extract_json(resp)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        def _str_list(key: str) -> List[str]:
            raw = data.get(key, [])
            if not isinstance(raw, list):
                return []
            return [v.strip() for v in raw if isinstance(v, str) and v.strip()]

        def _str_or_none(key: str) -> str:
            v = data.get(key)
            return v.strip() if isinstance(v, str) and v.strip() and v.strip().lower() != "null" else ""

        keywords = _str_list("keywords")
        context = _str_or_none("context")
        topic = _str_or_none("topic")

        chunk.metadata["keywords"] = keywords
        chunk.metadata["topic"] = topic
        chunk.metadata["context"] = context

        # Template the fields together — the LLM never composes this
        # sentence itself, which is the point (see module docstring).
        note_parts = []
        if keywords:
            note_parts.append("Keywords: " + ", ".join(keywords))
        if topic:
            note_parts.append("Topic: " + topic)
        if context:
            note_parts.append("Context: " + context)

        note = " | ".join(note_parts)
        chunk.metadata["note"] = note
        if note:
            chunk.lossless_restatement = chunk.lossless_restatement + "\n[Note] " + note

