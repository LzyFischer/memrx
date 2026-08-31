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

  - "note": LLM extracts keywords + a short context note per chunk
    (metadata["keywords"], metadata["note"]) and the note is appended
    directly after the raw chunk text, becoming part of
    entry.lossless_restatement — the single field MemoryStore embeds (see
    core/memory_store.py::add_batch). So this variant needs no special
    retrieval path at all; it's plain semantic search over raw-text+note,
    same as baseline/summary (core/retrieval2a.py). The raw text itself is
    never rewritten, only appended to — chunking.py's isolation argument
    (augmentation only ADDS on top of the same raw chunk) still holds.

Both variants make exactly ONE LLM call per chunk, the same order of
construction-time cost as summary's one-call-per-window (see
core/memory_builder.py) — this was a deliberate constraint, not an
afterthought, so augmentation stays comparable in cost to summary rather
than ballooning into a graph-dimension-style O(chunks) with multi-field
extraction.

"causal" was removed earlier and stays removed — see graph_builder.py /
graph="causal" for why (it's a graph concept, not a per-chunk one).
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
    """augmentation="note": one LLM call per chunk asking for keywords AND a
    short context note (what's being discussed, why it matters — the kind
    of gloss a human might jot in the margin). The note is appended after
    the raw chunk text, so it becomes part of what gets embedded — this is
    the variant's whole point (retrieval over raw text + note, not raw text
    alone). metadata["keywords"] / metadata["note"] keep the pieces
    available separately too, for inspection/debugging."""
    for chunk in chunks:
        prompt = f"""Read this dialogue excerpt and produce two things:
1. "keywords": 3-8 salient keywords/phrases (names, places, specific nouns, technical terms, numbers, dates).
2. "context": one or two plain sentences summarizing what's being discussed and why it might matter later — a short margin note, not a restatement of every line.

Dialogue:
{chunk.lossless_restatement}

Return ONLY a JSON object: {{"keywords": [...], "context": "..."}}"""
        try:
            resp = llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
            data = llm.extract_json(resp)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        raw_keywords = data.get("keywords", [])
        keywords = [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()] if isinstance(raw_keywords, list) else []
        context = data.get("context", "")
        context = context.strip() if isinstance(context, str) else ""

        chunk.metadata["keywords"] = keywords
        chunk.metadata["note"] = context

        note_parts = []
        if keywords:
            note_parts.append("Keywords: " + ", ".join(keywords))
        if context:
            note_parts.append("Context: " + context)
        if note_parts:
            chunk.lossless_restatement = chunk.lossless_restatement + "\n[Note] " + " | ".join(note_parts)
