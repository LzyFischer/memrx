"""augmentation=keywords: one LLM call per raw chunk extracts salient
keywords into metadata["keywords"]. The raw text is left untouched.

MemoryStore builds its BM25 index over these keywords (not the raw text), and
retrieval fuses that sparse channel with semantic search via RRF (see
core/retrieval.py), so the sparse channel reflects what the LLM judged
salient rather than plain term frequency.
"""
from typing import List

from core.entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list

_PROMPT = """Extract the 3-8 most retrieval-salient keywords or short phrases from this dialogue excerpt: names, places, specific nouns, technical terms, numbers, dates — the kind of terms someone would type into a search box to find this excerpt again. Do not include common/filler words.

Dialogue:
{text}

Return a JSON array of short strings. If nothing salient stands out, return []. Return ONLY the JSON array."""


def extract_keywords(llm: LLMClient, text: str) -> List[str]:
    try:
        resp = llm.chat_completion([{"role": "user", "content": _PROMPT.format(text=text)}],
                                   temperature=0.0)
        keywords = coerce_json_list(llm.extract_json(resp), context="keyword extraction")
    except Exception:
        keywords = []
    return [k.strip() for k in keywords if isinstance(k, str) and k.strip()]


def augment_keywords(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    for chunk in chunks:
        chunk.metadata["keywords"] = extract_keywords(llm, chunk.lossless_restatement)
