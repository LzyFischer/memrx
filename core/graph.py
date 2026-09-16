"""graph: chunk -> G(chunk)   (Mem0-style entity-linked memory)

Build: an LLM extracts the named entities of each raw chunk into
metadata["entities"]. Chunks that mention the same entity are linked through
that entity node (memory - entity - memory); the raw text is unchanged.

Retrieve (core/retrieval.py): entities of the store that occur in the query
give each chunk an entity score, which is mixed with the dense score:

    S_g(q, m) = sum_{e in E_q ∩ E_m} idf(e)
    S(q, m)   = alpha * minmax(S_dense) + (1 - alpha) * S_g / max S_g

idf down-weights entities that appear everywhere (e.g. the two speakers), so
no hard frequency cut-off is needed. A query that mentions no known entity is
ranked by dense similarity alone.
"""
import math
import re
from typing import Dict, List

import config
from core.chunking import map_chunks
from core.entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list

# Unchanged from the previous graph view, so stores cached from it still load.
_PROMPT = """Extract all named entities (people, places, organizations, specific objects/events) from this dialogue excerpt.

Dialogue:
{text}

Return a JSON array of entity name strings (deduplicated, singular canonical form). Return ONLY the JSON array."""


def _extract_entities(llm: LLMClient, text: str) -> List[str]:
    try:
        resp = llm.chat_completion([{"role": "user", "content": _PROMPT.format(text=text)}],
                                   temperature=0.0)
        entities = coerce_json_list(llm.extract_json(resp), context="entity extraction")
    except Exception:
        entities = []
    return sorted({str(e).strip().lower() for e in entities if str(e).strip()})


def extract_chunk_entities(chunks: List[MemoryEntry], llm: LLMClient) -> None:
    for chunk, ents in zip(chunks, map_chunks(lambda c: _extract_entities(llm, c.lossless_restatement),
                                              chunks, config.LLM_WORKERS)):
        chunk.metadata["entities"] = ents
    n_ent = len({e for c in chunks for e in c.metadata["entities"]})
    print(f"  [graph] {len(chunks)} chunks, {n_ent} distinct entities")


class EntityIndex:
    """entity -> chunk ids, with idf, and query-side entity matching."""

    def __init__(self, entries: Dict[str, MemoryEntry]):
        self.chunks_of: Dict[str, List[str]] = {}
        for eid, e in entries.items():
            for ent in e.metadata.get("entities", []):
                self.chunks_of.setdefault(ent, []).append(eid)
        n = max(1, len(entries))
        self.idf = {ent: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
                    for ent, ids in self.chunks_of.items()}
        self._patterns = {ent: re.compile(r"(?<!\w)" + re.escape(ent) + r"(?!\w)")
                          for ent in self.chunks_of if len(ent) >= 2}

    def query_entities(self, query: str) -> List[str]:
        q = query.lower()
        return [ent for ent, pat in self._patterns.items() if pat.search(q)]

    def scores(self, query: str) -> Dict[str, float]:
        """chunk id -> S_g(q, chunk); only chunks sharing an entity with the query."""
        out: Dict[str, float] = {}
        for ent in self.query_entities(query):
            for eid in self.chunks_of[ent]:
                out[eid] = out.get(eid, 0.0) + self.idf[ent]
        return out
