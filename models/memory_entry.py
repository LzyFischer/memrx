"""
Core data structures: MemoryEntry (atomic unit) and Dialogue (raw input).

Paper ref: Section 3.1 — Atomic Entries {m_k}
  m_k = F_theta(W_t) = Phi_time ∘ Phi_coref ∘ Phi_extract(W_t)
  Indexed via: I(m_k) = {v_k (semantic)}  — semantic-only retrieval
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class MemoryEntry(BaseModel):
    """
    Atomic memory unit, self-contained and disambiguated.

    Fields:
      - entry_id: stable unique id for deduplication
      - lossless_restatement: the only stored content. Must be a complete,
        self-contained sentence (no pronouns, absolute timestamps inlined),
        because it is the only text we embed and the only text the answerer
        sees at retrieval time.
      - metadata: optional bag for treatment-specific side info, added for
        the 2a ablation (augmentation / graph dimensions). Never touched by
        the baseline / summary-only pipeline, so it's a no-op for existing
        code. Conventions used by the current treatments:
          augmentation="keywords" -> metadata["keywords"] (List[str],
                                      LLM-extracted; drives MemoryStore's
                                      BM25 index, see core/memory_store.py)
          augmentation="note"     -> metadata["keywords"] (List[str]) and
                                      metadata["note"] (str), both
                                      LLM-extracted; the note is also
                                      appended onto lossless_restatement
                                      itself so it's part of the embedded
                                      text (core/augmentation_builder.py)
          graph=*                 -> metadata["node_id"], edges live in the
                                      MemoryStore.graph, not here
          summary="hierarchical"  -> metadata["level"] (0=fine, 1=session
                                      summary), metadata["children"] (List[str]
                                      of entry_id, only set on level-1 nodes)
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dialogue(BaseModel):
    """Raw input: a single conversational turn."""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None  # ISO 8601

    def __str__(self) -> str:
        prefix = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{prefix}{self.speaker}: {self.content}"
