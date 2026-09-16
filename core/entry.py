"""Core data structures: Dialogue (one raw turn) and MemoryEntry (one retrievable unit)."""
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """One retrievable memory unit.

    `lossless_restatement` is the text that is embedded and indexed; the reader
    sees it too unless metadata["display"] is set. `metadata` carries view-specific side info:
      dia_id_start / dia_id_end   source turn range (all views; used for recall)
      date                        session date of the chunk
      display                     augmentation: raw text shown to the reader instead
      attributes                  augmentation: the generated fields
      entities                    graph; drives the entity index
      summary_fallback            summary; True if the LLM output was unusable
    """

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Dialogue(BaseModel):
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{prefix}{self.speaker}: {self.content}"
