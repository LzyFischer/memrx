"""Raw chunking: the shared base unit for baseline / augmentation / graph.

Only the summary view rewrites text with an LLM. Augmentation only adds
metadata to a raw chunk and graph only adds edges between raw chunks, so each
view's difference from baseline is exactly its own treatment.
"""
from typing import List

from core.entry import Dialogue, MemoryEntry


def build_raw_chunks(dialogues: List[Dialogue], window_size: int = 5,
                     overlap: int = 0) -> List[MemoryEntry]:
    """Group consecutive turns into fixed windows, no LLM call."""
    step = max(1, window_size - overlap)
    chunks: List[MemoryEntry] = []
    for i in range(0, len(dialogues), step):
        window = dialogues[i:i + window_size]
        chunks.append(MemoryEntry(
            lossless_restatement="\n".join(str(d) for d in window),
            metadata={
                "dia_id_start": window[0].dialogue_id,
                "dia_id_end": window[-1].dialogue_id,
                "date": window[0].timestamp or "",
            },
        ))
    return chunks
