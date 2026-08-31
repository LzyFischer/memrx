"""
Raw chunking — shared base unit for baseline / augmentation / graph.

Design note:
-----------------------------------------------------------------
Only the *summary* dimension uses an LLM to rewrite dialogue windows into
paraphrased "lossless_restatement" text (see core/memory_builder.py).

The *baseline*, *augmentation*, and *graph* conditions all use the exact
same RAW, non-paraphrased chunking of the conversation — augmentation only
ADDS metadata on top of a raw chunk, and graph only ADDS edges on top of a
raw chunk. This way:
  - summary vs baseline isolates "does LLM restatement help"
  - augmentation vs baseline isolates "does adding keyword metadata, or a
    short appended note, to raw chunks help — holding the raw chunk text
    itself untouched (augmentation="note" only ever APPENDS after it, never
    rewrites it)"
  - graph vs baseline isolates "does adding structural edges between raw
    chunks help, holding the chunk text itself constant"
If augmentation/graph also silently rewrote the text via an LLM, any effect
we saw would be a mix of "restatement helped" + "the added structure
helped", which is exactly the confound the 2a design is trying to avoid.
"""
from typing import List

from models.memory_entry import MemoryEntry
from models.memory_entry import Dialogue


def build_raw_chunks(
    dialogues: List[Dialogue],
    window_size: int = 5,
    overlap: int = 0,
) -> List[MemoryEntry]:
    """Group consecutive turns into fixed windows, no LLM call.
    Each chunk's text is the literal concatenation of the turns it covers."""
    step = max(1, window_size - overlap)
    chunks: List[MemoryEntry] = []
    i = 0
    while i < len(dialogues):
        window = dialogues[i : i + window_size]
        if not window:
            break
        text = "\n".join(str(d) for d in window)
        first_id, last_id = window[0].dialogue_id, window[-1].dialogue_id
        chunk = MemoryEntry(
            lossless_restatement=text,
            metadata={
                "dia_id_start": first_id,
                "dia_id_end": last_id,
                "date": window[0].timestamp or "",
            },
        )
        chunks.append(chunk)
        i += step
    return chunks
