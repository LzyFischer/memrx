"""Raw chunking: the shared base unit for every view.

All three processing views work on exactly these chunks, one LLM call per chunk:
  summary       chunk -> C(chunk)          (raw text replaced by a structured summary)
  augmentation  chunk -> chunk + A(chunk)  (raw text kept, attributes appended)
  graph         chunk -> G(chunk)          (raw text kept, entities indexed)
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, TypeVar

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


T = TypeVar("T")


def map_chunks(fn: Callable[[MemoryEntry], T], chunks: List[MemoryEntry], workers: int = 8) -> List[T]:
    """fn over chunks with a thread pool (vLLM batches concurrent requests); order preserved."""
    if workers <= 1:
        return [fn(c) for c in chunks]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, chunks))
