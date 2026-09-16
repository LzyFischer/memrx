"""Build one MemoryStore per view (condition), optionally cached on disk."""
import os
from typing import List, Optional, Tuple

import config
from core.augmentation import augment_attributes
from core.chunking import build_raw_chunks
from core.conditions import Condition
from core.entry import Dialogue
from core.graph import extract_chunk_entities
from core.store import MemoryStore
from core.summary import build_summary_entries
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


def build_memory_store(dialogues: List[Dialogue], condition: Condition, llm: LLMClient,
                       embedding_model: EmbeddingModel, window_size: Optional[int] = None,
                       overlap: Optional[int] = None) -> MemoryStore:
    """Every view starts from the same raw chunks and processes each chunk."""
    window_size = window_size or config.WINDOW_SIZE
    overlap = config.OVERLAP_SIZE if overlap is None else overlap
    chunks = build_raw_chunks(dialogues, window_size=window_size, overlap=overlap)
    store = MemoryStore(embedding_model)

    if condition.dimension == "baseline":
        pass
    elif condition.dimension == "summary":
        chunks = build_summary_entries(chunks, llm)
    elif condition.dimension == "augmentation":
        augment_attributes(chunks, llm)
    elif condition.dimension == "graph":
        extract_chunk_entities(chunks, llm)
    else:
        raise ValueError(f"Unknown condition dimension: {condition.dimension}")
    store.add_batch(chunks)
    return store


# Bump when a view's construction changes, so stale cache files are not reused.
BUILD_VERSION = {"baseline": 1, "summary": 2, "augmentation": 2, "graph": 1}


def get_store(dialogues: List[Dialogue], condition: Condition, llm: LLMClient,
              embedding_model: EmbeddingModel, sample_id: str, cache_dir: Optional[str],
              window_size: int, overlap: int) -> Tuple[MemoryStore, bool]:
    """Returns (store, loaded_from_cache). cache_dir=None disables caching."""
    if cache_dir:
        v = BUILD_VERSION.get(condition.dimension, 1)
        path = os.path.join(cache_dir, f"{sample_id}__{condition.condition_id}"
                                       f"__w{window_size}o{overlap}{'' if v == 1 else f'__v{v}'}.pkl")
        if os.path.exists(path):
            return MemoryStore.load(path, embedding_model), True
    store = build_memory_store(dialogues, condition, llm, embedding_model, window_size, overlap)
    if cache_dir:
        store.save(path)
    return store, False
