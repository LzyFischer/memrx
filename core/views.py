"""Build one MemoryStore per view (condition), optionally cached on disk."""
import os
from typing import List, Optional, Tuple

import config
from core.augmentation import augment_keywords
from core.chunking import build_raw_chunks
from core.conditions import Condition
from core.entry import Dialogue
from core.graph import build_entity_graph
from core.store import MemoryStore
from core.summary import build_summary_entries
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


def build_memory_store(dialogues: List[Dialogue], condition: Condition, llm: LLMClient,
                       embedding_model: EmbeddingModel, window_size: Optional[int] = None,
                       overlap: Optional[int] = None) -> MemoryStore:
    window_size = window_size or config.WINDOW_SIZE
    overlap = config.OVERLAP_SIZE if overlap is None else overlap
    store = MemoryStore(embedding_model)

    if condition.dimension == "summary":
        store.add_batch(build_summary_entries(dialogues, llm, window_size, overlap))
        return store

    chunks = build_raw_chunks(dialogues, window_size=window_size, overlap=overlap)
    if condition.dimension == "baseline":
        store.add_batch(chunks)
    elif condition.dimension == "augmentation":
        augment_keywords(chunks, llm)
        store.add_batch(chunks)
    elif condition.dimension == "graph":
        store.add_batch(chunks)                  # entries must exist before edges are added
        build_entity_graph(store, chunks, llm)
    else:
        raise ValueError(f"Unknown condition dimension: {condition.dimension}")
    return store


def get_store(dialogues: List[Dialogue], condition: Condition, llm: LLMClient,
              embedding_model: EmbeddingModel, sample_id: str, cache_dir: Optional[str],
              window_size: int, overlap: int) -> Tuple[MemoryStore, bool]:
    """Returns (store, loaded_from_cache). cache_dir=None disables caching."""
    if cache_dir:
        path = os.path.join(cache_dir, f"{sample_id}__{condition.condition_id}"
                                       f"__w{window_size}o{overlap}.pkl")
        if os.path.exists(path):
            return MemoryStore.load(path, embedding_model), True
    store = build_memory_store(dialogues, condition, llm, embedding_model, window_size, overlap)
    if cache_dir:
        store.save(path)
    return store, False
