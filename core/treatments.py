"""
Treatment dispatcher for the 2a ablation.

build_memory_store(dialogues, condition, llm, embedding_model) -> MemoryStore

Reuses:
  - core/memory_builder.py's existing, already-tested extraction prompts for
    the summary dimension (session_level = single_entry_mode,
    fine_grained = adaptive_split_mode). We bypass its LanceDB coupling with
    a tiny in-memory collector so the same prompts populate our unified
    MemoryStore (see core/memory_store.py for why we need one shared
    backend across all 4 conditions).
  - core/chunking.py's raw (non-LLM) chunks for baseline / augmentation /
    graph, so augmentation and graph are measured as additions ON TOP OF
    the same raw text baseline, isolating each dimension's own effect.
"""
from typing import List

import config
from config_2a import Condition
from core.chunking import build_raw_chunks
from core.graph_builder import build_causal_graph, build_entity_graph, build_semantic_graph
from core.augmentation_builder import augment_keywords, augment_note
from core.memory_builder import MemoryBuilder
from core.memory_store import MemoryStore
from models.memory_entry import Dialogue
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


class _Collector:
    """Drop-in replacement for VectorStore's add_entries(), used only to let
    MemoryBuilder run its extraction prompts without touching LanceDB."""

    def __init__(self):
        self.collected = []

    def add_entries(self, entries):
        self.collected.extend(entries)


def _build_summary_entries(dialogues: List[Dialogue], variant: str, llm: LLMClient,
                            window_size: int, overlap: int):
    if variant == "session_level":
        collector = _Collector()
        builder = MemoryBuilder(
            llm, collector, window_size=window_size, overlap_size=overlap,
            single_entry_mode=True, enable_parallel_processing=False,
        )
        builder.add_dialogues(dialogues, auto_process=True)
        builder.process_remaining()
        return collector.collected

    if variant == "fine_grained":
        collector = _Collector()
        builder = MemoryBuilder(
            llm, collector, window_size=window_size, overlap_size=overlap,
            adaptive_split_mode=True, enable_parallel_processing=False,
        )
        builder.add_dialogues(dialogues, auto_process=True)
        builder.process_remaining()
        return collector.collected

    raise ValueError(f"Unknown summary variant: {variant}")


def build_memory_store(
    dialogues: List[Dialogue],
    condition: Condition,
    llm: LLMClient,
    embedding_model: EmbeddingModel,
    window_size: int = None,
    overlap: int = None,
) -> MemoryStore:
    window_size = window_size or config.WINDOW_SIZE
    overlap = overlap if overlap is not None else getattr(config, "OVERLAP_SIZE", 0)

    store = MemoryStore(embedding_model)

    # ---- baseline: raw chunks, no metadata, no edges -------------------
    if condition.dimension == "baseline":
        chunks = build_raw_chunks(dialogues, window_size=window_size, overlap=overlap)
        store.add_batch(chunks)
        return store

    # ---- summary: LLM-restated entries at the requested granularity ----
    if condition.dimension == "summary":
        entries = _build_summary_entries(dialogues, condition.summary, llm, window_size, overlap)
        store.add_batch(entries)
        return store

    # ---- augmentation: raw chunks + metadata (+ appended note for
    # "note"), no edges. Both variants make exactly one LLM call per chunk
    # (core/augmentation_builder.py) — "keywords" feeds MemoryStore's BM25
    # index, "note" gets folded into the embedded text itself.
    if condition.dimension == "augmentation":
        chunks = build_raw_chunks(dialogues, window_size=window_size, overlap=overlap)
        if condition.augmentation == "keywords":
            augment_keywords(chunks, llm)
        elif condition.augmentation == "note":
            augment_note(chunks, llm)
        else:
            raise ValueError(f"Unknown augmentation variant: {condition.augmentation}")
        store.add_batch(chunks)
        return store

    # ---- graph: raw chunks + edges, no metadata beyond what graph needs -
    if condition.dimension == "graph":
        chunks = build_raw_chunks(dialogues, window_size=window_size, overlap=overlap)
        store.add_batch(chunks)  # must be added first so entry_id lookups work for edges
        if condition.graph == "semantic":
            build_semantic_graph(store, top_n=3)
        elif condition.graph == "entity":
            build_entity_graph(store, chunks, llm)
        elif condition.graph == "causal":
            build_causal_graph(store, chunks, llm)
        else:
            raise ValueError(f"Unknown graph variant: {condition.graph}")
        return store

    raise ValueError(f"Unknown condition dimension: {condition.dimension}")
