"""
Memory Builder — Stage 1 + Stage 2

Stage 1 · Semantic Structured Compression (§3.1)
  Φ_gate(W) → {m_k}: sliding-window gating, de-linearization

Stage 2 · Online Semantic Synthesis (§3.2)
  Intra-session consolidation: related fragments are merged during the write phase

Each generated memory entry contains a single field — `lossless_restatement` —
which is a fully disambiguated, self-contained sentence (absolute timestamps
inlined, no pronouns).
"""
import concurrent.futures
from typing import List, Optional

import config
from database.vector_store import VectorStore
from models.memory_entry import Dialogue, MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list


class MemoryBuilder:
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: Optional[int] = None,
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        overlap_size: Optional[int] = None,
        single_entry_mode: bool = False,
        adaptive_split_mode: bool = False,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or config.WINDOW_SIZE
        self.overlap_size = (
            overlap_size if overlap_size is not None
            else getattr(config, "OVERLAP_SIZE", 0)
        )
        # Clamp overlap so step_size stays >= 1 (matters when window_size=1).
        self.overlap_size = max(0, min(self.overlap_size, self.window_size - 1))
        self.step_size = max(1, self.window_size - self.overlap_size)
        # When True the extraction prompt asks for exactly ONE memory entry
        # per window — used by the granularity ablation.
        self.single_entry_mode = single_entry_mode
        # When True the extraction prompt asks the model to SPLIT the window
        # into multiple retrieval-ready entries, with the model itself
        # deciding the number and granularity of the splits. This is the
        # NON-TARGET adaptive entry decomposition condition: no query, no
        # answer hint — the split is driven by the content alone.
        self.adaptive_split_mode = adaptive_split_mode
        if self.single_entry_mode and self.adaptive_split_mode:
            raise ValueError(
                "single_entry_mode and adaptive_split_mode are mutually exclusive"
            )

        self.enable_parallel_processing = (
            enable_parallel_processing
            if enable_parallel_processing is not None
            else getattr(config, "ENABLE_PARALLEL_PROCESSING", True)
        )
        self.max_parallel_workers = (
            max_parallel_workers
            if max_parallel_workers is not None
            else getattr(config, "MAX_PARALLEL_WORKERS", 4)
        )

        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None:
        self.dialogue_buffer.append(dialogue)
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True) -> None:
        use_parallel = (
            self.enable_parallel_processing
            and len(dialogues) > self.window_size * 2
        )
        if use_parallel:
            self._add_dialogues_parallel(dialogues)
        else:
            for d in dialogues:
                self.dialogue_buffer.append(d)
            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()

    def process_remaining(self) -> None:
        """Process any dialogues left in the buffer (call after add_dialogues)."""
        if self.dialogue_buffer:
            print(f"Processing remaining {len(self.dialogue_buffer)} dialogues")
            entries = self._generate_entries(self.dialogue_buffer)
            if entries:
                self.vector_store.add_entries(entries)
                self.processed_count += len(self.dialogue_buffer)
            self.dialogue_buffer = []

    def process_window(self) -> None:
        if not self.dialogue_buffer:
            return
        window = self.dialogue_buffer[: self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size :]
        print(f"Processing window: {len(window)} dialogues ({self.processed_count} processed so far)")
        entries = self._generate_entries(window)
        if entries:
            self.vector_store.add_entries(entries)
            self.processed_count += len(window)
        print(f"  → {len(entries)} memory entries generated")

    # ------------------------------------------------------------------
    # Parallel processing
    # ------------------------------------------------------------------

    def _add_dialogues_parallel(self, dialogues: List[Dialogue]) -> None:
        pre_existing = list(self.dialogue_buffer)
        windows: List[List[Dialogue]] = []
        try:
            self.dialogue_buffer.extend(dialogues)
            pos = 0
            while pos + self.window_size <= len(self.dialogue_buffer):
                windows.append(self.dialogue_buffer[pos : pos + self.window_size])
                pos += self.step_size
            remaining = self.dialogue_buffer[pos:]
            if remaining:
                windows.append(remaining)
            self.dialogue_buffer = []

            print(f"[Parallel] {len(windows)} batches, {self.max_parallel_workers} workers")
            self._process_windows_parallel(windows)

        except Exception as e:
            print(f"[Parallel] Failed ({e}), falling back to sequential")
            if not self.dialogue_buffer:
                self.dialogue_buffer = pre_existing + list(dialogues)
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def _process_windows_parallel(self, windows: List[List[Dialogue]]) -> None:
        all_entries: List[MemoryEntry] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as ex:
            futures = {
                ex.submit(self._generate_entries_worker, w, i + 1): i
                for i, w in enumerate(windows)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[Parallel] Window {idx + 1}: {len(entries)} entries")
                except Exception as e:
                    print(f"[Parallel] Window {idx + 1} failed: {e}")

        if all_entries:
            self.vector_store.add_entries(all_entries)
            self.processed_count += sum(len(w) for w in windows)
        print(f"[Parallel] Done — {len(all_entries)} total entries")

    def _generate_entries_worker(
        self, window: List[Dialogue], window_num: int
    ) -> List[MemoryEntry]:
        print(f"[Worker {window_num}] {len(window)} dialogues")
        entries = self._generate_entries(window)
        print(f"[Worker {window_num}] → {len(entries)} entries")
        return entries

    # ------------------------------------------------------------------
    # LLM extraction (Stage 1 + Stage 2 gating)
    # ------------------------------------------------------------------

    def _generate_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        dialogue_text = "\n".join(str(d) for d in dialogues)

        prompt = self._extraction_prompt(dialogue_text)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional information extraction assistant. "
                    "Extract structured, unambiguous facts from conversations. "
                    "Output valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response_format = (
            {"type": "json_object"}
            if getattr(config, "USE_JSON_FORMAT", False)
            else None
        )

        for attempt in range(3):
            try:
                response = self.llm_client.chat_completion(messages, temperature=0.1, response_format=response_format)
                entries = self._parse_response(response)
                self._stamp_source_range(entries, dialogues)
                return entries
            except Exception as e:
                if attempt < 2:
                    print(f"  Extraction attempt {attempt + 1}/3 failed: {e} — retrying")
                else:
                    print(f"  Extraction failed after 3 attempts: {e}")
                    return []
        return []

    def _parse_response(self, response: str) -> List[MemoryEntry]:
        data = coerce_json_list(
            self.llm_client.extract_json(response), context="memory_builder extraction"
        )
        entries = []
        for item in data:
            # Tolerate either {"lossless_restatement": "..."} or a bare string
            if isinstance(item, dict):
                text = item.get("lossless_restatement")
            elif isinstance(item, str):
                text = item
            else:
                text = None
            if not isinstance(text, str) or not text.strip():
                continue
            entries.append(MemoryEntry(lossless_restatement=text.strip()))
        # Safety net for the granularity ablation: if the LLM ignored the
        # "exactly one entry" instruction we keep only the first.
        if self.single_entry_mode and len(entries) > 1:
            entries = entries[:1]
        return entries

    @staticmethod
    def _stamp_source_range(entries: List[MemoryEntry], dialogues: List[Dialogue]) -> None:
        """Record which raw dialogue turns this window's entries were built
        from, using the same metadata keys as core/chunking.py::build_raw_chunks
        (dia_id_start / dia_id_end) so retrieval-recall analysis can locate
        gold-evidence turns regardless of which dimension produced the entry
        (2a preliminary experiment 3 — see eval/analysis/retrieval_recall.py).
        This is a window-level range, not a per-entry one: adaptive_split_mode
        can split one window into several entries, and there's no LLM-side
        signal for which sub-span of the window each split entry came from,
        so every entry generated from the same window gets the same range.
        """
        if not dialogues:
            return
        start, end = dialogues[0].dialogue_id, dialogues[-1].dialogue_id
        for e in entries:
            e.metadata.setdefault("dia_id_start", start)
            e.metadata.setdefault("dia_id_end", end)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _extraction_prompt(self, dialogue_text: str) -> str:
        if self.single_entry_mode:
            return self._single_entry_prompt(dialogue_text)
        if self.adaptive_split_mode:
            return self._adaptive_split_prompt(dialogue_text)
        return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. Write timestamps to all entries.
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured.
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow). Use full names and absolute ISO 8601 timestamps inline.
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence that includes all relevant subjects, objects, time, and location inline.

[Output Format]
Return a JSON array. Each element is a memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)"
  }},
  ...
]
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""

    def _single_entry_prompt(self, dialogue_text: str) -> str:
        """
        Granularity-ablation prompt: compress the entire window into ONE
        memory entry. Same disambiguation rules as the default prompt; only
        the entry-count requirement and the density emphasis differ.
        """
        return f"""
Your task is to compress the following dialogues into a SINGLE memory entry that preserves every piece of information necessary to answer downstream questions.

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. Write timestamps to the lossless restatement.
1. **Complete Coverage**:  Ensure ALL information in the dialogues is captured.
2. **Exactly ONE entry**: The output JSON array MUST contain exactly one element.
3. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow). Use full names and absolute ISO 8601 timestamps inline.
4. **Lossless Information**: The single lossless_restatement must be a self-contained, independently understandable text that includes all relevant information.

[Output Format]
Return a JSON array with EXACTLY ONE element:

```json
[
  {{
    "lossless_restatement": "Complete, dense, unambiguous restatement that includes every fact and time stamp from the window"
  }}
]
```

Now process the above dialogues. Return ONLY the JSON array (length 1), no other explanations.
"""

    def _adaptive_split_prompt(self, dialogue_text: str) -> str:
        """
        NON-TARGET adaptive entry decomposition prompt: split the window into
        multiple retrieval-ready memory entries. The model decides the number
        of entries and where the boundaries fall, based ONLY on the dialogue
        content — no target question or answer is provided. Disambiguation
        rules are identical to the other prompts; only the decomposition
        instruction differs.
        """
        return f"""
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. Write timestamps to all entries.
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured, add timestamps to all entries.
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow). Use full names and absolute ISO 8601 timestamps inline.
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence that includes all relevant subjects, objects, time, and location inline.

[Output Format]
Return a JSON array. Each element is a memory entry:

```json
[
  {{
    "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)"
  }},
  ...
]
```

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
"""
