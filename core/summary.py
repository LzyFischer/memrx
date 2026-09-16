"""summary: chunk -> ONE dense restatement (raw text is not kept).

Each raw chunk is compressed by the LLM into a single self-contained
lossless_restatement, so the view has the same number of units as baseline.
"""
from typing import List

import config
from core.chunking import map_chunks
from core.entry import MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list

_SYSTEM = ("You are a professional information extraction assistant. "
           "Extract structured, unambiguous facts from conversations. "
           "Output valid JSON only.")

_PROMPT = """Your task is to compress the following dialogues into a SINGLE memory entry that preserves every piece of information necessary to answer downstream questions.

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. Write timestamps to the lossless restatement with format YYYY-MM-DDTHH:MM:SS or null.
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


def _parse(llm: LLMClient, response: str) -> str:
    """The first element's lossless_restatement; "" if there is none."""
    for item in coerce_json_list(llm.extract_json(response), context="summary"):
        text = item.get("lossless_restatement") if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def summarize_chunk(llm: LLMClient, chunk: MemoryEntry, retries: int = 3) -> MemoryEntry:
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _PROMPT.format(dialogue_text=chunk.lossless_restatement)}]
    text = ""
    for attempt in range(retries):
        try:
            text = _parse(llm, llm.chat_completion(messages, temperature=0.1))
            if text:
                break
        except Exception as e:
            print(f"  summary attempt {attempt + 1}/{retries} failed: {e}")
    fallback = not text
    if fallback:
        # Never drop a chunk: an unparseable summary falls back to the raw text
        # so the view keeps full coverage (counted in metadata for inspection).
        text = chunk.lossless_restatement
    return MemoryEntry(lossless_restatement=text, metadata={**chunk.metadata, "summary_fallback": fallback})


def build_summary_entries(chunks: List[MemoryEntry], llm: LLMClient) -> List[MemoryEntry]:
    entries = map_chunks(lambda c: summarize_chunk(llm, c), chunks, config.LLM_WORKERS)
    n_fb = sum(e.metadata["summary_fallback"] for e in entries)
    print(f"  [summary] {len(chunks)} chunks -> {len(entries)} entries ({n_fb} fell back to raw)")
    return entries
