"""summary=session_level: each window of turns is compressed by the LLM into
exactly ONE self-contained memory entry.

Windowing matches the original MemoryBuilder's sequential path: slide a
`window_size` window with step `window_size - overlap` while a full window is
available, then process whatever is left in the buffer as a final window.
"""
from typing import List

from core.entry import Dialogue, MemoryEntry
from utils.llm_client import LLMClient, coerce_json_list

_SYSTEM = ("You are a professional information extraction assistant. "
           "Extract structured, unambiguous facts from conversations. "
           "Output valid JSON only.")

_PROMPT = """
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


def _windows(dialogues: List[Dialogue], window_size: int, overlap: int) -> List[List[Dialogue]]:
    overlap = max(0, min(overlap, window_size - 1))
    step = max(1, window_size - overlap)
    buf, out = list(dialogues), []
    while len(buf) >= window_size:
        out.append(buf[:window_size])
        buf = buf[step:]
    if buf:
        out.append(buf)
    return out


def _parse(llm: LLMClient, response: str) -> List[MemoryEntry]:
    entries = []
    for item in coerce_json_list(llm.extract_json(response), context="summary extraction"):
        text = item.get("lossless_restatement") if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            entries.append(MemoryEntry(lossless_restatement=text.strip()))
    return entries[:1]  # the prompt asks for exactly one; keep the first if it ignores that


def summarize_window(llm: LLMClient, window: List[Dialogue], retries: int = 3) -> List[MemoryEntry]:
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _PROMPT.format(
                    dialogue_text="\n".join(str(d) for d in window))}]
    for attempt in range(retries):
        try:
            entries = _parse(llm, llm.chat_completion(messages, temperature=0.1))
        except Exception as e:
            print(f"  summary attempt {attempt + 1}/{retries} failed: {e}")
            continue
        for e in entries:
            e.metadata.setdefault("dia_id_start", window[0].dialogue_id)
            e.metadata.setdefault("dia_id_end", window[-1].dialogue_id)
        return entries
    return []


def build_summary_entries(dialogues: List[Dialogue], llm: LLMClient,
                          window_size: int, overlap: int) -> List[MemoryEntry]:
    windows = _windows(dialogues, window_size, overlap)
    entries: List[MemoryEntry] = []
    for w in windows:
        entries.extend(summarize_window(llm, w))
    print(f"  [summary] {len(windows)} windows -> {len(entries)} entries")
    return entries
