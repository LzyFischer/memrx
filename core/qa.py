"""Reader side, shared by curation and the preliminary runs: format retrieved
entries into a context string, ask the LLM, parse out the answer."""
from typing import List

from core.entry import MemoryEntry
from utils.llm_client import LLMClient
from utils.locomo import build_qa_prompt

MAX_CONTEXT_CHARS = 6000


def format_context(entries: List[MemoryEntry], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, total = [], 0
    for i, e in enumerate(entries, 1):
        line = f"[{i}] {e.lossless_restatement}"
        total += len(line)
        if total > max_chars:
            break
        parts.append(line)
    return "\n".join(parts)


def answer_question(llm: LLMClient, context: str, question: str, category: int) -> str:
    """Returns the "answer" field of the JSON reply, or the raw reply if it does not parse."""
    prompt = build_qa_prompt(context, question, category)
    try:
        raw = llm.chat_completion([{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=1024).strip()
    except Exception as e:
        return f"[error] {e}"
    try:
        parsed = llm.extract_json(raw)
        if isinstance(parsed, dict):
            a = parsed.get("answer")
            if isinstance(a, str) and a.strip():
                return a.strip()
    except Exception:
        pass
    return raw


def gold_answer(qa: dict) -> str:
    """LoCoMo adversarial (category 5) questions have no gold answer string."""
    gold = qa.get("answer", "")
    if int(qa.get("category", 1)) == 5 and not gold:
        gold = "Not mentioned in the conversation"
    return str(gold)
