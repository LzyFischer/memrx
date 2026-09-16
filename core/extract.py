"""Shared helper for the per-chunk LLM extraction used by summary and augmentation:
ask for a JSON object of named list fields, then render it as sectioned text."""
from typing import Dict, List, Optional, Sequence, Tuple

from utils.llm_client import LLMClient


def extract_fields(llm: LLMClient, prompt: str, fields: Sequence[str],
                   retries: int = 2) -> Optional[Dict[str, List[str]]]:
    """Returns {field: [str, ...]} or None if no attempt produced a usable JSON object."""
    for _ in range(retries):
        try:
            resp = llm.chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
            data = llm.extract_json(resp)
        except Exception:
            continue
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            continue
        out = {}
        for f in fields:
            v = data.get(f) or []
            if isinstance(v, str):
                v = [v]
            out[f] = [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()]
        if any(out.values()):
            return out
    return None


def render(fields: Dict[str, List[str]], sections: Sequence[Tuple[str, str, bool]]) -> str:
    """sections: (field, heading, as_bullets). Empty fields are omitted.
    Bullets for sentence-like items, one '; '-joined line for short names."""
    lines = []
    for key, heading, bullets in sections:
        items = fields.get(key) or []
        if not items:
            continue
        if bullets:
            lines.append(f"{heading}:")
            lines.extend(f"- {x}" for x in items)
        else:
            lines.append(f"{heading}: {'; '.join(items)}")
    return "\n".join(lines)
