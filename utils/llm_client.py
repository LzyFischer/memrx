"""
LLM client: thin wrapper around the OpenAI-compatible API.

Handles retries, optional streaming, Qwen thinking mode, and robust JSON
extraction from raw LLM responses.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import config


# ---------------------------------------------------------------------------
# Qwen3 thinking-mode handling.
#
# Qwen3 (served locally via vLLM, e.g. Qwen/Qwen3-0.6B) has thinking mode ON
# by default. Two things can happen depending on how the vLLM server was
# started:
#
#   1. No --reasoning-parser flag: the chat template primes the assistant
#      turn with "<think>\n", and the model's generated text (returned in
#      message.content) is the thinking trace followed by "</think>" and
#      then the real answer, e.g.:
#          "Let me check the dates...\n</think>\n\nThe answer is 12 May 2023."
#      Note the OPENING <think> tag is often NOT echoed back (it was part of
#      the priming, not generated), only the CLOSING </think> is visible.
#
#   2. --reasoning-parser qwen3 (or similar) IS set: vLLM splits the two
#      itself, so message.content is already the clean final answer and the
#      thinking trace lives in message.reasoning_content instead.
#
# strip_thinking() below handles case 1 (and is a harmless no-op for case 2,
# since there's nothing to strip if content is already clean). We never read
# reasoning_content into the returned string, so case 2 is handled by
# omission.
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_ONLY_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> trace from a raw completion string."""
    if not text:
        return text

    cleaned = _THINK_BLOCK_RE.sub("", text)
    if cleaned == text and "</think>" in text:
        # Only the closing tag is present (opening tag was in the chat
        # template priming, not in the generated text) -> strip everything
        # up to and including the first </think>.
        cleaned = _THINK_CLOSE_ONLY_RE.sub("", text, count=1)
    return cleaned.strip()


def coerce_json_list(data: Any, context: str = "") -> list:
    """Small local models (e.g. Qwen3-0.6B) frequently ignore "return a JSON
    ARRAY" and wrap the array in an object instead — {"keywords": [...]},
    {"entries": [...]}, or even a bare single item with no array at all.
    Used by every extraction call-site (memory_builder / augmentation_builder
    / graph_builder) so the coercion logic lives in exactly one place.

    Returns [] (never raises) if nothing list-shaped can be salvaged, so
    callers can treat "extraction found nothing" and "extraction was
    malformed" the same way — both just mean "no bonus info this time",
    not a fatal error worth 3 retries."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        list_fields = [v for v in data.values() if isinstance(v, list)]
        if list_fields:
            return list_fields[0]
        # bare single object with no array anywhere -> treat it as a
        # one-element list (covers single_entry_mode's dict-not-array case)
        if data:
            return [data]
    if context:
        print(f"  [warn] {context}: could not coerce {type(data)} into a list, treating as empty")
    return []


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url if base_url is not None else getattr(config, "OPENAI_BASE_URL", None)
        self.enable_thinking = enable_thinking if enable_thinking is not None else getattr(config, "ENABLE_THINKING", False)
        self.use_streaming = use_streaming if use_streaming is not None else getattr(config, "USE_STREAMING", False)

        if self.base_url:
            print(f"LLM base URL: {self.base_url}")
        if self.enable_thinking:
            print("Deep thinking mode enabled")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        strip_think: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Returns the FINAL ANSWER only — thinking trace is stripped by default.

        Note on max_tokens with Qwen3 thinking mode: the thinking trace itself
        consumes generation budget BEFORE the answer, so if max_tokens is set
        too low the model can get cut off mid-think and never emit an answer
        (content ends up empty after stripping). Leave generous headroom
        (e.g. >= 512) for QA/extraction calls when thinking mode is on;
        config.ENABLE_THINKING=False callers can use tighter budgets.
        """
        raw, _ = self.chat_completion_with_thinking(
            messages, temperature, response_format, max_retries, max_tokens
        )
        return strip_thinking(raw) if strip_think else raw

    def chat_completion_with_thinking(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Returns (raw_content, reasoning_content). raw_content may still
        contain an inline <think>...</think> block (case 1 in the module
        docstring) — call strip_thinking() on it, or just use
        chat_completion() which does this for you."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Qwen official dashscope API: requires explicit enable_thinking param.
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        if is_qwen_api:
            use_thinking = self.use_streaming and self.enable_thinking and not response_format
            kwargs["extra_body"] = {"enable_thinking": use_thinking}
        elif not self.enable_thinking:
            # Local vLLM serving Qwen3: thinking is ON by default via the chat
            # template. If the caller wants it OFF (e.g. cheap metadata
            # extraction calls in augmentation/graph builders where a long
            # reasoning trace just burns latency for a 0.6B model), ask vLLM
            # to skip it at the template level instead of paying for the
            # tokens and stripping them after the fact. Requires vLLM's Qwen3
            # chat template to support chat_template_kwargs (true for recent
            # vLLM versions); if your server ignores this, strip_thinking()
            # in chat_completion() still cleans the output either way.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._collect_stream(**kwargs)
                else:
                    resp = self.client.chat.completions.create(**kwargs)
                    msg = resp.choices[0].message
                    content = msg.content or ""
                    # vLLM with --reasoning-parser puts the trace here instead
                    # of inline in content (case 2 in the module docstring).
                    reasoning = getattr(msg, "reasoning_content", "") or ""
                    return content, reasoning
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"LLM attempt {attempt + 1}/{max_retries} failed: {e}. Retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"LLM failed after {max_retries} attempts: {e}")

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed")

    def _collect_stream(self, **kwargs: Any) -> Tuple[str, str]:
        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        for chunk in self.client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content_chunks.append(delta.content)
            if getattr(delta, "reasoning_content", None):
                reasoning_chunks.append(delta.reasoning_content)
        return "".join(content_chunks), "".join(reasoning_chunks)

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> Any:
        """Robustly extract a JSON object/array from raw LLM output."""
        if not text or not text.strip():
            raise ValueError("Empty LLM response")

        text = strip_thinking(text.strip())  # defense-in-depth: harmless no-op if already clean

        # Strip common preamble phrases
        for prefix in ["Here's the JSON:", "JSON:", "Result:", "Output:", "Answer:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. ```json ... ``` block
        m = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(self._clean_json(m.group(1).strip()))
            except json.JSONDecodeError:
                pass

        # 3. Generic ``` ... ``` block
        m = re.search(r"```\w*\s*(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(self._clean_json(m.group(1).strip()))
            except json.JSONDecodeError:
                pass

        # 4. Balanced brace/bracket scan
        for start_char in ["{", "["]:
            result = self._balanced_extract(text, start_char)
            if result is not None:
                return result

        raise ValueError(f"Could not parse JSON from response (first 200 chars): {text[:200]}")

    @staticmethod
    def _clean_json(s: str) -> str:
        s = re.sub(r",(\s*[}\]])", r"\1", s)          # trailing commas
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)  # // comments
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # /* */ comments
        return s.strip()

    def _balanced_extract(self, text: str, start_char: str) -> Any:
        end_char = "}" if start_char == "{" else "]"
        idx = text.find(start_char)
        if idx == -1:
            return None
        depth, in_str, esc = 0, False, False
        for i in range(idx, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[idx: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(self._clean_json(candidate))
                        except json.JSONDecodeError:
                            return None
        return None
