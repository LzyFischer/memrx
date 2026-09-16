"""
LoCoMo data loading, QA prompt, and F1/EM scoring.

Get the data file from:
  https://github.com/snap-research/locomo/blob/main/data/locomo10.json
  (CC BY-NC 4.0) and place it at data/locomo10.json (or pass --data).
"""
import collections
import json
import math
import re
import string
from typing import Any, Dict, List, Tuple

import numpy as np

from core.entry import Dialogue

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def load_locomo(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_session_keys(sample: Dict[str, Any]) -> List[str]:
    conv = sample["conversation"]
    return sorted(
        [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)],
        key=lambda x: int(x.split("_")[1]),
    )


def build_dia_id_index(sample: Dict[str, Any]) -> Dict[str, int]:
    """Map each turn's original LoCoMo ``dia_id`` string (e.g. ``"D1:3"``) to
    the flat integer ``dialogue_id`` used by ``sample_to_dialogues`` below —
    same session order, same turn order, so the two stay in sync as long as
    both keep iterating ``conv[sk]`` in file order.

    Needed to resolve QA ``"evidence"`` lists (also given as ``"D1:3"``-style
    strings) against the flat ids stamped onto retrieved MemoryEntry objects
    via ``metadata["dia_id_start"/"dia_id_end"]`` (core/chunking.py,
    core/summary.py), for the retrieval-recall analysis.
    """
    conv = sample["conversation"]
    index: Dict[str, int] = {}
    dia_counter = 0
    for sk in _sorted_session_keys(sample):
        for turn in conv[sk]:
            raw_id = turn.get("dia_id")
            if raw_id:
                index[raw_id] = dia_counter
            dia_counter += 1
    return index


def evidence_flat_ids(qa: Dict[str, Any], dia_id_index: Dict[str, int]) -> List[int]:
    """Resolve a QA's gold ``evidence`` dia_id strings to flat integer ids.

    Returns ``[]`` when the QA has no (or an unresolvable) evidence list —
    notably LoCoMo's category-5 (adversarial) questions, which aren't
    annotated with evidence turns. Callers should treat an empty list as
    "retrieval recall not defined for this question", not as "recall = 0".
    """
    ids = []
    for raw in qa.get("evidence", []) or []:
        idx = dia_id_index.get(raw)
        if idx is not None:
            ids.append(idx)
    return ids


def sample_to_dialogues(sample: Dict[str, Any]) -> List[Dialogue]:
    """Flatten a LoCoMo sample's sessions into a chronological Dialogue list."""
    conv = sample["conversation"]
    dialogues: List[Dialogue] = []
    dia_counter = 0
    for sk in _sorted_session_keys(sample):
        date_str = conv.get(f"{sk}_date_time", "")
        for turn in conv[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            caption = turn.get("blip_caption") or turn.get("caption")
            if caption:
                text = f"{text} [shares a photo of {caption}]"
            dialogues.append(Dialogue(
                dialogue_id=dia_counter, speaker=speaker, content=text, timestamp=date_str,
            ))
            dia_counter += 1
    return dialogues


def build_qa_prompt(context: str, question: str, category: int = 1) -> str:
    # Same prompt for every category. Asking for "reasoning" before "answer"
    # stops small models from echoing a context line's "[N]" label as the answer.
    return f"""
Answer the user's question based on the provided context.

User Question: {question}

Relevant Context:
{context}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction: str, ground_truth: str) -> float:
    """Sentence BLEU-1: clipped unigram precision x brevity penalty, on the same
    normalised tokens as f1_score (lowercase, no punctuation or articles)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    matches = sum((collections.Counter(pred_tokens) & collections.Counter(gt_tokens)).values())
    precision = matches / len(pred_tokens)
    bp = 1.0 if len(pred_tokens) > len(gt_tokens) else math.exp(1 - len(gt_tokens) / len(pred_tokens))
    return bp * precision


QUERY_TYPES = [1, 2, 3, 4]   # the four reported LoCoMo types; 5 (adversarial) is left out


def print_type_table(title: str, rows) -> None:
    """rows: [(name, categories, {"f1": scores, "bleu1": scores})], one score per question.
    Prints mean F1 and BLEU-1 (x100) for each of the four query types and over types 1-4.
    The n= line counts the first row's questions."""
    name_w = max(28, max(len(r[0]) for r in rows) + 2)
    headers = [CATEGORY_NAMES[c] for c in QUERY_TYPES] + ["all 1-4"]

    def masks(cats):
        cats = np.asarray(cats)
        return [cats == c for c in QUERY_TYPES] + [np.isin(cats, QUERY_TYPES)]

    first = np.asarray(rows[0][1])
    print(f"\n{title}")
    print(" " * name_w + "".join(f"{h:>16s}" for h in headers))
    print(" " * name_w + "".join(f"{f'(n={int(m.sum())})':>16s}" for m in masks(first)))
    print(" " * name_w + "".join(f"{'F1':>8s}{'B1':>8s}" for _ in headers))
    for name, cats, vals in rows:
        cells = []
        for m in masks(cats):
            for key in ("f1", "bleu1"):
                x = np.asarray(vals[key], dtype=float)[m]
                x = x[np.isfinite(x)]
                cells.append(f"{100 * x.mean():>8.1f}" if len(x) else f"{'-':>8s}")
        print(f"{name:<{name_w}s}" + "".join(cells))
    n5 = int((first == 5).sum())
    if n5:
        print(f"  ({n5} adversarial questions not in this table)")


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def split_locomo(
    data: List[Dict[str, Any]], n_train: int = 2, n_val: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split LoCoMo conversations by their order in the source file.

    train = data[:n_train]
    val   = data[n_train : n_train+n_val]
    test  = data[n_train+n_val :]

    Positional, not random — LoCoMo10 only has 10 conversations, so a
    deterministic split keeps every run reproducible without needing to
    persist a seed or an index list.
    """
    return {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }
