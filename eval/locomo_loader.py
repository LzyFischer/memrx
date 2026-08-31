"""
LoCoMo data loading + scoring for the 2a ablation.

Reuses (near-verbatim) the normalize_answer / f1_score / ANSWER_PROMPT
logic already validated in run_locomo.py at the repo root, so numbers stay
comparable with the existing full-context baseline run.

Get the data file from:
  https://github.com/snap-research/locomo/blob/main/data/locomo10.json
  (CC BY-NC 4.0) and place it at data/locomo10.json (or pass --data).
"""
import collections
import json
import re
import string
from typing import Any, Dict, List, Tuple

from models.memory_entry import Dialogue

# Single, uniform answer-generation prompt for every category — no more
# per-category rule text. The JSON {reasoning, answer} shape is kept (see
# run_2a_locomo.py::run_one_qa, which parses out "answer" for scoring) since
# forcing a "reasoning" field first is what stops the model from just
# echoing a context line's "[N]" label as if that were the answer.
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def load_locomo(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_to_dialogues(sample: Dict[str, Any]) -> List[Dialogue]:
    """Flatten a LoCoMo sample's sessions into a chronological Dialogue list."""
    conv = sample["conversation"]
    session_keys = sorted(
        [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)],
        key=lambda x: int(x.split("_")[1]),
    )
    dialogues: List[Dialogue] = []
    dia_counter = 0
    for sk in session_keys:
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


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))
