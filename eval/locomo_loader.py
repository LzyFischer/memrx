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

ANSWER_PROMPT = {
    1: ("Based on the above conversations, write short answers for each of the "
        "following questions in a few words. Write the answers in the form of a "
        "short phrase for each question. Answer with exact words from the "
        "conversations whenever possible."),
    2: ("Based on the above conversations, write short answers for each of the "
        "following questions in a few words. Write the answers in the form of a "
        "short phrase for each question. Answer with exact words from the "
        "conversations whenever possible."),
    3: ("Based on the above conversations, write short answers for each of the "
        "following questions using DATE of CONVERSATION for reference. Write the "
        "answer in the form of a short phrase. The answers need to be "
        "grounded in the dates of the conversations. Answer with exact words "
        "from the conversations whenever possible."),
    4: ("Based on the above conversations, answer the following question. Use "
        "DATE of CONVERSATION to answer with an approximate date. Answer with "
        "exact words from the conversation whenever possible."),
    5: ("Based on the above conversations, answer the following question. "
        "Write the answer as \"Not mentioned in the conversation\" if the "
        "information is not present in the conversation. Otherwise write a "
        "short phrase as the answer."),
}
DEFAULT_ANSWER_PROMPT = ANSWER_PROMPT[1]
CATEGORY_NAMES = {1: "single_hop", 2: "multi_hop", 3: "temporal", 4: "open_domain", 5: "adversarial"}


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


def build_qa_prompt(context: str, question: str, category: int) -> str:
    instr = ANSWER_PROMPT.get(category, DEFAULT_ANSWER_PROMPT)
    return f"{context}\n\n{instr}\n\nQuestion: {question}\nAnswer:"


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
