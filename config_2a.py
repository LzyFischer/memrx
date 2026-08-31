"""
2a 实验配置：Does heterogeneous treatment effect exist?

设计：三个维度，每次只变一个维度，另外两个维度固定为 "none"（不使用该维度处理）。
baseline（三个维度都是 none）只需要跑一次，三组共享做对照组。

条件总数 = 1 baseline + 2 summary + 2 augmentation + 2 graph = 7 个条件。
（summary 原本有 session_level/fine_grained/hierarchical 三个变体，hierarchical
已移除；augmentation 原本有 temporal/keywords/causal 三个变体：causal 已移除
——causal 关系天然是"指向另一个 chunk 的指针"，单 chunk 内的因果文本增强容易
退化成对原文的空泛复述，temporal 也已移除——它的"query 时间意图推断 + 独立
日期匹配打分 + RRF 融合"机制相对 keywords/note 这两个变体来说不好三言两语
讲清楚，见 core/augmentation_builder.py。现在 augmentation 的两个变体是
keywords（LLM 抽关键词，喂给 BM25，和语义检索 RRF 融合）和 note（LLM 抽
关键词+上下文，直接拼进原文做纯语义检索），两者构建期都是每个 chunk 恰好
一次 LLM 调用，跟 summary 的单 window 一次调用同量级，见
core/augmentation_builder.py。graph 原本有 semantic/entity/causal 三个变体：
causal 已移除——见 core/graph_builder.py 顶部说明。）
"""
from dataclasses import dataclass
from typing import Literal

SummaryVariant = Literal["none", "session_level", "fine_grained"]
AugmentationVariant = Literal["none", "keywords", "note"]
GraphVariant = Literal["none", "semantic", "entity"]

@dataclass
class Condition:
    condition_id: str
    dimension: str          # "baseline" | "summary" | "augmentation" | "graph"
    summary: SummaryVariant = "none"
    augmentation: AugmentationVariant = "none"
    graph: GraphVariant = "none"

    def as_dict(self):
        return {
            "condition_id": self.condition_id,
            "dimension": self.dimension,
            "summary": self.summary,
            "augmentation": self.augmentation,
            "graph": self.graph,
        }


def build_condition_matrix() -> list:
    conditions = [Condition(condition_id="baseline", dimension="baseline")]

    for v in ["session_level", "fine_grained"]:
        conditions.append(Condition(condition_id=f"summary__{v}", dimension="summary", summary=v))

    for v in ["keywords", "note"]:
        conditions.append(Condition(condition_id=f"augmentation__{v}", dimension="augmentation", augmentation=v))

    for v in ["semantic", "entity"]:
        conditions.append(Condition(condition_id=f"graph__{v}", dimension="graph", graph=v))

    return conditions


# ---------------- 实验运行参数 ----------------
RETRIEVAL_TOP_K = 10
WINDOW_SIZE = 5      # raw chunk / summary window：LoCoMo 单条对话较长，5轮/窗比默认20更适合小样本 ablation
OVERLAP_SIZE = 0

DATA_PATH = "data/locomo10.json"
RESULTS_DIR = "results"

# LoCoMo adversarial (category 5) 没有标准 gold answer 算 F1，但保留在结果里，
# 用 "是否正确拒答" 来算 accuracy，而不是 F1（和主流评测惯例一致）。
EXCLUDE_ADVERSARIAL_FROM_F1 = False
