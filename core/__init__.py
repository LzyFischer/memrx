# 2a 实验只用到 MemoryBuilder（复用其 summary 抽取 prompt）。
# 原库的 HybridRetriever / AnswerGenerator 在这次 2a 实验里没有被用到，
# 打包时一并移除了，避免不必要的依赖耦合。
from .memory_builder import MemoryBuilder

__all__ = ["MemoryBuilder"]
