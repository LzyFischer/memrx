# 2a 实验只用到 MemoryBuilder（复用其 summary 抽取 prompt）。
#
# 惰性导入（PEP 562）：`from core import MemoryBuilder` 行为不变，但
# `from core.probe import ...` / `from core.pl_router import ...` 不再顺带把
# memory_builder -> database -> lancedb 整条链拉进来。router 是纯 numpy 的。
__all__ = ["MemoryBuilder"]


def __getattr__(name):
    if name == "MemoryBuilder":
        from .memory_builder import MemoryBuilder
        return MemoryBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
