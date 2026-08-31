"""
Minimal Okapi BM25 — no external dependency (rank-bm25 is already in
requirements.txt for users who want the battle-tested version; this
self-contained implementation avoids adding a hard dependency just for a
few hundred short documents per LoCoMo conversation, where performance
doesn't matter).

Used as the sparse-retrieval channel for augmentation="keywords" — see
core/retrieval2a.py. Mem0's production system runs BM25 as an independent
signal fused with semantic similarity (not folded into the embedding),
because BM25 is exactly strong where dense embeddings are weak: exact
names, technical terms/acronyms, dates, numbers, rare/discriminative
words. See docs.mem0.ai/core-concepts/memory-evaluation.
"""
import math
import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SimpleBM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if corpus_tokens else 0.0
        self.doc_freqs: List[Counter] = []
        df: Counter = Counter()
        for doc in corpus_tokens:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            for term in freqs:
                df[term] += 1
        n = len(corpus_tokens)
        self.idf: Dict[str, float] = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * len(self.corpus_tokens)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores
