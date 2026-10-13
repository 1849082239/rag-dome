"""混合检索：向量检索 + BM25 关键词检索，加权融合。

为什么需要混合？
纯向量检索理解语义，但对精确词（人名、术语、数字、型号）不敏感；
BM25 是经典关键词算法，正好补上这一块。两者融合是生产级 RAG 的常见做法。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import jieba

    def _tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))
except ImportError:  # 没装 jieba 时退化为按字切分，照样能跑
    def _tokenize(text: str) -> list[str]:
        return list(text)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class BM25:
    """经典 BM25 算法（简化版）：词频越高、文档越短、词越稀有，得分越高。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokenized = [_tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.n = len(self.tokenized)
        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.n, 1)
        df: dict[str, int] = {}
        for tokens in self.tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        # IDF：越稀有的词权重越高
        self.idf = {t: math.log((self.n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q_tokens = _tokenize(query)
        scores = []
        for tokens in self.tokenized:
            s, dl = 0.0, len(tokens)
            for t in q_tokens:
                if t not in self.idf:
                    continue
                f = tokens.count(t)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[t] * f * (self.k1 + 1) / denom
            scores.append(s)
        return scores


def _minmax_norm(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    """向量召回（多召回一些）+ BM25 全量打分 → 归一化加权融合 → 取 top_k。"""

    def __init__(self, store, embed_client, chunks, w_vector: float = 0.7, w_bm25: float = 0.3):
        self.store = store
        self.embed_client = embed_client
        self.chunks = chunks
        self.w_vector, self.w_bm25 = w_vector, w_bm25
        self.bm25 = BM25([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q_vec = self.embed_client.embed_query(query)
        vec_hits = self.store.search(q_vec, top_k=max(top_k * 4, 20))
        vec_map = {(h["source"], h["index"]): h["score"] for h in vec_hits}
        bm25_raw = self.bm25.score(query)

        idxs, vec_scores, bm25_scores = [], [], []
        for i, c in enumerate(self.chunks):
            key = (c.source, c.index)
            if key not in vec_map and bm25_raw[i] == 0:
                continue  # 两边都没命中，直接跳过
            idxs.append(i)
            vec_scores.append(vec_map.get(key, 0.0))
            bm25_scores.append(bm25_raw[i])

        vec_n = _minmax_norm(vec_scores)
        bm25_n = _minmax_norm(bm25_scores)
        fused = [self.w_vector * v + self.w_bm25 * b for v, b in zip(vec_n, bm25_n)]
        order = sorted(range(len(idxs)), key=lambda j: fused[j], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                text=self.chunks[idxs[j]].text,
                source=self.chunks[idxs[j]].source,
                score=round(fused[j], 4),
            )
            for j in order
        ]


def rerank(
    query: str,
    docs: list[RetrievedChunk],
    *,
    base_url: str,
    api_key: str,
    model: str,
    top_n: int,
) -> list[RetrievedChunk]:
    """精排（可选）：用 Rerank 模型对初排结果重新打分。

    Rerank 模型是 cross-encoder，query 和文档一起编码，
    精度高于向量检索，但慢且贵，所以只对 top_k 做精排。
    model 为空时跳过，直接返回原序。
    """
    if not model:
        return docs[:top_n]
    import requests

    resp = requests.post(
        f"{base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "query": query,
              "documents": [d.text for d in docs], "top_n": top_n},
        timeout=60,
    )
    resp.raise_for_status()
    results = sorted(resp.json()["results"], key=lambda r: r["relevance_score"], reverse=True)
    return [
        RetrievedChunk(text=docs[r["index"]].text,
                       source=docs[r["index"]].source,
                       score=round(r["relevance_score"], 4))
        for r in results
    ]
