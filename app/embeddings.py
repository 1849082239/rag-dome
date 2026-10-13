"""Embedding 客户端：把文本变成向量（语义的数值表示）。

两种模式，用 EMBEDDING_PROVIDER 切换：
- api：调 OpenAI 兼容接口的 embedding 服务（默认，速度快、不吃本地内存）
- local：sentence-transformers 本地推理（免费、不依赖网络，首次用时下载模型）
"""
from __future__ import annotations

from openai import OpenAI


class EmbeddingClient:
    """API 版：走 OpenAI 兼容接口。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量转向量。分批是为了避开服务商的单次条数/长度限制。"""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)  # 保证顺序不错乱
            vectors.extend([d.embedding for d in ordered])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbeddingClient:
    """本地版：sentence-transformers，懒加载，调用 embed 时才下载/加载模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embedding] 正在加载本地模型：{self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            print(f"[embedding] 模型加载完成，输出维度：{self._model.get_sentence_embedding_dimension()}")
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        model = self._load()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
            vectors.extend([v.tolist() for v in emb])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def create_embedding_client(settings):
    """工厂函数：按配置返回对应客户端，其余代码不用改。"""
    if settings.embedding_provider == "local":
        return LocalEmbeddingClient(settings.embed_model)
    return EmbeddingClient(settings.embed_base_url, settings.embed_api_key, settings.embed_model)
