"""向量库封装：Chroma（本地文件，开箱即用）或 Qdrant（独立服务），接口统一。

上层代码（pipeline / API / 评测）只认这 5 个方法，换向量库不用改业务代码：
reset / add / search / all_chunks / count
"""
from __future__ import annotations

import uuid

import chromadb

from .ingest import Chunk


class VectorStore:
    """Chroma 版：数据落盘到本地文件，无需额外服务。"""

    def __init__(self, persist_dir: str, collection_name: str):
        # PersistentClient 会把数据落盘，重启服务索引不丢失
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度，最常用
        )

    def reset(self) -> None:
        """清空重建（重新 ingest 时用）。"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.add(
            ids=[f"{c.source}#{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """按向量相似度取 top_k。Chroma 返回的是 cosine 距离，转成相似度分数。"""
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({
                "text": doc,
                "source": meta["source"],
                "index": meta["index"],
                "score": round(1 - dist, 4),
            })
        return hits

    def all_chunks(self) -> list[Chunk]:
        """读回全部 chunk（混合检索里的 BM25 需要全量文本）。"""
        data = self.collection.get(include=["documents", "metadatas"])
        chunks = [
            Chunk(text=doc, source=meta["source"], index=meta["index"])
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        return sorted(chunks, key=lambda c: (c.source, c.index))

    def count(self) -> int:
        return self.collection.count()


class QdrantStore:
    """Qdrant 版：需要先启动 Qdrant 服务（Docker 一行命令，见 README）。

    接口与 VectorStore 完全一致，上层无感知切换。
    注意：collection 创建时就固定了向量维度，换 embedding 模型必须重建索引。
    """

    def __init__(self, host: str, port: int, collection_name: str, vector_size: int):
        # qdrant_client 做延迟导入：只用 Chroma 的用户不用装这个依赖
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def reset(self) -> None:
        """清空重建（重新 ingest 时用）。"""
        from qdrant_client.models import Distance, VectorParams
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct
        if embeddings and len(embeddings[0]) != self.vector_size:
            raise ValueError(
                f"向量维度不匹配：embedding 输出 {len(embeddings[0])} 维，"
                f"但 Qdrant collection 是 {self.vector_size} 维。"
                f"请检查 EMBEDDING_DIM 配置是否与模型一致，或重建索引。"
            )
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={"text": c.text, "source": c.source, "index": c.index},
            )
            for c, vec in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        # qdrant-client 1.10+ 统一用 query_points 做向量查询（旧 search 已移除）
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": h.payload["text"],
                "source": h.payload["source"],
                "index": h.payload["index"],
                "score": round(float(h.score), 4),
            }
            for h in resp.points
        ]

    def all_chunks(self) -> list[Chunk]:
        """读回全部 chunk（混合检索里的 BM25 需要全量文本）。"""
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                chunks.append(Chunk(
                    text=p.payload["text"],
                    source=p.payload["source"],
                    index=p.payload["index"],
                ))
            if offset is None:
                break
        return sorted(chunks, key=lambda c: (c.source, c.index))

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count


def create_vector_store(settings):
    """工厂函数：按 VECTOR_STORE 配置返回对应实现，其余代码不用改。"""
    if settings.vector_store == "qdrant":
        return QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name,
            vector_size=settings.embedding_dim,
        )
    return VectorStore(settings.chroma_dir, settings.collection_name)
