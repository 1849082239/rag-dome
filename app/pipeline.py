"""索引构建流水线：加载文档 → 切分 → embedding → 存向量库。

被 API 的 /ingest 和 scripts/build_index.py 共用，
保证两种建索引的方式逻辑完全一致。
"""
from __future__ import annotations

from .config import require_api_key, settings
from .embeddings import create_embedding_client
from .ingest import chunk_documents, load_documents
from .vectorstore import create_vector_store


def build_index(doc_dir: str = "data/docs") -> dict:
    require_api_key()
    docs = load_documents(doc_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

    embed_client = create_embedding_client(settings)
    vectors = embed_client.embed([c.text for c in chunks])

    store = create_vector_store(settings)
    store.reset()
    store.add(chunks, vectors)

    return {"files": len(docs), "chunks": len(chunks),
            "message": f"索引构建完成：{len(docs)} 个文件 → {len(chunks)} 个 chunk"}
