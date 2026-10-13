"""FastAPI 服务：索引构建、问答、流式问答。

启动（在项目根目录）：
    uvicorn app.main:app --reload --port 8000
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import require_api_key, settings
from .embeddings import EmbeddingClient, create_embedding_client
from .generator import Generator
from .pipeline import build_index
from .retriever import HybridRetriever, RetrievedChunk, rerank
from .schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse, SourceInfo
from .vectorstore import QdrantStore, VectorStore, create_vector_store

state: dict = {}


def _make_retriever(store: VectorStore | QdrantStore, embed_client) -> HybridRetriever:
    chunks = store.all_chunks()
    if not chunks:
        raise HTTPException(status_code=400, detail="索引为空，请先调用 POST /ingest 构建索引")
    return HybridRetriever(store, embed_client, chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_api_key()
    store = create_vector_store(settings)
    embed_client = create_embedding_client(settings)
    generator = Generator(settings.llm_base_url, settings.llm_api_key, settings.chat_model)
    state.update(store=store, embed_client=embed_client, generator=generator)
    state["retriever"] = _make_retriever(store, embed_client) if store.count() else None
    yield


app = FastAPI(title="RAG Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 演示项目放开跨域，生产环境请收紧
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_retriever() -> HybridRetriever:
    retriever = state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=400, detail="索引为空，请先调用 POST /ingest 构建索引")
    return retriever


def _retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
    retriever = _get_retriever()
    contexts = retriever.retrieve(query, top_k=top_k)
    # 可选精排：.env 里配了 LLM_RERANK_MODEL 才会生效
    return rerank(
        query, contexts,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.rerank_model,
        top_n=top_k,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", indexed_chunks=state["store"].count())


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """（重新）构建索引：读取 data/docs 下的文档 → 切分 → embedding → 入库。"""
    result = build_index("data/docs")
    state["retriever"] = _make_retriever(state["store"], state["embed_client"])
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """一问一答：检索 → 精排 → 生成，一次返回。"""
    contexts = _retrieve(req.question, req.top_k)
    answer = state["generator"].generate(req.question, contexts)
    return QueryResponse(
        answer=answer,
        sources=[SourceInfo(source=c.source, score=c.score, text=c.text) for c in contexts],
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """流式问答：SSE 逐 token 返回，先推 sources，再推 token，最后 [DONE]。"""
    contexts = _retrieve(req.question, req.top_k)
    generator = state["generator"]

    def event_gen():
        yield f"data: {json.dumps({'sources': [{'source': c.source, 'score': c.score} for c in contexts]}, ensure_ascii=False)}\n\n"
        for token in generator.generate_stream(req.question, contexts):
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
