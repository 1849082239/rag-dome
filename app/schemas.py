"""API 数据模型。"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回几个片段")


class SourceInfo(BaseModel):
    source: str
    score: float
    text: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class IngestResponse(BaseModel):
    files: int
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
