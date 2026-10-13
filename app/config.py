"""配置管理：从环境变量（支持 .env 文件）读取所有可调参数。"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # OpenAI 兼容接口：硅基流动 / DeepSeek / OpenAI / 本地 Ollama 都能用
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("LLM_CHAT_MODEL", "deepseek-ai/DeepSeek-V3"))
    # Embedding 走独立配置：有些服务商（如 Groq）只提供对话模型、没有 embedding 接口，
    # 这时可以让对话和 embedding 分属不同服务商。不填则默认沿用上面的 LLM 配置。
    embed_base_url: str = field(default_factory=lambda: os.getenv("EMBED_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    embed_api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY") or os.getenv("LLM_API_KEY", ""))
    # Embedding 提供方：api（调接口）或 local（本地 sentence-transformers）
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "api"))
    embed_model: str = field(default_factory=lambda: os.getenv("LLM_EMBED_MODEL", "BAAI/bge-m3"))
    # 精排模型，为空则跳过重排（硅基流动可用 BAAI/bge-reranker-v2-m3）
    rerank_model: str = field(default_factory=lambda: os.getenv("LLM_RERANK_MODEL", ""))

    chroma_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "chroma_db"))
    collection_name: str = "rag_demo"

    # 向量库：chroma（本地文件，开箱即用）或 qdrant（需自行启动 Qdrant 服务）
    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "chroma"))
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    # embedding 输出维度，必须与模型一致（all-MiniLM-L6-v2=384，BAAI/bge-m3=1024）
    # Qdrant 建 collection 时用它定维度，Chroma 则不需要
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "80")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))


settings = Settings()


def require_api_key() -> None:
    if not settings.llm_api_key:
        raise RuntimeError("未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")
