# LangGraph 完整 RAG 链路：分类 → Qdrant 检索 → Groq 生成
# 对应 Dify 里那张 Chatflow 图
import requests
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient


# ================= 配置区：按你实际情况改 =================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_test"  # ← 去 http://localhost:6333/dashboard 看你的 collection 名，填这里

EMBED_URL = "http://jacob.hidencloud.com:24576/v1/embeddings"
EMBED_KEY = "ollama"
EMBED_MODEL = "bge-m3:latest"
TOP_K = 3
# ===========================================================


class State(TypedDict):
    question: str
    category: str
    contexts: list
    answer: str


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
qdrant = QdrantClient(url=QDRANT_URL)


def embed(text: str) -> list:
    """调 HidenCloud 的 bge-m3 把文本转成向量"""
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {EMBED_KEY}"},
        json={"model": EMBED_MODEL, "input": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def payload_text(payload: dict) -> str:
    """兼容不同的 payload 字段名"""
    for k in ("text", "content", "page_content", "document"):
        if payload.get(k):
            return payload[k]
    return str(payload)


# ---- 节点1：分类器（和 classifier.py 一样） ----
def classify_node(state: State) -> dict:
    prompt = (
        "把用户问题分到下面的一类，只返回类别名，不要解释：\n"
        "- price：价格、行情相关\n"
        "- standard：分类标准、质检相关\n"
        "- other：其他\n"
        f"问题：{state['question']}"
    )
    category = llm.invoke(prompt).content.strip().lower()
    if category not in ("price", "standard", "other"):
        category = "other"
    print(f"[分类] {category}")
    return {"category": category}


def route(state: State) -> Literal["price", "standard", "other"]:
    return state["category"]  # type: ignore


# ---- 节点2：检索，对应 Dify 的 Knowledge Retrieval ----
def retrieve_node(state: State) -> dict:
    vec = embed(state["question"])
    try:
        # qdrant-client 新版用 query_points，老版用 search，做个兼容
        if hasattr(qdrant, "query_points"):
            hits = qdrant.query_points(
                collection_name=COLLECTION_NAME, query=vec, limit=TOP_K
            ).points
        else:
            hits = qdrant.search(
                collection_name=COLLECTION_NAME, query_vector=vec, limit=TOP_K
            )
    except Exception as e:
        # 常见坑：collection 里是 MiniLM 的 384 维向量，和 bge-m3 的 1024 维对不上
        print(f"[检索失败] {e}")
        print("可能原因：collection 不存在，或向量维度不匹配（旧数据是 384 维，bge-m3 是 1024 维）")
        return {"contexts": []}
    contexts = [payload_text(h.payload) for h in hits]
    print(f"[检索] 召回 {len(contexts)} 段，最高分 {hits[0].score:.3f}" if hits else "[检索] 无结果")
    return {"contexts": contexts}


# ---- 节点3：生成，对应 Dify 的 LLM ----
def generate_node(state: State) -> dict:
    ctx = "\n\n".join(state["contexts"]) if state["contexts"] else "（未检索到相关资料）"
    prompt = f"根据以下资料回答问题，只基于资料回答，不要编造：\n{ctx}\n\n问题：{state['question']}"
    resp = llm.invoke(prompt)
    return {"answer": resp.content}


def other_node(state: State) -> dict:
    return {"answer": "抱歉，这个问题我暂时帮不上忙。"}


graph = StateGraph(State)
graph.add_node("classify", classify_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_node("other", other_node)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify", route, {"price": "retrieve", "standard": "retrieve", "other": "other"}
)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
graph.add_edge("other", END)
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"question": "破碎料多少钱一吨？"})
    print("\n最终回答：", result["answer"])
