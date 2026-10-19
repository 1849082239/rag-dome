# LangGraph 条件边：复刻 Dify 的 Question Classifier 节点
import os
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq


class State(TypedDict):
    question: str
    category: str
    answer: str


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ---- 节点1：分类器，对应 Dify 的 Question Classifier ----
def classify_node(state: State) -> dict:
    prompt = (
        "把用户问题分到下面的一类，只返回类别名，不要解释：\n"
        "- price：价格、行情相关\n"
        "- standard：分类标准、质检相关\n"
        "- other：其他\n"
        f"问题：{state['question']}"
    )
    resp = llm.invoke(prompt)
    category = resp.content.strip().lower()
    if category not in ("price", "standard", "other"):
        category = "other"  # 兜底：模型瞎答时默认进 other
    print(f"[分类结果] {category}")
    return {"category": category}


# ---- 路由函数：条件边看它的返回值决定走哪条路 ----
def route(state: State) -> Literal["price", "standard", "other"]:
    return state["category"]  # type: ignore


def price_node(state: State) -> dict:
    return {"answer": f"[价格类] {state['question']} → 下一步：去知识库查价格文档"}


def standard_node(state: State) -> dict:
    return {"answer": f"[标准类] {state['question']} → 下一步：去知识库查分类标准文档"}


def other_node(state: State) -> dict:
    return {"answer": "抱歉，这个问题我暂时帮不上忙。"}


graph = StateGraph(State)
graph.add_node("classify", classify_node)
graph.add_node("price", price_node)
graph.add_node("standard", standard_node)
graph.add_node("other", other_node)

graph.add_edge(START, "classify")
# 条件边：classify 之后看 route() 的返回值走不同分支，对应 Dify 分类器的三条输出线
graph.add_conditional_edges(
    "classify",
    route,
    {"price": "price", "standard": "standard", "other": "other"},
)
graph.add_edge("price", END)
graph.add_edge("standard", END)
graph.add_edge("other", END)
app = graph.compile()


if __name__ == "__main__":
    for q in ["破碎料多少钱一吨？", "重型废钢的标准是什么？", "今天天气怎么样？"]:
        result = app.invoke({"question": q})
        print("回答：", result["answer"], "\n")
