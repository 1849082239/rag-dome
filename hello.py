# LangGraph Hello World：单节点调 Groq，对应 Dify 里的 LLM 节点
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq


# ---- State：节点之间传递的数据，对应 Dify 里的变量 ----
class State(TypedDict):
    question: str
    answer: str


# ---- LLM：用 Groq 的 gpt-oss-120b，key 从环境变量读 ----
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ---- Node：一个节点就是一个函数，输入 State，返回要更新的字段 ----
def chat_node(state: State) -> dict:
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}


# ---- Graph：把节点和边拼起来，对应 Dify 画布 ----
graph = StateGraph(State)
graph.add_node("chat", chat_node)  # 注册节点
graph.add_edge(START, "chat")     # 开始 -> chat
graph.add_edge("chat", END)       # chat -> 结束
app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"question": "破碎料多少钱一吨？"})
    print("回答：", result["answer"])
