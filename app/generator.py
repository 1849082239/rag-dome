"""生成：把检索到的上下文拼进 Prompt，让 LLM 基于资料回答。

核心就三件事：
1. Prompt 约束模型"只根据资料回答，不知道就说不知道"（抗幻觉）；
2. 要求标注来源（可追溯，面试喜欢问）；
3. temperature 调低，让回答更稳定。
"""
from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请只根据下面提供的[参考资料]回答用户问题。\n"
    "- 如果资料中没有相关信息，请明确说\"根据现有资料无法回答\"，不要编造。\n"
    "- 回答要简洁准确，回答后用 [来源：文件名] 标注信息来源。"
)


def build_messages(query: str, contexts: list[RetrievedChunk]) -> list[dict]:
    refs = "\n\n".join(
        f"[资料{i + 1}｜来源：{c.source}]\n{c.text}"
        for i, c in enumerate(contexts)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[参考资料]\n{refs}\n\n[问题]\n{query}"},
    ]


class Generator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, query: str, contexts: list[RetrievedChunk], temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(query, contexts),
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def generate_stream(
        self, query: str, contexts: list[RetrievedChunk], temperature: float = 0.2
    ) -> Iterator[str]:
        """流式生成：token 一个个吐，前端可以打字机效果展示，体验好很多。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(query, contexts),
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
