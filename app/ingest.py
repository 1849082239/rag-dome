"""文档加载与文本切分。

RAG 第一步：把原始文档变成带元信息的小文本块（chunk）。
切分质量直接影响检索效果，是面试高频考点。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 分隔符按优先级从高到低排列：优先在段落、换行、中文标点处切分，
# 尽量不断开一个完整的语义单元。最后一个 "" 是兜底（按字切）。
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


@dataclass
class Chunk:
    text: str
    source: str  # 文件名
    index: int   # 在该文档中的顺序号


def load_documents(doc_dir: str | Path) -> list[tuple[str, str]]:
    """读取目录下所有 .txt / .md 文件，返回 [(文件名, 全文)]。"""
    docs = []
    for path in sorted(Path(doc_dir).glob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"{doc_dir} 下没有找到 .txt / .md 文档")
    return docs


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """递归字符切分（与 LangChain RecursiveCharacterTextSplitter 同思想）。

    1. 用最高优先级的分隔符把文本切开；
    2. 太长的片段用下一级分隔符继续切，直到每块 <= chunk_size；
    3. 合并小片段时尽量填满 chunk_size；
    4. 相邻块之间保留 chunk_overlap 的重叠，避免语义在边界处断裂。
    """
    separators = separators or SEPARATORS

    def _split(t: str, seps: list[str]) -> list[str]:
        sep, rest = seps[0], seps[1:]
        splits = t.split(sep) if sep else list(t)
        out: list[str] = []
        for s in splits:
            if len(s) <= chunk_size or not rest:
                out.append(s)
            else:
                out.extend(_split(s, rest))
        return out

    pieces = [p.strip() for p in _split(text, separators)]
    pieces = [p for p in pieces if p]

    chunks: list[str] = []
    buf = ""
    for p in pieces:
        candidate = f"{buf} {p}".strip() if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        # 新块以旧块尾部做 overlap 开头
        overlap = buf[-chunk_overlap:] if (chunk_overlap and buf) else ""
        buf = f"{overlap} {p}".strip()
        while len(buf) > chunk_size:  # 单个片段就超长，硬切
            chunks.append(buf[:chunk_size])
            buf = buf[chunk_size - chunk_overlap:] if chunk_overlap else buf[chunk_size:]
            buf = buf.strip()
    if buf:
        chunks.append(buf)
    return chunks


def chunk_documents(
    docs: list[tuple[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """把多个文档全部切分成 Chunk 列表。"""
    chunks: list[Chunk] = []
    for source, text in docs:
        for i, piece in enumerate(recursive_split(text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(text=piece, source=source, index=i))
    return chunks
