# rag-dome 学习项目：从 Dify 可视化工作流到 LangGraph 代码实现
# 记录一次完整 RAG 链路的学习路径：问题分类 → 知识检索 → LLM 生成

## 文件说明

| 文件 | 说明 |
|------|------|
| `hello.py` | LangGraph Hello World：单节点调 Groq，对应 Dify 的 LLM 节点 |
| `classifier.py` | 条件边复刻 Dify 的 Question Classifier：问题分类路由 |
| `rag_chain.py` | 完整 RAG 链路：分类 → Qdrant 检索 → Groq 生成（对应 Dify Chatflow）|
| `ingest.py` | 文档切分 → bge-m3 向量化 → 灌入 Qdrant（对应 Dify 的文档处理）|

## 环境要求

- Python 3.12 + venv
- Qdrant 本地运行（`localhost:6333`）
- Groq API Key（对话）、HidenCloud bge-m3（embedding，OpenAI 兼容接口）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="你的key"
python ingest.py      # 先灌库
python rag_chain.py   # 再跑完整链路
```

## Dify → LangGraph 概念映射

- Dify 节点 = LangGraph Node（函数）
- Dify 连线 = LangGraph 边（Edge）
- Question Classifier = 条件边（`add_conditional_edges`）
- Knowledge Retrieval = 检索函数（调 Qdrant）
- 变量传递（`{{sys.query}}`）= State 在节点间流转
