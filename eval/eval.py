"""
RAG 问答系统批量评测（黑盒版）

用法（在 langgraph-demo 的 venv 里跑）：
$env:GROQ_API_KEY="你的key"
python eval.py

被测系统：supplychain-all 的 Python ai-service（POST /api/ask）
裁判：Groq openai/gpt-oss-120b（LLM-as-Judge）
产物：控制台汇总 + eval_results.json（逐条明细）
"""
import json
import os
import re
import time
from collections import defaultdict

import requests

# ============ 配置区（按你的环境改） ============
API_URL = "http://localhost:8000/api/ask" # ai-service 地址
USER_ID = "eval-user" # X-User-ID：换成你有权限的测试用户
USER_GROUPS = "test-org" # X-User-Groups：换成能访问知识库的组织编码
JUDGE_MODEL = "openai/gpt-oss-120b"
PASS_SCORE = 4 # >=4 分算通过
# ================================================

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

JUDGE_PROMPT = """你是 RAG 问答系统的评测裁判。只依据下面的判分标准打分，不要脑补标准之外的知识。

{question}
{key_points}
{answer}

判分标准（1~5 分整数）：
- 5 分：期望要点全覆盖，事实正确，无编造
- 4 分：覆盖主要要点，基本正确，细节略有缺失
- 3 分：覆盖一半左右要点，或答非所问、泛泛而谈
- 2 分：只沾边，关键信息错误或大量编造
- 1 分：完全答错、拒答、或声称无数据但要点本应可查

特别规则：如果回答编造了期望要点和资料中都没有的具体数字/事实，在 reason 里标出"疑似幻觉"，最高给 2 分。

只输出 JSON，不要任何其他文字：
{{"score": 分数, "hits": ["命中的要点"], "reason": "一句理由"}}"""


def ask_system(question: str) -> dict:
"""调被测系统，返回 {answer, intent, references}，失败返回 {error}"""
headers = {
"Content-Type": "application/json",
"X-User-ID": USER_ID,
"X-User-Groups": USER_GROUPS,
}
try:
resp = requests.post(API_URL, json={"question": question}, headers=headers, timeout=90)
resp.raise_for_status()
data = resp.json().get("data") or {}
return {
"answer": data.get("answer", ""),
"intent": data.get("intent", ""),
"references": data.get("references") or [],
}
except Exception as e:
return {"error": str(e)}


def judge(question: str, key_points: list, answer: str) -> dict:
"""LLM 裁判打分，返回 {score, hits, reason}"""
prompt = JUDGE_PROMPT.format(
question=question,
key_points="；".join(key_points),
answer=answer or "（空回答）",
)
try:
resp = requests.post(
"https://api.groq.com/openai/v1/chat/completions",
headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
json={
"model": JUDGE_MODEL,
"messages": [{"role": "user", "content": prompt}],
"temperature": 0,
},
timeout=60,
)
resp.raise_for_status()
text = resp.json()["choices"][0]["message"]["content"]
m = re.search(r"\{.*\}", text, re.S)
return json.loads(m.group(0)) if m else {"score": 0, "reason": "裁判输出无法解析"}
except Exception as e:
return {"score": 0, "reason": f"裁判调用失败: {e}"}


def main():
if not GROQ_KEY:
print("先设置环境变量 GROQ_API_KEY")
return
with open("test_set.json", encoding="utf-8") as f:
cases = json.load(f)
print(f"共 {len(cases)} 条，开始评测（被测 {API_URL}）...\n")

results = []
for case in cases:
got = ask_system(case["question"])
if "error" in got:
print(f"[{case['id']}] 调用失败: {got['error']}")
results.append({**case, "error": got["error"]})
continue
verdict = judge(case["question"], case["key_points"], got["answer"])
expected = case["expected_intent"]
intent_ok = got["intent"] in expected if isinstance(expected, list) else got["intent"] == expected
row = {
**case,
"answer": got["answer"],
"got_intent": got["intent"],
"intent_ok": intent_ok,
"ref_count": len(got["references"]),
"score": verdict.get("score", 0),
"hits": verdict.get("hits", []),
"reason": verdict.get("reason", ""),
}
results.append(row)
print(f"[{case['id']}] {case['type']:9s} 意图{'✓' if intent_ok else '✗(' + str(got['intent']) + ')'} "
f"分数 {row['score']}/5 {verdict.get('reason', '')}")
time.sleep(1) # 给 Groq 免费额度留口气

# ===== 汇总：按类型分开看 =====
print("\n===== 成绩单 =====")
by_type = defaultdict(list)
for r in results:
if "score" in r:
by_type[r["type"]].append(r)
all_scores = []
for t, rows in sorted(by_type.items()):
scores = [r["score"] for r in rows]
all_scores += scores
intent_acc = sum(r["intent_ok"] for r in rows) / len(rows)
pass_rate = sum(s >= PASS_SCORE for s in scores) / len(scores)
print(f"{t:10s} 平均分 {sum(scores)/len(scores):.2f} 通过率 {pass_rate:.0%} 意图准确率 {intent_acc:.0%} (n={len(rows)})")
if all_scores:
print(f"{'总计':10s} 平均分 {sum(all_scores)/len(all_scores):.2f} 通过率 "
f"{sum(s >= PASS_SCORE for s in all_scores)/len(all_scores):.0%} (n={len(all_scores)})")

with open("eval_results.json", "w", encoding="utf-8") as f:
json.dump(results, f, ensure_ascii=False, indent=2)
print("\n明细已存 eval_results.json")


if __name__ == "__main__":
main()
