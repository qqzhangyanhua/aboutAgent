"""
RAG 评估：用 LLM-as-Judge 自动评估 RAG 质量
完整可运行示例 —— 对应文章第 12 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python rag_evaluator.py
"""
import os
import json
import re
from openai import OpenAI
import chromadb

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"


# ==================== 评估 Prompt ====================

FAITHFULNESS_PROMPT = """你是严格的事实核查员。
判断【回答】中的每个事实声明，是否能在【上下文】中找到依据。

【上下文】
{context}

【回答】
{answer}

逐条分析后，最后一行输出 JSON：
{{"claims": ["声明1", "声明2"], "supported": N, "total": N, "score": 0.xx}}"""

RELEVANCE_PROMPT = """你是相关性评估专家。
评估检索到的文档片段与用户问题的相关程度。

【问题】{question}
【检索到的文档】
{context}

对每个文档评分（0-1），1.0=直接回答，0.5=部分相关，0.0=完全无关。
最后一行输出 JSON：{{"scores": [0.x, 0.x], "avg_score": 0.xx}}"""

ANSWER_RELEVANCE_PROMPT = """你是回答质量评估专家。
评估回答是否直接回答了用户问题。

【问题】{question}
【回答】{answer}

从以下维度打分（0-1）：
1. 直接性：是否直接回答问题
2. 完整性：是否覆盖问题的各个方面
3. 简洁性：是否有冗余无关内容

最后一行输出 JSON：{{"directness": 0.x, "completeness": 0.x, "conciseness": 0.x, "overall": 0.x}}"""


# ==================== RAG 评估器 ====================

class RAGEvaluator:
    """RAG 质量评估器"""

    def _extract_json(self, text: str) -> dict:
        """从 LLM 输出中提取最后的 JSON"""
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {}

    def evaluate_faithfulness(self, context: str, answer: str) -> dict:
        """评估回答是否忠实于上下文"""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)}],
            temperature=0
        )
        return self._extract_json(resp.choices[0].message.content)

    def evaluate_context_relevance(self, question: str, context: str) -> dict:
        """评估检索到的文档与问题的相关性"""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": RELEVANCE_PROMPT.format(question=question, context=context)}],
            temperature=0
        )
        return self._extract_json(resp.choices[0].message.content)

    def evaluate_answer_relevance(self, question: str, answer: str) -> dict:
        """评估回答与问题的相关性"""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)}],
            temperature=0
        )
        return self._extract_json(resp.choices[0].message.content)

    def full_evaluate(self, question: str, context: str, answer: str) -> dict:
        """全维度评估"""
        return {
            "faithfulness": self.evaluate_faithfulness(context, answer),
            "context_relevance": self.evaluate_context_relevance(question, context),
            "answer_relevance": self.evaluate_answer_relevance(question, answer),
        }


# ==================== 模拟 RAG 系统 ====================

class SimpleRAG:
    """简单的 RAG 系统，用于评估演示"""

    def __init__(self):
        chroma = chromadb.Client()
        self.collection = chroma.get_or_create_collection("eval_docs", metadata={"hnsw:space": "cosine"})
        self._init_docs()

    def _init_docs(self) -> None:
        docs = [
            "年假规定：入职满1年不满10年，每年5天带薪年假。满10年不满20年，每年10天。当年未使用年假可顺延至次年3月31日。",
            "病假规定：需提供正规医院诊断证明。3天以内直属上级审批，3天以上部门总监审批。病假期间工资按基本工资80%发放。",
            "差旅住宿标准：一线城市每晚不超过500元，二线城市不超过350元，其他城市不超过250元。",
            "绩效考核：季度考核制。S级年终奖系数2.0，A级1.5，B级1.0，C级无年终奖。",
            "报销流程：费用发生后30天内提交。5000元以内部门经理审批，以上需财务总监审批。",
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        existing = self.collection.get()["ids"]
        new = [(id_, doc) for id_, doc in zip(ids, docs) if id_ not in existing]
        if new:
            self.collection.add(ids=[x[0] for x in new], documents=[x[1] for x in new])

    def query(self, question: str) -> tuple[str, str]:
        """返回 (context, answer)"""
        results = self.collection.query(query_texts=[question], n_results=2)
        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "基于参考资料回答问题。如果资料不足明确说明。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
            ],
            temperature=0
        )
        return context, resp.choices[0].message.content


# ==================== 运行演示 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    rag = SimpleRAG()
    evaluator = RAGEvaluator()

    test_cases = [
        "年假有几天？能不能顺延？",
        "出差去上海住酒店最多报多少？",
        "绩效S级有什么好处？",
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"问题：{question}")

        context, answer = rag.query(question)
        print(f"回答：{answer[:200]}...")

        scores = evaluator.full_evaluate(question, context, answer)
        print(f"\n评估结果：")
        for dimension, result in scores.items():
            print(f"  {dimension}: {result}")
