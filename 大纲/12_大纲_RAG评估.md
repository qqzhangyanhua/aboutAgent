# RAG 评估：怎么科学地衡量你的 RAG 到底好不好？

## 封面图提示词（2:1 比例，1232×616px）

> A clean tech illustration showing a RAG system being examined by quality control instruments. In the center, a RAG pipeline (document → search → AI answer) sits on an examination table. Around it, various measurement tools float: a thermometer showing a score, a ruler measuring accuracy, a magnifying glass checking details, a balance/scale weighing relevance. A robot lab technician in a white coat holds a clipboard with checkmarks and scores. Background: clean laboratory/clinical environment with soft blue-white gradient. No text, no watermark. Aspect ratio 2:1.

## 一句话定位

RAG 搭完之后，怎么知道它到底好不好？不能靠"感觉"——本文介绍 RAG 评估的核心指标、RAGAS 自动评估框架，以及如何自建评估数据集。

---

## 章节大纲

### 开头引入

- 切入点：前面 7 篇文章，我们从零搭了 RAG、优化了检索、加了记忆、上了图谱。你问我效果怎么样？"感觉还行"——这不行，工程师不能靠感觉
- 类比：写完代码要跑单元测试，做完产品要看数据指标。RAG 也一样，得有一套科学的评估方法
- 核心问题：用什么指标评估？怎么自动化评估？评估数据从哪来？

### 一、RAG 评估到底在评什么？

**要点：**
- RAG 系统可以拆成两个环节评估：
  1. **检索质量**（Retrieval）：找到的文档片段是否相关？是否全面？
  2. **生成质量**（Generation）：基于检索到的内容，回答是否正确？是否忠实于原文？
- 四个核心指标：
  - **Context Relevance**（上下文相关性）：检索到的文档和问题有多相关
  - **Context Recall**（上下文召回率）：正确答案所需的信息是否被检索到
  - **Faithfulness**（忠实度）：回答是否忠实于检索到的上下文（不幻觉）
  - **Answer Relevance**（答案相关性）：回答是否直接回答了问题
- 这四个指标覆盖了 RAG 的全链路质量

> **正文配图 1 提示词：**
> A pipeline diagram showing the RAG evaluation framework. A horizontal flow: Question → Retriever (magnifying glass icon) → Retrieved Chunks (document cards) → Generator (AI brain icon) → Answer (chat bubble). Below the Retriever section, two measurement gauges for "Context Relevance" and "Context Recall". Below the Generator section, two measurement gauges for "Faithfulness" and "Answer Relevance". Each gauge shows a green/yellow/red zone like a speedometer. Clean infographic style, white background. No text labels, just icons and gauges. Aspect ratio 16:9.

### 二、用 LLM 做自动评估（LLM-as-Judge）

**要点：**
- 传统评估方法（BLEU、ROUGE）对 RAG 不太适用——关注的是字面匹配，不是语义正确
- 现代方案：用 LLM 当裁判，评估每个维度的质量
- 核心思路：设计针对性的评估 prompt，让 LLM 打分

**核心代码片段：**

```python
FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。
请判断以下【回答】中的每一个事实声明，是否能在【上下文】中找到依据。

【上下文】
{context}

【回答】
{answer}

请逐条分析：
1. 列出回答中的每个事实声明
2. 判断每个声明是否有上下文支撑（支撑/无依据/矛盾）
3. 给出忠实度得分：有支撑的声明数 / 总声明数

输出 JSON：{{"claims": [...], "supported": N, "total": N, "score": 0.xx}}
"""

RELEVANCE_PROMPT = """你是一个相关性评估专家。
请评估以下检索到的文档片段与用户问题的相关程度。

【问题】{question}
【检索到的文档】{context}

对每个文档片段评分（0-1）：
- 1.0：直接回答了问题
- 0.5：部分相关
- 0.0：完全不相关

输出 JSON：{{"scores": [0.x, 0.x, ...], "avg_score": 0.xx}}
"""

class RAGEvaluator:
    def evaluate_faithfulness(self, context, answer):
        result = ask_llm(FAITHFULNESS_PROMPT.format(
            context=context, answer=answer))
        return json.loads(result)["score"]

    def evaluate_relevance(self, question, context):
        result = ask_llm(RELEVANCE_PROMPT.format(
            question=question, context=context))
        return json.loads(result)["avg_score"]
```

### 三、RAGAS 框架：开箱即用的 RAG 评估

**要点：**
- RAGAS（Retrieval Augmented Generation Assessment）是专门的 RAG 评估开源框架
- 内置了上面四个核心指标的评估逻辑
- 一行代码跑完评估，输出各维度得分

**核心代码片段：**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": [
        "公司的 AI 项目什么时候启动的？",
        "张三在公司负责什么？",
    ],
    "answer": [
        "AI 项目于 2024 年 Q1 启动。",
        "张三是技术部总监，负责 AI 项目。",
    ],
    "contexts": [
        ["AI 项目于 2024 年 Q1 正式启动，由技术部牵头。"],
        ["张三，技术部总监，2023年入职，负责推动 AI 项目。"],
    ],
    "ground_truth": [
        "AI 项目于 2024 年 Q1 启动。",
        "张三是技术部总监，负责推动公司的 AI 项目落地。",
    ]
}

dataset = Dataset.from_dict(eval_data)

# 一行评估
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88,
#  'context_precision': 0.90, 'context_recall': 0.85}
```

> **正文配图 2 提示词：**
> A dashboard/control panel illustration showing four circular gauge meters arranged in a 2x2 grid. Each gauge has a needle pointing to a score in the green zone. Top-left: gauge with a chain/link icon (Faithfulness). Top-right: gauge with a target/bullseye icon (Answer Relevance). Bottom-left: gauge with a search/magnifying glass icon (Context Precision). Bottom-right: gauge with a net/coverage icon (Context Recall). The overall dashboard has a "System Health" feel with an overall score displayed prominently. Clean flat design, dark dashboard background. No text, just icons and gauge visuals. Aspect ratio 16:9.

### 四、自建评估数据集

**要点：**
- RAGAS 需要"标准答案"（ground_truth），这些从哪来？
- 方法一：人工标注——最准确但最贵
- 方法二：用 LLM 自动生成 QA 对——快速但需要校验
- 方法三：从真实用户日志中挖掘——最贴近实际场景
- 建议：先用 LLM 批量生成，再人工校验 20% 的样本

**核心代码片段：**

```python
QA_GEN_PROMPT = """根据以下文档内容，生成 3 个高质量的问答对。

要求：
1. 问题要有实际意义，不能太简单（不能直接从一句话里找到答案）
2. 答案要准确，完全基于文档内容
3. 包含不同类型：事实型、推理型、总结型

文档内容：
{document}

输出 JSON 格式：
[{{"question": "...", "answer": "...", "type": "事实型/推理型/总结型"}}]
"""

def generate_eval_dataset(documents, sample_size=50):
    """批量生成评估数据集"""
    qa_pairs = []
    for doc in documents:
        generated = ask_llm(QA_GEN_PROMPT.format(document=doc))
        qa_pairs.extend(json.loads(generated))

    # 随机采样
    import random
    if len(qa_pairs) > sample_size:
        qa_pairs = random.sample(qa_pairs, sample_size)

    return qa_pairs

def run_evaluation(rag_system, eval_dataset):
    """对 RAG 系统跑完整评估"""
    results = []
    for qa in eval_dataset:
        # 用 RAG 系统回答
        rag_answer, retrieved_chunks = rag_system.query_with_context(qa["question"])
        results.append({
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "rag_answer": rag_answer,
            "contexts": retrieved_chunks,
        })
    return results
```

### 五、评估驱动优化：一个完整的迭代流程

**要点：**
- 评估不是做一次就完了，而是持续迭代的过程
- 流程：评估 → 定位问题 → 优化 → 重新评估 → 确认改善
- 不同指标低说明不同问题：
  - Context Relevance 低 → 检索策略有问题（Embedding 模型、切块方式）
  - Context Recall 低 → 检索不够全（增加 top_k、加混合搜索）
  - Faithfulness 低 → 生成有幻觉（改 prompt 约束、加事实校验）
  - Answer Relevance 低 → 答非所问（改 prompt 引导）

**核心代码片段（诊断）：**

```python
def diagnose(eval_results):
    """根据评估结果诊断问题"""
    diagnostics = []

    if eval_results["context_precision"] < 0.7:
        diagnostics.append({
            "issue": "检索精度不够",
            "suggestion": "尝试：1) 更好的 Embedding 模型 2) 加 Rerank 3) 调整切块大小"
        })
    if eval_results["context_recall"] < 0.7:
        diagnostics.append({
            "issue": "检索召回不足",
            "suggestion": "尝试：1) 增大 top_k 2) 加混合搜索(BM25) 3) 查询改写"
        })
    if eval_results["faithfulness"] < 0.8:
        diagnostics.append({
            "issue": "生成幻觉严重",
            "suggestion": "尝试：1) 强化 prompt 约束 2) 加引用要求 3) 降低 temperature"
        })
    if eval_results["answer_relevancy"] < 0.7:
        diagnostics.append({
            "issue": "答非所问",
            "suggestion": "尝试：1) 改进 prompt 模板 2) 加问题分类 3) 检查检索内容是否跑偏"
        })

    return diagnostics
```

> **正文配图 3 提示词：**
> A circular workflow/cycle diagram showing 4 stages in a continuous loop. Stage 1 (top): "Evaluate" - a clipboard with scores/checkmarks. Stage 2 (right): "Diagnose" - a stethoscope/medical icon examining a document. Stage 3 (bottom): "Optimize" - a wrench/gear icon adjusting settings. Stage 4 (left): "Re-evaluate" - the clipboard again but with improved scores (higher gauges). Circular arrows connect all stages, suggesting continuous improvement. In the center: a RAG system icon getting progressively better. Clean flat design, white background. No text. Aspect ratio 16:9.

### 六、线上评估：监控真实用户反馈

**要点：**
- 离线评估和线上效果可能有差距
- 线上信号采集：
  - 用户点赞/点踩
  - 追问率（用户紧接着又问了相关问题 = 第一次没回答好）
  - 引用点击率（用户有没有去看原文来源）
- 把线上信号反哺到评估数据集，形成闭环

### 七、总结对比

```
| 评估方式        | 优点              | 缺点              | 适合阶段          |
|----------------|------------------|--------------------|------------------|
| 人工评估        | 最准确            | 贵、慢、不可持续     | 初期校准           |
| LLM-as-Judge   | 快速、可自动化     | 存在偏差            | 开发迭代           |
| RAGAS 框架      | 标准化、开箱即用   | 需要 ground_truth   | 系统化评估         |
| 线上监控        | 贴近真实场景       | 信号滞后            | 生产环境           |
```

### 写在最后（系列收官）

- 系列回顾：从第一篇"从零开发一个智能体"开始，我们一步步搭了 RAG、排过坑、让智能体自主检索、加了记忆、比了框架、接了 MCP、搞了多智能体协作、选了向量数据库、学了规划、上了图谱，最后学会了评估
- 一句话总结整个系列的核心理念：先理解原理再用框架，先跑通再优化，先评估再迭代
- 结尾风格：不用"期待下一篇"，换成"这个系列就到这了。如果你从头跟到这里，恭喜你，你对 AI 智能体和 RAG 的理解已经超过了 90% 只看文档不写代码的人"

> **正文配图 4 提示词：**
> A panoramic timeline/roadmap illustration showing the entire article series journey. From left to right: 11 milestone markers along a winding path, each with a small icon representing its topic: 1) robot birth (agent), 2) book+search (basic RAG), 3) warning signs (pitfalls), 4) robot+bookshelf (agentic RAG), 5) brain/memory chip (memory), 6) three paths (frameworks), 7) USB plug (MCP), 8) robot team (multi-agent), 9) database cylinders (vector DB), 10) checklist/plan (planning), 11) graph network (graph RAG), and finally a trophy/graduation cap at the end (evaluation/completion). The path has a "YOU ARE HERE" marker at the last position. Warm sunset lighting suggesting completion. No text. Aspect ratio 16:9.
