# RAG 评估：怎么科学地衡量你的 RAG 到底好不好？

> **封面图提示词（2:1 比例，1232×616px）：**
> A clean tech illustration showing a RAG system being examined by quality control instruments. In the center, a RAG pipeline (document → search → AI answer) sits on an examination table. Around it, various measurement tools float: a thermometer showing a score, a ruler measuring accuracy, a magnifying glass checking details, a balance/scale weighing relevance. A robot lab technician in a white coat holds a clipboard with checkmarks and scores. Background: clean laboratory/clinical environment with soft blue-white gradient. No text, no watermark. Aspect ratio 2:1.

前面 7 篇，咱们从零搭了 RAG、优化了检索、加了记忆、上了图谱。你问我效果怎么样？

"感觉还行。"

这不行。工程师不能靠感觉。写完代码要跑单测，做完产品要看数据指标，RAG 也一样——得有一套科学的评估方法，才知道哪里好、哪里烂、该往哪儿优化。不然你改完切块、换完 Embedding、调完 prompt，到底有没有变好？说不清。老板问"咱们这个 RAG 准确率多少"，你总不能回一句"大概八成吧"。

这篇文章就讲三件事：**用什么指标评估？怎么自动化评估？评估数据从哪来？** 全是能落地的实操，代码直接能跑。看完你就能给自己的 RAG 搭一套评估流水线，不用再靠感觉了。

---

## 一、RAG 评估到底在评什么？

很多人评估 RAG 就一个办法：自己问几个问题，看看回答顺不顺眼。这跟"我跑了两遍没报错所以代码没问题"一样不靠谱。问题在于：你问的那几个问题有代表性吗？回答"看起来还行"和"事实正确"是一回事吗？换个人问、换个说法问，效果会不会崩？

RAG 系统可以拆成两个环节：**检索**和**生成**。评估也得覆盖这两块，不能只盯着最终回答。

**检索质量**：找到的文档片段相关吗？全面吗？有没有漏掉关键信息？——检索烂了，后面生成再聪明也白搭。

**生成质量**：基于检索到的内容，回答对不对？有没有瞎编？有没有答非所问？——检索再好，模型瞎编或者跑题，照样翻车。

业界常用的四个核心指标，把全链路都管住了：

| 指标 | 英文 | 评什么 | 属于哪一环 |
|------|------|--------|------------|
| 上下文相关性 | Context Relevance / Precision | 检索到的文档和问题有多相关 | 检索 |
| 上下文召回率 | Context Recall | 正确答案所需的信息有没有被检索到 | 检索 |
| 忠实度 | Faithfulness | 回答有没有忠实于检索到的上下文（不幻觉） | 生成 |
| 答案相关性 | Answer Relevance | 回答有没有直接回答用户问题 | 生成 |

举个例子：用户问"年假几天？"，检索到三段"年假规定""报销流程""技术规范"，那 Precision 就低——混进了无关内容。如果用户问"入职 8 年年假能顺延到几月？"，检索只找到"年假 5 天"却没找到"顺延至次年 3 月"，那 Recall 就低——关键信息漏了。生成阶段，如果模型回答"可以顺延到 6 月"而文档写的是 3 月，Faithfulness 就崩了；如果用户问年假它答报销，Answer Relevance 就崩了。

说白了：**检索**要"找得准、找得全"，**生成**要"不瞎编、不跑题"。

> **正文配图 1 提示词：**
> A pipeline diagram showing the RAG evaluation framework. A horizontal flow: Question → Retriever (magnifying glass icon) → Retrieved Chunks (document cards) → Generator (AI brain icon) → Answer (chat bubble). Below the Retriever section, two measurement gauges for "Context Relevance" and "Context Recall". Below the Generator section, two measurement gauges for "Faithfulness" and "Answer Relevance". Each gauge shows a green/yellow/red zone like a speedometer. Clean infographic style, white background. No text labels, just icons and gauges. Aspect ratio 16:9.

---

## 二、用 LLM 做自动评估（LLM-as-Judge）

传统的 BLEU、ROUGE 对 RAG 不太适用——它们看的是字面匹配，不是语义正确。你改个说法意思一样，BLEU 可能就掉分了。而且 RAG 的很多问题，比如"回答有没有瞎编"、"检索到的文档相关不相关"，根本没法用 n-gram 算，得靠语义理解。

现代做法：**用 LLM 当裁判**，让它按每个维度打分。LLM 能理解语义，能判断"回答是否忠实于上下文"、"检索到的文档是否相关"这类问题，比 n-gram 匹配靠谱多了。

思路很简单：设计针对性的评估 prompt，把「上下文」「回答」「问题」塞进去，让 LLM 输出结构化结果（比如 JSON），再解析成分数。成本也不高，gpt-4o-mini 评估一条样本也就几分钱，一百条评估集跑下来也就几块钱。比招人标注便宜两个数量级。

下面是一个完整的 `RAGEvaluator`，实现忠实度和相关性评估。你完全可以按自己的需求改 prompt，比如加"回答是否简洁"、"是否引用了原文"等维度：

```python
import os
import json
import re
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()

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

最后一行输出 JSON，格式：{{"claims": ["声明1", "声明2"], "supported": N, "total": N, "score": 0.xx}}
"""

RELEVANCE_PROMPT = """你是一个相关性评估专家。
请评估以下检索到的文档片段与用户问题的相关程度。

【问题】{question}
【检索到的文档】
{context}

对每个文档片段评分（0-1）：
- 1.0：直接回答了问题
- 0.5：部分相关
- 0.0：完全不相关

最后一行输出 JSON，格式：{{"scores": [0.x, 0.x], "avg_score": 0.xx}}
"""

class RAGEvaluator:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def evaluate_faithfulness(self, context, answer):
        """评估回答是否忠实于上下文"""
        prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content
        # 提取 JSON
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("score", 0.0)
        return 0.0

    def evaluate_relevance(self, question, context):
        """评估检索到的文档与问题的相关性"""
        prompt = RELEVANCE_PROMPT.format(question=question, context=context)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("avg_score", 0.0)
        return 0.0

# 模拟运行
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    context = "年假规定：入职满1年不满10年，每年5天带薪年假。入职满10年不满20年，每年10天。"
    answer = "根据公司规定，入职满5年的员工每年有5天年假。"
    question = "入职满5年有几天年假？"

    faith = evaluator.evaluate_faithfulness(context, answer)
    rel = evaluator.evaluate_relevance(question, context)
    print(f"忠实度: {faith:.2f}")
    print(f"上下文相关性: {rel:.2f}")
```

**模拟运行效果：**

```
忠实度: 0.95
上下文相关性: 0.85
```

LLM-as-Judge 的优点是快、可自动化，缺点是存在偏差——不同模型、不同 prompt 可能打出不同分。适合开发迭代时用，不适合当唯一真理。建议用同一套 prompt 和模型做横向对比，这样至少相对趋势是可信的。

---

## 三、RAGAS 框架：开箱即用

不想自己写评估 prompt？用 **RAGAS**（Retrieval Augmented Generation Assessment）就行。它把上面四个指标的评估逻辑都封装好了，底层也是 LLM-as-Judge，但 prompt 和指标都经过优化，一行代码跑完，省心。

先装依赖：

```bash
pip install openai ragas datasets
```

RAGAS 需要的评估数据格式是：`question`、`answer`、`contexts`、`ground_truth`。`ground_truth` 就是标准答案，用来算 Context Recall 等指标。注意 `contexts` 是二维的：每个问题对应一个 list，里面是检索到的多个文档块。下面这段代码复用前面文章的公司知识库（请假、报销、技术规范），直接构造了一个小评估集：

```python
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

os.environ["OPENAI_API_KEY"] = "你的API Key"

# 准备评估数据（复用前面文章的公司知识库内容）
eval_data = {
    "question": [
        "入职满5年有几天年假？",
        "出差住酒店一线城市报销标准是多少？",
        "前端项目用什么构建工具？",
    ],
    "answer": [
        "根据公司请假制度，入职满1年不满10年的员工每年享有5天带薪年假，所以入职满5年有5天年假。",
        "一线城市（北上广深）住宿费每晚不超过500元。",
        "公司规定前端构建工具统一使用 Vite，不再使用 Webpack。",
    ],
    "contexts": [
        ["年假规定：入职满1年不满10年的员工，每年享有5天带薪年假。入职满10年不满20年，每年10天。年假需提前3个工作日申请。"],
        ["住宿费：一线城市（北上广深）每晚不超过500元，其他城市每晚不超过350元。需提供酒店发票。"],
        ["构建工具使用 Vite，不再使用 Webpack。框架统一使用 React 18+。"],
    ],
    "ground_truth": [
        "入职满5年（属于1-10年区间）每年有5天带薪年假。",
        "一线城市住宿费每晚不超过500元。",
        "前端构建工具使用 Vite。",
    ]
}

dataset = Dataset.from_dict(eval_data)

# 一行评估
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
```

**示例输出：**

```
{'faithfulness': 0.92, 'answer_relevancy': 0.88,
 'context_precision': 0.91, 'context_recall': 0.87}
```

RAGAS 底层也是 LLM 打分，但它的 prompt 和指标计算方式都经过论文验证，比你自己拍脑袋写的评估 prompt 更稳定。而且四个指标一次性跑完，不用自己拼。注意：RAGAS 需要你传入 `answer` 和 `contexts`，也就是你的 RAG 系统实际产出的结果。所以完整流程是：用 RAG 回答每个评估问题 → 拿到回答和检索到的 chunks → 和 ground_truth 一起塞进 RAGAS → 跑评估。如果你用的是 LangChain、LlamaIndex 这类框架，一般都有 `query` 或 `retrieve` 的接口，能直接拿到 contexts，接上 RAGAS 就行。

> **正文配图 2 提示词：**
> A dashboard/control panel illustration showing four circular gauge meters arranged in a 2x2 grid. Each gauge has a needle pointing to a score in the green zone. Top-left: gauge with a chain/link icon (Faithfulness). Top-right: gauge with a target/bullseye icon (Answer Relevance). Bottom-left: gauge with a search/magnifying glass icon (Context Precision). Bottom-right: gauge with a net/coverage icon (Context Recall). The overall dashboard has a "System Health" feel with an overall score displayed prominently. Clean flat design, dark dashboard background. No text, just icons and gauge visuals. Aspect ratio 16:9.

---

## 四、自建评估数据集

RAGAS 要 `ground_truth`，这些标准答案从哪来？这是很多人卡住的地方。

三种路子：

| 方法 | 优点 | 缺点 | 适合场景 |
|------|------|------|----------|
| 人工标注 | 最准 | 贵、慢、不可持续 | 初期校准、关键样本 |
| LLM 生成 | 快、量大 | 需要校验 | 快速搭建评估集 |
| 用户日志 | 最贴近真实 | 信号滞后、需清洗 | 生产环境反哺 |

人工标注最靠谱，但一条 QA 对可能要几十块、几分钟，评估集一上百条就扛不住。用户日志最贴近真实场景，但得先上线、有流量，而且得清洗、打标。所以起步阶段，**用 LLM 批量生成，再人工抽 20% 校验**，是最务实的做法。一条文档能生成 3～5 个 QA 对，十篇文档就能凑出几十条评估集，够跑第一轮了。

下面这段代码，基于前面文章的公司知识库文档（请假、报销、技术规范），用 LLM 自动生成 QA 对：

```python
import os
import json
import random
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()

# 复用前面文章的公司知识库文档（简化版）
COMPANY_DOCS = [
    """公司请假制度：入职满1年不满10年每年5天年假，满10年不满20年每年10天，满20年以上每年15天。
年假需提前3个工作日申请。当年未用可顺延至次年3月31日。病假需诊断证明，3天内直属上级批，3天以上总监批。""",
    """差旅报销：一线城市住宿每晚不超过500元，其他城市350元。飞机经济舱、高铁二等座实报实销。
餐饮补贴每天100元。5000元以下直属上级批，5000-20000元总监批，20000元以上财务总监批。""",
    """前端技术规范：框架用React 18+，状态管理Zustand，UI库Ant Design 5.x，构建工具Vite。
TypeScript严格模式，禁止any。Git分支命名feature/xxx，Commit格式type(scope): description。""",
]

QA_GEN_PROMPT = """根据以下文档内容，生成 3 个高质量的问答对。

要求：
1. 问题要有实际意义，不能太简单（不能直接从一句话里找到答案）
2. 答案要准确，完全基于文档内容
3. 包含不同类型：事实型、推理型、总结型

文档内容：
{document}

输出 JSON 格式，不要其他文字：
[{{"question": "...", "answer": "...", "type": "事实型"}}, ...]
"""

def generate_eval_dataset(documents, sample_size=20):
    """批量生成评估数据集"""
    qa_pairs = []
    for doc in documents:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": QA_GEN_PROMPT.format(document=doc)}],
            temperature=0.3
        )
        text = resp.choices[0].message.content.strip()
        # 提取 JSON 数组
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start:end])
                qa_pairs.extend(items)
            except json.JSONDecodeError:
                pass

    if len(qa_pairs) > sample_size:
        qa_pairs = random.sample(qa_pairs, sample_size)
    return qa_pairs

# 生成并打印
eval_dataset = generate_eval_dataset(COMPANY_DOCS, sample_size=9)
for i, qa in enumerate(eval_dataset[:3]):
    print(f"[{i+1}] {qa.get('type', '')} | Q: {qa['question'][:40]}...")
```

**模拟运行效果：**

```
[1] 事实型 | Q: 入职满15年的员工每年有几天年假？...
[2] 推理型 | Q: 去上海出差住酒店，每晚最多能报多少？...
[3] 总结型 | Q: 前端项目的技术栈有哪些硬性要求？...
```

有了这些 QA 对，你就可以：用 RAG 系统回答每个问题 → 拿到 `answer` 和 `contexts` → 把 `ground_truth` 填进去 → 丢给 RAGAS 跑评估。

生成的时候注意两点：一是问题要有区分度，别全是"年假几天"这种一眼能答的；二是答案要严格基于文档，别让 LLM 自由发挥。生成完建议人工抽 20% 看一眼，把明显错的删掉或修正，否则评估集质量会拖累整体结论。

---

## 五、评估驱动优化：完整迭代流程

评估不是做一次就完事，而是**评估 → 诊断 → 优化 → 重评估**的循环。跟写代码一样：跑单测 → 发现问题 → 修 bug → 再跑单测。

每个指标低，说明的问题不一样，优化方向也不同。别一上来就乱调——先看指标，再对症下药：

| 指标低 | 可能原因 | 优化方向 |
|--------|----------|----------|
| Context Precision | 检索精度不够 | 换更好的 Embedding、加 Rerank、调切块 |
| Context Recall | 召回不足 | 增大 top_k、加混合搜索(BM25)、查询改写 |
| Faithfulness | 生成幻觉 | 强化 prompt 约束、加引用要求、降 temperature |
| Answer Relevance | 答非所问 | 改 prompt 模板、加问题分类、检查检索是否跑偏 |

有了诊断，优化就有方向了。别一上来就"我换个 Embedding 试试"——先看是 Precision 低还是 Recall 低，再决定是加 Rerank 还是增大 top_k。写一个 `diagnose` 函数，根据评估结果自动给建议：

```python
def diagnose(eval_results):
    """根据评估结果诊断问题"""
    diagnostics = []

    if eval_results.get("context_precision", 1) < 0.7:
        diagnostics.append({
            "issue": "检索精度不够",
            "suggestion": "尝试：1) 更好的 Embedding 模型 2) 加 Rerank 3) 调整切块大小"
        })
    if eval_results.get("context_recall", 1) < 0.7:
        diagnostics.append({
            "issue": "检索召回不足",
            "suggestion": "尝试：1) 增大 top_k 2) 加混合搜索(BM25) 3) 查询改写"
        })
    if eval_results.get("faithfulness", 1) < 0.8:
        diagnostics.append({
            "issue": "生成幻觉严重",
            "suggestion": "尝试：1) 强化 prompt 约束 2) 加引用要求 3) 降低 temperature"
        })
    if eval_results.get("answer_relevancy", 1) < 0.7:
        diagnostics.append({
            "issue": "答非所问",
            "suggestion": "尝试：1) 改进 prompt 模板 2) 加问题分类 3) 检查检索内容是否跑偏"
        })

    if not diagnostics:
        return [{"issue": "无", "suggestion": "各项指标正常，可继续观察或扩大评估集"}]
    return diagnostics

# 示例
results = {"context_precision": 0.65, "context_recall": 0.82, "faithfulness": 0.75, "answer_relevancy": 0.88}
for d in diagnose(results):
    print(f"❌ {d['issue']}\n   → {d['suggestion']}\n")
```

**模拟运行效果：**

```
❌ 检索精度不够
   → 尝试：1) 更好的 Embedding 模型 2) 加 Rerank 3) 调整切块大小

❌ 生成幻觉严重
   → 尝试：1) 强化 prompt 约束 2) 加引用要求 3) 降低 temperature
```

拿到诊断结果后，优先解决分数最低的那一项。比如 Faithfulness 只有 0.6，说明模型在瞎编，先别折腾检索，把 prompt 里"严格基于参考资料回答、未提及的如实告知"这类约束加强，再跑一轮评估，看分数有没有上来。有数据指导，优化才不会乱打。

> **正文配图 3 提示词：**
> A circular workflow/cycle diagram showing 4 stages in a continuous loop. Stage 1 (top): "Evaluate" - a clipboard with scores/checkmarks. Stage 2 (right): "Diagnose" - a stethoscope/medical icon examining a document. Stage 3 (bottom): "Optimize" - a wrench/gear icon adjusting settings. Stage 4 (left): "Re-evaluate" - the clipboard again but with improved scores (higher gauges). Circular arrows connect all stages, suggesting continuous improvement. In the center: a RAG system icon getting progressively better. Clean flat design, white background. No text. Aspect ratio 16:9.

---

## 六、线上评估：监控真实用户反馈

离线评估和线上效果可能有差距。用户真实行为才是最终裁判。

可以采集的线上信号：

- **点赞/点踩**：用户对回答满不满意，最直接。点踩的样本一定要捞出来分析，看看是检索错了还是生成错了。
- **追问率**：用户紧接着又问了相关问题，说明第一次没答好，需要补充。追问率高是 RAG 质量差的强信号。
- **引用点击率**：用户有没有点进去看原文来源——点得多说明回答可能不够清楚，或者用户想核实。如果引用点击率特别高，可能是回答太简略，或者用户对 AI 不够信任。

把这些信号存下来，定期反哺到评估数据集里——比如把"点踩 + 追问"的样本挑出来，人工补上 `ground_truth`，加进下一轮评估。这样评估集会越来越贴近真实用户问题，形成闭环。

线上监控和离线评估要配合用：离线评估帮你快速迭代、对比版本；线上监控帮你发现"评估集里没有但用户常问"的问题，及时补充到评估集里。两者缺一不可。

---

## 七、总结对比表

| 评估方式 | 优点 | 缺点 | 适合阶段 |
|----------|------|------|----------|
| 人工评估 | 最准确 | 贵、慢、不可持续 | 初期校准 |
| LLM-as-Judge | 快速、可自动化 | 存在偏差 | 开发迭代 |
| RAGAS 框架 | 标准化、开箱即用 | 需要 ground_truth | 系统化评估 |
| 线上监控 | 贴近真实场景 | 信号滞后 | 生产环境 |

实际落地时，建议这么搭配：初期用人工标注几十条做校准，开发阶段用 RAGAS + LLM 生成的数据集做迭代，上线后用线上信号持续扩充评估集。别指望一种方法包打天下。

还有一个点：评估集要定期更新。你的知识库文档会变，用户问法也会变。一季度review 一次，把过时的删掉、把新的高频问题加进去，评估才有意义。

---

## 八、写在最后（系列收官）

这个系列从第一篇「从零开发一个 AI 智能体」开始，咱们一步步搭了 RAG、排过坑、让智能体自主检索、加了记忆、比了框架、接了 MCP、搞了多智能体协作、选了向量数据库、学了规划、上了图谱，最后学会了评估。十一篇，从零到能跑、能优化、能评估，一条龙走下来。

一句话总结整个系列的核心理念：**先理解原理再用框架，先跑通再优化，先评估再迭代。** 别一上来就堆框架，先把最简版跑通；别一上来就追求完美，先把评估跑起来，才知道往哪儿使劲。很多人在 RAG 上踩坑，不是因为技术难，而是因为没评估——改完不知道好不好，只能凭感觉，越改越乱。

这个系列就到这了。如果你从头跟到这里，恭喜你——你对 AI 智能体和 RAG 的理解，已经超过了 90% 只看文档不写代码的人。

> **正文配图 4 提示词：**
> A panoramic timeline/roadmap illustration showing the entire article series journey. From left to right: 11 milestone markers along a winding path, each with a small icon representing its topic: 1) robot birth (agent), 2) book+search (basic RAG), 3) warning signs (pitfalls), 4) robot+bookshelf (agentic RAG), 5) brain/memory chip (memory), 6) three paths (frameworks), 7) USB plug (MCP), 8) robot team (multi-agent), 9) database cylinders (vector DB), 10) checklist/plan (planning), 11) graph network (graph RAG), and finally a trophy/graduation cap at the end (evaluation/completion). The path has a "YOU ARE HERE" marker at the last position. Warm sunset lighting suggesting completion. No text. Aspect ratio 16:9.

---

## 附录：整合版完整代码

下面是一份整合版代码，包含：LLM-as-Judge 评估、RAGAS 评估、自建数据集生成、诊断函数。依赖 `pip install openai ragas datasets`，基于 OpenAI API。

```python
"""
RAG 评估完整示例
依赖: pip install openai ragas datasets
"""
import os
import json
import re
import random
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()

# ========== 1. LLM-as-Judge 评估器 ==========
FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。判断【回答】中的事实是否能在【上下文】中找到依据。
【上下文】{context}
【回答】{answer}
最后一行输出 JSON：{{"supported": N, "total": N, "score": 0.xx}}"""

RELEVANCE_PROMPT = """评估检索文档与问题的相关程度。
【问题】{question}
【文档】{context}
最后一行输出 JSON：{{"scores": [0.x], "avg_score": 0.xx}}"""

class RAGEvaluator:
    def evaluate_faithfulness(self, context, answer):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)}],
            temperature=0
        )
        m = re.search(r'\{[^{}]*\}', resp.choices[0].message.content)
        return json.loads(m.group()).get("score", 0.0) if m else 0.0

    def evaluate_relevance(self, question, context):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": RELEVANCE_PROMPT.format(question=question, context=context)}],
            temperature=0
        )
        m = re.search(r'\{[^{}]*\}', resp.choices[0].message.content)
        return json.loads(m.group()).get("avg_score", 0.0) if m else 0.0

# ========== 2. RAGAS 评估 ==========
def run_ragas_eval():
    eval_data = {
        "question": ["入职满5年有几天年假？", "一线城市住宿报销标准？", "前端构建工具？"],
        "answer": [
            "入职满5年有5天年假。",
            "一线城市每晚不超过500元。",
            "使用 Vite。",
        ],
        "contexts": [
            ["入职满1年不满10年每年5天年假。"],
            ["一线城市每晚不超过500元。"],
            ["构建工具使用 Vite。"],
        ],
        "ground_truth": [
            "5天",
            "每晚不超过500元",
            "Vite",
        ]
    }
    ds = Dataset.from_dict(eval_data)
    return evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

# ========== 3. 自建评估数据集 ==========
COMPANY_DOCS = [
    "年假：1-10年5天，10-20年10天，20年以上15天。病假需诊断证明。",
    "差旅：一线城市住宿500元/晚，餐饮100元/天。5000元以下直属上级批。",
    "前端：React 18+，Vite 构建，TypeScript 严格模式。",
]

QA_GEN_PROMPT = """根据文档生成3个QA对，输出JSON数组：[{{"question":"...","answer":"..."}}]
文档：{document}"""

def generate_eval_dataset(docs, n=6):
    pairs = []
    for doc in docs:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": QA_GEN_PROMPT.format(document=doc)}],
            temperature=0.3
        )
        text = resp.choices[0].message.content
        start, end = text.find('['), text.rfind(']') + 1
        if start >= 0 and end > start:
            try:
                pairs.extend(json.loads(text[start:end]))
            except: pass
    return random.sample(pairs, min(n, len(pairs))) if pairs else []

# ========== 4. 诊断函数 ==========
def diagnose(results):
    diag = []
    if results.get("context_precision", 1) < 0.7:
        diag.append({"issue": "检索精度不够", "suggestion": "换 Embedding、加 Rerank、调切块"})
    if results.get("context_recall", 1) < 0.7:
        diag.append({"issue": "召回不足", "suggestion": "增大 top_k、加 BM25、查询改写"})
    if results.get("faithfulness", 1) < 0.8:
        diag.append({"issue": "生成幻觉", "suggestion": "强化 prompt、加引用、降 temperature"})
    if results.get("answer_relevancy", 1) < 0.7:
        diag.append({"issue": "答非所问", "suggestion": "改 prompt、加问题分类"})
    return diag or [{"issue": "无", "suggestion": "指标正常"}]

# ========== 运行 ==========
if __name__ == "__main__":
    print("=== LLM-as-Judge ===")
    ev = RAGEvaluator()
    print("忠实度:", ev.evaluate_faithfulness("年假5天", "入职满5年有5天年假"))
    print("相关性:", ev.evaluate_relevance("年假几天？", "年假5天"))

    print("\n=== RAGAS ===")
    # results = run_ragas_eval()
    # print(results)

    print("\n=== 生成评估集 ===")
    dataset = generate_eval_dataset(COMPANY_DOCS)
    for qa in dataset[:2]:
        print(f"Q: {qa['question'][:50]}...")

    print("\n=== 诊断 ===")
    for d in diagnose({"context_precision": 0.6, "faithfulness": 0.75}):
        print(f"{d['issue']} → {d['suggestion']}")
```

运行前请将 `OPENAI_API_KEY` 替换为你的实际 Key。RAGAS 评估部分会调用 OpenAI，产生少量 API 费用。
