# Graph RAG：让 AI 理解知识之间的关系

> **封面图提示词（2:1 比例，1232×616px）：**
> A tech illustration showing a knowledge graph floating in space. Multiple glowing nodes (circles) of different sizes connected by labeled edges/lines forming a web-like structure. Some nodes cluster into communities (highlighted with translucent colored bubbles - blue, green, orange). A robot stands on one node, using a telescope to look across the graph, with a search beam illuminating a path through connected nodes. Small document icons feed into the graph from the left side. Background: deep space/dark navy with subtle constellation patterns. No text, no watermark. Aspect ratio 2:1.

你问 RAG「张三是谁？」，它能找到包含张三的段落，答得头头是道。但你问「张三和李四是什么关系？」，它就傻了——张三的信息在第 3 段，李四的信息在第 15 段，普通向量检索根本不会把它们关联起来。

问题在哪？**知识不是孤立的碎片，而是一张网。** 普通 RAG 只会找「语义相似」的片段，不会找「逻辑关联」的路径。

这篇文章从普通 RAG 的局限说起，一步步搭到 Graph RAG，让你搞清楚：什么时候该用图、怎么用、以及微软那套进阶思路到底在干啥。

依赖就四个：`pip install openai chromadb networkx python-louvain`，代码都能跑。

---

## 测试文档（实体链：张三 → AI项目 → 创新奖 → 李四）

后面所有演示都基于下面 5 段文档。注意看：张三、AI 项目、创新奖、李四分在不同段落，但通过「负责」「获得」「评审」等关系连成一条链。普通 RAG 很难把这条链串起来，Graph RAG 可以。

```python
TEST_DOCUMENTS = [
    {"id": "doc1", "title": "技术部人事",
     "content": "张三是技术部总监，2023年加入公司，负责推动公司的AI项目落地。他直接向CTO王五汇报。技术部目前有研发、测试、运维三个小组。"},
    {"id": "doc2", "title": "AI项目进展",
     "content": "公司AI项目于2024年Q2正式启动，由技术部主导。项目采用大模型+RAG技术路线，目标是搭建企业级智能问答系统。项目在Q3获得了公司年度创新奖。"},
    {"id": "doc3", "title": "创新奖评审",
     "content": "公司年度创新奖每年Q4评选，评审委员会由各部门总监组成。李四是评审委员会主席，负责组织评审会议和最终结果公示。今年共有5个项目入围。"},
    {"id": "doc4", "title": "组织架构",
     "content": "王五是公司CTO，分管技术部和产品部。产品部总监赵六与张三在AI项目上紧密合作，负责需求定义和产品规划。公司采用扁平化管理，总监直接向CTO汇报。"},
    {"id": "doc5", "title": "项目协作",
     "content": "AI项目团队由技术部张三带队，产品部赵六担任产品经理。项目使用了OpenAI API和ChromaDB向量数据库。创新奖颁奖典礼上，李四为张三颁发了奖杯。"},
]
```

实体链一目了然：张三负责 AI 项目 → AI 项目获得创新奖 → 李四主持创新奖评审 → 李四给张三颁奖。问「张三和李四什么关系」，就得靠这条链来答。

---

## 一、普通 RAG 哪里不够用？

咱们先看三个普通 RAG 搞不定的场景。

**局限一：多跳推理**

答案分散在多个文档片段里，需要「推理链」才能拼出来。比如：

- 文档 1：张三是技术部总监，负责推动 AI 项目落地
- 文档 5：AI 项目在 Q3 获得了公司创新奖
- 文档 12：创新奖的评审委员会主席是李四

问题：「张三的工作和李四有什么关联？」

普通 RAG 只会检索到文档 1（和张三最相关），根本看不到李四那条线。它缺的不是「信息」，而是「连接」。

**局限二：全局总结**

「公司今年做了哪些大事？」这种问题，需要跨大量文档做汇总。向量检索只能给你一堆相似度高的片段，没法从整体视角做归纳。你塞再多 chunk 进 prompt，模型也是在「盲人摸象」。

**局限三：关系查询**

「A 和 B 有什么关系？」——这种问题天然需要理解实体间的连接。向量检索看的是「这段话和问题像不像」，不看「A 和 B 有没有边」。所以关系类问题，普通 RAG 基本靠蒙。

**根本原因**：向量检索只看「语义相似度」，不看「逻辑关联度」。它不知道张三、AI 项目、创新奖、李四是一条链上的。

打个比方：普通 RAG 像在图书馆里按「书名相似」找书，你要的是「和《三体》有续集关系的书」，它只会给你一堆名字里带「体」字的，不会顺着「三体→三体2→三体3」这条线找。Graph RAG 相当于先画一张「书与书的关系图」，再按图索骥。

> **正文配图 1 提示词：**
> A comparison illustration split vertically. Left side: a traditional search showing scattered document fragments/cards floating randomly, with a search beam only catching nearby fragments (missing connections). Right side: the same fragments now connected by visible relationship lines forming a graph/network structure, with a search beam following the connection lines to reach distant but related information. Left side looks flat and disconnected, right side looks rich and interconnected. Clean flat design. No text. Aspect ratio 16:9.

---

## 二、Graph RAG 的核心思路

Graph RAG 多了一层：**先把文档里的实体和关系抽出来，建成知识图谱，再和图一起做检索。**

整体流程可以概括成：

```
文档 → 实体抽取 → 关系抽取 → 构建知识图谱 → 图+向量联合检索 → 生成回答
```

知识图谱的基础概念就三个：

| 概念 | 含义 | 例子 |
|------|------|------|
| **实体（Entity）** | 人名、组织、项目、概念等 | 张三、AI项目、技术部 |
| **关系（Relation）** | 实体之间的连接 | 负责、参与、获得、组织 |
| **三元组（Triple）** | (实体1, 关系, 实体2) | (张三, 负责, AI项目) |

和普通 RAG 的区别：多了一层「关系」信息。检索时不仅能找到「像」的文档，还能沿着图的边找到「连」的实体。

> **正文配图 2 提示词：**
> A step-by-step process illustration showing three stages from left to right. Stage 1: Raw text documents being scanned by a magnifying glass, with highlighted entities (people, projects, organizations) popping out. Stage 2: The extracted entities become nodes, and relationships become edges, forming a small graph network being assembled like puzzle pieces. Stage 3: The complete graph with colored community clusters (groups of closely connected nodes in blue, green, orange bubbles). Arrows connect the stages. Clean flat design, white background. No text. Aspect ratio 16:9.

---

## 三、第一步：用 LLM 做实体和关系抽取

实体和关系怎么抽？传统做法是用 NER（命名实体识别）+ 关系分类模型，要标注数据、要训练，成本高。现在直接用 LLM 就行：给一段文本，让它输出三元组列表，格式统一成 JSON，方便后续处理。零样本就能用，换领域只要改改 prompt。

Prompt 设计决定了抽取质量。实体类型、关系类型写清楚，模型抽得才准。下面这个可以直接用：

```python
EXTRACT_PROMPT = """从以下文本中提取实体和关系。

规则：
1. 实体包括：人名、组织、项目、技术、产品、事件、职位
2. 关系包括：担任、负责、参与、使用、创建、合作、包含、汇报、获得、组织、颁发
3. 输出三元组格式，每个三元组 (实体1, 关系, 实体2)
4. 只输出 JSON 数组，不要其他内容

文本：
{text}

输出格式示例：
[{"entity1": "张三", "relation": "担任", "entity2": "技术部总监"}, {"entity1": "张三", "relation": "负责", "entity2": "AI项目"}]
"""

def extract_triples(text_chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text_chunk)}]
    )
    return json.loads(response.choices[0].message.content.strip())
```

**示例**：

输入：「张三是技术部总监，负责推动公司的 AI 项目落地。」

输出：
```json
[
  {"entity1": "张三", "relation": "担任", "entity2": "技术部总监"},
  {"entity1": "张三", "relation": "负责", "entity2": "AI项目"},
  {"entity1": "AI项目", "relation": "属于", "entity2": "公司"}
]
```

每个文档块都抽一遍，合并去重，就得到整份文档的三元组集合。

**跑一遍看看**（对 5 段测试文档做抽取）：

```
【第一步】实体和关系抽取...
  技术部人事: 抽取 4 个三元组
  AI项目进展: 抽取 3 个三元组
  创新奖评审: 抽取 3 个三元组
  组织架构: 抽取 4 个三元组
  项目协作: 抽取 5 个三元组

  共 19 个唯一三元组
```

抽出来的典型三元组包括：`(张三, 负责, AI项目)`、`(AI项目, 获得, 创新奖)`、`(李四, 担任, 评审委员会主席)`、`(李四, 颁发, 奖杯)` 等。有了这些，图就能建起来了。

---

## 四、第二步：构建知识图谱

有了三元组，用 NetworkX 建图就行。节点是实体，边是关系，边上可以挂 `relation` 属性。

```python
import networkx as nx
from community import community_louvain

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_triples(self, triples):
        for t in triples:
            self.graph.add_node(t["entity1"], type="entity")
            self.graph.add_node(t["entity2"], type="entity")
            self.graph.add_edge(t["entity1"], t["entity2"], relation=t["relation"])
        return self

    def detect_communities(self):
        """社区检测：把紧密关联的实体分组"""
        partition = community_louvain.best_partition(self.graph)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        return communities

    def get_subgraph(self, entity, hops=2):
        """获取某个实体的 N 跳子图"""
        if entity not in self.graph:
            return self.graph.subgraph([])
        nodes = {entity}
        current = {entity}
        for _ in range(hops):
            next_nodes = set()
            for n in current:
                next_nodes.update(self.graph.neighbors(n))
            nodes.update(next_nodes)
            current = next_nodes
        return self.graph.subgraph(nodes)
```

**社区检测**：用 Louvain 算法把关联紧密的节点聚成「社区」。后面做全局总结时，可以对每个社区生成摘要，再汇总，这是微软 GraphRAG 里 Global Search 的基础。

**`get_subgraph`**：从某个实体出发，取 N 跳内的邻居，得到一个小子图。检索时就用这个子图里的边作为「关系上下文」。

**跑一遍看看**：

```
【第二步】构建知识图谱...
  节点数: 12
  边数: 19
  社区数: 3
```

比如从「张三」出发 2 跳，能拿到：张三 ↔ 技术部总监、AI项目、王五、赵六；AI项目 ↔ 创新奖；创新奖 ↔ 李四。问「张三和李四什么关系」时，子图里已经包含这条路径了。

---

## 五、第三步：图 + 向量联合检索

检索流程分五步：

1. 用 LLM 从用户问题里提取关键实体
2. 在知识图谱里取这些实体的 N 跳子图
3. 同时做向量相似度检索
4. 把图信息和文档片段合并成上下文
5. 塞进 prompt 让模型生成回答

两种检索互补：向量找「语义相似」，图找「逻辑关联」。

```python
def extract_entities_from_question(question):
    """从问题中提取实体"""
    prompt = f"""从用户问题中提取可能的关键实体（人名、项目名、组织名等）。
问题：{question}
只输出 JSON 数组，如 ["张三", "李四"]。如果没有明显实体，输出 []。"""
    response = ask_llm(prompt)
    return json.loads(response)

class GraphRAG:
    def __init__(self, knowledge_graph, collection):
        self.kg = knowledge_graph
        self.collection = collection

    def query(self, question, top_k=3):
        # 1. 从问题中提取实体
        entities = extract_entities_from_question(question)

        # 2. 图检索：获取相关实体的子图
        graph_context = []
        for entity in entities[:3]:
            subgraph = self.kg.get_subgraph(entity, hops=2)
            if subgraph.number_of_edges() > 0:
                triples = [(u, d["relation"], v) for u, v, d in subgraph.edges(data=True)]
                graph_context.append(
                    f"关于 {entity} 的关系：\n" +
                    "\n".join([f"  {t[0]} --{t[1]}--> {t[2]}" for t in triples])
                )

        # 3. 向量检索
        query_embedding = get_embedding(question)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        vector_docs = vector_results["documents"][0]

        # 4. 合并上下文
        context = "【知识图谱信息】\n" + "\n\n".join(graph_context)
        context += "\n\n【相关文档片段】\n" + "\n---\n".join(vector_docs)

        # 5. 生成回答
        return ask_llm(f"根据以下信息回答问题：\n{context}\n\n问题：{question}")
```

**模拟运行效果**：

```
问题："张三的工作和李四有什么关联？"

[图检索]
关于 张三 的关系：
  张三 --负责--> AI项目
  张三 --担任--> 技术部总监
  AI项目 --获得--> 创新奖

关于 李四 的关系：
  李四 --担任--> 创新奖评审主席
  创新奖 --颁发给--> AI项目

[向量检索]
  "张三是技术部总监，负责推动 AI 项目落地……"
  "李四是评审委员会主席，负责组织评审会议……"

AI 回答：张三负责的 AI 项目获得了公司创新奖，而李四是创新奖评审委员会主席。
所以张三的工作成果（AI项目）由李四所在的评审委员会评定。
```

完整可运行代码在 `graph_rag_demo.py`，包含 5 段测试文档（公司组织/人事/项目，实体之间有链式关系），直接 `python graph_rag_demo.py` 即可跑通全流程。

---

## 六、微软 GraphRAG 的进阶思路

咱们上面实现的是「简化版」：图检索 + 向量检索，适合具体问题。微软开源的 GraphRAG 在「图 + 向量」基础上，多做了一层：**社区摘要 + 双模式检索**，专门解决「全局总结」类问题。

**两种查询模式**：

| 模式 | 适用场景 | 做法 |
|------|----------|------|
| **Local Search** | 具体问题（某人、某项目） | 从实体出发，遍历子图，和咱们上面实现的类似 |
| **Global Search** | 全局总结（公司大事、主题概览） | 从社区摘要出发，用 Map-Reduce 汇总 |

**社区摘要**：对每个社区里的实体和关系，用 LLM 生成一段概述。比如「技术部-产品部-AI项目」这个社区，摘要可能是「技术部与产品部合作推进 AI 项目，项目获创新奖」。

**Map-Reduce**：全局问题来了，先让每个社区根据摘要各自回答，再把各社区的回答汇总成最终答案。这样不用把整张图塞进 prompt，也能做「公司今年做了哪些大事」这类问题。

> **正文配图 3 提示词：**
> A dual-mode search illustration. Top half: "Local Search" - a spotlight/flashlight beam from a specific node illuminating nearby connected nodes in a graph, showing detailed local structure. Bottom half: "Global Search" - a satellite/bird's eye view of the entire graph showing colored community clusters, with a wide-angle lens capturing the big picture. Both halves share the same underlying graph structure but viewed differently. Clean flat design, dark background for contrast. No text. Aspect ratio 16:9.

---

## 七、Graph RAG vs 普通 RAG 对比

| 维度 | 普通 RAG | Graph RAG |
|------|----------|-----------|
| 检索依据 | 语义相似度 | 语义相似度 + 实体关系 |
| 多跳推理 | 不支持 | 支持（沿图路径推理） |
| 全局总结 | 很弱 | 社区摘要 + Map-Reduce |
| 关系查询 | 不支持 | 原生支持 |
| 构建成本 | 低（切块 + Embedding） | 高（实体抽取 + 图构建） |
| 适合数据 | 所有文本 | 实体关系丰富的数据 |
| 典型场景 | 知识问答、文档搜索 | 企业知识管理、人物关系分析 |

---

## 八、写在最后

Graph RAG 让检索「看到了知识之间的关系」，但不管用哪种 RAG 方案，最终都得回答一个问题——**你的 RAG 到底好不好？** 下一篇也是系列收官篇，咱们聊 RAG 评估。

最后说一句：Graph RAG 不是万能的。如果你的数据本身就是扁平的（比如一堆产品说明书），没有什么实体关系，普通 RAG 就够用了。选技术方案得看数据特性，不是越复杂越好。

> **正文配图 4 提示词：**
> An infographic showing a spectrum of RAG complexity from left to right. Left: Simple documents (manuals, FAQs) with a basic search icon → Regular RAG is sufficient (green check). Middle: Semi-structured data (reports, articles) with moderate connections → Hybrid approach (yellow neutral). Right: Highly interconnected data (org charts, research papers, legal cases) with dense entity relationships → Graph RAG shines (star icon). A gentle upward curve shows increasing complexity vs benefit. Clean infographic style, gradient background. No text. Aspect ratio 16:9.

---

## 附录：完整演示代码

`graph_rag_demo.py` 包含：

- **测试文档**：5 段公司组织/人事/项目相关文本，实体链：张三 → AI项目 → 创新奖 → 李四
- **实体抽取**：`extract_triples` + `EXTRACT_PROMPT`
- **知识图谱**：`KnowledgeGraph`（`add_triples` / `detect_communities` / `get_subgraph`）
- **Graph RAG**：`GraphRAG` 类，图检索 + 向量检索 + 合并生成

运行前设置环境变量 `OPENAI_API_KEY`，然后：

```bash
pip install openai chromadb networkx python-louvain
python graph_rag_demo.py
```
