# Graph RAG：让 AI 理解知识之间的关系

## 封面图提示词（2:1 比例，1232×616px）

> A tech illustration showing a knowledge graph floating in space. Multiple glowing nodes (circles) of different sizes connected by labeled edges/lines forming a web-like structure. Some nodes cluster into communities (highlighted with translucent colored bubbles - blue, green, orange). A robot stands on one node, using a telescope to look across the graph, with a search beam illuminating a path through connected nodes. Small document icons feed into the graph from the left side. Background: deep space/dark navy with subtle constellation patterns. No text, no watermark. Aspect ratio 2:1.

## 一句话定位

当普通 RAG 只能找到"相似的文档片段"时，Graph RAG 能找到"有关联的知识网络"——用实体抽取和知识图谱让检索理解知识之间的关系。

---

## 章节大纲

### 开头引入

- 切入点：你问 RAG"张三是谁？"，它能找到包含张三的段落。但你问"张三和李四是什么关系？"，它就傻了——因为张三的信息在第 3 段，李四的信息在第 15 段，普通向量检索根本不会把它们关联起来
- 核心问题：知识不是孤立的碎片，而是一张网。怎么让 RAG 理解实体之间的关系？
- 本文思路：从普通 RAG 的局限说起，一步步搭到 Graph RAG

### 一、普通 RAG 哪里不够用？

**要点：**
- 局限一：多跳推理——答案分散在多个文档片段中，需要"推理链"才能回答
- 局限二：全局总结——"公司今年做了哪些大事"这种需要跨大量文档总结的问题
- 局限三：关系查询——"A 和 B 有什么关系"，需要理解实体间的连接
- 根本原因：向量检索只看"语义相似度"，不看"逻辑关联度"

**反面案例：**
```
文档 1：张三是技术部总监，负责推动 AI 项目落地。
文档 5：AI 项目在 Q3 获得了公司创新奖。
文档 12：创新奖的评审委员会主席是李四。

问题："张三的工作和李四有什么关联？"
普通 RAG：只检索到文档 1（和张三最相关），回答不了这个问题 ❌
Graph RAG：张三 → AI 项目 → 创新奖 → 李四，链路清晰 ✅
```

> **正文配图 1 提示词：**
> A comparison illustration split vertically. Left side: a traditional search showing scattered document fragments/cards floating randomly, with a search beam only catching nearby fragments (missing connections). Right side: the same fragments now connected by visible relationship lines forming a graph/network structure, with a search beam following the connection lines to reach distant but related information. Left side looks flat and disconnected, right side looks rich and interconnected. Clean flat design. No text. Aspect ratio 16:9.

### 二、Graph RAG 的核心思路

**要点：**
- 整体流程：文档 → 实体抽取 → 关系抽取 → 构建知识图谱 → 图+向量联合检索 → 生成回答
- 知识图谱基础概念：
  - **实体（Entity）**：人名、地名、项目名、概念……
  - **关系（Relation）**：属于、负责、合作、创建……
  - **三元组（Triple）**：(张三, 负责, AI项目)
- 和普通 RAG 的关键区别：多了一层"关系"信息

### 三、第一步：用 LLM 做实体和关系抽取

**要点：**
- 用 LLM 从每个文档块中提取实体和关系
- 输出格式：三元组列表
- 关键：prompt 的设计决定了抽取质量

**核心代码片段：**

```python
EXTRACT_PROMPT = """从以下文本中提取实体和关系。

规则：
1. 实体包括：人名、组织、项目、技术、产品、事件
2. 关系包括：属于、负责、参与、使用、创建、合作、包含
3. 输出三元组格式：(实体1, 关系, 实体2)

文本：
{text}

请输出 JSON 格式：
[{{"entity1": "...", "relation": "...", "entity2": "..."}}]
"""

def extract_triples(text_chunk):
    response = ask_llm(EXTRACT_PROMPT.format(text=text_chunk))
    return json.loads(response)

# 示例输出
# 输入："张三是技术部总监，负责推动公司的 AI 项目落地。"
# 输出：[
#   {"entity1": "张三", "relation": "担任", "entity2": "技术部总监"},
#   {"entity1": "张三", "relation": "负责", "entity2": "AI项目"},
#   {"entity1": "AI项目", "relation": "属于", "entity2": "公司"}
# ]
```

### 四、第二步：构建知识图谱

**要点：**
- 用 NetworkX 构建图结构（轻量、纯 Python、适合原型）
- 节点 = 实体，边 = 关系
- 社区检测：用 Louvain 算法把关联紧密的实体聚类成"社区"
- 社区摘要：对每个社区生成一段概述（用于全局总结类问题）

**核心代码片段：**

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
            self.graph.add_edge(t["entity1"], t["entity2"],
                               relation=t["relation"])

    def detect_communities(self):
        """社区检测：把紧密关联的实体分组"""
        partition = community_louvain.best_partition(self.graph)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        return communities

    def get_subgraph(self, entity, hops=2):
        """获取某个实体的 N 跳子图"""
        nodes = set()
        current = {entity}
        for _ in range(hops):
            next_nodes = set()
            for n in current:
                next_nodes.update(self.graph.neighbors(n))
            nodes.update(current)
            current = next_nodes - nodes
        nodes.update(current)
        return self.graph.subgraph(nodes)
```

> **正文配图 2 提示词：**
> A step-by-step process illustration showing three stages from left to right. Stage 1: Raw text documents being scanned by a magnifying glass, with highlighted entities (people, projects, organizations) popping out. Stage 2: The extracted entities become nodes, and relationships become edges, forming a small graph network being assembled like puzzle pieces. Stage 3: The complete graph with colored community clusters (groups of closely connected nodes in blue, green, orange bubbles). Arrows connect the stages. Clean flat design, white background. No text. Aspect ratio 16:9.

### 五、第三步：图 + 向量联合检索

**要点：**
- 检索流程：
  1. 先用 LLM 从用户问题中提取关键实体
  2. 在知识图谱中找到这些实体的 N 跳子图
  3. 同时做向量相似度检索
  4. 合并两种检索结果作为上下文
- 两种检索互补：向量找"语义相似"，图找"逻辑关联"

**核心代码片段：**

```python
class GraphRAG:
    def __init__(self, knowledge_graph, vector_store):
        self.kg = knowledge_graph
        self.vs = vector_store

    def query(self, question):
        # 1. 从问题中提取实体
        entities = extract_entities_from_question(question)

        # 2. 图检索：获取相关实体的子图
        graph_context = []
        for entity in entities:
            subgraph = self.kg.get_subgraph(entity, hops=2)
            triples = [(u, d["relation"], v)
                       for u, v, d in subgraph.edges(data=True)]
            graph_context.append(
                f"关于 {entity} 的关系：\n" +
                "\n".join([f"  {t[0]} --{t[1]}--> {t[2]}" for t in triples])
            )

        # 3. 向量检索：获取相似文档片段
        vector_results = self.vs.search(question, top_k=3)

        # 4. 合并上下文
        context = "【知识图谱信息】\n" + "\n".join(graph_context)
        context += "\n\n【相关文档片段】\n" + "\n---\n".join(vector_results)

        # 5. 生成回答
        return ask_llm(f"根据以下信息回答问题：\n{context}\n\n问题：{question}")
```

**模拟运行效果：**
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

AI 回答：张三负责的 AI 项目获得了公司创新奖，而李四是创新奖评审委员会主席。
所以张三的工作成果（AI项目）由李四所在的评审委员会评定。 ✅
```

### 六、微软 GraphRAG 的进阶思路

**要点：**
- 微软开源的 GraphRAG 论文核心思路
- 两种查询模式：
  - **Local Search**：从实体出发，遍历子图，适合具体问题
  - **Global Search**：从社区摘要出发，适合全局总结问题
- 社区摘要的生成：对每个社区的实体和关系用 LLM 生成概述
- Map-Reduce 汇总：全局问题 → 每个社区各自回答 → 汇总

> **正文配图 3 提示词：**
> A dual-mode search illustration. Top half: "Local Search" - a spotlight/flashlight beam from a specific node illuminating nearby connected nodes in a graph, showing detailed local structure. Bottom half: "Global Search" - a satellite/bird's eye view of the entire graph showing colored community clusters, with a wide-angle lens capturing the big picture. Both halves share the same underlying graph structure but viewed differently. Clean flat design, dark background for contrast. No text. Aspect ratio 16:9.

### 七、Graph RAG vs 普通 RAG 对比

```
| 维度         | 普通 RAG            | Graph RAG              |
|-------------|--------------------|-----------------------|
| 检索依据     | 语义相似度           | 语义相似度 + 实体关系    |
| 多跳推理     | 不支持              | 支持（沿图路径推理）     |
| 全局总结     | 很弱                | 社区摘要 + Map-Reduce   |
| 关系查询     | 不支持              | 原生支持               |
| 构建成本     | 低（切块+Embedding） | 高（实体抽取+图构建）    |
| 适合数据     | 所有文本             | 实体关系丰富的数据       |
| 典型场景     | 知识问答、文档搜索    | 企业知识管理、人物关系分析 |
```

### 写在最后

- 衔接语：Graph RAG 让检索"看到了知识之间的关系"，但不管用哪种 RAG 方案，最终都得回答一个问题——"你的 RAG 到底好不好？"。下一篇也是系列收官篇，我们来聊 RAG 评估
- 关键观点：Graph RAG 不是万能的。如果你的数据本身就是扁平的（比如一堆产品说明书），没有什么实体关系，那普通 RAG 就足够了。选技术方案得看数据特性，不是越复杂越好

> **正文配图 4 提示词：**
> An infographic showing a spectrum of RAG complexity from left to right. Left: Simple documents (manuals, FAQs) with a basic search icon → Regular RAG is sufficient (green check). Middle: Semi-structured data (reports, articles) with moderate connections → Hybrid approach (yellow neutral). Right: Highly interconnected data (org charts, research papers, legal cases) with dense entity relationships → Graph RAG shines (star icon). A gentle upward curve shows increasing complexity vs benefit. Clean infographic style, gradient background. No text. Aspect ratio 16:9.
