#!/usr/bin/env python3
"""
Graph RAG 完整演示代码
依赖: pip install openai chromadb networkx python-louvain
"""

import os
import json
from openai import OpenAI
import chromadb
import networkx as nx

# python-louvain 包
try:
    import community.community_louvain as community_louvain
except ImportError:
    community_louvain = None

# ==================== 测试文档（实体之间有链式关系） ====================

TEST_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "技术部人事",
        "content": "张三是技术部总监，2023年加入公司，负责推动公司的AI项目落地。他直接向CTO王五汇报。技术部目前有研发、测试、运维三个小组。"
    },
    {
        "id": "doc2",
        "title": "AI项目进展",
        "content": "公司AI项目于2024年Q2正式启动，由技术部主导。项目采用大模型+RAG技术路线，目标是搭建企业级智能问答系统。项目在Q3获得了公司年度创新奖。"
    },
    {
        "id": "doc3",
        "title": "创新奖评审",
        "content": "公司年度创新奖每年Q4评选，评审委员会由各部门总监组成。李四是评审委员会主席，负责组织评审会议和最终结果公示。今年共有5个项目入围。"
    },
    {
        "id": "doc4",
        "title": "组织架构",
        "content": "王五是公司CTO，分管技术部和产品部。产品部总监赵六与张三在AI项目上紧密合作，负责需求定义和产品规划。公司采用扁平化管理，总监直接向CTO汇报。"
    },
    {
        "id": "doc5",
        "title": "项目协作",
        "content": "AI项目团队由技术部张三带队，产品部赵六担任产品经理。项目使用了OpenAI API和ChromaDB向量数据库。创新奖颁奖典礼上，李四为张三颁发了奖杯。"
    },
]

# ==================== 配置 ====================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"))


def get_embedding(text):
    """获取文本向量"""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def ask_llm(messages, model="gpt-4o-mini"):
    """调用 LLM"""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


# ==================== 第一步：实体和关系抽取 ====================

EXTRACT_PROMPT = """从以下文本中提取实体和关系。

规则：
1. 实体包括：人名、组织、项目、技术、产品、事件、职位
2. 关系包括：担任、负责、参与、使用、创建、合作、包含、汇报、获得、组织、颁发
3. 输出三元组格式，每个三元组 (实体1, 关系, 实体2)
4. 只输出 JSON 数组，不要其他内容

文本：
{text}

输出格式示例：
[{{"entity1": "张三", "relation": "担任", "entity2": "技术部总监"}}, {{"entity1": "张三", "relation": "负责", "entity2": "AI项目"}}]
"""


def extract_triples(text_chunk):
    """从文本块中抽取三元组"""
    prompt = EXTRACT_PROMPT.format(text=text_chunk)
    response = ask_llm(prompt)
    # 尝试解析 JSON
    try:
        # 处理可能的 markdown 代码块
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return []


# ==================== 第二步：构建知识图谱 ====================

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_triples(self, triples):
        for t in triples:
            if not isinstance(t, dict):
                continue
            e1 = t.get("entity1", "").strip()
            e2 = t.get("entity2", "").strip()
            rel = t.get("relation", "").strip()
            if e1 and e2:
                self.graph.add_node(e1, type="entity")
                self.graph.add_node(e2, type="entity")
                self.graph.add_edge(e1, e2, relation=rel)
        return self

    def detect_communities(self):
        """社区检测"""
        if community_louvain is None:
            return {}
        partition = community_louvain.best_partition(self.graph)
        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        return communities

    def get_subgraph(self, entity, hops=2):
        """获取实体的 N 跳子图"""
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


# ==================== 第三步：图 + 向量联合检索 ====================

ENTITY_EXTRACT_PROMPT = """从用户问题中提取可能的关键实体（人名、项目名、组织名等），用于知识图谱检索。

问题：{question}

只输出 JSON 数组，如 ["张三", "李四"]。如果没有明显实体，输出 []。"""


def extract_entities_from_question(question):
    """从问题中提取实体"""
    prompt = ENTITY_EXTRACT_PROMPT.format(question=question)
    response = ask_llm(prompt)
    try:
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        entities = json.loads(response.strip())
        return [e for e in entities if isinstance(e, str)] if entities else []
    except json.JSONDecodeError:
        return []


class GraphRAG:
    def __init__(self, knowledge_graph, vector_store, collection):
        self.kg = knowledge_graph
        self.collection = collection

    def query(self, question, top_k=3):
        # 1. 从问题中提取实体
        entities = extract_entities_from_question(question)

        # 2. 图检索
        graph_context = []
        for entity in entities[:3]:  # 最多取3个实体
            subgraph = self.kg.get_subgraph(entity, hops=2)
            if subgraph.number_of_edges() > 0:
                triples = [
                    (u, d.get("relation", ""), v)
                    for u, v, d in subgraph.edges(data=True)
                ]
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
        vector_docs = vector_results["documents"][0] if vector_results["documents"] else []

        # 4. 合并上下文
        context_parts = []
        if graph_context:
            context_parts.append("【知识图谱信息】\n" + "\n\n".join(graph_context))
        if vector_docs:
            context_parts.append("【相关文档片段】\n" + "\n---\n".join(vector_docs))

        context = "\n\n".join(context_parts)
        if not context:
            context = "（未找到相关信息）"

        # 5. 生成回答
        prompt = f"""根据以下信息回答问题。如果信息不足，请如实说明。

{context}

问题：{question}

请直接回答，不要编造。"""
        return ask_llm(prompt)


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("  Graph RAG 完整演示")
    print("=" * 60)

    # 1. 实体抽取
    print("\n【第一步】实体和关系抽取...")
    all_triples = []
    for doc in TEST_DOCUMENTS:
        triples = extract_triples(doc["content"])
        all_triples.extend(triples)
        print(f"  {doc['title']}: 抽取 {len(triples)} 个三元组")

    # 去重（简单按内容去重）
    seen = set()
    unique_triples = []
    for t in all_triples:
        key = (t.get("entity1"), t.get("relation"), t.get("entity2"))
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    print(f"\n  共 {len(unique_triples)} 个唯一三元组")

    # 2. 构建知识图谱
    print("\n【第二步】构建知识图谱...")
    kg = KnowledgeGraph()
    kg.add_triples(unique_triples)
    print(f"  节点数: {kg.graph.number_of_nodes()}")
    print(f"  边数: {kg.graph.number_of_edges()}")

    if community_louvain:
        communities = kg.detect_communities()
        print(f"  社区数: {len(communities)}")

    # 3. 建立向量索引
    print("\n【第三步】建立向量索引...")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="graph_rag_docs", metadata={"hnsw:space": "cosine"})
    for doc in TEST_DOCUMENTS:
        collection.add(
            ids=[doc["id"]],
            embeddings=[get_embedding(doc["content"])],
            documents=[doc["content"]],
            metadatas=[{"title": doc["title"]}]
        )
    print("  向量索引建立完成")

    # 4. Graph RAG 查询
    graph_rag = GraphRAG(kg, None, collection)

    test_questions = [
        "张三和李四是什么关系？",
        "AI项目是谁负责的？获得了什么奖？",
        "王五管哪些部门？",
    ]

    print("\n【第四步】Graph RAG 查询演示")
    print("-" * 60)
    for q in test_questions:
        print(f"\n问题: {q}")
        answer = graph_rag.query(q)
        print(f"回答: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
