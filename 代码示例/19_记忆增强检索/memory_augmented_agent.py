"""
记忆增强检索：情景记忆 + 语义记忆 + 记忆感知 RAG
完整可运行示例 —— 对应文章第 19 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python memory_augmented_agent.py
"""
import os
import json
import time
import math
from openai import OpenAI
import chromadb

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"


# ==================== 情景记忆 ====================

class EpisodicMemory:
    """情景记忆：存储具体的对话事件"""

    def __init__(self, chroma_client: chromadb.Client):
        self.collection = chroma_client.get_or_create_collection(
            "episodic_memory", metadata={"hnsw:space": "cosine"}
        )
        self._counter = 0

    def store(self, user_msg: str, assistant_msg: str) -> None:
        text = f"用户：{user_msg}\n助手：{assistant_msg}"
        meta = {
            "timestamp": time.time(),
            "user_msg": user_msg[:200],
            "assistant_msg": assistant_msg[:200],
        }
        self._counter += 1
        self.collection.add(
            ids=[f"ep_{self._counter}"],
            documents=[text],
            metadatas=[meta]
        )

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        memories = []
        if results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                memories.append({"content": doc, "metadata": meta})
        return memories


# ==================== 语义记忆 ====================

class SemanticMemory:
    """语义记忆：存储从对话中提炼的用户认知"""

    def __init__(self, chroma_client: chromadb.Client):
        self.collection = chroma_client.get_or_create_collection(
            "semantic_memory", metadata={"hnsw:space": "cosine"}
        )
        self._counter = 0

    def extract_and_store(self, conversation: str) -> list[str]:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    "从以下对话中提取关于用户的关键事实和偏好。"
                    "每条事实独占一行，用 - 开头。"
                    "只提取明确表达的事实，不要推测。"
                    "如果没有值得提取的信息，返回"无"。\n\n"
                    f"对话内容：\n{conversation}"
                )
            }],
            temperature=0
        )
        text = response.choices[0].message.content
        if "无" in text and len(text) < 10:
            return []

        facts = [
            line.strip().lstrip("- ")
            for line in text.strip().split("\n")
            if line.strip().startswith("-")
        ]

        for fact in facts:
            if len(fact) < 3:
                continue
            self._counter += 1
            existing = self.collection.query(query_texts=[fact], n_results=1)
            if existing["distances"] and existing["distances"][0] and existing["distances"][0][0] < 0.3:
                continue
            self.collection.add(
                ids=[f"sem_{self._counter}"],
                documents=[fact],
                metadatas=[{"timestamp": time.time(), "source": "extraction"}]
            )
        return facts

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        if results["documents"]:
            return results["documents"][0]
        return []


# ==================== 记忆增强智能体 ====================

class MemoryAugmentedAgent:
    """记忆增强智能体：融合记忆与检索"""

    def __init__(self):
        chroma = chromadb.Client()
        self.episodic = EpisodicMemory(chroma)
        self.semantic = SemanticMemory(chroma)
        self.knowledge = chroma.get_or_create_collection("agent_knowledge")
        self._init_knowledge()
        self.session_history: list[dict] = []

    def _init_knowledge(self) -> None:
        docs = [
            "PostgreSQL 是强大的关系型数据库，asyncpg 驱动性能极高，适合 Python 异步后端。",
            "FastAPI + PostgreSQL + asyncpg 是 Python 高性能后端的经典黄金组合。",
            "Django 是全功能 Web 框架，自带 ORM 和 Admin，适合快速开发复杂应用。",
            "Redis 是内存缓存，Python 可用 aioredis 做异步缓存层。",
            "Docker 是容器化部署的标准工具，配合 Docker Compose 管理多服务。",
            "Kubernetes 用于大规模容器编排，适合微服务架构。",
            "pytest 是 Python 最流行的测试框架，支持异步测试和参数化。",
            "MySQL 是最流行的开源数据库，Java 和 PHP 生态使用最广。",
        ]
        ids = [f"k_{i}" for i in range(len(docs))]
        existing = self.knowledge.get()["ids"]
        new_items = [(id_, doc) for id_, doc in zip(ids, docs) if id_ not in existing]
        if new_items:
            self.knowledge.add(
                ids=[x[0] for x in new_items],
                documents=[x[1] for x in new_items]
            )

    def _retrieve_with_memory(self, query: str) -> list[str]:
        """记忆增强的检索"""
        # 1. 对话感知改写（解决指代）
        if self.session_history:
            recent = self.session_history[-6:]
            history_text = "\n".join(
                f"{m['role']}：{m['content']}" for m in recent if m["role"] != "system"
            )
            rewrite_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": (
                        "把以下问题改写成独立的完整查询，解决指代和省略。"
                        "直接返回改写结果。\n\n"
                        f"对话历史：\n{history_text}\n\n问题：{query}"
                    )
                }],
                temperature=0
            )
            contextualized = rewrite_response.choices[0].message.content.strip()
        else:
            contextualized = query

        # 2. 语义记忆增强（加入用户偏好）
        user_facts = self.semantic.recall(contextualized, top_k=3)
        if user_facts:
            facts_text = "；".join(user_facts)
            enhanced = f"{contextualized}（用户偏好：{facts_text}）"
        else:
            enhanced = contextualized

        # 3. 检索知识库
        results = self.knowledge.query(query_texts=[enhanced], n_results=3)

        # 4. 检索情景记忆
        episodes = self.episodic.recall(query, top_k=2)

        retrieved: list[str] = []
        if results["documents"]:
            for doc in results["documents"][0]:
                retrieved.append(f"[知识库] {doc}")
        for ep in episodes:
            retrieved.append(f"[历史对话] {ep['content']}")

        return retrieved

    def chat(self, user_input: str) -> str:
        context_docs = self._retrieve_with_memory(user_input)

        user_facts = self.semantic.recall(user_input, top_k=5)
        user_profile = ""
        if user_facts:
            user_profile = "已知用户信息：\n" + "\n".join(f"- {f}" for f in user_facts) + "\n\n"

        context_text = "\n\n".join(context_docs) if context_docs else "暂无参考资料"

        messages = [{
            "role": "system",
            "content": (
                "你是一个技术顾问。基于参考资料和用户的已知偏好回答问题。\n"
                "如果知道用户的偏好，回答时要考虑这些偏好。\n"
                "如果参考资料不足，明确告知。"
            )
        }]
        messages.extend(self.session_history[-6:])

        user_message = f"{user_profile}参考资料：\n{context_text}\n\n问题：{user_input}"
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.7
        )
        answer = response.choices[0].message.content

        self.session_history.append({"role": "user", "content": user_input})
        self.session_history.append({"role": "assistant", "content": answer})

        self.episodic.store(user_input, answer)
        self.semantic.extract_and_store(f"用户：{user_input}\n助手：{answer}")

        return answer


# ==================== 记忆生命周期工具 ====================

def decay_score(base_score: float, hours_ago: float, half_life: float = 168.0) -> float:
    """记忆衰减函数（半衰期默认 7 天）"""
    return base_score * math.exp(-0.693 * hours_ago / half_life)


def merge_memories(memories: list[str]) -> list[str]:
    """合并相似或矛盾的记忆"""
    if len(memories) <= 1:
        return memories
    memories_text = "\n".join(f"{i+1}. {m}" for i, m in enumerate(memories))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": (
                "以下是关于同一个用户的多条记忆。请合并处理：\n"
                "1. 合并相似的条目\n"
                "2. 如果有矛盾，保留最新的（编号大的更新）\n"
                "3. 去掉冗余信息\n"
                "每条独占一行，用 - 开头。\n\n"
                f"记忆条目：\n{memories_text}"
            )
        }],
        temperature=0
    )
    text = response.choices[0].message.content
    return [line.strip().lstrip("- ") for line in text.strip().split("\n") if line.strip().startswith("-")]


# ==================== 运行演示 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    print("=" * 60)
    print("Demo 1：记忆衰减")
    print("=" * 60)
    for days in [0, 1, 7, 30, 90]:
        score = decay_score(0.9, hours_ago=days * 24)
        print(f"  {days:3d} 天前的记忆：0.90 → {score:.3f}")

    print("\n" + "=" * 60)
    print("Demo 2：记忆合并")
    print("=" * 60)
    raw = [
        "用户是 Python 开发者",
        "用户主要用 Python 做后端开发",
        "用户喜欢用 FastAPI",
        "用户之前用过 Django 但觉得太重",
        "用户对 Java 有抵触",
        "用户不喜欢 Java 的冗长语法",
    ]
    merged = merge_memories(raw)
    print(f"合并前 {len(raw)} 条 → 合并后 {len(merged)} 条：")
    for m in merged:
        print(f"  - {m}")

    print("\n" + "=" * 60)
    print("Demo 3：记忆增强智能体对话")
    print("=" * 60)
    agent = MemoryAugmentedAgent()

    print("\n--- 第一轮：建立用户画像 ---")
    for msg in [
        "我是 Python 后端开发，主要用 FastAPI，对性能很敏感",
        "数据库方面我一直用 PostgreSQL，很满意",
        "千万别给我推荐 Java 的东西",
    ]:
        print(f"\n用户：{msg}")
        answer = agent.chat(msg)
        print(f"助手：{answer[:200]}...")

    print("\n--- 第二轮：验证记忆增强效果 ---")
    for msg in [
        "推荐一个缓存方案",
        "部署方面有什么建议？",
    ]:
        print(f"\n用户：{msg}")
        answer = agent.chat(msg)
        print(f"助手：{answer[:300]}...")
