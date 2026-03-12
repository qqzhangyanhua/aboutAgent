"""
智能体记忆系统：从短期到长期
完整可运行示例 —— 对应文章第 05 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python memory_system.py
"""
import os
import time
from openai import OpenAI
import chromadb

client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
MODEL_NAME = "deepseek-chat"

# ==================== 滑动窗口 ====================
def sliding_window_memory(messages: list[dict], max_turns: int = 10) -> list[dict]:
    system_msg = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    if len(conversation) > max_turns * 2:
        conversation = conversation[-(max_turns * 2):]
    return system_msg + conversation

# ==================== 摘要压缩 ====================
class SummaryMemory:
    def __init__(self, max_recent: int = 6, summarize_threshold: int = 12):
        self.messages: list[dict] = []
        self.summary = ""
        self.max_recent = max_recent
        self.summarize_threshold = summarize_threshold

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.summarize_threshold:
            self._compress()

    def _compress(self) -> None:
        old_msgs = self.messages[:-self.max_recent]
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_msgs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"请将以下对话历史压缩成简洁摘要，保留关键信息（姓名、偏好、需求、待办），控制在200字以内：\n\n{old_text}"}],
            temperature=0
        )
        self.summary = response.choices[0].message.content
        self.messages = self.messages[-self.max_recent:]

    def get_messages(self, system_prompt: str) -> list[dict]:
        context = f"{system_prompt}\n\n【之前的对话摘要】{self.summary}" if self.summary else system_prompt
        return [{"role": "system", "content": context}] + self.messages

# ==================== 长期向量记忆 ====================
class LongTermMemory:
    def __init__(self, user_id: str):
        chroma = chromadb.Client()
        self.collection = chroma.get_or_create_collection(f"user_{user_id}", metadata={"hnsw:space": "cosine"})
        self._counter = 0

    def save_memory(self, content: str, memory_type: str = "preference") -> None:
        self._counter += 1
        self.collection.add(
            ids=[f"mem_{self._counter}"],
            documents=[content],
            metadatas=[{"type": memory_type, "timestamp": str(time.time())}]
        )

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(top_k, self.collection.count()))
        return results["documents"][0] if results["documents"] else []

    def build_context(self, query: str, system_prompt: str) -> str:
        memories = self.recall(query)
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            return f"{system_prompt}\n\n【关于这位用户的记忆】\n{memory_text}"
        return system_prompt

# ==================== 工作记忆 ====================
class WorkingMemory:
    def __init__(self):
        self.scratchpad = "（空）"

    def update_from_response(self, response_text: str) -> str:
        if "[SCRATCHPAD]" in response_text:
            parts = response_text.split("[SCRATCHPAD]")
            self.scratchpad = parts[1].strip()
            return parts[0].strip()
        return response_text

# ==================== 整合 MemoryManager ====================
class MemoryManager:
    def __init__(self, user_id: str):
        self.summary_mem = SummaryMemory(max_recent=6, summarize_threshold=12)
        self.long_term = LongTermMemory(user_id)
        self.working = WorkingMemory()

    def get_messages(self, user_query: str, base_system_prompt: str) -> list[dict]:
        prompt = self.long_term.build_context(user_query, base_system_prompt)
        scratchpad_section = f"你有一块草稿纸，当前内容：{self.working.scratchpad}"
        full_system = f"{scratchpad_section}\n\n{prompt}"
        return self.summary_mem.get_messages(full_system)

    def add_turn(self, user_msg: str, ai_response: str) -> str:
        self.summary_mem.add("user", user_msg)
        clean = self.working.update_from_response(ai_response)
        self.summary_mem.add("assistant", clean)
        # 提取关键信息存入长期记忆
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f'从以下用户消息中提取值得长期记住的信息（姓名、偏好、重要决定）。若无则输出"无"。\n\n用户消息：{user_msg}'}],
            temperature=0
        )
        extracted = resp.choices[0].message.content.strip()
        if extracted and extracted != "无" and len(extracted) > 2:
            self.long_term.save_memory(extracted)
        return clean

# ==================== 演示 ====================
if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    # Demo 1: 滑动窗口
    print("=" * 50)
    print("Demo 1：滑动窗口")
    print("=" * 50)
    msgs = [{"role": "system", "content": "你是助手"}]
    for i in range(15):
        msgs.append({"role": "user", "content": f"第{i+1}轮：{'我喜欢川菜' if i == 0 else '继续聊'}"})
        msgs.append({"role": "assistant", "content": f"第{i+1}轮回复"})
    trimmed = sliding_window_memory(msgs, max_turns=10)
    user_msgs = [m["content"] for m in trimmed if m["role"] == "user"]
    print(f"原始 15 轮，保留 {len(user_msgs)} 轮")
    print(f"最早保留：{user_msgs[0]}")

    # Demo 2: 工作记忆
    print("\n" + "=" * 50)
    print("Demo 2：工作记忆（Scratchpad）")
    print("=" * 50)
    wm = WorkingMemory()
    fake = "先算 3+5=8，再算 8*2=16。答案是 16。\n[SCRATCHPAD]\n已算出 3+5=8, 8*2=16, 答案 16"
    clean = wm.update_from_response(fake)
    print(f"给用户看的：{clean}")
    print(f"草稿纸：{wm.scratchpad}")

    # Demo 3: 完整记忆系统对话
    print("\n" + "=" * 50)
    print("Demo 3：完整记忆系统")
    print("=" * 50)
    manager = MemoryManager(user_id="demo_user")
    base_prompt = "你是一个贴心助手。"

    for user_msg in ["我叫阿杰，是 Python 开发，最近在学 Go", "Go 的并发和 Python 有啥区别？"]:
        msgs = manager.get_messages(user_msg, base_prompt)
        msgs.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(model=MODEL_NAME, messages=msgs, temperature=0.7)
        ai_text = resp.choices[0].message.content
        clean = manager.add_turn(user_msg, ai_text)
        print(f"\n用户：{user_msg}")
        print(f"助手：{clean[:200]}...")

    # 模拟跨会话
    manager2 = MemoryManager(user_id="demo_user")
    ctx = manager2.long_term.build_context("推荐学习资料", base_prompt)
    print(f"\n跨会话回忆：\n{ctx[:300]}")
