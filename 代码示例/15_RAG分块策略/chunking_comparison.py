"""
RAG 分块策略全攻略：5 种分块方法对比
完整可运行示例 —— 对应文章第 15 篇

运行（纯分块对比，无需 API key）：
    python chunking_comparison.py

运行（含检索质量对比，需 API key）：
    export DEEPSEEK_API_KEY="your_key"
    python chunking_comparison.py
"""
import os
import re
import chromadb

# ==================== 测试文档 ====================

TEST_DOC = """
Python Web 框架对比（2024版）

一、Django
Django 是 Python 生态里最成熟的全栈框架，自带 ORM、Admin、认证系统。
适合中大型项目，学习曲线稍陡，但文档完善。
安装命令：pip install django

二、Flask
Flask 是轻量级微框架，核心只有路由和模板，其他功能靠扩展。
适合小型项目和 API 服务，灵活度高。
安装命令：pip install flask

三、FastAPI
FastAPI 基于类型注解，自动生成 OpenAPI 文档，性能接近 Go。
适合需要高性能 API 的场景，异步支持完善。
安装命令：pip install fastapi uvicorn

四、选型建议
- 要快速上线：选 Django
- 要灵活可控：选 Flask
- 要高性能 API：选 FastAPI
"""


# ==================== 策略一：固定大小切块 ====================

def fixed_chunk(text: str, chunk_size: int = 200) -> list[str]:
    """固定大小切块：到字数就切"""
    text = text.strip()
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ==================== 策略二：固定大小 + 重叠 ====================

def overlap_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """带重叠的固定大小切块"""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ==================== 策略三：段落切块 ====================

def paragraph_chunk(text: str, min_length: int = 50) -> list[str]:
    """按段落分割（双换行）"""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < min_length * 3:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current.strip())
            current = p
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ==================== 策略四：按标题分块 ====================

def heading_chunk(text: str) -> list[str]:
    """按标题（一、二、三...）分块"""
    sections = re.split(r'(?=^[一二三四五六七八九十]+、)', text.strip(), flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if section and len(section) > 10:
            chunks.append(section)
    return chunks


# ==================== 策略五：句子级切块 ====================

def sentence_chunk(text: str, max_sentences: int = 3) -> list[str]:
    """按句子分割，每 N 句一块"""
    sentences = re.split(r'(?<=[。！？\n])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ==================== 对比展示 ====================

def compare_strategies(doc: str) -> None:
    """对比所有分块策略"""
    strategies = {
        "固定大小(200字)": fixed_chunk(doc, 200),
        "固定+重叠(200/50)": overlap_chunk(doc, 200, 50),
        "段落分块": paragraph_chunk(doc),
        "标题分块": heading_chunk(doc),
        "句子分块(3句)": sentence_chunk(doc, 3),
    }

    print("=" * 60)
    print("5 种分块策略对比")
    print("=" * 60)

    for name, chunks in strategies.items():
        print(f"\n--- {name} ---")
        print(f"块数：{len(chunks)}")
        for i, c in enumerate(chunks):
            preview = c[:80].replace("\n", " ")
            print(f"  块{i+1} ({len(c)}字): {preview}{'...' if len(c) > 80 else ''}")


def retrieval_comparison(doc: str) -> None:
    """检索质量对比（需要 chromadb）"""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n跳过检索质量对比（需要 DEEPSEEK_API_KEY）")
        return

    llm_client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
    chroma = chromadb.Client()

    strategies = {
        "固定大小": fixed_chunk(doc, 200),
        "标题分块": heading_chunk(doc),
        "段落分块": paragraph_chunk(doc),
    }

    test_queries = [
        "FastAPI 怎么安装？",
        "哪个框架适合做 API 服务？",
        "Django 的特点是什么？",
    ]

    print("\n" + "=" * 60)
    print("检索质量对比")
    print("=" * 60)

    for strategy_name, chunks in strategies.items():
        col_name = f"test_{strategy_name}"
        col = chroma.get_or_create_collection(col_name)
        ids = [f"c_{i}" for i in range(len(chunks))]
        existing = col.get()["ids"]
        new = [(id_, chunk) for id_, chunk in zip(ids, chunks) if id_ not in existing]
        if new:
            col.add(ids=[x[0] for x in new], documents=[x[1] for x in new])

        print(f"\n--- {strategy_name} ({len(chunks)} 块) ---")
        for query in test_queries:
            results = col.query(query_texts=[query], n_results=1)
            if results["documents"] and results["documents"][0]:
                top_doc = results["documents"][0][0][:60].replace("\n", " ")
                dist = results["distances"][0][0] if results["distances"] else -1
                print(f"  Q: {query:20s} → [{dist:.3f}] {top_doc}...")


# ==================== 运行 ====================

if __name__ == "__main__":
    compare_strategies(TEST_DOC)
    retrieval_comparison(TEST_DOC)
