# LangChain vs LlamaIndex vs 裸写 —— RAG 框架到底该不该用？

## 封面图提示词（2:1 比例，1232×616px）

> A tech illustration showing three distinct paths/roads diverging from a single starting point at the bottom. Left path: a complex highway interchange with many layers and exits, representing LangChain (complex but powerful). Middle path: a streamlined elevated railway on pillars going straight to a destination, representing LlamaIndex (specialized and direct). Right path: a simple dirt/hiking trail with a person walking freely, representing bare Python (flexible but manual). All three paths lead to the same destination building at the top labeled with a search/answer icon. Background: soft gradient sky. No text, no watermark. Aspect ratio 2:1.

## 一句话定位

同一个 RAG 任务，用裸写 Python、LangChain、LlamaIndex 三种方式分别实现，直观对比代码量、灵活性、学习成本，帮你做技术选型。

---

## 章节大纲

### 开头引入

- 切入点：前面几篇我们全是裸写的——50 行代码搞定 RAG，根本没碰任何框架。但面试或者技术方案里总有人问"你用的什么框架"，好像不用框架就不专业似的
- 核心问题：LangChain 和 LlamaIndex 这些框架到底解决了什么问题？什么时候该用、什么时候不该用？
- 本文思路：同一个任务，三种实现，代码摆出来你自己看

### 一、先搞清楚这俩框架是干嘛的

**LangChain：**
- 定位：AI 应用开发的"瑞士军刀"，啥都能干
- 核心概念：Chain（链式调用）、Agent（智能体）、Memory（记忆）、Retriever（检索器）
- 特点：功能全、生态大、抽象层多、学习曲线陡
- 争议：过度封装，简单任务反而写更多代码

**LlamaIndex：**
- 定位：专注 RAG 的垂直框架
- 核心概念：Document（文档）、Index（索引）、QueryEngine（查询引擎）、Retriever
- 特点：数据接入能力强（支持几十种文档格式）、RAG 流程开箱即用
- 争议：做 RAG 之外的事不太行

**裸写：**
- 就是直接调 OpenAI API + ChromaDB，我们前面文章一直在做的事
- 优点：完全可控、理解底层原理
- 缺点：什么都得自己造轮子

> **正文配图 1 提示词：**
> A Venn diagram with three overlapping circles. Left circle (blue): shows icons for chains, agents, memory, tools - representing LangChain's broad capabilities. Right circle (green): shows icons for documents, indexes, parsers - representing LlamaIndex's data-focused approach. Bottom circle (orange): shows a simple Python logo with minimal icons - representing bare code. The overlapping center area shows a RAG icon (search + chat bubble). Each circle has a different size suggesting different scope. Clean flat design, white background. No text. Aspect ratio 16:9.

### 二、任务定义：搭一个"公司知识库问答"

- 统一任务：加载 3 篇公司文档 → 切块 → 向量化 → 存储 → 用户问答
- 统一测试问题：3 个相同的问题，对比回答质量
- 统一环境：Python 3.11 + OpenAI API

### 三、裸写实现（我们的老朋友）

**要点：**
- 代码行数：~50 行
- 需要自己处理：文档加载、切块、Embedding 调用、向量库管理、Prompt 拼接、生成调用
- 每一步完全透明，出了问题知道在哪

**核心代码片段：**

```python
# 裸写版：完全控制每一步
def rag_bare(question, documents):
    # 1. 切块（自己写）
    chunks = []
    for doc in documents:
        chunks.extend(overlap_chunk(doc, chunk_size=300, overlap=50))

    # 2. 向量化 + 存储（直接调 API + ChromaDB）
    collection = chroma_client.get_or_create_collection("bare_rag")
    for i, chunk in enumerate(chunks):
        collection.add(ids=[f"c{i}"], documents=[chunk],
                       embeddings=[get_embedding(chunk)])

    # 3. 检索
    results = collection.query(
        query_embeddings=[get_embedding(question)], n_results=3)

    # 4. 生成（自己拼 prompt）
    context = "\n---\n".join(results["documents"][0])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user",
                   "content": f"根据资料回答：\n{context}\n\n问题：{question}"}])
    return response.choices[0].message.content
```

### 四、LangChain 实现

**要点：**
- 代码行数：~30 行（但 import 一大堆）
- 大量概念需要学：TextSplitter、VectorStore、RetrievalQA Chain
- 改默认行为的时候反而更麻烦（比如自定义 prompt、换检索策略）

**核心代码片段：**

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# LangChain 版
def rag_langchain(question, documents):
    # 1. 切块（用内置 TextSplitter）
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(documents)

    # 2. 向量化 + 存储（一行搞定）
    vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

    # 3. 检索 + 生成（Chain 封装）
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))

    return qa_chain.invoke(question)["result"]
```

**踩坑提醒：**
- LangChain 版本更新极快，半年前的教程代码大概率跑不通
- 想自定义 prompt 模板？得学 PromptTemplate + Chain 组合
- Debug 的时候调用栈很深，出了问题不好定位

> **正文配图 2 提示词：**
> A side-by-side code comparison visualization. Three vertical panels arranged horizontally. Left panel (orange border): short, clean code blocks with Python logo, labeled with a "DIY" wrench icon. Middle panel (blue border): medium code with many import lines at top and chain-like connected blocks, labeled with a chain-link icon. Right panel (green border): compact code with document/index icons, labeled with an index/book icon. Below all three: identical output/answer result shown as a chat bubble, proving same result. Clean flat design. No actual code text, just abstract representations. Aspect ratio 16:9.

### 五、LlamaIndex 实现

**要点：**
- 代码行数：~20 行（RAG 场景下最简洁）
- 数据加载能力很强：PDF、Word、网页、数据库，内置几十种 Reader
- 对于纯 RAG 场景，上手确实最快

**核心代码片段：**

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# LlamaIndex 版
def rag_llamaindex(question, documents):
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding()

    # 1. 加载文档 → 自动切块 → 自动向量化 → 自动建索引
    docs = [Document(text=d) for d in documents]
    index = VectorStoreIndex.from_documents(docs)

    # 2. 检索 + 生成（一行）
    query_engine = index.as_query_engine(similarity_top_k=3)
    return str(query_engine.query(question))
```

### 六、三种方案正面对比

```
| 维度         | 裸写 Python      | LangChain          | LlamaIndex         |
|-------------|-----------------|--------------------|--------------------|
| 代码行数     | ~50 行           | ~30 行（import 多） | ~20 行             |
| 学习成本     | 低（会 Python 就行）| 高（概念多、API 变动快）| 中等               |
| 灵活性       | 完全可控          | 需要理解抽象层才能改 | RAG 内灵活，RAG 外受限 |
| 调试难度     | 低               | 高（调用栈深）       | 中等               |
| 数据源支持   | 自己写解析器       | 有 DocumentLoader  | 最强（几十种 Reader）|
| 生态         | 无               | 最大               | RAG 垂直方向很全    |
| 版本稳定性   | 自己代码自己控     | 差（Breaking change 多）| 较好              |
| 适合场景     | 学习原理、轻量项目  | 复杂 AI 应用        | 专注 RAG 的项目     |
```

### 七、选型决策树

**要点：**
- 如果你在学习 RAG 原理 → 裸写
- 如果你的项目只需要 RAG 问答 → LlamaIndex
- 如果你的项目涉及 Agent + RAG + 记忆 + 多种工具 → LangChain
- 如果你的项目对性能和可控性要求极高 → 裸写
- 如果你不确定 → 先裸写搞清楚原理，再决定要不要引入框架

> **正文配图 3 提示词：**
> A decision tree flowchart. Starting node at top: a question mark icon. First branch: "Only RAG?" - Yes leads to LlamaIndex icon (green), No continues. Second branch: "Need Agent + Tools + Memory?" - Yes leads to LangChain icon (blue), No continues. Third branch: "Need full control?" - Yes leads to Python icon (orange). Each end node has a small recommendation badge. Clean flowchart style, white background, rounded rectangles for decisions, colored endpoints. No text labels, icons and arrows only. Aspect ratio 16:9.

### 写在最后

- 衔接语：了解了框架选型之后，下一篇我们来看 MCP 协议——Anthropic 搞的一套标准化工具接入方案，让智能体加工具变得像插 USB 一样简单
- 关键观点：框架不是越多越好，你对底层理解越深，用框架才越自如。不理解原理就上框架，出了问题连 Debug 都不知道从哪下手

> **正文配图 4 提示词：**
> An illustration showing three different workshop/workbench setups. Left: a clean minimal workbench with basic tools (hammer, screwdriver) - representing bare Python. Middle: a large complex workshop with conveyor belts and automated machinery - representing LangChain. Right: a specialized precision workstation with focused tools for one task - representing LlamaIndex. A person stands in front of all three, thinking, with a thought bubble containing a simple product (a completed RAG application). Clean illustrated style, warm lighting. No text. Aspect ratio 16:9.
