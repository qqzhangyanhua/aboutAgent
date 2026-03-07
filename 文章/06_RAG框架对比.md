# LangChain vs LlamaIndex vs 裸写 —— RAG 框架到底该不该用？

> **封面图提示词（2:1 比例）：**
> A tech illustration showing three distinct paths/roads diverging from a single starting point at the bottom. Left path: a complex highway interchange with many layers and exits, representing LangChain (complex but powerful). Middle path: a streamlined elevated railway on pillars going straight to a destination, representing LlamaIndex (specialized and direct). Right path: a simple dirt/hiking trail with a person walking freely, representing bare Python (flexible but manual). All three paths lead to the same destination building at the top labeled with a search/answer icon. Background: soft gradient sky. No text, no watermark. Aspect ratio 2:1.

---

前面几篇咱们全是裸写的——50 行代码搞定 RAG，根本没碰任何框架。但面试或者技术方案里总有人问"你用的什么框架"，好像不用框架就不专业似的。

说实话，这个问题挺烦的。你明明把 RAG 跑通了、效果也调好了，对方一句"没用 LangChain 啊"就能让你怀疑人生。那 LangChain 和 LlamaIndex 这些框架到底解决了什么问题？什么时候该用、什么时候不该用？

今天咱们用同一个任务，三种实现，代码摆出来你自己看。看完你就知道：框架不是银弹，选对了事半功倍，选错了全是坑。

---

## 一、先搞清楚这俩框架是干嘛的

在写代码之前，咱先捋清楚这三个选项各自的定位。不然你很容易被各种概念绕晕。

**LangChain** —— 瑞士军刀，啥都能干。Chain（链式调用）、Agent（智能体）、Memory（记忆）、Retriever（检索器）……你想得到的 AI 应用场景，它都能插一脚。功能全、生态大，GitHub 十几万 star，各种集成一应俱全。但抽象层多，学习曲线陡。你写个简单 RAG 可能只要 30 行，但光搞懂 TextSplitter、VectorStore、Retriever、Chain 这几个概念就得半天。争议在于：过度封装，简单任务反而写更多代码；版本更新极快，半年前的教程大概率跑不通。

**LlamaIndex** —— 专注 RAG 的垂直框架。Document、Index、QueryEngine、Retriever，核心概念就围着"数据怎么进、怎么查"转。数据接入能力强，PDF、Word、网页、数据库、Notion、Confluence……几十种 Reader 开箱即用。RAG 流程一行 `VectorStoreIndex.from_documents` 就搞定。争议在于：做 RAG 之外的事不太行，你要搞 Agent、加工具、做多轮对话，就得另找方案。

**裸写** —— 就是直接调 OpenAI API + ChromaDB，咱们前面文章一直在做的事。没有中间层，每一步你都能看见、能改。完全可控、理解底层原理，缺点是啥都得自己造轮子：切块自己写、向量化自己调、prompt 自己拼。

> **正文配图 1 提示词：**
> A Venn diagram with three overlapping circles. Left circle (blue): shows icons for chains, agents, memory, tools - representing LangChain's broad capabilities. Right circle (green): shows icons for documents, indexes, parsers - representing LlamaIndex's data-focused approach. Bottom circle (orange): shows a simple Python logo with minimal icons - representing bare code. The overlapping center area shows a RAG icon (search + chat bubble). Each circle has a different size suggesting different scope. Clean flat design, white background. No text. Aspect ratio 16:9.

---

## 二、任务定义：搭一个"公司知识库问答"

为了公平对比，咱们把任务定死：加载 3 篇公司文档（请假制度、报销制度、技术规范）→ 切块 → 向量化 → 存储 → 用户问答。

统一测试问题：同一个问题，比如"病假工资怎么算？超过30天怎么办？"，三种实现都回答一遍，对比效果。

统一环境：Python 3.11 + OpenAI API + ChromaDB。依赖就这些：`pip install openai chromadb langchain langchain-openai langchain-community llama-index llama-index-llms-openai llama-index-embeddings-openai`。

---

## 三、裸写实现（咱们的老朋友）

裸写版代码行数大概 50 行。需要自己处理：文档加载、切块、Embedding 调用、向量库管理、Prompt 拼接、生成调用。每一步完全透明，出了问题你知道在哪——断点打在 `get_embedding` 就是向量化的问题，打在 `collection.query` 就是检索的问题。

用咱们之前文章里的 `overlap_chunk`，相邻块之间有 50 字重叠，避免一句话被硬生生切断。切完直接调 OpenAI Embedding 接口，塞进 ChromaDB，检索时取 top 3，拼进 prompt 让 gpt-4o-mini 回答。

```python
def overlap_chunk(text, chunk_size=300, overlap=50):
    """带重叠的分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def rag_bare(question, documents, chroma_client, client, get_embedding):
    # 1. 切块（自己写）
    chunks = []
    for doc in documents:
        chunks.extend(overlap_chunk(doc["content"], chunk_size=300, overlap=50))

    # 2. 向量化 + 存储（直接调 API + ChromaDB）
    collection = chroma_client.get_or_create_collection("bare_rag")
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"c{i}"],
            documents=[chunk],
            embeddings=[get_embedding(chunk)]
        )

    # 3. 检索
    results = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=3
    )

    # 4. 生成（自己拼 prompt）
    context = "\n---\n".join(results["documents"][0])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"根据资料回答：\n{context}\n\n问题：{question}"}]
    )
    return response.choices[0].message.content
```

**运行效果：**

```
问题：病假工资怎么算？超过30天怎么办？

回答：根据公司请假制度：
1. 病假期间工资按基本工资的80%发放。
2. 连续病假超过30天的，按公司长期病假政策处理。
```

---

## 四、LangChain 实现

LangChain 版代码行数大概 30 行，但 import 一大堆。你需要先搞懂几个概念：TextSplitter 负责切块、VectorStore 负责存向量、Retriever 负责检索、Chain 负责把检索和生成串起来。

`RecursiveCharacterTextSplitter` 是 LangChain 自带的切块器，会按段落、句子、词逐级尝试切，比咱们的 `overlap_chunk` 更智能一点。`Chroma.from_documents` 一行搞定向量化+存储。`create_retrieval_chain` 把 Retriever 和 LLM 绑在一起，你传问题进去，它自动检索、拼 context、调模型、返回答案。

听起来省事，但改默认行为的时候反而更麻烦。比如你想换检索策略（不用 top-k，改用 MMR 多样性检索）、或者自定义 prompt 模板，就得去翻文档学 PromptTemplate、Chain 的组合方式。而且 LangChain 的 API 变动频繁，`RetrievalQA` 已经 deprecated，现在推荐用 `create_retrieval_chain`，你照着老教程写很可能跑不起来。

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def rag_langchain(question, documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = [doc["content"] for doc in documents]
    splits = splitter.create_documents(texts)

    vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "根据以下资料回答：\n{context}\n\n问题：{input}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, docs_chain)

    result = chain.invoke({"input": question})
    return result["answer"]
```

**踩坑提醒：** LangChain 版本更新极快，半年前的教程代码大概率跑不通。想自定义 prompt 模板？得学 PromptTemplate + Chain 组合。Debug 的时候调用栈很深，出了问题不好定位。

**运行效果：** 同上，回答一致。三种实现面对同一个问题，检索到的文档块大同小异，最终答案也差不多。

> **正文配图 2 提示词：**
> A side-by-side code comparison visualization. Three vertical panels arranged horizontally. Left panel (orange border): short, clean code blocks with Python logo, labeled with a "DIY" wrench icon. Middle panel (blue border): medium code with many import lines at top and chain-like connected blocks, labeled with a chain-link icon. Right panel (green border): compact code with document/index icons, labeled with an index/book icon. Below all three: identical output/answer result shown as a chat bubble, proving same result. Clean flat design. No actual code text, just abstract representations. Aspect ratio 16:9.

---

## 五、LlamaIndex 实现

LlamaIndex 版代码行数大概 20 行，RAG 场景下最简洁。`Settings.llm` 和 `Settings.embed_model` 设好全局配置，`Document` 包一层文本，`VectorStoreIndex.from_documents` 自动完成切块、向量化、建索引，`as_query_engine` 一行搞定检索+生成。

数据加载能力是它的强项。PDF、Word、网页、数据库、Notion、Confluence……内置几十种 Reader，你不需要自己写解析器。纯 RAG 场景，上手确实最快。但如果你要做 Agent、加工具调用、搞多轮对话记忆，LlamaIndex 就有点力不从心了，得配合别的库用。

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def rag_llamaindex(question, documents):
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding()

    docs = [Document(text=d["content"]) for d in documents]
    index = VectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine(similarity_top_k=3)
    return str(query_engine.query(question))
```

**运行效果：** 同上，回答一致。

---

## 六、三种方案正面对比

把三个维度摊开看，差异就很明显了。下面这个表可以当技术选型时的速查手册。

| 维度 | 裸写 Python | LangChain | LlamaIndex |
|------|-------------|-----------|------------|
| 代码行数 | ~50 行 | ~30 行（import 多） | ~20 行 |
| 学习成本 | 低（会 Python 就行） | 高（概念多、API 变动快） | 中等 |
| 灵活性 | 完全可控 | 需要理解抽象层才能改 | RAG 内灵活，RAG 外受限 |
| 调试难度 | 低 | 高（调用栈深） | 中等 |
| 数据源支持 | 自己写解析器 | 有 DocumentLoader | 最强（几十种 Reader） |
| 生态 | 无 | 最大 | RAG 垂直方向很全 |
| 版本稳定性 | 自己代码自己控 | 差（Breaking change 多） | 较好 |
| 适合场景 | 学习原理、轻量项目 | 复杂 AI 应用 | 专注 RAG 的项目 |

裸写的优势是零黑盒，你写的每一行都在掌控之中。LangChain 的生态最大，但版本稳定性差，今天能跑的代码明天可能就报错。LlamaIndex 在 RAG 垂直方向做得最专，数据源多、上手快，但扩展性不如 LangChain。

> **正文配图 3 提示词：**
> A decision tree flowchart. Starting node at top: a question mark icon. First branch: "Only RAG?" - Yes leads to LlamaIndex icon (green), No continues. Second branch: "Need Agent + Tools + Memory?" - Yes leads to LangChain icon (blue), No continues. Third branch: "Need full control?" - Yes leads to Python icon (orange). Each end node has a small recommendation badge. Clean flowchart style, white background, rounded rectangles for decisions, colored endpoints. No text labels, icons and arrows only. Aspect ratio 16:9.

---

## 七、选型决策树

怎么选？按你的需求来：

- **学 RAG 原理** → 裸写。把切块、向量化、检索、生成每一步都自己写一遍，底层逻辑就通了。这时候上框架反而会挡住你的视线。

- **只做 RAG 问答** → LlamaIndex。文档多、格式杂、想快速上线，LlamaIndex 的 Reader + Index + QueryEngine 一条龙最省心。

- **Agent + RAG + 记忆 + 多种工具** → LangChain。你要做的是"会查资料、会用工具、能记对话"的智能体，LangChain 的 Chain、Agent、Memory 就是为这种场景设计的。

- **对性能和可控性要求极高** → 裸写。生产环境要压延迟、控成本、调检索策略，框架的抽象层会变成负担。裸写你能精确控制每一次 API 调用。

- **不确定** → 先裸写搞清楚原理，再决定要不要引入框架。千万别一上来就堆框架，出了问题连 Debug 都不知道从哪下手。

> **正文配图 4 提示词：**
> An illustration showing three different workshop/workbench setups. Left: a clean minimal workbench with basic tools (hammer, screwdriver) - representing bare Python. Middle: a large complex workshop with conveyor belts and automated machinery - representing LangChain. Right: a specialized precision workstation with focused tools for one task - representing LlamaIndex. A person stands in front of all three, thinking, with a thought bubble containing a simple product (a completed RAG application). Clean illustrated style, warm lighting. No text. Aspect ratio 16:9.

---

## 八、写在最后

了解了框架选型之后，下一篇咱们来看 MCP 协议——Anthropic 搞的一套标准化工具接入方案，让智能体加工具变得像插 USB 一样简单。

最后说一句：框架不是越多越好。你对底层理解越深，用框架才越自如。不理解原理就上框架，出了问题连 Debug 都不知道从哪下手。先把裸写跑通，再按需求选框架，这才是正道。

---

## 附录：完整可运行代码

```bash
pip install openai chromadb langchain langchain-openai langchain-community llama-index llama-index-llms-openai llama-index-embeddings-openai
```

```python
"""
LangChain vs LlamaIndex vs 裸写 —— 三种 RAG 实现对比
同一批测试文档、同一个测试问题，对比三种实现
"""
import os
import chromadb
from openai import OpenAI

# ==================== 环境配置 ====================
os.environ["OPENAI_API_KEY"] = "你的API Key"  # 替换为实际 Key
client = OpenAI()
chroma_client = chromadb.Client()

# ==================== 测试文档（3篇公司文档） ====================
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "请假制度",
        "content": """公司请假制度（2024年修订版）

一、年假规定
1. 入职满1年不满10年的员工，每年享有5天带薪年假。
2. 入职满10年不满20年的员工，每年享有10天带薪年假。
3. 入职满20年以上的员工，每年享有15天带薪年假。
4. 年假需提前3个工作日在OA系统中提交申请，直属上级审批通过后生效。
5. 年假可以分次使用，但每次不少于半天。
6. 当年未使用的年假可以顺延至次年第一季度，逾期作废。

二、病假规定
1. 病假需提供正规医院出具的诊断证明和病假条。
2. 3天以内的病假由直属上级审批。
3. 3天以上的病假需由部门总监审批。
4. 病假期间工资按基本工资的80%发放。
5. 连续病假超过30天的，按公司长期病假政策处理。

三、事假规定
1. 事假为无薪假期，按日扣除工资。
2. 事假每次不超过3天，需提前2个工作日申请。
3. 全年事假累计不超过15天。
4. 事假审批流程与病假相同。"""
    },
    {
        "id": "doc2",
        "title": "报销制度",
        "content": """公司费用报销制度

一、差旅报销标准
1. 交通费：飞机经济舱、高铁二等座实报实销，需提供电子行程单或车票。
2. 住宿费：一线城市（北上广深）每晚不超过500元，其他城市每晚不超过350元。
3. 餐饮补贴：国内出差每天100元，无需提供发票。
4. 市内交通：出租车或网约车实报实销，需提供行程截图。

二、报销流程
1. 出差结束后5个工作日内提交报销申请。
2. 在OA系统中填写报销单，附上所有票据的电子扫描件。
3. 报销金额5000元以下由直属上级审批。
4. 报销金额5000元至20000元需部门总监审批。
5. 报销金额20000元以上需财务总监审批。
6. 审批通过后，财务部在10个工作日内打款至员工工资卡。

三、日常办公费用报销
1. 办公用品：单次500元以下可先购后报，超过500元需提前申请。
2. 团建费用：按每人每季度200元的标准，需提供活动照片和参与人员名单。
3. 培训费用：需提前提交培训申请，经部门总监批准后方可报销。"""
    },
    {
        "id": "doc3",
        "title": "技术规范",
        "content": """前端技术开发规范

一、技术栈要求
1. 框架统一使用 React 18+，不允许在新项目中使用 Vue 或 Angular。
2. 状态管理使用 Zustand，不建议使用 Redux（历史项目除外）。
3. UI 组件库统一使用 Ant Design 5.x。
4. CSS 方案使用 Tailwind CSS 或 CSS Modules，禁止使用行内样式。
5. 构建工具使用 Vite，不再使用 Webpack。

二、代码规范
1. 使用 TypeScript 严格模式，禁止使用 any 类型。
2. 组件文件使用 PascalCase 命名，如 UserProfile.tsx。
3. 工具函数使用 camelCase 命名，如 formatDate.ts。
4. 常量使用 UPPER_SNAKE_CASE 命名，如 MAX_RETRY_COUNT。
5. 每个组件文件不超过 300 行，超过需拆分子组件。
6. 必须编写单元测试，核心业务组件覆盖率不低于 80%。

三、Git 规范
1. 分支命名：feature/xxx、bugfix/xxx、hotfix/xxx。
2. Commit 信息格式：type(scope): description，如 feat(user): add login page。
3. 禁止直接向 main 分支提交代码，必须通过 PR 合入。
4. PR 至少需要 1 位同事 Code Review 并批准后方可合入。
5. 合入后删除源分支，保持仓库整洁。"""
    },
]

# ==================== 裸写实现 ====================
def overlap_chunk(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def rag_bare(question):
    documents = DOCUMENTS
    chunks = []
    for doc in documents:
        chunks.extend(overlap_chunk(doc["content"], chunk_size=300, overlap=50))

    collection = chroma_client.get_or_create_collection("bare_rag")
    for i, chunk in enumerate(chunks):
        collection.add(ids=[f"c{i}"], documents=[chunk], embeddings=[get_embedding(chunk)])

    results = collection.query(query_embeddings=[get_embedding(question)], n_results=3)
    context = "\n---\n".join(results["documents"][0])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"根据资料回答：\n{context}\n\n问题：{question}"}]
    )
    return response.choices[0].message.content

# ==================== LangChain 实现 ====================
def rag_langchain(question):
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = [doc["content"] for doc in DOCUMENTS]
    splits = splitter.create_documents(texts)

    vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "根据以下资料回答：\n{context}\n\n问题：{input}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, docs_chain)

    result = chain.invoke({"input": question})
    return result["answer"]

# ==================== LlamaIndex 实现 ====================
def rag_llamaindex(question):
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding()

    docs = [Document(text=d["content"]) for d in DOCUMENTS]
    index = VectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine(similarity_top_k=3)
    return str(query_engine.query(question))

# ==================== 运行对比 ====================
if __name__ == "__main__":
    TEST_QUESTION = "病假工资怎么算？超过30天怎么办？"

    print("=" * 60)
    print("裸写版")
    print("=" * 60)
    print(rag_bare(TEST_QUESTION))

    print("\n" + "=" * 60)
    print("LangChain 版")
    print("=" * 60)
    print(rag_langchain(TEST_QUESTION))

    print("\n" + "=" * 60)
    print("LlamaIndex 版")
    print("=" * 60)
    print(rag_llamaindex(TEST_QUESTION))
```
