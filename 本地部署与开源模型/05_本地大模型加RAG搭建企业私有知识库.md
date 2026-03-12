# 本地大模型 + RAG：搭建企业私有知识库

老板把你叫进办公室："咱们的产品文档、技术规范、制度流程一大堆，能不能搞个 AI 问答系统？员工问啥都能答，省得天天翻文档。"你点点头，刚想说用 ChatGPT 接个 API 就行，老板补了一句："对了，数据不能出公司网络，合规那边盯得紧。"

得，云端 API 直接 pass。那怎么办？答案就是：**本地大模型 + RAG**。数据在你们自己的服务器上跑，一根网线都不往外连，照样能搭出一个像模像样的知识库问答系统。这篇文章就手把手教你从零搭起来。

---

## 一、为什么企业需要私有知识库

先说清楚一件事：不是所有公司都需要私有部署。小团队、非敏感行业、文档量不大，用 Dify 接个 API、或者直接 ChatGPT 企业版，省心又省钱。但下面这几类场景，私有知识库几乎是刚需。

**合规红线**。金融、医疗、政府、军工，数据出境有明文规定。你把客户信息、病历、内部文件往 OpenAI 的服务器一传，合规部门能跟你急。本地部署，数据不出机房，这是底线。

**商业机密**。产品路线图、未公开的专利、竞品分析报告，这些玩意儿扔到别人的云上，你睡得着吗？私有知识库相当于把保险柜搬进自己家，钥匙自己攥着。

**成本可控**。API 按 token 计费，用量一大，账单看着肉疼。本地跑起来之后，电费之外没有边际成本，想怎么问就怎么问，不用盯着余额发愁。文档量上去、用户量上去，成本曲线是平的，这是自建的最大优势。

**定制化需求**。商业产品往往要迁就通用场景，你们公司可能有独特的文档格式、审批流程、权限体系。自己搭，想怎么改就怎么改，没有"这个功能我们暂时不支持"的憋屈。

说白了，**私有知识库 = 数据主权 + 成本可控 + 灵活定制**。老板说"数据不能出网"，不是矫情，是现实约束。咱们就在这个约束里把事办了。

---

## 二、架构设计：从文档到回答的完整链路

RAG 的全称是 **Retrieval-Augmented Generation**，检索增强生成。名字唬人，但原理不复杂：**别让大模型凭记忆答题，给它开卷**。

打个比方：微调是让学生把整本教材背下来，RAG 是让学生带着教材进考场。需要的时候现场查，查到了再答。你说哪个划算？

整条链路拆开看，就七个步骤：

```
原始文档 → 切块（Chunk）→ Embedding 向量化 → 存入向量库 → 用户提问 → 检索相似块 → 塞进 Prompt → 本地 LLM 生成回答
```

**第一步：文档切块**。一篇 50 页的 PDF 不能整篇扔给大模型，上下文窗口撑不住。得切成几百字的小段，每段是一个完整的语义单元。切得好不好，直接决定后面检索准不准。

**第二步：Embedding 向量化**。把每段文字变成一串数字（比如 768 维的浮点数组）。语义相近的文字，向量也相近。"年假 5 天"和"带薪年假五天"的向量距离就很小。想象一个巨大的坐标系，Embedding 模型干的活就是帮每段文字找到"正确的位置"。

**第三步：存入向量库**。把这些向量存进专门的数据库，支持按相似度快速检索。ChromaDB、Milvus、Qdrant 都干这事儿。

**第四步：用户提问**。员工问"入职满 5 年有几天年假？"

**第五步：检索**。把问题也转成向量，在库里搜最相似的几块。比如搜到请假制度里"入职满 1 年不满 10 年享有 5 天年假"那段。

**第六步：塞进 Prompt**。把检索到的几段原文拼起来，和问题一起塞给大模型："根据以下资料回答：……"

**第七步：本地 LLM 生成**。大模型基于这些材料生成回答，数据全程在本地，一根线都没往外连。

整条链路的精髓就一句话：**搜索 + 大模型**。先搜到相关内容，再让大模型基于搜到的东西生成回答。模型本身不需要"见过"你们的文档，每次提问时现场查就行。

---

## 三、技术选型：用啥搭、为啥选

搭建一套本地 RAG，需要几个关键组件：**大模型**、**Embedding 模型**、**向量库**、**应用框架**。下面按组件说选型。

**大模型：Ollama + Qwen**

Ollama 是"大模型界的 Docker"，一条命令拉模型、跑起来，不用你操心 CUDA、量化这些破事。**Qwen2.5** 系列中英文都不错，7B 量化版在 16GB 内存的机器上就能跑，适合企业知识库这种"理解 + 生成"的场景。如果显存紧张，可以上 **Qwen2.5:0.5B** 或 **3B**，效果会打折扣，但能跑起来。

**向量库：ChromaDB 或 Milvus**

**ChromaDB** 轻量、单机够用、Python 原生支持好，适合文档量在万级以内的场景。**Milvus** 适合大规模、高并发，文档量上十万、百万级，或者需要分布式部署，选它。起步阶段 ChromaDB 就够了，后面扩容再迁也不迟。

**Embedding 模型：BGE**

**BGE**（BAAI General Embedding）是智源开源的文本向量模型，中文效果好，支持本地部署。**bge-small-zh** 模型小（100M 参数）、速度快，适合资源有限的场景；**bge-base-zh** 效果更好，但算力要求高一点。用 `sentence-transformers` 库加载，一行代码搞定。

**应用框架**

可以自己写 Python 脚本，也可以用 **LangChain**、**LlamaIndex** 这类框架。框架的好处是分块、检索、Prompt 模板都帮你封装好了，坏处是多一层抽象，出问题不好排查。建议先用纯 Python 把流程跑通，再考虑上框架。

---

## 四、实战搭建步骤（完整代码）

下面是一套**最小可运行**的代码，依赖少、逻辑清晰，跑通之后你再按需扩展。

### 4.1 环境准备

```bash
pip install ollama chromadb sentence-transformers
```

Ollama 需要单独安装，去 https://ollama.com 下载，装完后执行 `ollama pull qwen2.5:7b` 拉模型。

### 4.2 准备测试文档

```python
documents = [
    {
        "id": "doc1",
        "title": "请假制度",
        "content": """公司请假制度（2024年修订版）
一、年假规定
1. 入职满1年不满10年的员工，每年享有5天带薪年假。
2. 入职满10年不满20年的员工，每年享有10天带薪年假。
3. 年假需提前3个工作日在OA系统中提交申请，直属上级审批通过后生效。
二、病假规定
1. 病假需提供正规医院出具的诊断证明和病假条。
2. 3天以上的病假需由部门总监审批。"""
    },
    {
        "id": "doc2",
        "title": "报销制度",
        "content": """公司费用报销制度
一、差旅报销标准
1. 交通费：飞机经济舱、高铁二等座实报实销。
2. 住宿费：一线城市（北上广深）每晚不超过500元，其他城市不超过400元。
3. 伙食补助：每人每天100元，按出差天数计算。"""
    }
]
```

实际场景中，这些可以是 PDF、Word、Markdown，用 `PyPDF2`、`python-docx` 等库解析成文本即可。

### 4.3 完整 RAG 流程

```python
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# 1. 加载 Embedding 模型（首次运行会自动下载）
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 初始化向量库
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

def chunk_text(text, chunk_size=300, overlap=50):
    """按固定大小切块，带重叠避免截断语义"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index(docs):
    """建索引：切块 → 向量化 → 入库"""
    for doc in docs:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            embedding = embed_model.encode(chunk).tolist()
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["title"]}]
            )
        print(f"已入库：{doc['title']}，{len(chunks)} 块")
    print(f"索引完成，共 {collection.count()} 块")

def rag_query(question, top_k=3):
    """RAG 问答：检索 → 拼 Prompt → 本地 LLM 生成"""
    # 检索
    query_embedding = embed_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 拼 Prompt
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""你是一个企业知识库助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请如实说"未找到相关制度规定"。

参考资料：
{context}

用户问题：{question}

请直接回答，不要编造。"""

    # 调用本地 Ollama
    response = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# 执行
build_index(documents)
answer = rag_query("入职满5年有几天年假？")
print(answer)
```

跑起来之后，数据流全程在本地：文档在你机器上、Embedding 在你机器上、向量库在你机器上、大模型也在你机器上。**一根网线都没往外连**。

---

## 五、效果优化：从"能跑"到"好用"

搭起来只是第一步，想答得准、答得稳，还得在几个关键环节下功夫。

**分块策略**

按固定字数切是最省事的，但容易把一句话劈成两半。比如"3 天以上病假需部门总监审批"被切成"3 天以上"和"病假需部门总监审批"，检索到其中一块也答不对。更好的做法是**按段落或按语义边界切**：遇到换行、标题、列表项就切一刀，保证每个块是完整的语义单元。块大小建议 200–500 字，块与块之间可以重叠 50–100 字，避免关键信息落在边界上。

**Embedding 选型**

BGE 有多个版本：`bge-small-zh` 快但效果一般，`bge-base-zh` 效果好但慢一点，`bge-large-zh` 效果最好但吃显存。文档量不大（几千块以内）用 small 就行，追求准确率可以上 base。如果你的文档是纯英文，可以考虑 `bge-base-en`，中文场景还是用 zh 系列。

**Rerank 重排序**

向量检索是按相似度排序的，但"相似"不一定"相关"。比如用户问"年假几天"，可能检索到"病假几天"的块，因为都包含"假"和"几天"。加一层 **Rerank**：用一个小模型对检索到的 Top 10 做二次打分，把真正相关的排到前面，再只取 Top 3 塞给大模型。Rerank 能显著提升准确率，代价是多一次模型调用。可以用 `BAAI/bge-reranker-base` 这类专门的重排序模型。

**Prompt 模板**

模板里要明确三点：**角色**（你是企业知识库助手）、**约束**（严格根据参考资料，不要编造）、**格式**（直接回答，必要时引用出处）。还可以加一句"如果资料中没有相关信息，请说未找到"，避免大模型瞎编。模板调好了，幻觉能少一大半。

---

## 六、部署方案：Docker Compose 一键部署

开发阶段在本地跑没问题，真要给全公司用，得考虑部署。下面是一套 **Docker Compose** 方案，把 Ollama、向量库、应用服务打包在一起，一条命令拉起整套环境。

```yaml
# docker-compose.yml
version: "3.8"
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma

  rag-app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    depends_on:
      - ollama
      - chromadb

volumes:
  ollama_data:
  chroma_data:
```

没有 GPU 的机器，把 `ollama` 服务的 `deploy.resources` 那一段删掉，用 CPU 跑，慢一点但能跑。应用服务（`rag-app`）需要自己写一个简单的 FastAPI 或 Flask 接口，把上面的 `rag_query` 封装成 HTTP API，前端或企业微信接进来就能用。

部署前记得先 `ollama pull qwen2.5:7b`，否则第一次请求会卡在模型下载上。

---

## 七、跟商业方案的对比

最后聊聊：自己搭 vs 用现成的，到底怎么选。

**私有部署（本文方案）**

优势：数据完全在本地，合规无忧；成本可控，电费之外没有边际成本；可定制，想加什么功能自己改。劣势：要自己维护、自己调优，出问题得自己兜着；效果取决于你的调参水平，没有"开箱即用"的保证。

**Dify / Coze 等低代码平台**

优势：拖拽搭流程、接各种模型 API、有现成的知识库组件，上手快。劣势：数据要传到他们的服务器（除非你买私有部署版），Embedding、检索逻辑是黑盒，想深度优化得看他们支不支持。适合快速验证、对数据合规要求不高的团队。

**ChatGPT 企业版 / 微软 Copilot**

优势：能力强、体验好、企业级安全和合规。劣势：贵，而且数据虽然承诺不出境，但终究在别人的云上，有些行业就是过不了审。适合预算充足、对数据主权不那么敏感的大企业。

**一句话总结**：数据不能出网、预算有限、愿意折腾，选本地 RAG；要快、要省心、数据可以上云，选 Dify 或商业产品；不差钱、要最好的体验，选企业版 ChatGPT。没有银弹，看你的约束条件。

---

## 结尾

本地大模型 + RAG，本质上是在"数据不出网"这个硬约束下，用开源工具搭一座桥：让大模型能"读到"你们的私有文档，又不让数据离开你们的机房。技术栈不复杂，Ollama、ChromaDB、BGE 都是成熟方案，照着文章跑一遍，一个下午能搭出雏形。

剩下的就是调优：分块怎么切、检索取几条、Prompt 怎么写、要不要加 Rerank。这些细节决定了系统是"能跑"还是"好用"。多试几次，多看看答错的 case，慢慢就能摸到门道。

老板要的是结果：员工问问题，系统能答上来，数据还不出公司。这套方案，能做到。
