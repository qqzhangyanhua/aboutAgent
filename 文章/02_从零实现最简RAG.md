# 50 行 Python，让 AI 学会"翻书找答案"—— 从零实现最简 RAG

> **封面图提示词（2:1 比例，1232×616px）：**
> A clean, modern tech illustration in flat design style. On the left side, a large open book with glowing pages, documents floating out of it. On the right side, a friendly robot arm reaching toward the book, holding a magnifying glass. Between them, dotted lines and small vector icons (search icon, database cylinder, sparkle) connect the book to the robot. Background: soft gradient from deep blue (#1a1a2e) to teal (#16213e). Accent colors: warm orange highlights on the magnifying glass and key elements. Minimal, no text, no watermark. Aspect ratio 2:1.

上篇文章我们用几十行代码造了个会"自我反省"的 AI 智能体。评论区好多人问：能不能让它学会读我自己的文档？比如公司内部的产品手册、技术文档、会议纪要……

答案是：能，而且不难。这个技术叫 **RAG**（Retrieval-Augmented Generation，检索增强生成）。名字唬人，但原理嘛……你看完可能会觉得"就这？"

这篇文章带你从零搭三个版本，一步步升级：

- **V1** — 最朴素版，能从文档里找答案就行
- **V2** — 加上分块重叠 + 相似度过滤，准确率直接翻倍
- **V3** — 再加 Rerank 重排序，让最相关的内容排到最前面

依赖很少，`pip install openai chromadb` 就够了，代码直接能跑。

---

## 一、为什么需要 RAG？

先说一个你一定遇到过的场景：

> 你问 ChatGPT："我们公司的报销流程是什么？"
>
> ChatGPT 回："很抱歉，我无法获取贵公司的内部信息……"

废话，它当然不知道。大模型的知识来自训练数据，你公司的内部文档它压根没见过。

那怎么办？两条路：

**路线 A：微调（Fine-tuning）** — 把你的数据喂给模型重新训练。相当于让大模型"死记硬背"你的文档。代价是贵（几十到几百美元一次）、慢（几小时起步）、每次文档更新都得重新训练。

**路线 B：RAG** — 不改模型，改输入。每次用户提问的时候，先从你的文档库里搜出最相关的几段话，塞进 prompt 里，让大模型"开卷考试"。代价极低（只需要一次向量化），实时更新（文档变了重新入库就行）。

打个比方：微调是让学生把整本教材背下来，RAG 是让学生带着教材进考场。你说哪个划算？

说白了：**RAG = 搜索 + 大模型**。先搜到相关内容，再让大模型基于搜到的东西生成回答。

---

## 二、RAG 的核心流程

不管外面包了多花哨的框架，RAG 拆开就两个阶段：

> **正文配图 1 提示词：**
> A clean infographic-style diagram showing the RAG pipeline. Left section labeled "Indexing": icons of documents → scissors (chunking) → grid of numbers (embedding) → database cylinder. Right section labeled "Querying": a question mark → grid of numbers → database with search lines → a chat bubble with answer. Connected by arrows flowing left to right. Flat design, white background, blue and orange accent colors. No text labels, icons only. Aspect ratio 16:9.

### 阶段一：建索引（离线做一次）

```
原始文档 → 切成小块（Chunk） → 每块转成向量（Embedding） → 存进向量数据库
```

**切块**：一篇 10 页的文档不能整篇扔给大模型（会超 token 限制），得切成几百字的小段。

**向量化**：把每段文字变成一串数字（比如 1536 维的浮点数数组）。语义相近的文字，向量也相近。"今天天气真好"和"今天阳光明媚"的向量距离就很小。

**存储**：把这些向量存进专门的向量数据库（ChromaDB、FAISS、Qdrant 等），方便后面快速搜索。

### 阶段二：检索 + 生成（每次提问都做）

```
用户提问 → 问题转成向量 → 在向量库中搜最相似的几段 → 把这些段落塞进 prompt → 大模型生成回答
```

就这么简单。整个 RAG 的精髓就是一句话：**别让大模型凭记忆答题，给它"开卷"。**

---

## 三、准备工作（三个版本共用）

### 3.1 环境准备

```bash
pip install openai chromadb
```

```python
import os
import json
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
# 如果用 DeepSeek 等国产模型，取消下面的注释并修改
# client = OpenAI(base_url="https://api.deepseek.com", api_key="你的Key")
client = OpenAI()
```

> **注意**：如果你用的是国内的大模型（比如通义千问、DeepSeek），只需要把 `base_url` 换成对应的地址就行。Embedding 接口各家都有，调用方式几乎一样。

### 3.2 准备测试文档

我们模拟一个"公司知识库"，包含几篇内部文档。在实际场景中，这些可以是 PDF、Word、Markdown 等任何格式，这里为了演示方便直接用字符串：

```python
documents = [
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
    {
        "id": "doc4",
        "title": "新人入职指南",
        "content": """新员工入职指南

一、入职第一天
1. 上午 9:00 到前台报到，HR 会带你完成入职手续。
2. 领取工牌、笔记本电脑和办公用品。
3. 电脑预装了企业微信、OA 系统、Git 客户端和 VS Code。
4. 中午 HR 会带你认识团队成员，并安排一位导师（Buddy）。
5. 下午完成企业微信激活和 OA 系统账号开通。

二、入职第一周
1. 阅读公司文化手册和部门业务介绍文档。
2. 完成信息安全培训（线上课程，约2小时）。
3. 和导师一起熟悉项目代码仓库和开发流程。
4. 参加周五的团队周会，了解当前迭代进展。

三、试用期（3个月）
1. 试用期工资为正式工资的 90%。
2. 第一个月结束时进行一次非正式的 1v1 反馈。
3. 试用期满前两周由导师和主管进行转正评估。
4. 转正答辩需准备 PPT，内容包括：工作总结、学习收获、未来计划。
5. 转正通过后工资调整为100%，并开始享受完整的福利待遇。

四、常用系统
1. OA 系统：请假、报销、审批都在这里。地址 oa.company.com。
2. Git 仓库：代码托管在内部 GitLab，地址 git.company.com。
3. 文档平台：技术文档和产品文档在 Confluence，地址 wiki.company.com。
4. 沟通工具：日常沟通用企业微信，技术讨论用 Slack。"""
    },
    {
        "id": "doc5",
        "title": "绩效考核制度",
        "content": """绩效考核管理办法

一、考核周期
1. 公司采用季度考核制，每季度末进行一次绩效评估。
2. 年终综合评定取四个季度的加权平均分。

二、考核维度
1. 业务成果（50%）：KPI 完成情况，以可量化的指标为主。
2. 专业能力（20%）：技术能力、业务理解、问题解决能力。
3. 协作与沟通（15%）：跨部门协作、知识分享、团队贡献。
4. 文化价值观（15%）：主动性、责任心、创新意识。

三、考核流程
1. 员工先进行自评，填写当季工作总结和自我打分。
2. 直属上级根据实际表现进行评分，并撰写评语。
3. 部门总监进行校准，确保部门内评分标准一致。
4. HR 汇总结果并进行公司级校准。
5. 最终结果在下季度第一周内反馈给员工。

四、考核结果应用
1. S（卓越，前10%）：晋升优先、年终奖系数 2.0、额外期权奖励。
2. A（优秀，前30%）：年终奖系数 1.5、加薪幅度 15%-20%。
3. B（良好，中间50%）：年终奖系数 1.0、加薪幅度 5%-10%。
4. C（待改进，后10%）：制定改进计划（PIP），无年终奖。
5. 连续两个季度 C 评分，公司有权解除劳动合同。"""
    }
]
```

这五篇文档涵盖了请假、报销、技术规范、入职和绩效——一个典型的公司知识库。

---

## 四、V1：最朴素版 — 能找到答案就行

V1 的思路极其直白：把文档切块 → 转向量 → 存进去 → 用户提问时搜最相似的块 → 塞进 prompt 里让大模型回答。

### 4.1 文档切块

最简单的切块方式：按固定字数切。

```python
def simple_chunk(text, chunk_size=200):
    """按固定大小切块"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
```

### 4.2 向量化 + 存储

用 OpenAI 的 Embedding 接口把文字变成向量，然后存进 ChromaDB：

```python
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs_v1")

def get_embedding(text):
    """调用 OpenAI Embedding 接口"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_index_v1(documents):
    """V1：简单切块 + 入库"""
    print("正在建立索引...")
    for doc in documents:
        chunks = simple_chunk(doc["content"], chunk_size=200)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            embedding = get_embedding(chunk)
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["title"]}]
            )
        print(f"  ✅ {doc['title']}：切成 {len(chunks)} 块")
    print(f"索引建立完成，共 {collection.count()} 个文档块")
```

### 4.3 检索 + 生成回答

```python
def search_v1(query, top_k=3):
    """在向量库中搜索最相关的文档块"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]

def rag_query_v1(question):
    """V1 完整的 RAG 问答流程"""
    print(f"\n{'='*60}")
    print(f"  V1 朴素 RAG")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    # 第一步：检索
    chunks, metadatas = search_v1(question, top_k=3)
    print(f"\n📚 检索到 {len(chunks)} 段相关内容：")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        print(f"\n  [{i+1}] 来源：{meta['source']}")
        print(f"      {chunk[:80]}...")

    # 第二步：把检索到的内容塞进 prompt
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，请如实告知。

参考资料：
{context}

用户问题：{question}

请直接回答，不要编造参考资料中没有的信息。"""

    # 第三步：让大模型生成回答
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n✅ 回答：\n{answer}")
    return answer
```

### 4.4 跑一下看看

```python
# 先建索引（只需要跑一次）
build_index_v1(documents)

# 测试几个问题
rag_query_v1("入职满5年有几天年假？")
rag_query_v1("出差住酒店的报销标准是多少？")
rag_query_v1("前端项目用什么构建工具？")
```

**运行效果：**

```
正在建立索引...
  ✅ 请假制度：切成 5 块
  ✅ 报销制度：切成 4 块
  ✅ 技术规范：切成 5 块
  ✅ 新人入职指南：切成 5 块
  ✅ 绩效考核制度：切成 4 块
索引建立完成，共 23 个文档块

============================================================
  V1 朴素 RAG
  问题: 入职满5年有几天年假？
============================================================

📚 检索到 3 段相关内容：

  [1] 来源：请假制度
      公司请假制度（2024年修订版）一、年假规定 1. 入职满1年不满10年的员工...

  [2] 来源：请假制度
      5. 年假可以分次使用，但每次不少于半天。6. 当年未使用的年假可以...

  [3] 来源：新人入职指南
      新员工入职指南 一、入职第一天...

✅ 回答：
根据公司请假制度，入职满1年不满10年的员工，每年享有5天带薪年假。
所以入职满5年的员工，每年有5天年假。
```

V1 跑起来了！但仔细看输出，有几个不太对劲的地方：

1. **切块太粗暴**——按 200 字硬切，可能把一句话从中间劈开。"入职满10年"可能被切到下一块里去了。
2. **检索结果没有过滤**——召回的第三段"新人入职指南"跟年假问题根本无关，白白浪费 token。
3. **没有重排序**——向量搜索找到的"最相似"不一定是"最相关"，有时候需要精排一下。

这三个问题，V2 来解决。

---

## 五、V2：加上分块重叠 + 相似度过滤

### 5.1 改进切块：加重叠区域

V1 的硬切问题怎么解决？很自然的想法——让相邻的块之间有一段重叠。一句话即使被切开了，在重叠区里还能保住完整的上下文。

> **正文配图 2 提示词：**
> A side-by-side visual comparison diagram. Left side: three rectangular blocks placed end-to-end with a visible crack/gap between them, labeled with a red X. Right side: three rectangular blocks slightly overlapping each other like shingles on a roof, with the overlap zone highlighted in a soft yellow glow, labeled with a green checkmark. Clean flat design, white background, blocks in blue tones. No text. Aspect ratio 16:9.

```python
def overlap_chunk(text, chunk_size=200, overlap=50):
    """带重叠的分块：相邻块之间有 overlap 个字符的重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # 关键：每次只前进 (chunk_size - overlap)
    return chunks
```

用一张图理解：

```
V1（硬切，无重叠）：
  [----块1----][----块2----][----块3----]
                ↑ 这里断裂了

V2（有重叠）：
  [----块1----]
          [----块2----]
                  [----块3----]
          ↑ 重叠区保住了上下文
```

### 5.2 加相似度过滤

V1 召回的结果里混入了不相关的内容。解决办法：给相似度设一个阈值，低于阈值的直接扔掉。

```python
collection_v2 = chroma_client.create_collection(
    name="company_docs_v2",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

def build_index_v2(documents):
    """V2：重叠切块 + 入库"""
    print("正在建立 V2 索引...")
    for doc in documents:
        chunks = overlap_chunk(doc["content"], chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_v2_chunk_{i}"
            embedding = get_embedding(chunk)
            collection_v2.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["title"]}]
            )
        print(f"  ✅ {doc['title']}：切成 {len(chunks)} 块（带重叠）")
    print(f"V2 索引建立完成，共 {collection_v2.count()} 个文档块")

def search_v2(query, top_k=5, threshold=0.6):
    """带相似度过滤的检索"""
    query_embedding = get_embedding(query)
    results = collection_v2.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    filtered_chunks = []
    filtered_metas = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1 - distance  # ChromaDB 余弦距离转相似度
        if similarity >= threshold:
            filtered_chunks.append(doc)
            filtered_metas.append({**meta, "similarity": round(similarity, 3)})

    return filtered_chunks, filtered_metas

def rag_query_v2(question):
    """V2 改进版 RAG：重叠分块 + 相似度过滤"""
    print(f"\n{'='*60}")
    print(f"  V2 改进版 RAG")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    chunks, metadatas = search_v2(question, top_k=5, threshold=0.3)

    if not chunks:
        print("\n❌ 未找到足够相关的文档内容")
        return "抱歉，知识库中没有找到与您的问题相关的内容。"

    print(f"\n📚 检索到 {len(chunks)} 段相关内容（已过滤低相关度）：")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        print(f"\n  [{i+1}] 来源：{meta['source']}（相似度：{meta['similarity']}）")
        print(f"      {chunk[:80]}...")

    context = "\n\n---\n\n".join(chunks)
    prompt = f"""请根据以下参考资料回答用户的问题。

要求：
1. 只基于参考资料中的信息回答，不要编造
2. 如果参考资料不足以回答问题，如实告知
3. 回答要准确、简洁、有条理

参考资料：
{context}

用户问题：{question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n✅ 回答：\n{answer}")
    return answer
```

**测试：**

```python
build_index_v2(documents)
rag_query_v2("病假工资怎么算？超过30天怎么办？")
```

**运行效果：**

```
============================================================
  V2 改进版 RAG
  问题: 病假工资怎么算？超过30天怎么办？
============================================================

📚 检索到 3 段相关内容（已过滤低相关度）：

  [1] 来源：请假制度（相似度：0.82）
      二、病假规定 1. 病假需提供正规医院出具的诊断证明和病假条。2. 3天以内...

  [2] 来源：请假制度（相似度：0.71）
      ...4. 病假期间工资按基本工资的80%发放。5. 连续病假超过30天的...

  [3] 来源：请假制度（相似度：0.45）
      三、事假规定 1. 事假为无薪假期...

✅ 回答：
根据公司请假制度：
1. 病假期间工资按基本工资的80%发放。
2. 连续病假超过30天的，按公司长期病假政策处理。

需要注意的是，病假需要提供正规医院的诊断证明和病假条。
3天以内由直属上级审批，3天以上需部门总监审批。
```

对比 V1，V2 的改进很明显：

- 分块重叠后，"病假工资80%"和"超过30天"这两个信息点不再被切断
- 相似度过滤后，不相关的"新人入职指南"没有再混进来
- 每条检索结果都带了相似度分数，一目了然

不过还有个小毛病：向量搜索的排序不是完美的。有时候相似度 0.45 的那段（事假规定）排在前面，真正相关的反而靠后。接下来 V3 用 Rerank 来收拾这个问题。

---

## 六、V3：加上 Rerank 重排序

### 什么是 Rerank？

向量搜索是"粗筛"——速度快但不够精准，就像你在图书馆电脑上搜，搜出来一堆沾边的。

Rerank 是"精排"——拿到粗筛结果后，用一个更强的模型挨个看"这段内容到底跟用户的问题有多大关系"，然后重新排个序。

> **正文配图 3 提示词：**
> A funnel-shaped diagram illustrating the Rerank concept. Top of funnel: many scattered document icons (6-8 pieces) in light gray, representing raw search results. Middle of funnel: a filter/sieve layer with a magnifying glass icon. Bottom of funnel: 3 neatly stacked document icons in vibrant blue, representing the top-ranked results. An arrow on the side labeled with stars indicates quality improvement. Flat design, white background, blue and gold accents. No text. Aspect ratio 16:9.

```
V2 流程：  用户提问 → 向量搜索（粗筛） → 塞进 prompt
V3 流程：  用户提问 → 向量搜索（粗筛） → Rerank（精排） → 取 top N → 塞进 prompt
```

实际项目中 Rerank 通常用专门的模型（比如 Cohere Rerank、BGE-Reranker），但这里我们用大模型来模拟这个过程，让你理解原理：

```python
def rerank_with_llm(query, chunks, top_n=3):
    """用大模型对检索结果做重排序"""
    if len(chunks) <= top_n:
        return list(range(len(chunks)))

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\n[段落{i+1}]\n{chunk[:150]}...\n"

    prompt = f"""你是一个文档相关性评估专家。用户的问题是："{query}"

以下是从知识库中检索到的多个段落，请根据与用户问题的相关性进行排序。

{chunks_text}

请按相关性从高到低排列段落编号，只输出编号列表，格式如：1,3,2,5,4
不要输出任何解释，只输出逗号分隔的编号。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        order_str = response.choices[0].message.content.strip()
        order = [int(x.strip()) - 1 for x in order_str.split(",")]
        return order[:top_n]
    except:
        return list(range(top_n))

def rag_query_v3(question):
    """V3 完整版 RAG：重叠分块 + 相似度过滤 + Rerank"""
    print(f"\n{'='*60}")
    print(f"  V3 Rerank 版 RAG")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    # 第一步：粗筛——向量检索，多召回一些
    chunks, metadatas = search_v2(question, top_k=6, threshold=0.25)

    if not chunks:
        return "抱歉，知识库中没有找到相关内容。"

    print(f"\n📚 粗筛：向量检索召回 {len(chunks)} 段")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        print(f"  [{i+1}] {meta['source']}（相似度：{meta['similarity']}）- {chunk[:50]}...")

    # 第二步：精排——Rerank 重排序
    print(f"\n🔄 精排：Rerank 重排序中...")
    reranked_indices = rerank_with_llm(question, chunks, top_n=3)

    reranked_chunks = [chunks[i] for i in reranked_indices]
    reranked_metas = [metadatas[i] for i in reranked_indices]

    print(f"\n📚 精排后 Top {len(reranked_chunks)} 段：")
    for i, (chunk, meta) in enumerate(zip(reranked_chunks, reranked_metas)):
        print(f"  [{i+1}] {meta['source']}（相似度：{meta['similarity']}）- {chunk[:50]}...")

    # 第三步：生成回答
    context = "\n\n---\n\n".join(reranked_chunks)
    prompt = f"""请根据以下参考资料回答用户的问题。

要求：
1. 严格基于参考资料回答，不编造信息
2. 如果信息不足，如实告知
3. 如果涉及数字、金额、时间等关键信息，务必准确引用
4. 回答要结构清晰，重点突出

参考资料：
{context}

用户问题：{question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n✅ 回答：\n{answer}")
    return answer
```

**用一个稍微复杂的问题测试：**

```python
rag_query_v3("我是新入职的员工，试用期工资打几折？转正后怎么评定绩效？拿到S评分有什么奖励？")
```

**运行效果：**

```
============================================================
  V3 Rerank 版 RAG
  问题: 我是新入职的员工，试用期工资打几折？转正后怎么评定绩效？拿到S评分有什么奖励？
============================================================

📚 粗筛：向量检索召回 6 段
  [1] 新人入职指南（相似度：0.72）- 三、试用期（3个月）1. 试用期工资为正式...
  [2] 绩效考核制度（相似度：0.65）- 四、考核结果应用 1. S（卓越，前10%）...
  [3] 绩效考核制度（相似度：0.58）- 一、考核周期 1. 公司采用季度考核制...
  [4] 新人入职指南（相似度：0.51）- 一、入职第一天 1. 上午9:00到前台...
  [5] 请假制度（相似度：0.31）- 一、年假规定...
  [6] 报销制度（相似度：0.27）- 三、日常办公费用报销...

🔄 精排：Rerank 重排序中...

📚 精排后 Top 3 段：
  [1] 新人入职指南（相似度：0.72）- 三、试用期（3个月）1. 试用期工资为正式...
  [2] 绩效考核制度（相似度：0.65）- 四、考核结果应用 1. S（卓越，前10%）...
  [3] 绩效考核制度（相似度：0.58）- 一、考核周期 1. 公司采用季度考核制...

✅ 回答：
根据公司制度，为您解答以下三个问题：

**1. 试用期工资**
试用期为3个月，期间工资为正式工资的 90%（即打九折）。

**2. 转正后绩效评定**
公司采用季度考核制，从四个维度评分：
- 业务成果（50%）：KPI 完成情况
- 专业能力（20%）：技术和问题解决能力
- 协作沟通（15%）：跨部门协作和知识分享
- 文化价值观（15%）：主动性、责任心、创新意识

流程为：员工自评 → 上级评分 → 部门校准 → 公司校准 → 反馈结果。

**3. S 评分奖励**
S 评分为"卓越"级别（前10%），可获得：
- 晋升优先考虑
- 年终奖系数 2.0
- 额外期权奖励
```

这个问题横跨两篇文档（新人入职指南 + 绩效考核制度），V3 先粗筛拿到了 6 段候选，然后 Rerank 把最靠谱的 3 段捞出来，"请假制度"和"报销制度"这俩不沾边的就被淘汰了。

---

## 七、三个版本放一起看

```
┌────────┬──────────────────┬───────────────────┬────────────────────────┐
│        │  V1 朴素版        │  V2 改进版         │  V3 Rerank 版          │
├────────┼──────────────────┼───────────────────┼────────────────────────┤
│ 切块   │ 固定大小硬切      │ + 滑动窗口重叠     │ 同 V2                  │
│ 检索   │ 向量搜索 Top K   │ + 相似度阈值过滤   │ + Rerank 重排序        │
│ 准确率 │ 凑合             │ 明显提升           │ 再上一个台阶           │
│ 成本   │ 最低             │ 基本一样           │ 多一次 LLM 调用        │
│ 适合   │ 快速验证         │ 生产环境基础版      │ 对准确率要求高的场景    │
└────────┴──────────────────┴───────────────────┴────────────────────────┘
```

---

## 八、完整代码（直接复制就能跑）

```python
import os
from openai import OpenAI
import chromadb

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()
chroma_client = chromadb.Client()

# ==================== 测试文档 ====================

documents = [
    {
        "id": "doc1", "title": "请假制度",
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
        "id": "doc2", "title": "报销制度",
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
        "id": "doc3", "title": "技术规范",
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
    {
        "id": "doc4", "title": "新人入职指南",
        "content": """新员工入职指南

一、入职第一天
1. 上午 9:00 到前台报到，HR 会带你完成入职手续。
2. 领取工牌、笔记本电脑和办公用品。
3. 电脑预装了企业微信、OA 系统、Git 客户端和 VS Code。
4. 中午 HR 会带你认识团队成员，并安排一位导师（Buddy）。
5. 下午完成企业微信激活和 OA 系统账号开通。

二、入职第一周
1. 阅读公司文化手册和部门业务介绍文档。
2. 完成信息安全培训（线上课程，约2小时）。
3. 和导师一起熟悉项目代码仓库和开发流程。
4. 参加周五的团队周会，了解当前迭代进展。

三、试用期（3个月）
1. 试用期工资为正式工资的 90%。
2. 第一个月结束时进行一次非正式的 1v1 反馈。
3. 试用期满前两周由导师和主管进行转正评估。
4. 转正答辩需准备 PPT，内容包括：工作总结、学习收获、未来计划。
5. 转正通过后工资调整为100%，并开始享受完整的福利待遇。

四、常用系统
1. OA 系统：请假、报销、审批都在这里。地址 oa.company.com。
2. Git 仓库：代码托管在内部 GitLab，地址 git.company.com。
3. 文档平台：技术文档和产品文档在 Confluence，地址 wiki.company.com。
4. 沟通工具：日常沟通用企业微信，技术讨论用 Slack。"""
    },
    {
        "id": "doc5", "title": "绩效考核制度",
        "content": """绩效考核管理办法

一、考核周期
1. 公司采用季度考核制，每季度末进行一次绩效评估。
2. 年终综合评定取四个季度的加权平均分。

二、考核维度
1. 业务成果（50%）：KPI 完成情况，以可量化的指标为主。
2. 专业能力（20%）：技术能力、业务理解、问题解决能力。
3. 协作与沟通（15%）：跨部门协作、知识分享、团队贡献。
4. 文化价值观（15%）：主动性、责任心、创新意识。

三、考核流程
1. 员工先进行自评，填写当季工作总结和自我打分。
2. 直属上级根据实际表现进行评分，并撰写评语。
3. 部门总监进行校准，确保部门内评分标准一致。
4. HR 汇总结果并进行公司级校准。
5. 最终结果在下季度第一周内反馈给员工。

四、考核结果应用
1. S（卓越，前10%）：晋升优先、年终奖系数 2.0、额外期权奖励。
2. A（优秀，前30%）：年终奖系数 1.5、加薪幅度 15%-20%。
3. B（良好，中间50%）：年终奖系数 1.0、加薪幅度 5%-10%。
4. C（待改进，后10%）：制定改进计划（PIP），无年终奖。
5. 连续两个季度 C 评分，公司有权解除劳动合同。"""
    }
]

# ==================== 工具函数 ====================

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

def simple_chunk(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def overlap_chunk(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ==================== V1 朴素版 ====================

collection_v1 = chroma_client.create_collection(name="company_docs_v1")

def build_index_v1(docs):
    print("建立 V1 索引...")
    for doc in docs:
        for i, chunk in enumerate(simple_chunk(doc["content"])):
            collection_v1.add(
                ids=[f"{doc['id']}_c{i}"], embeddings=[get_embedding(chunk)],
                documents=[chunk], metadatas=[{"source": doc["title"]}]
            )
    print(f"V1 索引完成，共 {collection_v1.count()} 块")

def rag_v1(question):
    print(f"\n{'='*50}\n  V1 朴素版 | 问题: {question}\n{'='*50}")
    qe = get_embedding(question)
    r = collection_v1.query(query_embeddings=[qe], n_results=3)
    chunks, metas = r["documents"][0], r["metadatas"][0]
    for i, (c, m) in enumerate(zip(chunks, metas)):
        print(f"  [{i+1}] {m['source']}: {c[:60]}...")
    context = "\n---\n".join(chunks)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "user", "content": f"根据以下资料回答问题，不要编造。\n\n资料：\n{context}\n\n问题：{question}"}
    ])
    print(f"\n✅ {resp.choices[0].message.content}")

# ==================== V2 改进版 ====================

collection_v2 = chroma_client.create_collection(name="company_docs_v2", metadata={"hnsw:space": "cosine"})

def build_index_v2(docs):
    print("建立 V2 索引...")
    for doc in docs:
        for i, chunk in enumerate(overlap_chunk(doc["content"])):
            collection_v2.add(
                ids=[f"{doc['id']}_v2c{i}"], embeddings=[get_embedding(chunk)],
                documents=[chunk], metadatas=[{"source": doc["title"]}]
            )
    print(f"V2 索引完成，共 {collection_v2.count()} 块")

def rag_v2(question):
    print(f"\n{'='*50}\n  V2 改进版 | 问题: {question}\n{'='*50}")
    qe = get_embedding(question)
    r = collection_v2.query(query_embeddings=[qe], n_results=5, include=["documents","metadatas","distances"])
    chunks, metas, scores = [], [], []
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        sim = 1 - dist
        if sim >= 0.3:
            chunks.append(doc); metas.append(meta); scores.append(sim)
    for i, (c, m, s) in enumerate(zip(chunks, metas, scores)):
        print(f"  [{i+1}] {m['source']}(相似度{s:.2f}): {c[:60]}...")
    context = "\n---\n".join(chunks)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "user", "content": f"根据以下资料回答问题，不编造。\n\n资料：\n{context}\n\n问题：{question}"}
    ])
    print(f"\n✅ {resp.choices[0].message.content}")

# ==================== V3 Rerank 版 ====================

def rerank(query, chunks, top_n=3):
    if len(chunks) <= top_n:
        return list(range(len(chunks)))
    text = "".join(f"\n[段落{i+1}]\n{c[:150]}...\n" for i, c in enumerate(chunks))
    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
        {"role": "user", "content": f'问题："{query}"\n{text}\n按相关性从高到低排列段落编号，只输出逗号分隔的编号：'}
    ])
    try:
        return [int(x.strip())-1 for x in resp.choices[0].message.content.strip().split(",")][:top_n]
    except:
        return list(range(top_n))

def rag_v3(question):
    print(f"\n{'='*50}\n  V3 Rerank版 | 问题: {question}\n{'='*50}")
    qe = get_embedding(question)
    r = collection_v2.query(query_embeddings=[qe], n_results=6, include=["documents","metadatas","distances"])
    chunks, metas = [], []
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        if 1-dist >= 0.25:
            chunks.append(doc); metas.append(meta)
    print(f"  粗筛：{len(chunks)} 段")
    order = rerank(question, chunks)
    chunks = [chunks[i] for i in order]
    metas = [metas[i] for i in order]
    print(f"  精排后 Top {len(chunks)}：")
    for i, (c, m) in enumerate(zip(chunks, metas)):
        print(f"    [{i+1}] {m['source']}: {c[:60]}...")
    context = "\n---\n".join(chunks)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "user", "content": f"根据以下资料回答，涉及数字务必准确，不编造。\n\n资料：\n{context}\n\n问题：{question}"}
    ])
    print(f"\n✅ {resp.choices[0].message.content}")

# ==================== 运行 ====================

if __name__ == "__main__":
    build_index_v1(documents)
    build_index_v2(documents)

    q = "我是新入职的员工，试用期工资打几折？转正后怎么评定绩效？拿到S评分有什么奖励？"
    print("\n" + "🟢" * 25)
    print("三个版本处理同一个问题")
    print("🟢" * 25)

    rag_v1(q)
    rag_v2(q)
    rag_v3(q)
```

---

## 九、接下来可以折腾啥？

跑通这三个版本，RAG 的核心套路你就摸清了。想继续折腾的话，几个方向可以挖：

**分块策略还能更聪明** — 本文按字数切已经够用了，但实际项目里可以按段落切、按标题切，甚至用语义分块（让模型自己判断该在哪里切）。分块这一步做好了，后面怎么调都事半功倍。

**Rerank 换专业选手** — 用大模型做 Rerank 是为了演示原理，生产上建议用专门的 Reranker（Cohere Rerank、BGE-Reranker 之类），又快又准。

**上混合检索** — 纯向量检索有个死穴：对关键词和数字不敏感。搜"5天年假"可能搜不到那个"5"。加上 BM25 关键词检索做混合搜索，这个问题就解决了。下篇文章会专门讲。

**接真实文档** — PDF、Word、网页怎么解析？推荐 `unstructured` 或 `LlamaIndex` 的 Document Loader，格式转换这种脏活它们干得挺好。

**加对话记忆** — 现在每次提问都是独立的。加上对话历史管理，用户就可以追问"那事假呢？"而不用把前面的上下文再说一遍。

---

## 写在最后

回头看这三个版本，核心思路就一句话：**先搜再答**。V1 解决"能不能搜到"，V2 解决"搜得准不准"，V3 解决"排得对不对"。

仔细想想，这跟你自己查资料其实一模一样——先搜一堆结果出来（V1），扫一眼标题把不靠谱的划掉（V2），最后点进去仔细读读哪个最对路（V3）。RAG 不过是把这套操作给自动化了而已。

上篇文章我们造了个"会用工具的 AI"，这篇造了个"会查资料的 AI"。下一篇，我们把这俩合体——让智能体**自己决定什么时候该查资料、查哪个知识库、查完不满意再换个姿势查**。那就是 Agentic RAG，也是这个系列的终章。

> **正文配图 4 提示词：**
> A visual evolution roadmap showing three stages from left to right. Stage 1: a simple magnifying glass over a single document (labeled V1). Stage 2: the magnifying glass now has a filter/funnel attached, over multiple documents with some crossed out (labeled V2). Stage 3: the magnifying glass combined with a small brain icon, documents neatly ranked with numbers 1-2-3 (labeled V3). Connected by curved arrows. Clean flat design, dark navy background, neon blue and orange highlights. No text labels. Aspect ratio 16:9.

评论区聊，下篇见。
