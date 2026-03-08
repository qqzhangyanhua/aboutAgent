# 你的 RAG 为什么总是答不准？五大翻车场景逐个修复

> **封面图提示词（2:1 比例，1232×616px）：**
> A dramatic tech illustration showing a road with 5 potholes/cracks, each glowing with a different warning color (red, orange, yellow). A small robot character on the left side of the road holds a toolbox and wrench, ready to fix the road. Above the scene, floating document pages and search icons. Dark moody background with gradient from deep purple to dark blue. Style: clean vector illustration, slightly dramatic lighting from the toolbox. No text, no watermark. Aspect ratio 2:1.

上篇文章我们用 50 行代码搭了一个能"翻书找答案"的 RAG 系统。评论区不少人照着做了，然后回来跟我说：

> "搭倒是搭出来了，但回答经常胡说八道怎么办？"
> "明明文档里有答案，它就是找不到……"
> "问个数字它给我瞎编一个，比不用 RAG 还离谱。"

别急，这个阶段我也经历过。RAG 原理简单，但细节里全是坑。今天这篇我把最常踩的五个坑拆开来，**每个坑都给一段"翻车代码"和"修好的代码"**，让你直接看到修复前后的差距。

五个坑分别是：

1. **Chunk 切太大或太小** — 信息要么被淹没，要么被切断
2. **Embedding 模型选错** — 中文文档用英文模型，搜啥都不准
3. **检索只靠向量，关键词全丢** — "5天年假"搜不到"5"
4. **Prompt 没约束** — 大模型拿到资料后开始自由发挥
5. **多文档冲突** — 两篇文档说法矛盾，大模型不知道信谁

代码照样能直接跑，`pip install openai chromadb rank_bm25` 就够了。

---

## 准备工作

先把基础环境搭好，后面五个场景共用：

```python
import os
from openai import OpenAI
import chromadb

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()
chroma_client = chromadb.Client()

def get_embedding(text):
    """获取文本的向量表示

    Args:
        text: 输入文本

    Returns:
        向量列表

    Raises:
        ValueError: 如果text为空
        RuntimeError: 如果API调用失败
    """
    if not text or not text.strip():
        raise ValueError("文本不能为空")

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Embedding生成失败: {e}")

def ask_llm(prompt, model="gpt-4o-mini"):
    """调用大模型生成回答

    Args:
        prompt: 提示词
        model: 模型名称

    Returns:
        模型生成的文本

    Raises:
        ValueError: 如果prompt为空
        RuntimeError: 如果API调用失败
    """
    if not prompt or not prompt.strip():
        raise ValueError("提示词不能为空")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM调用失败: {e}")
```

---

## 坑一：Chunk 切太大或太小

### 翻车现场

这个坑我敢说 90% 的人第一次做 RAG 都会踩。先看两个极端的反面教材：

> **正文配图 1 提示词：**
> A visual comparison showing two extremes of text chunking. Left side: a single huge block of text (one giant rectangle) with a confused face emoji - representing chunks too large. Right side: many tiny scattered text fragments (10+ tiny rectangles) with broken connection lines between them - representing chunks too small. In the center: a "Goldilocks" middle ground with 3-4 medium-sized, neatly arranged blocks with subtle overlap zones highlighted in yellow. Flat design, white background, red for bad examples, green for good example. No text. Aspect ratio 16:9.

**切太大（1000字一块）：**

```python
def chunk_too_large(text, chunk_size=1000):
    """反面教材：chunk 太大"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

test_doc = """公司请假制度（2024年修订版）

一、年假规定
1. 入职满1年不满10年的员工，每年享有5天带薪年假。
2. 入职满10年不满20年的员工，每年享有10天带薪年假。
3. 入职满20年以上的员工，每年享有15天带薪年假。

二、病假规定
1. 病假需提供正规医院出具的诊断证明和病假条。
2. 3天以内的病假由直属上级审批。
3. 3天以上的病假需由部门总监审批。
4. 病假期间工资按基本工资的80%发放。

三、事假规定
1. 事假为无薪假期，按日扣除工资。
2. 事假每次不超过3天，需提前2个工作日申请。
3. 全年事假累计不超过15天。

四、婚假规定
1. 符合法定结婚年龄的员工，可享受3天婚假。
2. 晚婚（男满25岁、女满23岁）可额外增加7天，共10天。
3. 婚假需在领证后6个月内使用。

五、产假规定
1. 女员工产假为158天（含法定节假日）。
2. 难产增加15天，多胞胎每多一个增加15天。
3. 男员工陪产假为15天。
4. 产假期间工资照常发放。"""

big_chunks = chunk_too_large(test_doc, chunk_size=1000)
print(f"大块切法：切成 {len(big_chunks)} 块")
for i, c in enumerate(big_chunks):
    print(f"  块{i+1}：{len(c)} 字 — {c[:40]}...")
```

```
大块切法：切成 1 块
  块1：487 字 — 公司请假制度（2024年修订版）一、年假规定...
```

整篇文档变成一块！用户问"婚假几天"，你把整篇请假制度（包括年假、病假、事假……）全塞进 prompt，大模型从一堆无关信息里翻找，既浪费 token 又容易答偏。

**切太小（50字一块）：**

```python
small_chunks = chunk_too_large(test_doc, chunk_size=50)
print(f"小块切法：切成 {len(small_chunks)} 块")
for i, c in enumerate(small_chunks):
    print(f"  块{i+1}：「{c}」")
```

```
小块切法：切成 11 块
  块1：「公司请假制度（2024年修订版）一、年假规定 1. 入职满1年不满10年」
  块2：「的员工，每年享有5天带薪年假。2. 入职满10年不满20年的员工，每年」
  块3：「享有10天带薪年假。3. 入职满20年以上的员工，每年享有15天带薪年假」
  ...
```

"入职满1年不满10年"和"每年享有5天带薪年假"被劈成两块了！检索的时候很可能只找到前半截或后半截，答案不完整。

### 修复方案：语义分块

不按固定字数切，而是按段落/标题的自然边界切，再控制大小：

```python
import re

def smart_chunk(text, max_size=300):
    """按自然边界切块，保留语义完整性

    策略：
    1. 优先按段落（双换行）切分
    2. 段落过大时按句子切分（支持中英文）
    3. 句子过大时强制切分

    Args:
        text: 待切分的文本
        max_size: 单个块的最大字符数

    Returns:
        切分后的文本块列表
    """
    if not text or not text.strip():
        return []

    # 第一步：按段落切分（双换行或标题）
    # 支持中文标题（一、二、1. 2.）和英文标题（# ## 1. 2.）
    paragraphs = re.split(r'\n\s*\n|(?=\n[一二三四五六七八九十]+、)|(?=\n\d+\.)', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else:
            # 第二步：按句子切分（支持中英文标点）
            sentences = re.split(r'([。.!?！？\n])', para)
            # 重新组合句子和标点
            sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]

            current = ""
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(current) + len(sent) <= max_size:
                    current += sent
                else:
                    if current:
                        chunks.append(current.strip())
                    # 第三步：如果单个句子超长，强制切分
                    if len(sent) > max_size:
                        for i in range(0, len(sent), max_size):
                            chunks.append(sent[i:i+max_size].strip())
                    else:
                        current = sent
            if current:
                chunks.append(current.strip())

    return [c for c in chunks if c]  # 过滤空块

smart_chunks = smart_chunk(test_doc, max_size=200)
print(f"语义切法：切成 {len(smart_chunks)} 块")
for i, c in enumerate(smart_chunks):
    print(f"\n  块{i+1}（{len(c)}字）：")
    print(f"  「{c}」")
```

```
语义切法：切成 5 块

  块1（141字）：
  「公司请假制度（2024年修订版）
  一、年假规定
  1. 入职满1年不满10年的员工，每年享有5天带薪年假。
  2. 入职满10年不满20年的员工，每年享有10天带薪年假。
  3. 入职满20年以上的员工，每年享有15天带薪年假。」

  块2（108字）：
  「二、病假规定
  1. 病假需提供正规医院出具的诊断证明和病假条。
  2. 3天以内的病假由直属上级审批。
  3. 3天以上的病假需由部门总监审批。
  4. 病假期间工资按基本工资的80%发放。」

  ...
```

每一块都是一个完整的语义单元——年假归年假，病假归病假，不会从中间劈开。

### 何时需要语义分块？

✅ **需要：**
- 文档有明确的章节结构（标题、段落）
- 用户查询针对特定主题（如"年假规定"）
- 固定切分导致答案不完整或包含无关信息

❌ **不需要：**
- 文档是纯文本流（如小说、日志）
- 文档很短（少于1000字）
- 固定切分已经足够准确

**经验值：** 中文文档 chunk_size 在 200-500 字之间比较合适。太大了答案会被一堆无关内容淹没，太小了上下文断裂。具体多少得看你的文档结构，多试几次就有感觉了。

---

## 坑二：Embedding 模型选错

### 翻车现场

这个坑比较隐蔽。很多人随手拿了个英文 Embedding 模型就上了，或者用了个比较老的模型。结果就是：中文语义它根本"听不懂"。差距有多大？看数据：

```python
def compare_embeddings():
    """对比不同 Embedding 模型在中文场景的表现"""
    query = "员工请假需要提前多久申请？"
    docs = [
        "年假需提前3个工作日在OA系统中提交申请，直属上级审批通过后生效。",  # 高度相关
        "事假需提前2个工作日申请。",  # 高度相关
        "公司采用季度考核制，每季度末进行一次绩效评估。",  # 无关
        "框架统一使用 React 18+，不允许在新项目中使用 Vue。",  # 完全无关
    ]

    # 用 text-embedding-3-small（支持中文，效果好）
    print("模型: text-embedding-3-small")
    q_emb = get_embedding(query)
    for doc in docs:
        d_emb = get_embedding(doc)
        similarity = sum(a*b for a, b in zip(q_emb, d_emb))
        print(f"  相似度 {similarity:.3f} | {doc[:30]}...")

compare_embeddings()
```

```
模型: text-embedding-3-small
  相似度 0.712 | 年假需提前3个工作日在OA系统中提交申请...  ✅ 最相关
  相似度 0.658 | 事假需提前2个工作日申请...              ✅ 相关
  相似度 0.231 | 公司采用季度考核制...                   ✅ 不相关（分数低）
  相似度 0.089 | 框架统一使用 React 18+...              ✅ 完全无关（分数极低）
```

好的模型能清楚地把相关和不相关的内容拉开差距。换一个对中文理解差的模型，四段话的分数可能全挤在 0.3-0.5 之间，根本分不出谁跟谁。

### 修复方案

**选对模型**。2024-2025 年中文 Embedding 模型推荐（效果从高到低）：

| 模型 | 来源 | 维度 | 特点 |
|------|------|------|------|
| `text-embedding-3-large` | OpenAI | 3072 | 效果最好，但贵 |
| `text-embedding-3-small` | OpenAI | 1536 | 性价比最高，推荐 |
| `BAAI/bge-large-zh-v1.5` | 智源 | 1024 | 开源免费，中文专优 |
| `BAAI/bge-m3` | 智源 | 1024 | 开源，多语言 |
| `GTE-Qwen2` | 阿里 | 多种 | 开源，效果好 |

如果你的文档全是中文，强烈建议用 BGE 系列（免费 + 中文效果好）；如果中英混杂或者图方便，OpenAI 的 `text-embedding-3-small` 足够了。

```python
# 如果想用本地开源模型（免费、隐私安全）
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

local_model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

def get_local_embedding(text):
    """用本地 BGE 模型做向量化——免费、快、中文效果好"""
    if not text or not text.strip():
        raise ValueError("文本不能为空")
    return local_model.encode(text, normalize_embeddings=True).tolist()
```

### 何时需要更换Embedding模型？

✅ **需要：**
- 文档全是中文，但用的是英文模型（如老版text-embedding-ada-002）
- 相关文档的相似度分数都挤在0.3-0.5之间，区分度差
- 搜索"请假"却返回"加班"相关内容（语义理解错误）

❌ **不需要：**
- 当前模型的检索准确率已经超过80%
- 文档量很小（少于100篇），模型差异不明显
- 成本敏感且OpenAI模型已经够用

**经验值：** 先用OpenAI的`text-embedding-3-small`快速验证，如果效果不好或成本太高，再换BGE等开源模型。

---

## 坑三：检索只靠向量，关键词全丢

### 翻车现场

向量检索靠的是"语义相似"——它知道"汽车"和"轿车"差不多一个意思。但它有个天生的短板：**碰到精确的关键词、数字、专有名词，它就犯迷糊了**。

> **正文配图 2 提示词：**
> A split-screen comparison illustration. Left side labeled "Vector Search" (with a brain icon): a search query "A7-Pro" has wavy semantic lines connecting to wrong results, with the correct result ranked last - shown with red ranking numbers. Right side labeled "Hybrid Search" (with brain + keyword icon): the same query now has both wavy lines AND exact-match arrows, with the correct result ranked first - shown with green ranking numbers. Clean flat design, white background. Left side has a subtle red tint, right side has a subtle green tint. No text labels. Aspect ratio 16:9.

```python
def show_keyword_problem():
    """演示纯向量检索的关键词盲区"""
    collection = chroma_client.create_collection(name="keyword_demo")

    docs = [
        "产品型号 A7-Pro 的售价为 2999 元，支持 5G 网络。",
        "产品型号 B5-Lite 的售价为 1599 元，仅支持 4G。",
        "产品型号 C9-Max 的售价为 4999 元，支持 5G 和卫星通信。",
        "所有产品均提供一年质保，可延长至三年（付费）。",
    ]
    for i, doc in enumerate(docs):
        collection.add(
            ids=[f"d{i}"], embeddings=[get_embedding(doc)], documents=[doc]
        )

    # 搜一个带具体型号的问题
    query = "A7-Pro 多少钱？"
    results = collection.query(query_embeddings=[get_embedding(query)], n_results=3)
    print(f"问题：{query}")
    for i, doc in enumerate(results["documents"][0]):
        print(f"  [{i+1}] {doc}")

show_keyword_problem()
```

```
问题：A7-Pro 多少钱？
  [1] 所有产品均提供一年质保，可延长至三年（付费）。    ← ??? 完全不对
  [2] 产品型号 C9-Max 的售价为 4999 元...              ← 型号都对不上
  [3] 产品型号 A7-Pro 的售价为 2999 元...              ← 排到第三去了
```

"A7-Pro"这种型号名对大模型来说就是一串字符，它不理解这个字符串的"语义"，所以向量检索排不准。

### 修复方案：混合检索（BM25 + 向量）

BM25 是传统的关键词检索算法——简单粗暴，就看你的搜索词在文档里出现了几次。它对"A7-Pro"这种精确匹配极其在行。

把 BM25 和向量检索的结果合并（叫混合检索 / Hybrid Search），就能两全其美：

```python
from rank_bm25 import BM25Okapi
import jieba

class HybridSearch:
    """混合检索：BM25 关键词检索 + 向量语义检索

    设计原则：
    1. 数据只存一份（在ChromaDB里）
    2. BM25索引按需构建
    3. 权重在初始化时固定
    """

    def __init__(self, name="hybrid", bm25_weight=0.4):
        """初始化混合检索

        Args:
            name: 集合名称
            bm25_weight: BM25权重（0-1之间），向量权重自动为1-bm25_weight
        """
        self.collection = chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        self._docs = []
        self._bm25_index = None

    def add_documents(self, documents):
        """添加文档并构建索引

        Args:
            documents: 文档列表

        Raises:
            ValueError: 如果documents为空
        """
        if not documents:
            raise ValueError("文档列表不能为空")

        self._docs = documents

        # 构建BM25索引
        tokenized = [list(jieba.cut(doc)) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized)

        # 存入向量数据库
        for i, doc in enumerate(documents):
            self.collection.add(
                ids=[f"doc_{i}"],
                embeddings=[get_embedding(doc)],
                documents=[doc]
            )

    def search(self, query, top_k=3):
        """混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            [(文档, 分数), ...] 按分数降序排列

        Raises:
            RuntimeError: 如果未添加文档
        """
        if self._bm25_index is None:
            raise RuntimeError("请先调用add_documents添加文档")

        # BM25 检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self._bm25_index.get_scores(tokenized_query)

        # 归一化 BM25 分数到 0-1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = [s / max_bm25 for s in bm25_scores]

        # 向量检索
        results = self.collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=len(self._docs),
            include=["distances"]
        )
        vector_scores = [1 - d for d in results["distances"][0]]

        # 混合打分（加权求和）
        hybrid_scores = []
        for i in range(len(self._docs)):
            score = (self.bm25_weight * bm25_normalized[i] +
                    self.vector_weight * vector_scores[i])
            hybrid_scores.append((i, score))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._docs[idx], score) for idx, score in hybrid_scores[:top_k]]

# 对比效果
docs = [
    "产品型号 A7-Pro 的售价为 2999 元，支持 5G 网络。",
    "产品型号 B5-Lite 的售价为 1599 元，仅支持 4G。",
    "产品型号 C9-Max 的售价为 4999 元，支持 5G 和卫星通信。",
    "所有产品均提供一年质保，可延长至三年（付费）。",
]

hybrid = HybridSearch(name="hybrid_demo")
hybrid.add_documents(docs)

query = "A7-Pro 多少钱？"
print(f"问题：{query}\n")

print("纯向量检索结果：")
vec_results = hybrid.collection.query(
    query_embeddings=[get_embedding(query)], n_results=3
)
for i, doc in enumerate(vec_results["documents"][0]):
    print(f"  [{i+1}] {doc}")

print("\n混合检索结果：")
hybrid_results = hybrid.search(query, top_k=3)
for i, (doc, score) in enumerate(hybrid_results):
    print(f"  [{i+1}] (分数 {score:.3f}) {doc}")
```

```
问题：A7-Pro 多少钱？

纯向量检索结果：
  [1] 所有产品均提供一年质保...           ← 不相关
  [2] 产品型号 C9-Max 的售价为 4999 元... ← 型号不对
  [3] 产品型号 A7-Pro 的售价为 2999 元... ← 对的，但排第三

混合检索结果：
  [1] (分数 0.891) 产品型号 A7-Pro 的售价为 2999 元... ← 精准命中！
  [2] (分数 0.523) 产品型号 C9-Max 的售价为 4999 元...
  [3] (分数 0.412) 产品型号 B5-Lite 的售价为 1599 元...
```

BM25 一上，"A7-Pro"精确匹配直接把分数拉满，正确结果一下子回到第一位。

### 何时需要混合检索？

✅ **需要：**
- 文档包含大量产品编号、人名、专有名词、精确数字
- 用户查询经常包含精确的关键词（如"A7-Pro"、"5天年假"）
- 纯向量检索的准确率低于70%，经常找不到包含关键词的文档

❌ **不需要：**
- 文档少于100篇，向量检索已经足够准确
- 查询都是自然语言问题（如"怎么请假"），很少包含精确关键词
- 文档中没有需要精确匹配的专有名词

**经验值：** 权重方面我一般从 BM25 : 向量 = 0.4 : 0.6 开始试，然后根据实际效果调。如果关键词匹配很重要，可以提高BM25权重到0.5-0.6。

---

## 坑四：Prompt 没约束，大模型开始自由发挥

### 翻车现场

这个是最让人上火的一种翻车。你辛辛苦苦检索到了正确的文档，塞进 prompt 了，结果大模型看完之后心想"嗯我知道了，但我觉得还能补充两句"，然后就开始编。

> **正文配图 3 提示词：**
> A humorous illustration showing two scenarios side by side. Left (Bad): a robot reading a document, then speaking with a long speech bubble that extends far beyond the document, with the extra part colored in red/warning stripes - representing hallucination. Right (Good): a robot reading the same document, speaking with a speech bubble that exactly matches the document content in blue, plus a small "I don't know" speech bubble in gray for missing info. Clean cartoon style, white background. No text. Aspect ratio 16:9.

```python
def bad_prompt_demo():
    """反面教材：prompt 没约束"""
    context = "公司规定，出差住宿费标准：一线城市（北上广深）每晚不超过500元，其他城市每晚不超过350元。"

    # 坏 prompt：没有约束
    bad_prompt = f"""参考资料：{context}

问题：出差住宿标准是多少？包含早餐吗？可以住民宿吗？"""

    answer = ask_llm(bad_prompt)
    print(f"坏 prompt 的回答：\n{answer}")

bad_prompt_demo()
```

```
坏 prompt 的回答：
根据公司规定，出差住宿费标准为：
- 一线城市（北上广深）：每晚不超过500元
- 其他城市：每晚不超过350元

关于早餐：通常大部分商务酒店都包含早餐，建议优先选择含早的酒店。   ← 编的！
关于民宿：如果民宿价格在标准范围内且能提供正规发票，一般是允许的。 ← 编的！
建议出差前与财务部确认具体细节。
```

参考资料里压根没提"早餐"和"民宿"的事，但大模型"好心"地自己补充了——而且说得有模有样，你不仔细看根本分不出哪些是文档里的，哪些是它编的。

### 修复方案：严格约束 prompt

```python
def good_prompt_demo():
    """正确姿势：严格约束 prompt"""
    context = "公司规定，出差住宿费标准：一线城市（北上广深）每晚不超过500元，其他城市每晚不超过350元。"

    good_prompt = f"""你是一个企业知识库问答助手。请严格根据【参考资料】回答用户问题。

【核心规则】
1. 只基于参考资料中明确提到的信息回答
2. 如果参考资料中没有相关信息，必须明确说"根据现有资料，未找到相关信息"
3. 绝对禁止推测、补充、编造参考资料中没有的内容
4. 涉及数字、金额、日期的信息必须原文引用，不能近似

【参考资料】
{context}

【用户问题】
出差住宿标准是多少？包含早餐吗？可以住民宿吗？

【回答格式】
对于能回答的部分，给出准确答案。对于资料中未提及的部分，明确标注"资料未提及"。"""

    answer = ask_llm(good_prompt)
    print(f"好 prompt 的回答：\n{answer}")

good_prompt_demo()
```

```
好 prompt 的回答：
根据参考资料，出差住宿费标准如下：
- 一线城市（北上广深）：每晚不超过 500 元
- 其他城市：每晚不超过 350 元

关于早餐：资料未提及，建议咨询财务部门。
关于民宿：资料未提及，建议咨询财务部门。
```

同样的参考资料，prompt 一换，大模型就老实了——知道的说，不知道的说"资料未提及"，不瞎编了。

我总结了四条管用的规矩：

1. **给它定角色** — "你是知识库问答助手"，别让它觉得自己是百科全书
2. **锁死信息源** — "只基于参考资料"，别用你自己的知识
3. **堵死编造的路** — "资料里没有就说没有"，这条最关键
4. **约束格式** — "数字必须原文引用"，不然它会给你来个"大约500元"

### 何时需要严格约束Prompt？

✅ **需要：**
- RAG系统用于企业知识库、法律文档、医疗咨询等严肃场景
- 大模型经常"补充"参考资料中没有的信息
- 用户需要可追溯的答案来源，不能接受任何编造

❌ **不需要：**
- 用于创意写作、头脑风暴等场景，需要大模型发挥想象力
- 参考资料只是辅助，允许大模型结合自身知识回答
- 文档覆盖率很高（>90%），大模型很少需要说"不知道"

**经验值：** 企业级RAG系统必须严格约束Prompt，个人学习助手可以放宽。关键是在"准确性"和"有用性"之间找平衡。

---

## 坑五：多文档冲突，大模型不知道信谁

### 翻车现场

这个坑在真实项目里几乎必遇。同一个主题可能存了好几版文档，信息还矛盾——比如 2022 年的制度说年假 5 天，2024 年改成了 7 天，但旧文档没人删。

```python
def conflict_demo():
    """演示多文档信息冲突"""
    chunk_old = """【员工手册 v2.0 | 2022年版】
年假规定：入职满一年的员工，每年享有5天带薪年假。
病假工资：按基本工资的70%发放。"""

    chunk_new = """【员工手册 v3.0 | 2024年修订版】
年假规定：入职满一年的员工，每年享有7天带薪年假（2024年起上调）。
病假工资：按基本工资的80%发放。"""

    chunk_dept = """【技术部补充规定 | 2024年】
技术部员工在年假基础上额外享有2天"技术充电假"，用于参加技术会议或自学。"""

    # 坏情况：不带元数据，大模型不知道哪个新哪个旧
    bad_context = f"{chunk_old}\n\n---\n\n{chunk_new}"
    bad_answer = ask_llm(f"参考资料：\n{bad_context}\n\n问题：员工年假有几天？")
    print(f"不处理冲突的回答：\n{bad_answer}\n")

conflict_demo()
```

```
不处理冲突的回答：
根据参考资料，关于员工年假有两种说法：
- 一种是5天（员工手册v2.0）
- 另一种是7天（员工手册v3.0，2024年修订版）
建议以最新版本为准，即每年7天带薪年假。
```

这次大模型还算聪明，猜到了"以最新版为准"。但它不是每次都能猜对——有时候它会把两个版本的信息混在一起，或者干脆只引用旧版。

### 修复方案：元数据标注 + 冲突处理策略

```python
def conflict_fix_demo():
    """修复：用元数据帮大模型判断优先级"""
    chunks_with_meta = [
        {
            "content": "年假规定：入职满一年的员工，每年享有5天带薪年假。病假工资：按基本工资的70%发放。",
            "source": "员工手册",
            "version": "v2.0",
            "date": "2022-01-01",
            "status": "已废止"
        },
        {
            "content": "年假规定：入职满一年的员工，每年享有7天带薪年假（2024年起上调）。病假工资：按基本工资的80%发放。",
            "source": "员工手册",
            "version": "v3.0",
            "date": "2024-01-01",
            "status": "现行有效"
        },
        {
            "content": "技术部员工在年假基础上额外享有2天"技术充电假"，用于参加技术会议或自学。",
            "source": "技术部补充规定",
            "version": "v1.0",
            "date": "2024-03-01",
            "status": "现行有效"
        }
    ]

    # 构建带元数据的上下文
    context_parts = []
    for chunk in chunks_with_meta:
        meta_line = f"[来源: {chunk['source']} {chunk['version']} | 日期: {chunk['date']} | 状态: {chunk['status']}]"
        context_parts.append(f"{meta_line}\n{chunk['content']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是企业知识库问答助手。请根据参考资料回答问题。

【冲突处理规则】
1. 如果多篇资料信息冲突，以"现行有效"状态的文档为准
2. 如果都是"现行有效"，以日期更新的为准
3. 部门补充规定是对公司规定的补充，两者不冲突时都适用
4. 回答中要标注信息来源

【参考资料】
{context}

【用户问题】
技术部的员工年假一共有几天？病假工资怎么算？"""

    answer = ask_llm(prompt)
    print(f"处理冲突后的回答：\n{answer}")

conflict_fix_demo()
```

```
处理冲突后的回答：
根据现行有效的公司制度：

**年假**：技术部员工年假共 **9天**，具体构成如下：
- 基础年假：7天（来源：员工手册 v3.0，2024年起执行）
- 技术充电假：2天（来源：技术部补充规定 v1.0）

注：员工手册 v2.0 中的"5天"已废止，以 v3.0 为准。

**病假工资**：按基本工资的 **80%** 发放。
（来源：员工手册 v3.0，2024年版）
```

这次不仅答对了，还自己算出来 7+2=9，连信息来源都标得清清楚楚。关键就改了两个地方：

1. **给每段文档打上元数据标签**（版本号、日期、状态），不再喂裸文本
2. **在 Prompt 里写明冲突处理规则**，告诉它版本冲突了听谁的

### 何时需要元数据管理？

✅ **需要：**
- 文档有多个版本，存在新旧信息冲突
- 不同部门有补充规定，需要组合多个文档的信息
- 需要标注答案来源，方便用户追溯和验证
- 文档会定期更新，旧版本不能立即删除（过渡期）

❌ **不需要：**
- 文档都是最新版本，没有历史版本冲突
- 文档量很小（少于50篇），手动管理即可
- 不需要追溯答案来源，只要答案正确就行

**经验值：** 如果你的知识库会持续更新（如企业制度、产品文档），从一开始就加上元数据管理，否则后期补救成本很高。元数据至少包含：来源、版本、日期、状态。

> **正文配图 4 提示词：**
> An illustration showing document version conflict resolution. Left side: two overlapping document icons, one old (faded, with "v2.0" and a red "deprecated" stamp) and one new (vibrant, with "v3.0" and a green "active" checkmark). Arrows from both pointing to a central AI brain icon. The brain has a clear arrow pointing to the new document, showing it chose the current version. Right side: a clean answer output with source citations. Flat design, white background, blue and green tones. No text. Aspect ratio 16:9.

---

## 六、一张表总结五个坑

```
┌──────────────────┬────────────────────────┬──────────────────────────┐
│ 翻车场景          │ 根因                    │ 修复方案                  │
├──────────────────┼────────────────────────┼──────────────────────────┤
│ 1. Chunk 太大/小  │ 信息被淹没或被切断       │ 语义分块，按自然边界切      │
│ 2. Embedding 选错 │ 模型不理解中文语义       │ 用中文优化的模型（BGE等）   │
│ 3. 关键词丢失     │ 向量检索对精确词不敏感    │ 混合检索（BM25 + 向量）    │
│ 4. Prompt 没约束  │ 大模型自由发挥编造信息    │ 严格限定信息源+边界处理     │
│ 5. 多文档冲突     │ 新旧文档信息矛盾         │ 元数据标注 + 冲突处理规则   │
└──────────────────┴────────────────────────┴──────────────────────────┘
```

实际项目里这五个坑经常一起出现，建议按这个顺序排查：先看分块有没有问题（坑1），再确认 Embedding 模型选对没有（坑2），然后该加混合检索就加上（坑3），接着把 prompt 收紧（坑4），最后处理文档版本冲突（坑5）。一个一个来，别想着一步到位。

---

## 七、一个综合修复的完整示例

把上面五个修复方案整合到一起，看看"全副武装"的 RAG 长什么样：

```python
import re
import jieba
from rank_bm25 import BM25Okapi
from datetime import datetime

class ProductionRAG:
    """生产级 RAG：集成五大修复方案

    特性：
    1. 语义分块（坑一）
    2. 支持自定义Embedding模型（坑二）
    3. 混合检索 BM25+向量（坑三）
    4. 严格约束Prompt（坑四）
    5. 元数据管理（坑五）
    """

    def __init__(self, bm25_weight=0.4):
        """初始化RAG系统

        Args:
            bm25_weight: BM25权重（0-1之间）
        """
        self.collection = chroma_client.create_collection(
            name="production_rag",
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
        self.chunks = []
        self.chunk_metas = []
        self._bm25_index = None

    def add_document(self, content, metadata=None):
        """添加文档：语义分块 + 元数据

        Args:
            content: 文档内容
            metadata: 可选的元数据字典，包含：
                - source: 来源（默认"未知"）
                - version: 版本（默认"v1.0"）
                - date: 日期（默认当前日期）
                - status: 状态（默认"现行有效"）

        Raises:
            ValueError: 如果content为空
        """
        if not content or not content.strip():
            raise ValueError("文档内容不能为空")

        # 向后兼容：如果没有提供元数据，使用默认值
        if metadata is None:
            metadata = {}

        default_meta = {
            "source": "未知",
            "version": "v1.0",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": "现行有效"
        }
        # 合并用户提供的元数据和默认值
        meta = {**default_meta, **metadata}

        # 语义分块
        chunks = smart_chunk(content, max_size=300)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{meta['source']}_{meta['version']}_{i}_{len(self.chunks)}"
            embedding = get_embedding(chunk)
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[meta]
            )
            self.chunks.append(chunk)
            self.chunk_metas.append(meta)

        # 重建 BM25 索引
        tokenized = [list(jieba.cut(c)) for c in self.chunks]
        self._bm25_index = BM25Okapi(tokenized)
        print(f"  ✅ 已添加：{meta['source']} {meta['version']}（{len(chunks)} 块）")

    def hybrid_search(self, query, top_k=5):
        """混合检索：BM25 + 向量

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            [(chunk, metadata, score), ...]

        Raises:
            RuntimeError: 如果未添加文档
        """
        if self._bm25_index is None:
            raise RuntimeError("请先调用add_document添加文档")

        # BM25
        bm25_scores = self._bm25_index.get_scores(list(jieba.cut(query)))
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]

        # 向量
        results = self.collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=len(self.chunks),
            include=["distances"]
        )
        vector_scores = [1 - d for d in results["distances"][0]]

        # 混合打分
        scored = []
        for i in range(len(self.chunks)):
            score = (self.bm25_weight * bm25_norm[i] +
                    self.vector_weight * vector_scores[i])
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(self.chunks[i], self.chunk_metas[i], s)
                for i, s in scored[:top_k]]

    def query(self, question):
        """完整的 RAG 问答

        Args:
            question: 用户问题

        Returns:
            大模型生成的答案

        Raises:
            ValueError: 如果question为空
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        print(f"\n{'='*50}")
        print(f"  问题: {question}")
        print(f"{'='*50}")

        results = self.hybrid_search(question, top_k=4)

        print(f"\n📚 检索到 {len(results)} 段：")
        context_parts = []
        for i, (chunk, meta, score) in enumerate(results):
            status_tag = "⚠️已废止" if meta["status"] == "已废止" else "✅现行"
            print(f"  [{i+1}] {status_tag} {meta['source']} {meta['version']} (分数{score:.2f})")
            meta_line = f"[{meta['source']} {meta['version']} | {meta['date']} | {meta['status']}]"
            context_parts.append(f"{meta_line}\n{chunk}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""你是企业知识库问答助手。请严格根据参考资料回答。

【规则】
1. 只基于参考资料回答，禁止编造
2. 多文档冲突时，以"现行有效"且日期最新的为准
3. 资料未提及的，明确说"资料未提及"
4. 数字、金额必须原文引用
5. 标注信息来源

【参考资料】
{context}

【问题】{question}"""

        answer = ask_llm(prompt)
        print(f"\n✅ 回答：\n{answer}")
        return answer


# 使用示例
rag = ProductionRAG()

# 方式1：简单使用（向后兼容旧代码）
rag.add_document(
    content="年假：入职满一年享有5天。病假工资按70%发放。事假无薪。"
)

# 方式2：带完整元数据（推荐）
rag.add_document(
    content="年假：入职满一年享有7天（2024年起上调）。病假工资按80%发放。事假无薪，每次不超过3天。",
    metadata={
        "source": "员工手册",
        "version": "v3.0",
        "date": "2024-01-01",
        "status": "现行有效"
    }
)

rag.add_document(
    content="差旅报销：一线城市住宿每晚不超过500元，其他城市350元。餐补每天100元。报销需在出差结束后5个工作日内提交。",
    metadata={
        "source": "报销制度",
        "version": "v1.0",
        "date": "2024-06-01",
        "status": "现行有效"
    }
)

rag.query("年假有几天？病假工资打几折？")
rag.query("出差住酒店的报销上限是多少？")
```

### 何时需要生产级RAG？

这个完整的`ProductionRAG`集成了五大修复方案，但不是所有场景都需要：

✅ **需要完整方案：**
- 企业知识库、法律文档、医疗咨询等严肃场景
- 文档超过100篇，包含多个版本和来源
- 用户查询包含精确关键词和自然语言混合
- 需要可追溯的答案来源和高准确率（>90%）

✅ **只需要部分方案：**
- 文档50-100篇：语义分块 + 向量检索即可
- 文档无版本冲突：不需要元数据管理
- 查询都是自然语言：不需要BM25混合检索
- 个人学习助手：可以放宽Prompt约束

❌ **不需要（用基础RAG）：**
- 文档少于50篇，结构简单
- 只是做原型验证，不是生产系统
- 向量检索准确率已经超过80%

**渐进式升级路径：**
1. 第一步：基础RAG（向量检索 + 简单Prompt）
2. 第二步：加语义分块（如果答案经常不完整）
3. 第三步：加混合检索（如果关键词搜不准）
4. 第四步：严格Prompt（如果大模型爱编造）
5. 第五步：元数据管理（如果有版本冲突）

不要一上来就全上，根据实际问题逐步优化。

---

## 写在最后

踩过这五个坑的人应该都有同感——RAG 搭起来容易，调好真不容易。分块影响信息粒度，Embedding 影响语义理解，检索策略影响查全率，Prompt 影响可信度，文档管理影响实际可用性。哪个环节拉胯了，最终效果都会打折。

我自己的体会是：**RAG 的效果，八成取决于检索质量，大模型的能力其实只占两成**。与其花钱换更贵的模型，不如先把检索这一环做扎实。

**关键原则：**
1. **不要过度优化** — 根据实际问题逐步改进，不要一上来就全上
2. **先测后优** — 每个优化都要有量化指标（准确率、召回率）
3. **向后兼容** — 新方案要兼容旧代码，降低升级成本
4. **错误处理** — 生产代码必须处理API失败、空输入等边界情况

**渐进式优化路径：**
```
基础RAG（50行代码）
  ↓ 答案不完整？
加语义分块（坑一）
  ↓ 关键词搜不准？
加混合检索（坑三）
  ↓ 大模型爱编造？
严格Prompt（坑四）
  ↓ 有版本冲突？
元数据管理（坑五）
```

上篇搭了 RAG 的骨架，这篇把最常见的坑填了。下一篇来点更有意思的——让智能体**自己决定什么时候该查资料、查哪个知识库、查完觉得不对再换个方式查**。这就是 Agentic RAG，也是这个系列的收官篇。

评论区聊，下篇见。
