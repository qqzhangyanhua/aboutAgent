# RAG 向量数据库有哪些？怎么选？—— 2025 全景图 + 选型决策指南

## 封面图提示词

> A wide panoramic tech illustration showing a landscape of interconnected database icons arranged like a city skyline. In the foreground, a developer figure stands at a crossroads looking at multiple paths, each leading to different database buildings: some small and cozy (embedded), some tall corporate towers (cloud-native), some industrial warehouses (distributed), and some buildings attached to larger structures (database extensions). Vector arrows float in the sky like birds. Soft gradient background from deep purple to blue. Clean flat design, no text, no watermark. Aspect ratio 2:1.

---

## 一、开头：为什么这个问题这么让人头疼

做 RAG 的人，迟早会被这个问题困住：**向量数据库到底该用哪个？**

你去搜一圈，ChromaDB、FAISS、Qdrant、Milvus、Pinecone、Weaviate、pgvector、LanceDB、Elasticsearch……光名字就能列出十几个。每个的官网都说自己是最好的、最快的、最适合 RAG 的。

更让人抓狂的是，它们不是同一类东西：有的是嵌入式库，有的是独立服务，有的是云服务，有的压根就是传统数据库加了个向量扩展。你说拿来怎么比？

之前那篇文章咱们已经拿 ChromaDB、FAISS、Qdrant、Milvus 做了性能实测，今天换个角度——不拘泥于几个具体选手，而是把整个向量数据库生态画一张全景图，帮你搞清楚**有哪些类别、每个类别里谁是主流、你的场景该落到哪个类别里**。

先下一个结论：**没有"最好的"向量数据库，只有"最适合你当前阶段"的。**

---

## 二、向量数据库的四大类别

把市面上所有向量数据库按部署方式分，就四类。理解了这个分类，选型就清晰一半。

| 类别 | 特点 | 代表选手 | 一句话 |
|------|------|----------|--------|
| **嵌入式/库** | 不需要起服务，import 就用 | ChromaDB、FAISS、LanceDB | 最省事，但扛不住大规模 |
| **独立服务** | 需要部署服务端，有 REST/gRPC API | Qdrant、Weaviate、Milvus | 生产级，功能全 |
| **云原生/托管** | SaaS 服务，不用管运维 | Pinecone、Zilliz Cloud、Qdrant Cloud | 花钱买省心 |
| **传统数据库扩展** | 在现有数据库上加向量能力 | pgvector、Elasticsearch、OpenSearch | 不想多引入一个组件 |

**配图 1 提示词**：一张四象限图。X 轴从"轻量"到"重型"，Y 轴从"自托管"到"托管"。左下：嵌入式（ChromaDB/FAISS/LanceDB 图标），右下：独立服务（Qdrant/Weaviate/Milvus 图标），右上：云原生（Pinecone/Zilliz 图标），左上：数据库扩展（PostgreSQL+向量图标）。每个象限配不同颜色，卡通风格。

---

## 三、嵌入式 / 库：开发阶段的最佳拍档

### 3.1 ChromaDB —— Python 生态的默认选择

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=["doc1", "doc2"],
    documents=["RAG 是检索增强生成", "向量数据库存储 embedding"],
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
)

results = collection.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=1)
print(results["documents"])
```

**优点**：零配置，内存模式或持久化模式随便切；自带 embedding 函数，不需要单独调 API 也能跑；支持元数据过滤。

**缺点**：性能天花板明显，100K 以上就开始吃力；单机架构，没有分布式能力；没有 GPU 加速。

**适合**：RAG 原型、Demo、小团队内部工具、教学。

### 3.2 FAISS —— 纯速度怪物

```python
import faiss
import numpy as np

dim = 384
index = faiss.IndexHNSWFlat(dim, 32)

vectors = np.random.randn(10000, dim).astype(np.float32)
index.add(vectors)

query = np.random.randn(1, dim).astype(np.float32)
distances, indices = index.search(query, k=5)
print(indices[0])
```

Meta 开源，C++ 底层，有 GPU 版本。严格来说不是数据库——没有 CRUD，没有元数据，没有持久化（得自己 `faiss.write_index`）。纯粹就是一个高性能向量检索库。

**优点**：速度最快，没有之一；索引类型最丰富（Flat、IVF、HNSW、PQ 等十几种）；GPU 版本能把百万级检索压到亚毫秒。

**缺点**：不是数据库，啥都得自己拼；没有元数据过滤，做 RAG 很不方便；Python API 不够 Pythonic。

**适合**：离线批处理、纯向量召回、对延迟极度敏感的场景、ML 研究。

### 3.3 LanceDB —— 新生代，基于列式存储

```bash
pip install lancedb
```

```python
import lancedb
import numpy as np

db = lancedb.connect("./my_lance_db")

data = [
    {"id": "doc1", "text": "RAG 检索增强", "vector": np.random.randn(384).tolist()},
    {"id": "doc2", "text": "向量数据库选型", "vector": np.random.randn(384).tolist()},
]
table = db.create_table("docs", data)

query_vec = np.random.randn(384).tolist()
results = table.search(query_vec).limit(5).to_list()
for r in results:
    print(r["id"], r["text"], r["_distance"])
```

LanceDB 是 2023 年才冒出来的新选手，基于 Lance 列式数据格式，嵌入式部署但性能不错。核心卖点是**零拷贝访问**和**版本化数据管理**，跟 Parquet 类似的思路但专门为向量检索优化。

**优点**：嵌入式但性能接近独立服务；原生支持多模态数据（图片、视频 embedding 都行）；数据自带版本管理；支持 SQL 风格的过滤。

**缺点**：生态还不够成熟；社区比 ChromaDB 小；生产案例偏少。

**适合**：多模态 RAG、数据科学场景、想要嵌入式但嫌 ChromaDB 性能不够的。

---

## 四、独立服务：生产环境的主力军

### 4.1 Qdrant —— 中小规模的最优解

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=np.random.randn(384).tolist(), payload={"category": "tech" if i % 2 == 0 else "hr"})
    for i in range(100)
]
client.upsert(collection_name="my_docs", points=points)

hits = client.search(
    collection_name="my_docs",
    query_vector=np.random.randn(384).tolist(),
    query_filter=Filter(must=[FieldCondition(key="category", match=MatchValue(value="tech"))]),
    limit=5,
)
for h in hits:
    print(h.id, h.score, h.payload)
```

Rust 写的，性能好、内存效率高。REST API + gRPC 双协议，Python/Go/Rust/JS 多语言 SDK。功能全面：元数据过滤、多向量支持（同一条数据存多个 embedding）、分片复制、快照备份。

**优点**：Rust 底层，单机性能优秀；过滤能力强，支持嵌套 JSON 过滤；有 Qdrant Cloud 托管服务；API 设计简洁。

**缺点**：分布式能力不如 Milvus；中文社区相对小；大规模（十亿级）场景经验偏少。

**适合**：10 万到千万级的生产 RAG、需要元数据过滤、中小团队。

### 4.2 Weaviate —— GraphQL 风格的搜索引擎

```bash
docker run -p 8080:8080 -p 50051:50051 semitechnologies/weaviate
```

```python
import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()

collection = client.collections.create(
    name="Document",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    properties=[
        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ],
)

collection.data.insert(
    properties={"title": "RAG 入门", "content": "检索增强生成是一种..."},
    vector=[0.1] * 384,
)

response = collection.query.near_vector(
    near_vector=[0.1] * 384,
    limit=5,
    return_metadata=wvc.query.MetadataQuery(distance=True),
)
for obj in response.objects:
    print(obj.properties, obj.metadata.distance)

client.close()
```

Weaviate 的特色是**内置向量化**——你可以直接存文本，它自动调用配置好的 embedding 模型（OpenAI、Cohere、HuggingFace 等）帮你做向量化，不用自己调 API。另外它支持混合搜索（向量 + BM25 关键词），做 RAG 时一步到位。

**优点**：内置 vectorizer 模块，文本进去向量自动出来；原生混合搜索；GraphQL API 查询灵活；多租户支持好。

**缺点**：部署偏重，资源消耗比 Qdrant 大；学习曲线稍高（schema 定义比较啰嗦）；Java/Go 混合架构，排查问题有门槛。

**适合**：需要混合搜索的 RAG、多租户 SaaS、对数据建模有要求的项目。

### 4.3 Milvus —— 十亿级数据的分布式方案

```bash
# 生产部署需要 docker-compose，包含 etcd + MinIO + Milvus
# 轻量体验可用 Milvus Lite：
pip install pymilvus
```

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_lite.db")

client.create_collection(
    collection_name="my_docs",
    dimension=384,
    metric_type="COSINE",
)

data = [
    {"id": i, "vector": np.random.randn(384).tolist(), "category": "tech" if i % 2 == 0 else "hr"}
    for i in range(1000)
]
client.insert(collection_name="my_docs", data=data)

results = client.search(
    collection_name="my_docs",
    data=[np.random.randn(384).tolist()],
    limit=5,
    output_fields=["category"],
)
for hits in results:
    for hit in hits:
        print(hit["id"], hit["distance"], hit["entity"])
```

Milvus 是 LF AI & Data 基金会的毕业项目，专门为十亿级向量检索设计。架构分为 Proxy、QueryNode、DataNode、IndexNode 四层，可以独立扩缩容。生产部署需要 etcd（元数据）+ MinIO（对象存储）+ Milvus 服务本身。

新版 Milvus 2.4+ 推出了 Milvus Lite，可以像 SQLite 一样嵌入式运行，开发阶段不用搞 Docker 集群了。

**优点**：分布式架构，十亿级数据实测过；索引类型丰富（HNSW、IVF、DiskANN 等）；Zilliz Cloud 提供全托管服务；社区活跃，中文文档完善。

**缺点**：部署复杂度是所有选手里最高的；资源消耗大；小规模数据用它是"杀鸡用牛刀"。

**适合**：千万到十亿级、有运维团队、需要分布式的场景。

---

## 五、云原生 / 托管服务：花钱买省心

### 5.1 Pinecone —— 最早的向量数据库 SaaS

```bash
pip install pinecone
```

```python
from pinecone import Pinecone, ServerlessSpec
import numpy as np

pc = Pinecone(api_key="your-api-key")

pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("my-index")
index.upsert(
    vectors=[
        {"id": "doc1", "values": np.random.randn(384).tolist(), "metadata": {"category": "tech"}},
        {"id": "doc2", "values": np.random.randn(384).tolist(), "metadata": {"category": "hr"}},
    ]
)

results = index.query(
    vector=np.random.randn(384).tolist(),
    top_k=5,
    include_metadata=True,
    filter={"category": {"$eq": "tech"}},
)
for match in results["matches"]:
    print(match["id"], match["score"], match["metadata"])
```

Pinecone 是最早一批做向量数据库 SaaS 的，2021 年就开始了。全托管，不需要操心部署、扩容、备份。API 设计简洁，Python SDK 体验好。

**优点**：零运维，API 开箱即用；自动扩缩容；Serverless 模式按量付费，小规模很便宜；全球多区域部署。

**缺点**：数据在别人的云上，合规敏感场景不适用；厂商锁定，迁移成本高；免费额度有限（100K 向量、1 个索引）；国内访问可能需要代理。

**适合**：海外项目、初创团队快速上线、不想操心基础设施的。

### 5.2 其他托管服务

| 服务 | 背后的开源项目 | 亮点 |
|------|---------------|------|
| **Zilliz Cloud** | Milvus | Milvus 官方云服务，有免费额度 |
| **Qdrant Cloud** | Qdrant | Qdrant 官方云服务，部署简单 |
| **Weaviate Cloud** | Weaviate | Weaviate 官方 SaaS |
| **MongoDB Atlas Vector Search** | MongoDB | 现有 Mongo 用户无缝接入 |
| **Supabase Vector** | pgvector | 现有 Supabase 用户直接用 |

如果你已经在用某个云服务（比如 Supabase、MongoDB Atlas），直接用它自带的向量能力是阻力最小的路径。不用多引入一个服务，不用多管一套运维。

---

## 六、传统数据库扩展：不想多一个组件的选择

### 6.1 pgvector —— PostgreSQL 的向量扩展

```sql
-- 安装扩展
CREATE EXTENSION vector;

-- 创建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding vector(384)
);

-- 创建索引（HNSW）
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- 插入数据
INSERT INTO documents (title, content, embedding)
VALUES ('RAG 入门', '检索增强生成...', '[0.1, 0.2, ...]');

-- 查询最相似的 5 条
SELECT id, title, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

pgvector 是 PostgreSQL 的扩展插件，让你在原有的 PG 数据库里直接存向量、做相似搜索。最大的好处是**不用引入新的基础设施**——你的业务数据和向量数据在同一个库里，JOIN 查询、事务一致性都天然支持。

**优点**：零额外基础设施；SQL 查询，前端后端都会用；天然支持事务和 ACID；HNSW 和 IVFFlat 两种索引。

**缺点**：百万级以上性能明显不如专业向量数据库；HNSW 索引构建慢且占内存；不支持 GPU 加速；向量检索和普通查询共享 PG 的资源，可能互相影响。

**适合**：已有 PostgreSQL 的项目、数据量不超过百万、不想增加架构复杂度的。

### 6.2 Elasticsearch / OpenSearch

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

es.indices.create(
    index="my_docs",
    body={
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    },
)

es.index(
    index="my_docs",
    body={"title": "RAG 入门", "content": "检索增强生成...", "embedding": [0.1] * 384},
)

results = es.search(
    index="my_docs",
    body={
        "knn": {
            "field": "embedding",
            "query_vector": [0.1] * 384,
            "k": 5,
            "num_candidates": 50,
        }
    },
)
```

Elasticsearch 从 8.0 版本开始原生支持 `dense_vector` 类型和 KNN 搜索。如果你的项目已经在用 ES 做全文搜索，加上向量搜索做**混合检索**（BM25 + 向量）是最自然的选择。OpenSearch 是 ES 的分叉版本，向量能力类似。

**优点**：全文搜索 + 向量搜索一把梭；生态成熟，运维经验丰富；分布式架构，天然支持集群；Kibana 可视化。

**缺点**：向量搜索性能不如专业向量数据库；资源消耗大（Java 嘛）；纯向量场景用 ES 是大材小用。

**适合**：已有 ES 集群、需要混合检索、日志/搜索平台加 AI 能力的。

---

## 七、全景对比表

把所有主流选手摊开看：

| 数据库 | 类别 | 语言 | 部署 | 元数据过滤 | 混合搜索 | 云服务 | 百万级性能 | 学习成本 |
|--------|------|------|------|-----------|---------|--------|-----------|---------|
| **ChromaDB** | 嵌入式 | Python | pip install | ✓ | ✗ | Chroma Cloud | 一般 | 极低 |
| **FAISS** | 库 | C++/Python | pip install | ✗ | ✗ | ✗ | 极快 | 中 |
| **LanceDB** | 嵌入式 | Rust/Python | pip install | ✓ | ✗ | LanceDB Cloud | 较好 | 低 |
| **Qdrant** | 独立服务 | Rust | Docker | ✓ | ✓（实验） | Qdrant Cloud | 快 | 中 |
| **Weaviate** | 独立服务 | Go | Docker | ✓ | ✓ | Weaviate Cloud | 快 | 中高 |
| **Milvus** | 独立服务 | Go/C++ | Docker 集群 | ✓ | ✓ | Zilliz Cloud | 极快 | 高 |
| **Pinecone** | 云原生 | — | SaaS | ✓ | ✓ | 自身 | 快 | 低 |
| **pgvector** | DB 扩展 | C | PG 插件 | ✓（SQL） | ✓（配合 tsvector） | Supabase/Neon | 一般 | 低 |
| **Elasticsearch** | DB 扩展 | Java | 集群 | ✓ | ✓（原生） | Elastic Cloud | 一般 | 中高 |

**配图 2 提示词**：一张雷达图，六个维度：性能、易用性、功能完整性、可扩展性、成本、生态。9 个不同颜色的多边形重叠在一起，分别代表 9 个数据库。旁边配颜色图例。科技风深色背景。

---

## 八、选型决策树

别看上面列了一堆，实际选的时候走三步就够了：

### 第一步：你的数据规模

```
数据量 < 10 万 → 嵌入式（ChromaDB / LanceDB）
数据量 10 万 ~ 1000 万 → 独立服务（Qdrant / Weaviate）
数据量 > 1000 万 → 分布式（Milvus）或云托管（Pinecone / Zilliz）
```

### 第二步：你的技术栈现状

```
已有 PostgreSQL → 先试 pgvector，不够再换
已有 Elasticsearch → 直接用 ES 的 dense_vector
已有 MongoDB → MongoDB Atlas Vector Search
啥都没有 → 看第三步
```

### 第三步：你的团队和场景

```
一个人撸原型 → ChromaDB
小团队上生产 → Qdrant
不想管运维 → Pinecone / Zilliz Cloud
需要混合搜索 → Weaviate 或 Elasticsearch
数据十亿级 → Milvus + 运维团队
对延迟极度敏感 → FAISS + GPU
```

**配图 3 提示词**：一棵选型决策树流程图。根节点"数据规模？"，分三条路。每条路下面再根据"技术栈？"和"场景？"继续分支。叶子节点是具体的数据库名称。整洁的流程图风格，蓝色调。

### 一个更直接的推荐

你问我推荐什么？不绕弯子：

- **大多数人**：先用 **ChromaDB** 搞定原型，上生产切 **Qdrant**。这条路验证过的人最多，踩坑最少。
- **已有 PG 数据库的团队**：先试 **pgvector**，数据量上百万再考虑换。
- **海外项目 / 不想运维**：**Pinecone** Serverless，花点钱买省心。
- **数据量真的很大（千万级以上）**：**Milvus** 或 **Zilliz Cloud**，别的选手到这个量级都会吃力。

---

## 九、常见误区

### 误区 1：上来就搞 Milvus

你的数据才几万条，Milvus 的分布式架构需要 etcd + MinIO + 多个节点，光部署就够折腾半天。用 ChromaDB 十分钟就能跑通的事，用 Milvus 可能得搞一天。

**原则：按当前数据规模选，别按"未来可能有多大"选。** 真到那个规模的时候，迁移的工作量远小于现在过度设计的维护成本。

### 误区 2：把向量数据库当传统数据库用

向量数据库不是关系数据库，别指望它做复杂 JOIN、事务、聚合。它的核心能力就两个：存向量、搜相似。元数据过滤只是辅助功能，复杂查询还是得靠传统数据库。

如果你的场景既要向量搜索又要复杂查询，pgvector 可能是最好的折中——向量和结构化数据在同一个 PG 里，SQL 一把梭。

### 误区 3：纠结性能差异

ChromaDB 查一次 15ms，Qdrant 查一次 8ms，FAISS 查一次 2ms——这点差距在整个 RAG 链路里根本感知不到。你调一次 LLM 就是 500ms ~ 2s，向量检索那几毫秒的差异完全可以忽略。

**真正影响 RAG 效果的是分块策略、embedding 模型选择、检索策略（要不要加 rerank），而不是向量数据库快了几毫秒。**

### 误区 4：忽略混合搜索

纯向量搜索有个死穴：对精确关键词和数字不敏感。用户搜"合同编号 HT-2024-0815"，向量搜索可能搜出一堆"合同"相关但编号完全对不上的结果。加上 BM25 关键词搜索做混合检索，这个问题就解决了。

支持原生混合搜索的：Weaviate、Elasticsearch、Pinecone。其他的需要自己在应用层拼。

---

## 十、迁移策略：别把自己锁死

不管你现在用哪个，迟早可能要换。最佳实践是**抽象一层接口**，让业务代码不直接依赖具体的向量数据库。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
    text: str


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5, filter_meta: dict | None = None) -> list[SearchResult]:
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        ...


class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "default"):
        import chromadb
        self._client = chromadb.Client()
        self._col = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        self._col.add(ids=ids, embeddings=vectors.tolist(), documents=texts, metadatas=metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5, filter_meta: dict | None = None) -> list[SearchResult]:
        kwargs: dict = {"query_embeddings": [query_vector.tolist()], "n_results": k, "include": ["documents", "metadatas", "distances"]}
        if filter_meta:
            kwargs["where"] = filter_meta
        r = self._col.query(**kwargs)
        results: list[SearchResult] = []
        for doc_id, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            results.append(SearchResult(id=doc_id, score=1 - dist, metadata=meta, text=doc))
        return results

    def delete(self, ids: list[str]) -> None:
        self._col.delete(ids=ids)


class QdrantStore(VectorStore):
    def __init__(self, collection_name: str = "default", dim: int = 384):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        self._client = QdrantClient(host="localhost", port=6333)
        self._col_name = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict] | None = None) -> None:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(id=idx, vector=v.tolist(), payload={**(m or {}), "text": t})
            for idx, v, t, m in zip(range(len(ids)), vectors, texts, metadatas or [{}] * len(ids))
        ]
        self._client.upsert(collection_name=self._col_name, points=points)

    def search(self, query_vector: np.ndarray, k: int = 5, filter_meta: dict | None = None) -> list[SearchResult]:
        hits = self._client.search(collection_name=self._col_name, query_vector=query_vector.tolist(), limit=k)
        return [SearchResult(id=str(h.id), score=h.score, metadata=h.payload, text=h.payload.get("text", "")) for h in hits]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self._col_name, points_selector={"points": [int(i) for i in ids]})


def create_store(backend: str = "chroma", **kwargs) -> VectorStore:
    stores = {"chroma": ChromaStore, "qdrant": QdrantStore}
    store_cls = stores.get(backend)
    if store_cls is None:
        raise ValueError(f"不支持的后端: {backend}，可选: {list(stores.keys())}")
    return store_cls(**kwargs)


# 业务代码只依赖 VectorStore 接口
store = create_store("chroma")              # 开发
# store = create_store("qdrant", dim=384)   # 生产
```

核心思路：业务层只调 `add`、`search`、`delete`，底层实现随时切。新增一个后端只需要实现三个方法。

**配图 4 提示词**：一张架构图。上层是"业务代码"方块，中间是一个"VectorStore 抽象层"方块（标注 add/search/delete），下面分出四条线通向 ChromaDB、Qdrant、Milvus、pgvector 四个方块。箭头方向从上到下。简洁扁平风格，蓝色调。

---

## 十一、2025 年的趋势

几个值得关注的方向：

**1. 混合搜索成为标配** — 纯向量搜索不够用了，向量 + 关键词 + 元数据过滤的混合检索正在成为所有向量数据库的必备功能。Weaviate 和 ES 先跑了一步，Qdrant 和 Milvus 也在跟进。

**2. 嵌入式数据库崛起** — LanceDB、ChromaDB 这类嵌入式方案越来越受欢迎。原因很简单：大多数 RAG 应用的数据量根本用不上分布式，能少一个服务就少一个服务。

**3. 传统数据库纷纷加向量** — PostgreSQL（pgvector）、MongoDB（Atlas Vector Search）、Redis（Redis Stack）、SQLite（sqlite-vss）……几乎所有主流数据库都在加向量能力。未来"向量数据库"可能不再是一个独立品类，而是每个数据库的标配功能。

**4. Serverless / 按量付费** — Pinecone Serverless、Zilliz Serverless、Qdrant Cloud 都在推按量付费模式。对小团队和独立开发者来说，这意味着起步成本接近零。

**5. 多模态向量** — 不只存文本 embedding，还要存图片、音频、视频的 embedding。LanceDB 在这方面走得比较前面。

---

## 十二、写在最后

向量数据库选型没有标准答案，但有标准思路：

1. **从你的数据规模出发** — 10 万以下用嵌入式，百万级用独立服务，千万级以上用分布式或云托管。
2. **从你的技术栈出发** — 已有 PG 就先试 pgvector，已有 ES 就用 ES 的向量能力，啥都没有再从头选。
3. **从你的团队出发** — 一个人就别搞 Milvus 集群，花钱买 Pinecone 比自己运维划算。

记住最重要的一点：**RAG 效果好不好，90% 取决于分块策略、embedding 模型、检索策略和 prompt 工程。向量数据库只是存和搜的工具，别在这上面纠结太久。**先用 ChromaDB 把 RAG 跑通，效果调好了再换，什么时候都来得及。

---

## 正文配图提示词汇总

1. **配图 1**：四象限分类图，X 轴轻量→重型，Y 轴自托管→托管，四类数据库分布其中。
2. **配图 2**：雷达图，9 个数据库在性能/易用性/功能/扩展性/成本/生态六维度对比。
3. **配图 3**：选型决策树流程图，按数据规模→技术栈→场景三层分流。
4. **配图 4**：抽象层架构图，业务代码 → VectorStore 接口 → 多种数据库实现。
