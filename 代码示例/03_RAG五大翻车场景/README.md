# RAG 五大翻车场景 - 代码示例

本目录包含文章《你的 RAG 为什么总是答不准？五大翻车场景逐个修复》的完整代码示例。

## 目录结构

```
03_RAG五大翻车场景/
├── README.md                    # 本文件
├── requirements.txt             # 依赖包
├── v1_chunk_problem.py          # 坑一：Chunk 切分问题
├── v2_embedding_problem.py      # 坑二：Embedding 模型选择
├── v3_hybrid_search.py          # 坑三：混合检索
├── v4_prompt_control.py         # 坑四：Prompt 约束
├── v5_metadata_conflict.py      # 坑五：多文档冲突
└── production_rag.py            # 完整的生产级 RAG
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
# macOS/Linux
export DEEPSEEK_API_KEY="your_deepseek_key"
export SILICONFLOW_API_KEY="your_siliconflow_key"

# Windows
set DEEPSEEK_API_KEY=your_deepseek_key
set SILICONFLOW_API_KEY=your_siliconflow_key
```

### 3. 运行示例

每个文件都可以独立运行：

```bash
# 坑一：Chunk 切分问题
python v1_chunk_problem.py

# 坑二：Embedding 模型选择
python v2_embedding_problem.py

# 坑三：混合检索
python v3_hybrid_search.py

# 坑四：Prompt 约束
python v4_prompt_control.py

# 坑五：多文档冲突
python v5_metadata_conflict.py

# 完整的生产级 RAG
python production_rag.py
```

## 文件说明

### v1_chunk_problem.py - 坑一：Chunk 切分问题

**演示问题：**
- 切太大：信息被淹没在一堆无关内容里
- 切太小：上下文被切断，答案不完整

**修复方案：**
- 语义分块：按自然段落边界切分，保留语义完整性

**运行效果：**
- 对比固定大小切分 vs 语义切分
- 实际检索效果对比

---

### v2_embedding_problem.py - 坑二：Embedding 模型选择

**演示问题：**
- 用英文模型处理中文文档，语义理解差
- 相关和不相关的文档分数都挤在一起，区分度低

**修复方案：**
- 使用中文优化的 Embedding 模型（如 BCE）

**运行效果：**
- 展示好的模型如何拉开相关/不相关文档的分数差距
- 推荐的 Embedding 模型列表

---

### v3_hybrid_search.py - 坑三：混合检索

**演示问题：**
- 向量检索对精确关键词（产品型号、人名、数字）不敏感
- 搜索 "A7-Pro" 可能找不到包含 "A7-Pro" 的文档

**修复方案：**
- 混合检索：BM25 关键词检索 + 向量语义检索

**运行效果：**
- 对比纯向量检索 vs 混合检索
- 展示不同权重的效果

---

### v4_prompt_control.py - 坑四：Prompt 约束

**演示问题：**
- 大模型看完参考资料后"好心"补充资料里没有的信息
- 编造的内容和真实内容混在一起，用户分不清

**修复方案：**
- 严格约束 Prompt：只基于参考资料回答
- 资料里没有的，明确说"资料未提及"

**运行效果：**
- 对比没约束 vs 严格约束的 Prompt
- 展示四条管用的规矩

---

### v5_metadata_conflict.py - 坑五：多文档冲突

**演示问题：**
- 同一主题有多个版本的文档，信息矛盾
- 大模型不知道该信哪个，可能混用新旧信息

**修复方案：**
- 元数据管理：给每段文档打上版本号、日期、状态标签
- 冲突处理规则：在 Prompt 里明确告诉大模型如何处理冲突

**运行效果：**
- 对比不处理冲突 vs 元数据管理
- 展示如何正确组合多个文档的信息

---

### production_rag.py - 完整的生产级 RAG

**集成内容：**
1. 语义分块（坑一）
2. 中文优化的 Embedding 模型（坑二）
3. 混合检索 BM25+向量（坑三）
4. 严格约束 Prompt（坑四）
5. 元数据管理（坑五）

**特性：**
- 向后兼容：支持简单调用和完整元数据两种方式
- 错误处理：完善的输入验证和异常处理
- 可配置：BM25 权重、chunk 大小等参数可调

**使用示例：**

```python
from production_rag import ProductionRAG

# 创建 RAG 实例
rag = ProductionRAG(bm25_weight=0.4)

# 方式1：简单使用（向后兼容）
rag.add_document(content="年假：入职满一年享有5天。")

# 方式2：带完整元数据（推荐）
rag.add_document(
    content="年假：入职满一年享有7天（2024年起上调）。",
    metadata={
        "source": "员工手册",
        "version": "v3.0",
        "date": "2024-01-01",
        "status": "现行有效"
    }
)

# 查询
rag.query("年假有几天？")
```

## API Key 获取

### DeepSeek API
- 官网：https://platform.deepseek.com/
- 用途：对话生成（大模型）
- 价格：¥1/百万 tokens（非常便宜）

### 硅基流动 API
- 官网：https://siliconflow.cn/
- 用途：Embedding（向量化）
- 价格：免费额度充足

## 何时需要这些优化？

### ✅ 需要完整方案：
- 企业知识库、法律文档、医疗咨询等严肃场景
- 文档超过 100 篇，包含多个版本和来源
- 用户查询包含精确关键词和自然语言混合
- 需要可追溯的答案来源和高准确率（>90%）

### ✅ 只需要部分方案：
- 文档 50-100 篇：语义分块 + 向量检索即可
- 文档无版本冲突：不需要元数据管理
- 查询都是自然语言：不需要 BM25 混合检索
- 个人学习助手：可以放宽 Prompt 约束

### ❌ 不需要（用基础 RAG）：
- 文档少于 50 篇，结构简单
- 只是做原型验证，不是生产系统
- 向量检索准确率已经超过 80%

## 渐进式升级路径

```
基础 RAG（50行代码）
  ↓ 答案不完整？
加语义分块（坑一）
  ↓ 关键词搜不准？
加混合检索（坑三）
  ↓ 大模型爱编造？
严格 Prompt（坑四）
  ↓ 有版本冲突？
元数据管理（坑五）
```

不要一上来就全上，根据实际问题逐步优化！

## 常见问题

### Q1: 为什么用 DeepSeek 而不是 OpenAI？
A: DeepSeek 价格便宜（¥1/百万 tokens vs OpenAI 的 $15/百万 tokens），对中文支持好，适合学习和原型验证。生产环境可以换成 OpenAI 或其他模型。

### Q2: 可以用本地模型吗？
A: 可以。Embedding 可以用 `sentence-transformers` 加载本地 BGE 模型，LLM 可以用 Ollama 等本地部署方案。参考 v2_embedding_problem.py 中的注释。

### Q3: BM25 权重怎么调？
A: 从 0.4 开始试，如果关键词匹配很重要，提高到 0.5-0.6；如果语义理解更重要，降低到 0.3。没有固定答案，根据实际效果调整。

### Q4: Chunk 大小怎么定？
A: 中文文档 200-500 字比较合适。太大了答案会被无关内容淹没，太小了上下文断裂。具体多少得看你的文档结构，多试几次。

### Q5: 代码可以直接用于生产吗？
A: 可以，但建议根据实际需求调整：
- 添加日志记录
- 添加性能监控
- 使用持久化的向量数据库（如 Qdrant、Milvus）
- 添加缓存机制
- 添加并发控制

## 相关资源

- 文章：[你的 RAG 为什么总是答不准？五大翻车场景逐个修复](../文章/03_RAG五大翻车场景.md)
- 上一篇：[02_从零实现最简RAG](../02_从零实现最简RAG/)
- 下一篇：04_Agentic_RAG智能体自主检索

## 许可证

MIT License
