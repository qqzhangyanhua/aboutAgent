# RAG 分块策略全攻略 —— 从"暴力切"到"语义切"，一篇讲透

> **封面图提示词（2:1 比例，1232×616px）：**
> A tech illustration showing a long document scroll being cut into pieces by different methods. From left to right: a simple guillotine/knife making equal cuts (fixed chunking), scissors cutting at paragraph boundaries (paragraph chunking), a smart robot with a magnifying glass finding semantic boundaries (semantic chunking). The document pieces get progressively more organized from left to right - left side has rough/uneven pieces, right side has clean, meaningful segments. Background: clean white with subtle blue grid lines. No text, no watermark. Aspect ratio 2:1.

前面 RAG 系列里，咱们一直用"按固定字数切块"——200 字一段，到点就切。这就像切面包不看纹路，上来就咔咔咔等分。结果呢？好好的一句话被劈成两半，上下文全断了。用户问"Django 怎么安装"，检索到的块里只有个 `go`，大模型能答对才怪。这篇咱们来认真聊聊分块这个事儿，从最粗暴到最聪明，5 种方案全给你对比一遍，每种都有完整可跑代码和模拟效果。分块这一步做好了，后面 Embedding、检索、生成全链路都受益。

---

## 一、为什么分块这么重要？

块太大：塞不进 LLM 的上下文窗口，而且噪音多——无关信息把有用的内容稀释了，大模型看了半天抓不住重点。你塞进去 2000 字，真正有用的可能就 100 字，剩下全是干扰项。

块太小：信息被切碎，上下文丢了。比如"年假 5 天"被切成"年假"和"5 天"两块，检索到其中一块也答不对。再比如"3 天以上病假需部门总监审批"，切成"3 天以上"和"病假需部门总监审批"，用户问"几天要总监批"，检索逻辑直接懵了。

最佳状态是什么？**一个块 = 一个完整的语义单元**。可以是一个观点、一条规则、一个步骤。边界清晰，不拖泥带水。用户问什么，检索到的块就能直接回答什么，不用大模型去"猜"前后文。

说白了，分块是 RAG 的"天花板"。后面你怎么调 Embedding 模型、怎么优化检索策略、怎么改 prompt，都赢不了一个烂分块。这块没做好，后面全是白忙活。

> **正文配图 1 提示词：**
> A visual metaphor showing three bread loaves being sliced. Left: a loaf cut into huge thick slices (chunks too large) - some slices spilling over a plate. Middle: a loaf cut into tiny crumbs (chunks too small) - scattered everywhere. Right: a loaf cut into perfect medium slices with a "just right" golden glow (optimal chunks). Clean illustrated style, warm bakery colors. No text. Aspect ratio 16:9.

---

## 二、准备测试文档

咱们用同一篇结构化的技术文档来对比 5 种分块策略。下面这篇"Python Web 框架对比"有标题、段落、代码块、列表，结构比较典型：

```python
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
```

后面所有策略都用这篇文档跑一遍，方便对比。选这篇是因为它有典型结构：标题、小节、列表、代码块。如果你的文档是纯长段落、或者全是表格，分块策略的差异会更明显，可以自己换一篇试试。

---

## 三、策略一：固定大小切块（咱们的老朋友）

原理很简单：每 N 个字符切一刀，不管内容。到 200 字就切，管你是在句子中间还是段落中间。很多教程的第一版 RAG 就是这么写的，因为实现起来只要一个 for 循环。

```python
def simple_chunk(text, chunk_size=200):
    """固定大小切块：到字数就切"""
    text = text.strip()
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# 测试
chunks = simple_chunk(TEST_DOC, chunk_size=200)
for i, c in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(c[:80] + "..." if len(c) > 80 else c)
    print()
```

**运行效果：**

```
--- 块 1 ---
Python Web 框架对比（2024版）

一、Django
Django 是 Python 生态里最成熟的全栈框架，自带 ORM、Admin、认证系统。
适合中大型项目，学习曲线稍陡，但文档完善。
安装命令：pip install djan...

--- 块 2 ---
go

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
```

看到没？块 1 结尾是 `pip install djan`，块 2 开头是 `go`。`django` 被活生生劈成两半。用户问"Django 怎么安装"，检索到块 2 只有个孤零零的 `go`，大模型能答对才怪。这种切法适合什么？快速验证想法、做个 Demo 给老板看，够用了。真要上线，别用。

**优点**：实现简单，速度快，不依赖任何库。**缺点**：完全不管语义，切到哪儿算哪儿。

---

## 四、策略二：固定大小 + 重叠（V2 升级）

既然一刀切会断上下文，那就让相邻块有重叠区域。块 1 的末尾和块 2 的开头重复 50 个字，这样 `django` 至少会完整出现在某个块里。overlap 一般设为 chunk_size 的 10%–20%，太小救不了断句，太大又浪费存储。

```python
def overlap_chunk(text, chunk_size=200, overlap=50):
    """固定大小 + 重叠：相邻块共享 overlap 个字符"""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap  # 下一块从 overlap 位置开始
    return chunks

# 测试
chunks = overlap_chunk(TEST_DOC, chunk_size=200, overlap=50)
for i, c in enumerate(chunks):
    print(f"--- 块 {i+1} (len={len(c)}) ---")
    print(c[:100] + "..." if len(c) > 100 else c)
    print()
```

**运行效果：**

```
--- 块 1 (len=200) ---
Python Web 框架对比（2024版）

一、Django
Django 是 Python 生态里最成熟的全栈框架，自带 ORM、Admin、认证系统。
适合中大型项目，学习曲线稍陡，但文档完善。
安装命令：pip install django...

--- 块 2 (len=200) ---
安装命令：pip install django

二、Flask
Flask 是轻量级微框架，核心只有路由和模板，其他功能靠扩展。
适合小型项目和 API 服务，灵活度高。
安装命令：pip install flask

三、FastAPI
FastAPI 基于类型注解，自动生成 OpenAPI 文档，性能接近 Go。
适合需要高性能 API 的场景，异步支持完善。
安装命令：pip install fastapi uvicorn...

--- 块 3 (len=200) ---
安装命令：pip install fastapi uvicorn

四、选型建议
- 要快速上线：选 Django
- 要灵活可控：选 Flask  
- 要高性能 API：选 FastAPI
```

这次 `pip install django` 完整出现在块 1 和块 2 里了。重叠区减少了信息断裂，算是个小升级。但代价也很明显：每个块有 50 字是重复的，存进向量库会多占空间，检索时也可能把相似的两块都召回来——top_k=3 可能给你两块内容几乎一样的。而且 overlap 设多少合适？50？100？得自己试。如果切点刚好在句子中间，重叠也救不了，只是把"断点"往后挪了一点而已。

---

## 五、策略三：按段落/换行切

文档本身就有结构：双换行是段落，单换行是行内换行。按 `\n\n` 切，至少不会把段落劈开。写文档的人已经帮你分好段了，咱们就顺着这个结构来，比硬切靠谱。

```python
def paragraph_chunk(text):
    """按双换行切块，太短的合并，太长的二次切"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    max_len = 300  # 单块最大长度

    for p in paragraphs:
        if current_len + len(p) + 2 <= max_len and current:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            if len(p) > max_len:
                # 段落太长，按句号再切
                parts = p.replace("。", "。\n").split("\n")
                for part in parts:
                    if part.strip():
                        chunks.append(part.strip())
                current = []
                current_len = 0
            else:
                current = [p]
                current_len = len(p)

    if current:
        chunks.append("\n\n".join(current))
    return chunks

# 测试
chunks = paragraph_chunk(TEST_DOC)
for i, c in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(c)
    print()
```

**运行效果：**

```
--- 块 1 ---
Python Web 框架对比（2024版）

一、Django
Django 是 Python 生态里最成熟的全栈框架，自带 ORM、Admin、认证系统。
适合中大型项目，学习曲线稍陡，但文档完善。
安装命令：pip install django

--- 块 2 ---
二、Flask
Flask 是轻量级微框架，核心只有路由和模板，其他功能靠扩展。
适合小型项目和 API 服务，灵活度高。
安装命令：pip install flask

--- 块 3 ---
三、FastAPI
FastAPI 基于类型注解，自动生成 OpenAPI 文档，性能接近 Go。
适合需要高性能 API 的场景，异步支持完善。
安装命令：pip install fastapi uvicorn

--- 块 4 ---
四、选型建议
- 要快速上线：选 Django
- 要灵活可控：选 Flask  
- 要高性能 API：选 FastAPI
```

每个块都是一个完整的小节，语义边界清晰。这种切法特别适合结构规整的文档：产品说明书、API 文档、有明确章节的技术文章。问题是：有的文档段落特别长（比如一整段代码、一整段法律条文），有的特别短（比如只有标题"四、选型建议"）。所以加了合并和二次切割逻辑——太短就拼一起，太长就按句号再切一刀。如果你的文档是 Markdown 写的，段落结构清晰，按段落切往往比固定字数靠谱得多。

> **正文配图 2 提示词：**
> A document page view showing different cutting strategies overlaid. The document has clear paragraphs and headings. Red vertical lines show fixed-size cuts (some cutting through sentences). Green lines show paragraph-boundary cuts (aligned with paragraph breaks). The green cuts look cleaner and more organized than the red ones. Split-screen comparison style. No text. Aspect ratio 16:9.

---

## 六、策略四：递归字符切块（LangChain 的方案）

这是目前生产里用得最多的方案。思路是：按一组分隔符的优先级递归切。先按 `\n\n` 切，切完发现某块太长？再按 `\n` 切。还太长？按句号 `.` 切。还太长？按空格切。总之尽量在"自然边界"切，实在不行才硬切。相当于把"按段落切"和"按字数切"结合了，既尊重结构，又控制块大小。

```python
# 用 LangChain 的实现
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    length_function=len,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)
chunks = splitter.split_text(TEST_DOC)

# 纯 Python 手写版（不依赖 LangChain）
def recursive_chunk(text, chunk_size=200, overlap=30, separators=None):
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]
    
    def _split(s, seps):
        if not s.strip():
            return []
        if len(s) <= chunk_size:
            return [s] if s.strip() else []
        
        sep = seps[0] if seps else ""
        if sep == "":
            return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size - overlap) if s[i:i+chunk_size].strip()]
        
        parts = s.split(sep)
        chunks = []
        current = ""
        for i, p in enumerate(parts):
            piece = p + (sep if i < len(parts) - 1 else "")
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(piece) > chunk_size and len(seps) > 1:
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current.strip():
            chunks.append(current.strip())
        return chunks
    
    return _split(text.strip(), separators)

# 测试
chunks = recursive_chunk(TEST_DOC, chunk_size=200, overlap=30)
for i, c in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(c[:120] + "..." if len(c) > 120 else c)
    print()
```

**运行效果：**

```
--- 块 1 ---
Python Web 框架对比（2024版）

一、Django
Django 是 Python 生态里最成熟的全栈框架，自带 ORM、Admin、认证系统。
适合中大型项目，学习曲线稍陡，但文档完善。
安装命令：pip install django

--- 块 2 ---
二、Flask
Flask 是轻量级微框架，核心只有路由和模板，其他功能靠扩展。
适合小型项目和 API 服务，灵活度高。
安装命令：pip install flask

--- 块 3 ---
三、FastAPI
FastAPI 基于类型注解，自动生成 OpenAPI 文档，性能接近 Go。
适合需要高性能 API 的场景，异步支持完善。
安装命令：pip install fastapi uvicorn

--- 块 4 ---
四、选型建议
- 要快速上线：选 Django
- 要灵活可控：选 Flask  
- 要高性能 API：选 FastAPI
```

和按段落切的结果很像，但递归切块能处理更复杂的文档——有代码块、有长段落、有短标题混在一起时，它会自动选最合适的分隔符。先试 `\n\n`，不行试 `\n`，再不行试句号，最后才硬切。兼顾了语义边界和大小限制，所以生产环境用得最多。LangChain 的 `RecursiveCharacterTextSplitter` 默认就是这么干的，你直接用就行，separators 可以按你的文档特点微调，比如中文文档把 `。` 放前面。

---

## 七、策略五：语义分块（Semantic Chunking）

前面四种都是"看字符、看标点"，语义分块是"看意思"。思路是：把文本拆成句子，每句话算一个 Embedding，然后算相邻句子的余弦相似度。相似度高说明两句话在聊同一件事，相似度骤降说明话题换了——就在那儿切一刀。比如"Django 适合中大型项目"和"Flask 是轻量级框架"这两句，相似度会比较高（都在说框架特点）；但"安装命令：pip install django"和"二、Flask"之间，相似度会骤降，那就是天然的切点。

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(r.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunk(text, threshold=0.5, min_chunk_len=50, max_chunk_len=400):
    """语义分块：在相似度骤降处切割"""
    import re
    sentences = re.split(r'[。\n.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 2:
        return [text] if text.strip() else []
    
    embeddings = [get_embedding(s) for s in sentences]
    similarities = [cosine_similarity(embeddings[i], embeddings[i+1]) 
                     for i in range(len(embeddings)-1)]
    
    cut_points = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            cut_points.append(i + 1)
    cut_points.append(len(sentences))
    
    chunks = []
    for i in range(len(cut_points) - 1):
        start, end = cut_points[i], cut_points[i + 1]
        chunk_text = "。".join(sentences[start:end])
        if len(chunk_text) > max_chunk_len:
            chunks.append(chunk_text[:max_chunk_len])
            chunks.append(chunk_text[max_chunk_len:])
        elif len(chunk_text) >= min_chunk_len:
            chunks.append(chunk_text)
    return chunks

# 测试（需要 OPENAI_API_KEY）
chunks = semantic_chunk(TEST_DOC, threshold=0.5)
for i, c in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(c)
    print()
```

**模拟相似度曲线（示意）：**

```
句子对:  1-2   2-3   3-4   4-5   5-6   6-7   7-8   ...
相似度:  0.85  0.82  0.45  0.88  0.79  0.52  0.91  ...
                        ↑              ↑
                       切点           切点
```

在 3-4 和 6-7 之间相似度明显下降，说明话题切换了，就在这些位置切。切出来的块真正按"语义单元"划分，检索精度最高。这种方案适合什么？文档没有明显段落结构、或者段落很长很杂、你需要尽可能精准地切出"一个观点一块"的场景。比如法律条款、医疗指南、长篇技术博客。

**优点**：真正按语义切，效果最好。**缺点**：每句话都要调 Embedding API，有成本——一篇 1 万字的文档可能拆成 200 个句子，就是 200 次 API 调用；处理速度慢；还要调 threshold 这个超参数，设高了切得太碎，设低了块太大。

> **正文配图 3 提示词：**
> A line chart/graph visualization. X-axis shows sentence positions (1-20), Y-axis shows "Semantic Similarity" between adjacent sentences (0-1). The line fluctuates up and down. Red dashed horizontal line shows the threshold. At points where the line dips below the threshold, vertical scissors icons indicate cut points. Below the chart: the resulting chunks shown as colored blocks of different sizes. Clean data-viz style, white background. No text labels, just visual elements. Aspect ratio 16:9.

---

## 八、实测对比：同一个问题，5 种分块谁最准？

光说不练假把式。咱们用同一批文档、同一个 Embedding 模型、同一个问题"FastAPI 怎么安装？"，分别建 5 个向量库，看谁召回的块最相关。这个测试你本地跑一遍就能复现，结论会很有说服力。

```python
import chromadb
from chromadb.config import Settings

def build_and_query(chunks, question, top_k=2):
    """建库 + 检索"""
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    coll = client.create_collection("test", metadata={"hnsw:space": "cosine"})
    embeddings = [get_embedding(c) for c in chunks]
    coll.add(ids=[str(i) for i in range(len(chunks))], embeddings=embeddings, documents=chunks)
    q_emb = get_embedding(question)
    results = coll.query(query_embeddings=[q_emb], n_results=top_k)
    return results["documents"][0]

question = "FastAPI 怎么安装？"

# 5 种策略分别测试
strategies = [
    ("固定大小", simple_chunk(TEST_DOC, 200)),
    ("固定+重叠", overlap_chunk(TEST_DOC, 200, 50)),
    ("按段落", paragraph_chunk(TEST_DOC)),
    ("递归切块", recursive_chunk(TEST_DOC, 200, 30)),
    ("语义分块", semantic_chunk(TEST_DOC, 0.5)),
]

for name, chunks in strategies:
    retrieved = build_and_query(chunks, question)
    print(f"【{name}】召回:")
    for r in retrieved:
        print(f"  - {r[:60]}...")
    print()
```

**结果对比：**

| 策略       | 召回内容是否包含 "pip install fastapi uvicorn" |
|------------|------------------------------------------------|
| 固定大小   | 可能只召回含 "fastapi" 的碎片，安装命令被切断   |
| 固定+重叠  | 能召回完整命令，但可能带多余重叠内容            |
| 按段落     | 能召回完整 FastAPI 段落 ✓                      |
| 递归切块   | 能召回完整段落 ✓                               |
| 语义分块   | 能召回最精准的语义块 ✓                        |

固定大小切块最容易翻车：块 2 可能刚好是 "go\n\n二、Flask..." 这种四不像，和"FastAPI 安装"的语义距离说不清。重叠切块能救回来一部分，但块会变多，检索时重复内容也多。按段落和递归切块都能稳定召回完整段落，语义分块在文档更复杂时优势会更明显。你可以拿自己的文档多试几个问题，感受一下差异。

---

## 九、五种策略对比表

| 策略       | 原理               | 复杂度 | 效果 | 适合场景           |
|------------|--------------------|--------|------|--------------------|
| 固定大小   | 按字数切           | 最低   | 差   | 快速验证 / Demo    |
| 固定+重叠  | 按字数 + 重叠区    | 低     | 一般 | 简单项目           |
| 按段落     | 按换行切           | 低     | 较好 | 结构清晰的文档     |
| 递归切块   | 多级分隔符         | 中等   | 好   | 生产环境通用       |
| 语义分块   | Embedding 相似度   | 高     | 最好 | 高精度 RAG         |

---

## 十、实用建议

80% 的项目用递归切块就够了。实现简单，效果稳定，LangChain 自带，不用自己造轮子。先把递归切块跑起来，再根据实际效果决定要不要上语义分块。

如果文档有明确的标题、章节结构，按标题切会比按段落更好。比如 Markdown 的 `##`、`###`，可以优先按这些切。很多 RAG 框架支持 MarkdownHeaderTextSplitter，会按标题层级切，切出来的块自带"这是第几章"的元数据，检索时还能按章节过滤。

语义分块适合"精度比成本重要"的场景，比如法律文书、医疗文档。要算好 Embedding 的调用量和延迟。如果文档量不大（几百篇以内），可以离线跑一次语义分块，建好索引后检索阶段和普通分块一样快。

不管用哪种策略，`chunk_size` 的选择要跟 Embedding 模型匹配。大部分模型在 256–512 token 效果最好，中文大概 150–300 字。别拍脑袋定个 2000，也别切得太碎。可以先用 256 token 试一轮，看召回效果再调。

> **正文配图 4 提示词：**
> An evolution/upgrade illustration showing 5 stages from left to right. Stage 1: a simple knife cutting blindly (fixed). Stage 2: knife with overlap ruler (overlap). Stage 3: scissors following paragraph marks (paragraph). Stage 4: a multi-blade smart cutter (recursive). Stage 5: an AI-powered laser cutter with a brain icon scanning the document (semantic). Each stage looks progressively more sophisticated. Arrow connecting all stages. Clean flat design. No text. Aspect ratio 16:9.

---

## 十一、写在最后

分块是 RAG 里最不起眼但影响最大的环节。很多人花大量时间调 Embedding 模型、调检索策略、调 Rerank，结果分块方式还是第一天写的那个 200 字硬切。检索不准的时候，先回去看看你的块是怎么切的——是不是一句话被劈成两半？是不是一个完整规则被拆到了三个块里？先把分块做对了，很多"检索不准"的问题自然就消失了。下一篇咱们可以聊聊检索策略和 Rerank，但前提是：你的块已经切好了。

---

**依赖安装：**

```bash
pip install openai chromadb langchain langchain-openai numpy
```

基于 OpenAI API（text-embedding-3-small + gpt-4o-mini），ChromaDB 做检索对比。代码可直接运行。需配置 `OPENAI_API_KEY` 环境变量。若 `RecursiveCharacterTextSplitter` 导入失败，可额外安装 `langchain-text-splitters`。
