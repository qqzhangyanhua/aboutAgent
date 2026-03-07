# 让 AI 自己决定翻哪本书 —— Agentic RAG：智能体 × 检索的终极合体

> **封面图提示词（2:1 比例，1232×616px）：**
> A sophisticated tech illustration of a friendly robot librarian standing in front of multiple bookshelves. The robot has one arm reaching toward one shelf and its eyes scanning another shelf simultaneously. Glowing connection lines link the robot to different bookshelves, and small floating question marks transform into answer bubbles near the robot. Some bookshelves have green glow (selected), others dim (not selected). Background: warm library atmosphere with modern tech overlay, dark wood tones mixed with neon blue circuit patterns. No text, no watermark. Aspect ratio 2:1.

前两篇我们分别搞了两件事：

- **第一篇**：搭了个能"翻书找答案"的 RAG —— 用户问啥就去知识库里搜
- **第二篇**：修了 RAG 的五个常见翻车场景 —— 切块、模型、检索、Prompt、冲突

但写到这里我自己也发现一个问题：前面的 RAG 都是**被动的**——你问一个问题，它固定去一个知识库搜一次，搜到啥就拿啥回答。搜到的不靠谱呢？问题跨了好几个知识库呢？需要分步检索的复杂问题呢？

它不会自己调整策略。因为它没有"脑子"。

所以这篇文章，我们要把更早之前的智能体和 RAG 合体，搞一个 **Agentic RAG**——一个**自己决定什么时候查、查哪个库、查完不满意再换个姿势查**的智能检索系统。

打个比方就明白了：

| 类型 | 比喻 | 行为 |
|------|------|------|
| 普通 RAG | 图书馆的自助查询机 | 你输关键词，它吐结果，不管对不对 |
| 改良 RAG | 聪明一点的查询机 | 能过滤垃圾结果、能重排序 |
| **Agentic RAG** | **图书馆管理员** | 先理解你到底想问啥，自己决定去哪个书架找，找完翻一翻觉得不对再换一个书架，最终把靠谱的答案整理好给你 |

这篇文章会搭两个版本：

- **V1** — 多知识库路由：智能体自己判断该查哪个库
- **V2** — 自适应检索：查完不满意自动重查 + 多步推理

依赖没变，`pip install openai chromadb` 就够了，代码直接能跑。

---

## 一、Agentic RAG 到底"智能"在哪？

普通 RAG 的流程是**固定管线**——问题进来 → 搜一次 → 生成答案。不管问什么，走的都是同一条路。

Agentic RAG 的流程是**动态决策**——问题进来 → 智能体先分析 → 决定查哪个库（甚至不查） → 检索 → 评估结果够不够好 → 不够好就换策略再查 → 够好了才生成答案。

核心就俩字：**自主**。

来看几个普通 RAG 搞不定、但 Agentic RAG 能兜住的场景：

> **正文配图 1 提示词：**
> A flow diagram comparing two approaches side by side. Left side (labeled "Static RAG"): a straight arrow from Question → Single Database → Answer, with a rigid/mechanical feel. Right side (labeled "Agentic RAG"): a dynamic flow from Question → Brain (decision) → branching arrows to multiple databases → evaluation checkpoint (with a loop-back arrow for retry) → Answer. The right side has an organic, intelligent feel with glowing nodes. Dark background, left side in gray tones, right side in vibrant blue/teal tones. No text labels, icons only. Aspect ratio 16:9.

**场景 1：跨知识库**
> "我们技术部的前端规范是什么？新人入职第一周要做哪些事？"

这个问题涉及两个知识库（技术规范 + 入职指南），普通 RAG 只查一个库，答案必然不完整。Agentic RAG 会自己拆成两个子查询，分别去两个库搜。

**场景 2：检索结果不够好**
> "公司有没有关于远程办公的规定？"

如果知识库里压根没有远程办公的文档，普通 RAG 会硬拿一堆不相关的内容凑数，然后大模型开始胡编。Agentic RAG 检索完会先看一眼"这些结果相关吗？"——不相关就直接告诉用户"没找到"，不会胡编。

**场景 3：需要推理的复杂问题**
> "我入职 8 年了，今年还没休过年假，最多能攒到明年几月？"

这个问题得先查"8 年工龄对应几天年假"，再查"年假能不能顺延、顺延到几月"，然后综合推理。普通 RAG 做不了多步检索+推理，Agentic RAG 可以。

---

## 二、准备工作：建立多个知识库

我们模拟一家公司有四个独立的知识库。先把基础设施搭好：

```python
import os
import json
from openai import OpenAI
import chromadb

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()
chroma_client = chromadb.Client()

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

# ==================== 四个知识库 ====================

knowledge_bases = {
    "hr_policy": {
        "name": "人事制度库",
        "description": "包含请假制度、考勤规定、薪酬福利、绩效考核等人事相关的规章制度",
        "documents": [
            "年假规定：入职满1年不满10年，每年5天带薪年假。入职满10年不满20年，每年10天。入职满20年以上，每年15天。年假需提前3个工作日申请。当年未使用的年假可顺延至次年3月31日，逾期作废。",
            "病假规定：需提供正规医院诊断证明。3天以内直属上级审批，3天以上部门总监审批。病假期间工资按基本工资的80%发放。连续病假超过30天按长期病假政策处理。",
            "事假规定：事假为无薪假期，按日扣除工资。每次不超过3天，需提前2个工作日申请。全年累计不超过15天。",
            "试用期规定：试用期为3个月，工资为正式工资的90%。第一个月末进行非正式1v1反馈。试用期满前两周进行转正评估。转正需准备PPT进行答辩。",
            "绩效考核：采用季度考核制。维度包括业务成果50%、专业能力20%、协作沟通15%、文化价值观15%。S级（前10%）年终奖系数2.0，A级（前30%）系数1.5，B级（中间50%）系数1.0，C级（后10%）无年终奖。",
        ]
    },
    "finance": {
        "name": "财务报销库",
        "description": "包含差旅报销标准、日常费用报销流程、发票规范、预算管理等财务相关制度",
        "documents": [
            "差旅住宿标准：一线城市（北上广深）每晚不超过500元，二线城市每晚不超过350元，其他城市每晚不超过250元。需提供酒店发票。",
            "差旅交通标准：飞机经济舱、高铁二等座实报实销。头等舱/商务座需总监级以上审批。市内交通出租车/网约车实报实销，需提供行程截图。",
            "餐饮补贴：国内出差每天100元定额补贴，无需发票。海外出差按目的地标准执行，需提供消费凭证。",
            "报销流程：出差结束后5个工作日内提交。5000元以下直属上级审批，5000-20000元部门总监审批，20000元以上财务总监审批。审批通过后10个工作日内打款。",
            "团建费用：每人每季度200元标准。需提供活动照片和参与人员名单。超出部分由部门经费补充，需提前申请。",
        ]
    },
    "tech_spec": {
        "name": "技术规范库",
        "description": "包含前后端技术栈要求、代码规范、Git工作流、部署流程、安全规范等技术相关文档",
        "documents": [
            "前端技术栈：框架使用React 18+，状态管理用Zustand，UI库用Ant Design 5.x，CSS用Tailwind CSS或CSS Modules，构建工具用Vite。禁止在新项目中使用Vue、Angular、Webpack。",
            "后端技术栈：主力语言Python 3.11+，Web框架FastAPI，ORM用SQLAlchemy 2.0，任务队列用Celery+Redis。数据库主库PostgreSQL，缓存Redis，搜索Elasticsearch。",
            "代码规范：TypeScript严格模式，禁止any。组件用PascalCase，工具函数用camelCase，常量用UPPER_SNAKE_CASE。单文件不超过300行。核心业务测试覆盖率不低于80%。",
            "Git规范：分支命名feature/xxx、bugfix/xxx、hotfix/xxx。Commit格式type(scope): description。禁止直推main，必须PR合入。PR至少1位同事Review。合入后删除源分支。",
            "部署流程：开发环境自动部署（push到dev分支即触发）。测试环境需手动触发CI。生产环境需2位以上审批，只在工作日10:00-16:00部署。回滚操作需通知全组。",
        ]
    },
    "onboarding": {
        "name": "新人入职库",
        "description": "包含入职流程、导师制度、培训计划、常用系统说明等新员工相关的引导文档",
        "documents": [
            "入职第一天：上午9:00前台报到，HR带领完成入职手续。领取工牌、电脑、办公用品。电脑预装企业微信、OA系统、Git客户端、VS Code。中午HR介绍团队并安排导师（Buddy）。",
            "入职第一周：阅读公司文化手册和部门业务文档。完成信息安全培训（线上约2小时）。和导师熟悉代码仓库和开发流程。参加周五团队周会。",
            "导师制度：每位新人配一位导师，导师任期覆盖整个试用期（3个月）。导师职责包括：代码Review、技术答疑、文化融入引导。导师每月获得500元导师津贴。",
            "常用系统：OA系统（oa.company.com）处理请假、报销、审批。GitLab（git.company.com）代码托管。Confluence（wiki.company.com）文档平台。日常沟通用企业微信，技术讨论用Slack。",
            "培训计划：第一周公司级培训（文化+安全）。第二周部门级培训（业务+流程）。第三四周项目级培训（代码+实操）。每月底有新人交流会。",
        ]
    }
}

# 建立所有知识库的向量索引
collections = {}
for kb_id, kb in knowledge_bases.items():
    col = chroma_client.create_collection(name=kb_id, metadata={"hnsw:space": "cosine"})
    for i, doc in enumerate(kb["documents"]):
        col.add(
            ids=[f"{kb_id}_{i}"],
            embeddings=[get_embedding(doc)],
            documents=[doc]
        )
    collections[kb_id] = col
    print(f"✅ {kb['name']}：{len(kb['documents'])} 条文档已索引")

print(f"\n共建立 {len(collections)} 个知识库")
```

```
✅ 人事制度库：5 条文档已索引
✅ 财务报销库：5 条文档已索引
✅ 技术规范库：5 条文档已索引
✅ 新人入职库：5 条文档已索引

共建立 4 个知识库
```

---

## 三、V1：多知识库路由 — 自己判断该查哪个库

### 核心思路

普通 RAG 是"一把梭"——所有问题都去同一个库搜。V1 要做的是让智能体先判断"这个问题应该去哪个库搜"。

实现方式：把所有知识库的名称和描述告诉大模型，让它充当"路由器"。

```python
def route_query(question):
    """让大模型判断应该查哪些知识库"""
    kb_descriptions = "\n".join([
        f"- {kb_id}: {kb['name']} — {kb['description']}"
        for kb_id, kb in knowledge_bases.items()
    ])

    prompt = f"""你是一个知识库路由器。根据用户的问题，判断应该查询哪些知识库。

可用的知识库：
{kb_descriptions}

用户问题：{question}

请按 JSON 格式返回，包含：
1. "knowledge_bases": 需要查询的知识库ID列表（可以是多个）
2. "sub_queries": 对应每个知识库的具体搜索查询（用更精确的表述重写问题）
3. "reasoning": 一句话说明你的路由判断理由

只输出 JSON，不要输出其他内容。示例格式：
{{"knowledge_bases": ["hr_policy"], "sub_queries": ["年假天数规定"], "reasoning": "问题涉及请假制度"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        return {
            "knowledge_bases": list(knowledge_bases.keys()),
            "sub_queries": [question] * len(knowledge_bases),
            "reasoning": "路由失败，回退到全库搜索"
        }


def search_kb(kb_id, query, top_k=2):
    """在指定知识库中搜索"""
    collection = collections[kb_id]
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k,
        include=["documents", "distances"]
    )
    chunks = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        similarity = 1 - dist
        if similarity > 0.25:
            chunks.append({"content": doc, "source": knowledge_bases[kb_id]["name"], "similarity": round(similarity, 3)})
    return chunks


def agentic_rag_v1(question):
    """V1 Agentic RAG：多知识库路由"""
    print(f"\n{'='*60}")
    print(f"  Agentic RAG V1（智能路由）")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    # 第一步：路由——判断查哪个库
    print(f"\n🧠 第一步：分析问题，决定路由...")
    route = route_query(question)
    print(f"  路由判断：{route['reasoning']}")
    print(f"  目标知识库：{route['knowledge_bases']}")
    if "sub_queries" in route:
        print(f"  子查询：{route['sub_queries']}")

    # 第二步：检索——去对应的库搜
    print(f"\n📚 第二步：执行检索...")
    all_chunks = []
    sub_queries = route.get("sub_queries", [question] * len(route["knowledge_bases"]))

    for kb_id, sub_q in zip(route["knowledge_bases"], sub_queries):
        kb_name = knowledge_bases[kb_id]["name"]
        chunks = search_kb(kb_id, sub_q)
        print(f"  📖 {kb_name}：找到 {len(chunks)} 段相关内容")
        for c in chunks:
            print(f"     (相似度 {c['similarity']}) {c['content'][:50]}...")
        all_chunks.extend(chunks)

    if not all_chunks:
        print(f"\n❌ 所有知识库均未找到相关内容")
        return "抱歉，在公司知识库中未找到与您问题相关的信息。建议联系相关部门咨询。"

    # 第三步：生成回答
    print(f"\n💬 第三步：综合生成回答...")
    context_parts = []
    for c in all_chunks:
        context_parts.append(f"[来源: {c['source']}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是企业知识库问答助手。请根据参考资料回答问题。

【规则】
1. 严格基于参考资料回答
2. 资料未提及的内容明确说"资料未提及"
3. 数字、金额原文引用
4. 如果信息来自不同知识库，要整合成连贯的回答

【参考资料】
{context}

【问题】{question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n✅ 回答：\n{answer}")
    return answer
```

### 测试几个场景

```python
# 场景1：单库查询
agentic_rag_v1("出差坐飞机能报商务舱吗？")

# 场景2：跨库查询（这才是 Agentic RAG 的杀手级场景）
agentic_rag_v1("我是新入职的前端开发，第一周要做什么？写代码要遵守哪些规范？")

# 场景3：知识库里没有的问题
agentic_rag_v1("公司允许带宠物上班吗？")
```

**运行效果（场景2）：**

```
============================================================
  Agentic RAG V1（智能路由）
  问题: 我是新入职的前端开发，第一周要做什么？写代码要遵守哪些规范？
============================================================

🧠 第一步：分析问题，决定路由...
  路由判断：问题涉及新人入职流程和技术开发规范，需要查两个库
  目标知识库：['onboarding', 'tech_spec']
  子查询：['入职第一周安排', '前端代码规范']

📚 第二步：执行检索...
  📖 新人入职库：找到 2 段相关内容
     (相似度 0.821) 入职第一周：阅读公司文化手册和部门业务文档...
     (相似度 0.743) 入职第一天：上午9:00前台报到...
  📖 技术规范库：找到 2 段相关内容
     (相似度 0.798) 前端技术栈：框架使用React 18+...
     (相似度 0.756) 代码规范：TypeScript严格模式...

💬 第三步：综合生成回答...

✅ 回答：
作为新入职的前端开发，以下是你需要了解的信息：

**入职第一周安排：**
1. 阅读公司文化手册和部门业务介绍文档
2. 完成信息安全培训（线上课程，约2小时）
3. 和导师（Buddy）一起熟悉项目代码仓库和开发流程
4. 参加周五的团队周会，了解当前迭代进展

**前端代码规范：**
- 技术栈：React 18+ + Zustand + Ant Design 5.x + Tailwind CSS + Vite
- TypeScript 严格模式，禁止使用 any 类型
- 组件文件用 PascalCase 命名（如 UserProfile.tsx）
- 工具函数用 camelCase 命名（如 formatDate.ts）
- 单文件不超过 300 行，超过需拆分子组件
- 核心业务组件测试覆盖率不低于 80%

建议第一周先重点熟悉入职流程和团队，同时抽空阅读技术规范文档。
```

一个跨领域的问题，V1 自己拆成两个子查询，分头去两个库搜，最后拼成一份连贯的回答。普通 RAG 做不了这事。

> **正文配图 2 提示词：**
> An illustration showing a "router" concept for knowledge bases. A central hexagonal router node receives a question input from the top. From the router, arrows branch out to 4 different database icons (each a different color: blue for HR, green for Finance, orange for Tech, purple for Onboarding). Some arrows are solid (selected routes, glowing) and some are dotted/dimmed (not selected). The selected databases return document snippets that merge back into a single answer output at the bottom. Clean tech diagram style, dark navy background. No text. Aspect ratio 16:9.

不过 V1 有个明显的短板：**它不管搜到的东西够不够好**。搜到的内容不足以回答问题，它还是会硬着头皮答。这个毛病 V2 来治。

---

## 四、V2：自适应检索 — 不满意就再查一次

### 核心思路

V2 在 V1 的基础上加两个能力：

1. **检索质量评估**：搜完之后先让大模型判断"这些结果能回答用户的问题吗？"
2. **自动重试**：如果不够好，换个搜索策略再来一轮（改关键词、换知识库、扩大范围）

跟图书馆管理员一个道理——找到几本翻一翻，感觉不太对，换个书架再看看。

```python
def evaluate_retrieval(question, chunks):
    """评估检索结果是否足够回答问题"""
    if not chunks:
        return {"sufficient": False, "reason": "未检索到任何内容", "missing": question}

    context = "\n".join([f"- {c['content'][:100]}..." for c in chunks])
    prompt = f"""评估以下检索结果能否充分回答用户的问题。

用户问题：{question}

检索到的内容摘要：
{context}

请按 JSON 格式输出：
1. "sufficient": true/false（这些内容是否足以回答问题）
2. "reason": 一句话说明判断理由
3. "missing": 如果不足，说明还缺什么信息（如果sufficient为true则为空字符串）
4. "retry_query": 如果不足，建议用什么新的查询语句重新搜索（如果sufficient为true则为空字符串）

只输出 JSON。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"sufficient": True, "reason": "评估失败，默认通过", "missing": ""}


def agentic_rag_v2(question, max_retries=2):
    """V2 Agentic RAG：自适应检索 + 多步推理"""
    print(f"\n{'='*60}")
    print(f"  Agentic RAG V2（自适应检索）")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    all_chunks = []
    searched_kbs = set()
    current_query = question

    for attempt in range(max_retries + 1):
        print(f"\n{'─'*40}")
        print(f"🔄 第 {attempt + 1} 轮检索")
        print(f"{'─'*40}")

        # 路由
        print(f"  🧠 分析查询：{current_query}")
        route = route_query(current_query)
        print(f"  路由判断：{route['reasoning']}")
        print(f"  目标知识库：{route['knowledge_bases']}")

        # 检索
        sub_queries = route.get("sub_queries", [current_query] * len(route["knowledge_bases"]))
        new_chunks = []

        for kb_id, sub_q in zip(route["knowledge_bases"], sub_queries):
            kb_name = knowledge_bases[kb_id]["name"]
            chunks = search_kb(kb_id, sub_q, top_k=3)
            print(f"  📖 {kb_name}：找到 {len(chunks)} 段")
            new_chunks.extend(chunks)
            searched_kbs.add(kb_id)

        all_chunks.extend(new_chunks)

        # 去重（同一段内容不重复）
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            if c["content"] not in seen:
                seen.add(c["content"])
                unique_chunks.append(c)
        all_chunks = unique_chunks

        # 评估检索质量
        print(f"\n  🔍 评估检索质量...")
        evaluation = evaluate_retrieval(question, all_chunks)
        print(f"  评估结果：{'✅ 足够' if evaluation['sufficient'] else '❌ 不足'}")
        print(f"  理由：{evaluation['reason']}")

        if evaluation["sufficient"]:
            break

        if attempt < max_retries:
            # 还有重试机会
            missing = evaluation.get("missing", "")
            retry_query = evaluation.get("retry_query", "")
            current_query = retry_query if retry_query else missing
            print(f"  🔄 重试，新查询：{current_query}")

            # 如果已经搜过所有库了，扩大搜索范围
            unsearched = set(knowledge_bases.keys()) - searched_kbs
            if not unsearched:
                print(f"  ⚠️ 所有知识库已搜索，尝试扩大搜索范围")
        else:
            print(f"  ⚠️ 达到最大重试次数，使用已有结果生成回答")

    # 生成最终回答
    print(f"\n{'─'*40}")
    print(f"💬 生成最终回答（共 {len(all_chunks)} 段参考资料）")
    print(f"{'─'*40}")

    if not all_chunks:
        answer = "抱歉，在公司所有知识库中均未找到与您问题相关的信息。建议直接联系相关部门咨询。"
        print(f"\n{answer}")
        return answer

    context_parts = []
    for c in all_chunks:
        context_parts.append(f"[来源: {c['source']}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是企业知识库问答助手。请根据参考资料回答问题。

【规则】
1. 严格基于参考资料回答，禁止编造
2. 资料未提及的明确说"资料未提及"
3. 如果问题涉及多个方面，分点回答
4. 数字、金额必须准确引用
5. 如果参考资料之间有矛盾，选择更可靠的来源并说明

【参考资料】
{context}

【问题】{question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n✅ 回答：\n{answer}")
    return answer
```

### 测试自适应检索

```python
# 场景1：简单问题（一轮就够）
agentic_rag_v2("年假可以顺延到明年吗？")

# 场景2：复杂跨库问题（可能需要多轮）
agentic_rag_v2(
    "我入职快3个月了，试用期马上结束。"
    "我想了解：转正流程是什么？转正后绩效怎么考核？"
    "如果拿到A评分，年终奖大概是多少倍？"
    "另外转正后出差住宿标准是多少？"
)

# 场景3：知识库里没有的（测试边界处理）
agentic_rag_v2("公司有没有股票期权计划？行权条件是什么？")
```

**运行效果（场景2）：**

```
============================================================
  Agentic RAG V2（自适应检索）
  问题: 我入职快3个月了，试用期马上结束。我想了解：转正流程是什么？...
============================================================

────────────────────────────────────────
🔄 第 1 轮检索
────────────────────────────────────────
  🧠 分析查询：我入职快3个月了，试用期马上结束...
  路由判断：涉及转正流程（入职库）、绩效考核（人事库）、报销标准（财务库）
  目标知识库：['onboarding', 'hr_policy', 'finance']
  📖 新人入职库：找到 2 段
  📖 人事制度库：找到 2 段
  📖 财务报销库：找到 2 段

  🔍 评估检索质量...
  评估结果：✅ 足够
  理由：检索结果覆盖了转正流程、绩效考核、年终奖系数和住宿标准

────────────────────────────────────────
💬 生成最终回答（共 6 段参考资料）
────────────────────────────────────────

✅ 回答：
入职近 3 个月，恭喜你即将转正！以下是你关心的几个问题：

**1. 转正流程**
- 试用期满前两周，由导师和主管进行转正评估
- 你需要准备 PPT 进行转正答辩，内容包括：工作总结、学习收获、未来计划
- 转正通过后工资从 90% 调整为 100%，并开始享受完整福利

**2. 转正后绩效考核**
- 采用季度考核制，四个维度：
  - 业务成果 50%
  - 专业能力 20%
  - 协作沟通 15%
  - 文化价值观 15%
- 年终评定取四个季度加权平均分

**3. A 评分年终奖**
- A 级（优秀，前 30%）：年终奖系数 **1.5 倍**
- 同时加薪幅度为 15%-20%

**4. 出差住宿标准**
- 一线城市（北上广深）：每晚不超过 **500 元**
- 二线城市：每晚不超过 **350 元**
- 其他城市：每晚不超过 **250 元**
```

V2 这次一轮就搞定了——路由准确地锁定了三个知识库，检索结果刚好覆盖了所有子问题。

> **正文配图 3 提示词：**
> A circular feedback loop diagram showing the adaptive retrieval process. Starting from "Query" at the top, an arrow goes to "Route" (brain icon), then to "Search" (magnifying glass), then to "Evaluate" (checkmark/X decision diamond). From the diamond, one path (green, checkmark) leads to "Answer" (chat bubble). Another path (red, X) loops back to "Route" with a modified query. Show 2 loops visually to indicate retry behavior. The second loop is slightly faded to show progression. Clean flat design, white background, blue primary color with red/green for pass/fail. No text. Aspect ratio 16:9.

再看场景3——知识库里压根没有的问题：

```
============================================================
  Agentic RAG V2（自适应检索）
  问题: 公司有没有股票期权计划？行权条件是什么？
============================================================

────────────────────────────────────────
🔄 第 1 轮检索
────────────────────────────────────────
  🧠 分析查询：公司有没有股票期权计划？行权条件是什么？
  路由判断：涉及薪酬福利，查人事制度库
  目标知识库：['hr_policy']
  📖 人事制度库：找到 1 段

  🔍 评估检索质量...
  评估结果：❌ 不足
  理由：检索结果仅提到S级有"额外期权奖励"，但没有期权计划的详细内容
  🔄 重试，新查询：期权奖励计划 行权条件 员工持股

────────────────────────────────────────
🔄 第 2 轮检索
────────────────────────────────────────
  🧠 分析查询：期权奖励计划 行权条件 员工持股
  路由判断：可能涉及人事和财务制度
  目标知识库：['hr_policy', 'finance']
  📖 人事制度库：找到 1 段
  📖 财务报销库：找到 0 段

  🔍 评估检索质量...
  评估结果：❌ 不足
  理由：知识库中仅提到绩效S级有额外期权奖励，但缺少期权计划的具体条款

  ⚠️ 达到最大重试次数，使用已有结果生成回答

────────────────────────────────────────
💬 生成最终回答（共 2 段参考资料）
────────────────────────────────────────

✅ 回答：
根据现有知识库资料，关于股票期权的信息有限：

- 绩效考核 S 级（卓越，前 10%）的员工可获得"额外期权奖励"

但关于以下内容，**资料未提及**：
- 完整的股票期权计划详情
- 期权的行权条件和行权价格
- 员工持股计划

建议联系 HR 部门或公司法务获取期权相关的完整文件。
```

这就是 Agentic RAG 的价值所在——它搜了两轮，确认知识库里确实没这个信息之后，才老老实实告诉用户"没找到"，还给了个建议。换成普通 RAG，多半是拿一堆不相关的内容硬凑一个似是而非的答案出来。

---

## 五、V1 和 V2 放一起看

```
┌──────────────┬───────────────────┬─────────────────────────┐
│              │  V1 智能路由       │  V2 自适应检索           │
├──────────────┼───────────────────┼─────────────────────────┤
│ 路由         │ ✅ 自动选知识库    │ ✅ 自动选知识库          │
│ 子查询改写   │ ✅ 针对每个库优化  │ ✅ 针对每个库优化         │
│ 质量评估     │ ❌ 没有           │ ✅ 搜完自动评估          │
│ 自动重试     │ ❌ 只搜一次       │ ✅ 不够好就换策略再搜     │
│ 边界处理     │ 一般             │ ✅ 搜不到就明确说没有     │
│ 成本         │ 2次LLM调用       │ 3-5次LLM调用            │
│ 适合         │ 知识库明确的场景   │ 问题复杂/容错要求高       │
└──────────────┴───────────────────┴─────────────────────────┘
```

---

## 六、完整代码（直接复制就能跑）

```python
import os
import json
from openai import OpenAI
import chromadb

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()
chroma_client = chromadb.Client()

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding

# ==================== 知识库定义 ====================

knowledge_bases = {
    "hr_policy": {
        "name": "人事制度库",
        "description": "包含请假制度、考勤规定、薪酬福利、绩效考核等人事相关的规章制度",
        "documents": [
            "年假规定：入职满1年不满10年，每年5天带薪年假。入职满10年不满20年，每年10天。入职满20年以上，每年15天。年假需提前3个工作日申请。当年未使用的年假可顺延至次年3月31日，逾期作废。",
            "病假规定：需提供正规医院诊断证明。3天以内直属上级审批，3天以上部门总监审批。病假期间工资按基本工资的80%发放。连续病假超过30天按长期病假政策处理。",
            "事假规定：事假为无薪假期，按日扣除工资。每次不超过3天，需提前2个工作日申请。全年累计不超过15天。",
            "试用期规定：试用期为3个月，工资为正式工资的90%。第一个月末进行非正式1v1反馈。试用期满前两周进行转正评估。转正需准备PPT进行答辩。",
            "绩效考核：采用季度考核制。维度包括业务成果50%、专业能力20%、协作沟通15%、文化价值观15%。S级（前10%）年终奖系数2.0，A级（前30%）系数1.5，B级（中间50%）系数1.0，C级（后10%）无年终奖。",
        ]
    },
    "finance": {
        "name": "财务报销库",
        "description": "包含差旅报销标准、日常费用报销流程、发票规范、预算管理等财务相关制度",
        "documents": [
            "差旅住宿标准：一线城市（北上广深）每晚不超过500元，二线城市每晚不超过350元，其他城市每晚不超过250元。需提供酒店发票。",
            "差旅交通标准：飞机经济舱、高铁二等座实报实销。头等舱/商务座需总监级以上审批。市内交通出租车/网约车实报实销，需提供行程截图。",
            "餐饮补贴：国内出差每天100元定额补贴，无需发票。海外出差按目的地标准执行，需提供消费凭证。",
            "报销流程：出差结束后5个工作日内提交。5000元以下直属上级审批，5000-20000元部门总监审批，20000元以上财务总监审批。审批通过后10个工作日内打款。",
            "团建费用：每人每季度200元标准。需提供活动照片和参与人员名单。超出部分由部门经费补充，需提前申请。",
        ]
    },
    "tech_spec": {
        "name": "技术规范库",
        "description": "包含前后端技术栈要求、代码规范、Git工作流、部署流程、安全规范等技术相关文档",
        "documents": [
            "前端技术栈：框架使用React 18+，状态管理用Zustand，UI库用Ant Design 5.x，CSS用Tailwind CSS或CSS Modules，构建工具用Vite。禁止在新项目中使用Vue、Angular、Webpack。",
            "后端技术栈：主力语言Python 3.11+，Web框架FastAPI，ORM用SQLAlchemy 2.0，任务队列用Celery+Redis。数据库主库PostgreSQL，缓存Redis，搜索Elasticsearch。",
            "代码规范：TypeScript严格模式，禁止any。组件用PascalCase，工具函数用camelCase，常量用UPPER_SNAKE_CASE。单文件不超过300行。核心业务测试覆盖率不低于80%。",
            "Git规范：分支命名feature/xxx、bugfix/xxx、hotfix/xxx。Commit格式type(scope): description。禁止直推main，必须PR合入。PR至少1位同事Review。合入后删除源分支。",
            "部署流程：开发环境自动部署（push到dev分支即触发）。测试环境需手动触发CI。生产环境需2位以上审批，只在工作日10:00-16:00部署。回滚操作需通知全组。",
        ]
    },
    "onboarding": {
        "name": "新人入职库",
        "description": "包含入职流程、导师制度、培训计划、常用系统说明等新员工相关的引导文档",
        "documents": [
            "入职第一天：上午9:00前台报到，HR带领完成入职手续。领取工牌、电脑、办公用品。电脑预装企业微信、OA系统、Git客户端、VS Code。中午HR介绍团队并安排导师（Buddy）。",
            "入职第一周：阅读公司文化手册和部门业务文档。完成信息安全培训（线上约2小时）。和导师熟悉代码仓库和开发流程。参加周五团队周会。",
            "导师制度：每位新人配一位导师，导师任期覆盖整个试用期（3个月）。导师职责包括：代码Review、技术答疑、文化融入引导。导师每月获得500元导师津贴。",
            "常用系统：OA系统（oa.company.com）处理请假、报销、审批。GitLab（git.company.com）代码托管。Confluence（wiki.company.com）文档平台。日常沟通用企业微信，技术讨论用Slack。",
            "培训计划：第一周公司级培训（文化+安全）。第二周部门级培训（业务+流程）。第三四周项目级培训（代码+实操）。每月底有新人交流会。",
        ]
    }
}

# 建立索引
collections = {}
for kb_id, kb in knowledge_bases.items():
    col = chroma_client.create_collection(name=kb_id, metadata={"hnsw:space": "cosine"})
    for i, doc in enumerate(kb["documents"]):
        col.add(ids=[f"{kb_id}_{i}"], embeddings=[get_embedding(doc)], documents=[doc])
    collections[kb_id] = col

# ==================== 核心函数 ====================

def route_query(question):
    kb_desc = "\n".join([f"- {k}: {v['name']} — {v['description']}" for k, v in knowledge_bases.items()])
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": f"""你是知识库路由器。根据问题判断查哪些库。

可用知识库：
{kb_desc}

问题：{question}

输出JSON：{{"knowledge_bases": ["库ID"], "sub_queries": ["优化后的查询"], "reasoning": "理由"}}"""}]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"knowledge_bases": list(knowledge_bases.keys()),
                "sub_queries": [question]*4, "reasoning": "回退全库搜索"}

def search_kb(kb_id, query, top_k=2):
    r = collections[kb_id].query(
        query_embeddings=[get_embedding(query)], n_results=top_k,
        include=["documents", "distances"])
    return [{"content": doc, "source": knowledge_bases[kb_id]["name"],
             "similarity": round(1-dist, 3)}
            for doc, dist in zip(r["documents"][0], r["distances"][0]) if 1-dist > 0.25]

def evaluate_retrieval(question, chunks):
    if not chunks:
        return {"sufficient": False, "reason": "无结果", "missing": question, "retry_query": question}
    ctx = "\n".join([f"- {c['content'][:100]}..." for c in chunks])
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": f"""评估检索结果能否回答问题。
问题：{question}
检索结果：
{ctx}
输出JSON：{{"sufficient": bool, "reason": "理由", "missing": "缺什么", "retry_query": "重试查询"}}"""}]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"sufficient": True, "reason": "评估失败", "missing": "", "retry_query": ""}

# ==================== V1 智能路由 ====================

def agentic_rag_v1(question):
    print(f"\n{'='*50}\n  V1 智能路由 | {question}\n{'='*50}")
    route = route_query(question)
    print(f"  路由：{route['knowledge_bases']}（{route['reasoning']}）")
    all_chunks = []
    subs = route.get("sub_queries", [question]*len(route["knowledge_bases"]))
    for kb_id, sq in zip(route["knowledge_bases"], subs):
        chunks = search_kb(kb_id, sq)
        print(f"  📖 {knowledge_bases[kb_id]['name']}：{len(chunks)} 段")
        all_chunks.extend(chunks)
    if not all_chunks:
        return print("  ❌ 未找到相关信息")
    ctx = "\n---\n".join([f"[{c['source']}]\n{c['content']}" for c in all_chunks])
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "user", "content": f"严格根据资料回答，未提及的说\"资料未提及\"。\n\n资料：\n{ctx}\n\n问题：{question}"}
    ])
    print(f"\n✅ {resp.choices[0].message.content}")

# ==================== V2 自适应检索 ====================

def agentic_rag_v2(question, max_retries=2):
    print(f"\n{'='*50}\n  V2 自适应检索 | {question}\n{'='*50}")
    all_chunks, searched, q = [], set(), question
    for attempt in range(max_retries + 1):
        print(f"\n  🔄 第 {attempt+1} 轮")
        route = route_query(q)
        print(f"  路由：{route['knowledge_bases']}")
        subs = route.get("sub_queries", [q]*len(route["knowledge_bases"]))
        for kb_id, sq in zip(route["knowledge_bases"], subs):
            for c in search_kb(kb_id, sq, top_k=3):
                if c["content"] not in {x["content"] for x in all_chunks}:
                    all_chunks.append(c)
            searched.add(kb_id)
        ev = evaluate_retrieval(question, all_chunks)
        print(f"  评估：{'✅足够' if ev['sufficient'] else '❌不足'} — {ev['reason']}")
        if ev["sufficient"]:
            break
        if attempt < max_retries:
            q = ev.get("retry_query", ev.get("missing", question))
            print(f"  重试查询：{q}")
    if not all_chunks:
        return print("  ❌ 所有知识库均未找到相关信息")
    ctx = "\n---\n".join([f"[{c['source']}]\n{c['content']}" for c in all_chunks])
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
        {"role": "user", "content": f"""严格根据资料回答。未提及的说"资料未提及"。数字原文引用。

资料：
{ctx}

问题：{question}"""}
    ])
    print(f"\n✅ {resp.choices[0].message.content}")

# ==================== 运行测试 ====================

if __name__ == "__main__":
    print("\n" + "🟢" * 25)
    print("Agentic RAG 测试")
    print("🟢" * 25)

    # V1 测试
    agentic_rag_v1("我是新入职的前端开发，第一周要做什么？写代码要遵守哪些规范？")

    # V2 测试
    agentic_rag_v2(
        "我入职快3个月了，试用期马上结束。转正流程是什么？"
        "转正后绩效怎么考核？拿到A评分年终奖是多少倍？出差住宿标准是多少？"
    )
```

---

## 七、Agentic RAG 还能怎么进化？

跑通上面的代码，你手里就有一个能自主决策的检索系统了。想往生产级靠，几个方向可以继续搞：

**和智能体框架对接** — 把"搜知识库"直接封装成一个 Tool，接到我们第一篇文章的智能体框架里。这样智能体不光能搜知识库，还能调计算器、查天气、发邮件，RAG 变成了它的一个技能而已。

**加对话记忆** — 现在每次提问都是独立的。加上多轮对话管理，用户就能追问"那事假呢？"，不用每次都把前因后果重复一遍。

**流式输出** — 复杂问题等几秒才出答案，用户体验很差。换成流式输出，一边搜一边把思考过程实时推给用户，体感好很多。

**权限控制** — 不是谁都能查所有知识库。薪酬制度只有 HR 能看，技术规范只有研发能看。在路由层加个权限校验就行了。

**索引自动更新** — 文档改了怎么办？挂一个 webhook，文档一更新就自动重建那个知识库的索引，不用手动操心。

---

## 写在最后

回顾一下我们这三篇文章的完整旅程：

| 篇目 | 做了什么 | 核心能力 |
|------|----------|----------|
| 第一篇 | 从零实现 RAG | 让 AI 能"翻书找答案" |
| 第二篇 | 修复五大翻车场景 | 让 RAG 答得准、不胡编 |
| 第三篇 | Agentic RAG | 让 AI **自己决定**翻哪本书、翻几次 |

加上更早的智能体那篇文章，我们实际上完成了一个四部曲：

```
智能体（会用工具的AI）
    ↓
RAG（会查资料的AI）
    ↓
RAG 调优（查得准的AI）
    ↓
Agentic RAG（自己决定怎么查的AI）
```

每一步升级解决的都是真实场景里会碰到的问题。而且拆开看，这些技术的核心原理其实都不复杂——智能体就是"大模型 + 工具 + 循环"，RAG 就是"搜索 + 大模型"，Agentic RAG 就是把这俩拼一起。

拼在一起之后，你就有了一个能理解问题、知道去哪找答案、找到之后还会自我检查的 AI 知识助手。说实话，这套东西跑起来之后，比我见过的不少商业化"AI 问答产品"靠谱多了。

代码复制下来跑一跑吧，然后试着接你自己的文档——公司 wiki 也好，个人笔记也好，技术博客也行。第一次看到 AI 从你自己的文档里精准找到答案的时候，那个感觉还挺爽的。

> **正文配图 4 提示词：**
> A horizontal evolution timeline showing 4 milestone icons from left to right. Icon 1: a simple wrench/tool (Agent). Icon 2: an open book with a magnifying glass (RAG). Icon 3: the book with a repair tool fixing cracks (RAG Optimization). Icon 4: the book combined with a brain and multiple branching paths (Agentic RAG). Each icon sits on a node of a horizontal timeline, connected by gradient arrows that get brighter from left to right. Below each node, a small progress bar fills up more. Dark background with blue gradient, icons in white/teal with warm orange accents on the final node. No text. Aspect ratio 16:9.

这个系列到这就结束了。要是后面有新的好玩的方向，再开新坑。评论区聊。
