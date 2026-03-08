"""
V2 改进版 RAG：分块重叠 + 相似度过滤

相比 V1 的改进：
- 滑动窗口切块（相邻块有重叠区域，避免断句）
- 使用余弦相似度（只看方向不看长度，更适合文本）
- 相似度阈值过滤（过滤掉不相关的噪音结果）

运行方式：
    pip install openai chromadb
    export DEEPSEEK_API_KEY="your_key_here"
    export SILICONFLOW_API_KEY="your_key_here"
    python v2_improved_rag.py
"""
import os
from openai import OpenAI
import chromadb

# ==================== 环境配置 ====================

# DeepSeek API（用于对话生成）
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise ValueError(
        "请设置环境变量 DEEPSEEK_API_KEY\n"
        "  macOS/Linux: export DEEPSEEK_API_KEY='your_key_here'\n"
        "  Windows:     set DEEPSEEK_API_KEY=your_key_here"
    )

# 硅基流动 API（用于 embedding）
siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
if not siliconflow_key:
    raise ValueError(
        "请设置环境变量 SILICONFLOW_API_KEY\n"
        "  macOS/Linux: export SILICONFLOW_API_KEY='your_key_here'\n"
        "  Windows:     set SILICONFLOW_API_KEY=your_key_here"
    )

# 对话模型客户端
chat_client = OpenAI(base_url="https://api.deepseek.com", api_key=deepseek_key)
CHAT_MODEL = "deepseek-chat"

# Embedding 模型客户端
embedding_client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key=siliconflow_key)
EMBEDDING_MODEL = "netease-youdao/bce-embedding-base_v1"

# ==================== 测试文档（模拟公司知识库） ====================

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

# ==================== 工具函数 ====================


def get_embedding(text):
    """调用硅基流动 Embedding 接口，将文本转为向量"""
    response = embedding_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def overlap_chunk(text, chunk_size=200, overlap=50):
    """带重叠的滑动窗口切块

    相邻块之间有 overlap 个字符的重叠区域。
    好处：一句话即使被切开了，在重叠区里还能保住完整的上下文。

    参数选择建议：
    - chunk_size: 中文 200-500 字。太小丢上下文，太大降精度。
    - overlap: 一般取 chunk_size 的 10%-25%。

    示意图：
      [----块1----]
              [----块2----]
                      [----块3----]
              ↑ 重叠区保住了上下文
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # 每次只前进 (chunk_size - overlap)
    return chunks


# ==================== 索引构建 ====================

chroma_client = chromadb.Client()

# 使用余弦相似度（cosine）而非默认的 L2 距离
# 余弦只看向量"方向"不看"长度"，更适合文本语义比较
collection = chroma_client.create_collection(
    name="company_docs_v2",
    metadata={"hnsw:space": "cosine"}
)


def build_index(documents):
    """V2：重叠切块 + 入库"""
    print("正在建立 V2 索引...")
    for doc in documents:
        chunks = overlap_chunk(doc["content"], chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_v2_chunk_{i}"
            embedding = get_embedding(chunk)
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["title"]}]
            )
        print(f"  {doc['title']}：切成 {len(chunks)} 块（带重叠）")
    print(f"V2 索引建立完成，共 {collection.count()} 个文档块\n")


# ==================== 检索 + 生成 ====================


def search(query, top_k=5, threshold=0.3):
    """带相似度过滤的检索

    参数：
    - top_k: 先召回多少条候选（多召回一些，后面再过滤）
    - threshold: 相似度阈值，低于此值的结果直接丢弃

    阈值选择建议（text-embedding-3-small 模型下）：
    - 0.5 以上：语义明显相关
    - 0.3-0.5：沾点边
    - 0.3 以下：基本是噪音
    建议先设 0.4 跑一批测试，再根据漏检/误检比例调整。
    """
    query_embedding = get_embedding(query)
    results = collection.query(
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

    chunks, metadatas = search(question, top_k=5, threshold=0.3)

    if not chunks:
        print("\n未找到足够相关的文档内容")
        return "抱歉，知识库中没有找到与您的问题相关的内容。"

    print(f"\n检索到 {len(chunks)} 段相关内容（已过滤低相关度）：")
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

    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"\n回答：\n{answer}")
    return answer


# ==================== 运行测试 ====================

if __name__ == "__main__":
    build_index(documents)

    # 测试：这个问题涉及的信息在 V1 中可能被切断
    rag_query_v2("病假工资怎么算？超过30天怎么办？")
    rag_query_v2("入职满5年有几天年假？")
    rag_query_v2("出差住酒店的报销标准是多少？")

