"""
Agentic RAG：智能体自主检索
完整可运行示例 —— 对应文章第 04 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python agentic_rag.py
"""
import os
import json
from openai import OpenAI
import chromadb

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"

# ==================== 知识库初始化 ====================
# 4个知识库，每个有5-6条模拟文档

knowledge_bases_config = {
    "hr_policy": {
        "name": "人事制度库",
        "description": "包含请假制度、考勤规定、薪酬福利、绩效考核等人事相关规章制度",
        "documents": [
            "年假规定：入职满1年不满10年，每年5天带薪年假。满10年不满20年，每年10天。满20年以上，每年15天。当年未使用年假可顺延至次年3月31日。",
            "病假规定：需提供正规医院诊断证明。3天以内直属上级审批，3天以上部门总监审批。病假期间工资按基本工资80%发放。",
            "事假规定：事假为无薪假期，按日扣除工资。每次不超过3天，全年累计不超过15天。",
            "试用期规定：试用期为3个月，工资为正式工资的90%。试用期满前两周进行转正评估。",
            "绩效考核：采用季度考核制。S级（前10%）年终奖系数2.0，A级系数1.5，B级系数1.0，C级无年终奖。",
        ]
    },
    "finance": {
        "name": "财务报销库",
        "description": "包含差旅报销标准、日常费用报销流程、发票规范等财务相关制度",
        "documents": [
            "差旅住宿标准：一线城市每晚不超过500元，二线城市不超过350元，其他城市不超过250元。需提供酒店发票。",
            "餐饮补贴标准：国内出差每日餐补100元，无需发票。海外出差按目的地标准执行。",
            "交通报销：优先公共交通，打车需注明事由。高铁二等座可报，飞机经济舱可报。",
            "报销流程：费用发生后30天内提交报销申请。5000元以内部门经理审批，5000元以上需财务总监审批。",
        ]
    },
    "tech_standard": {
        "name": "技术规范库",
        "description": "包含代码规范、技术选型标准、安全规范等技术相关文档",
        "documents": [
            "代码规范：Python 项目统一使用 Black 格式化，类型注解必须完整。PR 需至少一人 Review。",
            "技术选型：后端优先 Python/FastAPI，前端统一 React+TypeScript。数据库首选 PostgreSQL。",
            "安全规范：密码必须加盐哈希存储。API 必须走 HTTPS。敏感操作需二次确认。",
            "部署规范：使用 Docker 容器化部署，CI/CD 基于 GitHub Actions。生产环境变更需走审批。",
        ]
    },
    "onboarding": {
        "name": "入职指南库",
        "description": "包含新员工入职流程、培训计划、办公设备申领等新人相关指南",
        "documents": [
            "第一天：HR 带领熟悉办公环境，领取工牌和电脑。完成系统账号开通（邮箱、VPN、内部工具）。",
            "第一周：阅读公司 handbook，参加新人培训。与 buddy 进行 1v1，了解团队工作方式。",
            "第一个月：完成部门内所有系统的权限申请。参与至少一个小型项目。月末与直属上级 1v1 反馈。",
            "试用期目标：3个月内完成至少一个独立负责的项目。试用期满前两周进行转正答辩。",
        ]
    }
}

def init_knowledge_bases() -> dict[str, chromadb.Collection]:
    """初始化所有知识库"""
    chroma = chromadb.Client()
    collections = {}
    for kb_id, config in knowledge_bases_config.items():
        col = chroma.get_or_create_collection(kb_id, metadata={"hnsw:space": "cosine"})
        ids = [f"{kb_id}_{i}" for i in range(len(config["documents"]))]
        existing = col.get()["ids"]
        new_items = [(id_, doc) for id_, doc in zip(ids, config["documents"]) if id_ not in existing]
        if new_items:
            col.add(ids=[x[0] for x in new_items], documents=[x[1] for x in new_items])
        collections[kb_id] = col
    return collections


# ==================== 知识库路由 ====================

def route_to_knowledge_base(query: str) -> list[str]:
    """用 LLM 判断该查哪些知识库"""
    kb_descriptions = "\n".join(
        f"- {kb_id}: {config['description']}"
        for kb_id, config in knowledge_bases_config.items()
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": (
                f"判断以下问题应该查询哪些知识库。可以选多个。\n\n"
                f"可用知识库：\n{kb_descriptions}\n\n"
                f"问题：{query}\n\n"
                f"只返回知识库 ID 列表，用逗号分隔，不要其他内容。"
            )
        }],
        temperature=0
    )
    ids = [x.strip() for x in response.choices[0].message.content.strip().split(",")]
    return [x for x in ids if x in knowledge_bases_config]


# ==================== 检索与质量评估 ====================

def retrieve_from_kb(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[dict]:
    """从指定知识库检索"""
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    docs = []
    if results["documents"]:
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            docs.append({"content": doc, "distance": dist})
    return docs


def evaluate_retrieval_quality(query: str, docs: list[dict]) -> dict:
    """评估检索结果质量"""
    if not docs:
        return {"sufficient": False, "reason": "未检索到任何文档"}
    docs_text = "\n".join(f"- {d['content'][:200]}" for d in docs)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": (
                f"评估以下检索结果能否回答用户问题。\n\n"
                f"问题：{query}\n\n检索结果：\n{docs_text}\n\n"
                f"只返回 JSON：{{\"sufficient\": true/false, \"reason\": \"原因\"}}"
            )
        }],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(text)
    except:
        return {"sufficient": True, "reason": "无法解析评估结果，默认接受"}


# ==================== Agentic RAG 智能体 ====================

class AgenticRAG:
    """Agentic RAG：自主决定检索策略的智能体"""

    def __init__(self):
        self.collections = init_knowledge_bases()

    def query(self, question: str) -> str:
        print(f"\n{'='*50}")
        print(f"问题：{question}")

        # 1. 路由到相关知识库
        target_kbs = route_to_knowledge_base(question)
        print(f"路由到知识库：{target_kbs}")

        # 2. 检索
        all_docs: list[dict] = []
        for kb_id in target_kbs:
            if kb_id in self.collections:
                docs = retrieve_from_kb(self.collections[kb_id], question)
                for d in docs:
                    d["source"] = knowledge_bases_config[kb_id]["name"]
                all_docs.extend(docs)
                print(f"  从 {knowledge_bases_config[kb_id]['name']} 检索到 {len(docs)} 条")

        # 3. 评估检索质量
        quality = evaluate_retrieval_quality(question, all_docs)
        print(f"质量评估：{quality}")

        if not quality.get("sufficient", True) and len(target_kbs) < len(self.collections):
            print("检索结果不足，扩大搜索范围...")
            for kb_id, col in self.collections.items():
                if kb_id not in target_kbs:
                    extra_docs = retrieve_from_kb(col, question, top_k=2)
                    for d in extra_docs:
                        d["source"] = knowledge_bases_config[kb_id]["name"]
                    all_docs.extend(extra_docs)

        # 4. 生成回答
        context = "\n\n".join(
            f"[来源：{d['source']}] {d['content']}" for d in all_docs
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是企业知识助手。基于参考资料回答问题，注明信息来源。如果资料不足，明确说明。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    rag = AgenticRAG()

    questions = [
        "年假有几天？能顺延吗？",
        "出差去上海住酒店最多报多少钱？",
        "新员工入职第一周要做什么？技术规范在哪看？",
        "公司的绩效考核怎么评？S级有什么好处？",
    ]

    for q in questions:
        answer = rag.query(q)
        print(f"\n回答：{answer}\n")
