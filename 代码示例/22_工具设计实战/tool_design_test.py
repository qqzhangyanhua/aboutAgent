"""
工具设计实战：大工具 vs 小工具选择准确率对比测试
完整可运行示例 —— 对应文章第 22 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python tool_design_test.py
"""
import os
import json
from typing import Callable, Any
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"


# ==================== 方案 A：一个大工具 ====================

big_tool = [{
    "type": "function",
    "function": {
        "name": "order_system",
        "description": "订单系统操作，通过 action 参数区分操作类型",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["query", "cancel", "refund", "track", "update_address"],
                    "description": "操作类型"
                },
                "order_id": {"type": "string", "description": "订单号"},
                "reason": {"type": "string", "description": "原因（退款时必填）"},
                "new_address": {"type": "string", "description": "新地址（改地址时必填）"}
            },
            "required": ["action", "order_id"]
        }
    }
}]


# ==================== 方案 B：五个小工具 ====================

small_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "取消指定订单",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_refund",
            "description": "提交退款申请",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"},
                    "reason": {"type": "string", "description": "退款原因"}
                },
                "required": ["order_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_logistics",
            "description": "查询包裹物流状态",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_address",
            "description": "修改订单的收货地址",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号"},
                    "new_address": {"type": "string", "description": "新的收货地址"}
                },
                "required": ["order_id", "new_address"]
            }
        }
    },
]


# ==================== 测试函数 ====================

def test_tool_selection(tools: list[dict], label: str) -> dict[str, int]:
    """测试大模型的工具选择准确率"""
    test_cases = [
        ("我的订单 12345 现在什么状态？", "query"),
        ("我想取消订单 67890", "cancel"),
        ("订单 11111 我要退款，商品有质量问题", "refund"),
        ("快递到哪了？订单号 22222", "track"),
        ("我搬家了，订单 33333 改个地址到北京海淀区", "update_address"),
    ]

    expected_map_big = {
        "query": ("order_system", "query"),
        "cancel": ("order_system", "cancel"),
        "refund": ("order_system", "refund"),
        "track": ("order_system", "track"),
        "update_address": ("order_system", "update_address"),
    }

    expected_map_small = {
        "query": ("query_order", None),
        "cancel": ("cancel_order", None),
        "refund": ("apply_refund", None),
        "track": ("track_logistics", None),
        "update_address": ("update_address", None),
    }

    is_big = len(tools) == 1
    expected_map = expected_map_big if is_big else expected_map_small

    correct = 0
    total = len(test_cases)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    for query, expected_key in test_cases:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            temperature=0
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = json.loads(call.function.arguments)
            tool_name = call.function.name

            exp_name, exp_action = expected_map[expected_key]

            if is_big:
                actual_action = args.get("action", "")
                is_correct = tool_name == exp_name and actual_action == exp_action
                display = f"{tool_name}(action={actual_action})"
            else:
                is_correct = tool_name == exp_name
                display = f"{tool_name}({args})"

            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1
            print(f"  {status} {query[:30]:30s} → {display}")
        else:
            print(f"  ❌ {query[:30]:30s} → 未调用工具")

    accuracy = correct / total * 100
    print(f"\n  准确率：{correct}/{total} = {accuracy:.0f}%")
    return {"correct": correct, "total": total}


# ==================== 工具拆分判断 ====================

def should_split_tool(tool_info: dict) -> dict[str, Any]:
    """根据规则判断工具是否应该拆分"""
    params = tool_info.get("parameters", {}).get("properties", {})
    description = tool_info.get("description", "")

    signals: list[str] = []
    score = 0

    if len(params) > 5:
        signals.append(f"参数过多（{len(params)} 个）")
        score += 2

    if any(p.get("enum") for p in params.values()):
        for name, p in params.items():
            if p.get("enum") and len(p["enum"]) > 3:
                signals.append(f"参数 '{name}' 有 {len(p['enum'])} 个枚举值")
                score += 2

    if len(description) > 100:
        signals.append("描述过长，可能职责过多")
        score += 1

    and_or_count = description.count("和") + description.count("或") + description.count("以及")
    if and_or_count >= 2:
        signals.append(f"描述中有 {and_or_count} 个"和/或"，暗示多职责")
        score += 1

    return {
        "should_split": score >= 2,
        "score": score,
        "signals": signals,
        "recommendation": "建议拆分" if score >= 2 else "暂时可以不拆"
    }


# ==================== Skill 编排 ====================

class SkillEngine:
    """Skill 引擎：组合多个原子工具成高层意图"""

    def __init__(self):
        self.skills: dict[str, dict] = {}
        self.tools: dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        self.tools[name] = func

    def register_skill(self, name: str, description: str, steps: list[str]) -> None:
        self.skills[name] = {"description": description, "steps": steps}

    def execute_skill(self, skill_name: str, context: dict) -> dict:
        if skill_name not in self.skills:
            return {"error": f"未知 Skill: {skill_name}"}
        skill = self.skills[skill_name]
        print(f"\n  执行 Skill: {skill_name}")
        for step_name in skill["steps"]:
            if step_name not in self.tools:
                return {"error": f"未知工具: {step_name}"}
            print(f"    → {step_name}")
            result = self.tools[step_name](context)
            context.update(result)
        return context


# ==================== 运行演示 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    # Demo 1：工具选择准确率对比
    print("\n" + "=" * 60)
    print("Demo 1：大工具 vs 小工具 - 选择准确率对比")
    print("=" * 60)
    result_a = test_tool_selection(big_tool, "方案 A：一个大工具")
    result_b = test_tool_selection(small_tools, "方案 B：五个小工具")

    print(f"\n{'='*50}")
    print(f"  对比结果")
    print(f"{'='*50}")
    print(f"  方案 A（大工具）：{result_a['correct']}/{result_a['total']}")
    print(f"  方案 B（小工具）：{result_b['correct']}/{result_b['total']}")

    # Demo 2：工具拆分判断
    print(f"\n{'='*60}")
    print("Demo 2：工具拆分判断")
    print(f"{'='*60}")
    analysis = should_split_tool(big_tool[0]["function"])
    print(f"  分析 order_system:")
    print(f"    应拆分：{analysis['should_split']}")
    print(f"    得分：{analysis['score']}")
    for s in analysis["signals"]:
        print(f"    ⚠️  {s}")
    print(f"    建议：{analysis['recommendation']}")

    # Demo 3：Skill 编排
    print(f"\n{'='*60}")
    print("Demo 3：Skill 编排演示")
    print(f"{'='*60}")
    engine = SkillEngine()

    engine.register_tool("check_refund_eligible", lambda ctx: {
        "eligible": True,
        "max_refund": ctx.get("order_amount", 100)
    })
    engine.register_tool("calculate_refund", lambda ctx: {
        "refund_amount": ctx["max_refund"] * 0.9
    })
    engine.register_tool("submit_refund", lambda ctx: {
        "refund_id": "RF-20250310-001",
        "status": "已提交",
        "amount": ctx["refund_amount"]
    })

    engine.register_skill(
        name="自动退款",
        description="检查退款资格 → 计算退款金额 → 提交退款",
        steps=["check_refund_eligible", "calculate_refund", "submit_refund"]
    )

    result = engine.execute_skill("自动退款", {
        "order_id": "ORD-12345",
        "order_amount": 299.0,
        "reason": "商品质量问题"
    })
    print(f"\n  退款结果：")
    print(f"    退款单号：{result.get('refund_id')}")
    print(f"    状态：{result.get('status')}")
    print(f"    金额：¥{result.get('refund_amount', 0):.2f}")
