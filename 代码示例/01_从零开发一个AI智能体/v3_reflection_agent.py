"""
V3 反思版智能体：带自我审查和纠错的智能体
"""
import json
import os
from openai import OpenAI

# ==================== 环境配置 ====================
# 从环境变量读取API Key，使用前请先设置：
# export DEEPSEEK_API_KEY="your_key_here"
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
MODEL_NAME = "deepseek-chat"  # 使用的模型名称

# ==================== 工具函数（模拟真实API） ====================

def get_weather(city: str) -> str:
    """模拟查天气"""
    fake_weather = {
        "北京": "晴，25°C，微风，湿度40%",
        "上海": "多云，22°C，东南风3级，湿度65%",
        "广州": "雷阵雨，30°C，闷热，湿度85%",
        "深圳": "阴转小雨，28°C，湿度78%",
        "杭州": "晴，24°C，西北风2级，湿度50%",
    }
    return fake_weather.get(city, f"{city}：暂无天气数据")

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

def search_news(keyword: str) -> str:
    """模拟搜索新闻"""
    fake_news = {
        "AI": "【最新】OpenAI发布GPT-5，推理能力大幅提升；Google推出Gemini 2.5 Pro；"
              "国内多家大模型厂商宣布降价，AI应用开发成本持续下降。",
        "天气": "【预警】未来三天华南地区将迎来强降雨，气象台发布暴雨黄色预警。",
        "股市": "【财经】A股三大指数集体高开，AI概念股领涨，成交额突破万亿。",
        "广州": "【城市】广州地铁新线路即将开通，出行更便捷；广交会下周开幕，预计吸引全球客商。",
        "出差": "【商务】各地商务活动逐步恢复，酒店预订量环比上涨30%。",
    }
    for key, news in fake_news.items():
        if key in keyword:
            return news
    return f"未找到关于'{keyword}'的相关新闻"

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """模拟查汇率"""
    rates = {
        ("USD", "CNY"): 7.24,
        ("EUR", "CNY"): 7.86,
        ("JPY", "CNY"): 0.048,
        ("GBP", "CNY"): 9.15,
        ("CNY", "USD"): 0.138,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate:
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"
    return f"暂不支持 {from_currency} → {to_currency} 的汇率查询"

# ==================== 工具说明书（JSON Schema） ====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气情况，包括温度、风力、湿度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京、上海"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持加减乘除、括号、幂运算等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 (3+5)*2"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "根据关键词搜索最新新闻资讯",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词，如 AI、天气、股市"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "查询两种货币之间的汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "源货币代码，如 USD、EUR、CNY"},
                    "to_currency": {"type": "string", "description": "目标货币代码，如 CNY、USD"}
                },
                "required": ["from_currency", "to_currency"]
            }
        }
    }
]

# ==================== 工具执行器 ====================

available_tools = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_news": search_news,
    "get_exchange_rate": get_exchange_rate,
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name in available_tools:
        return available_tools[tool_name](**arguments)
    return f"未知工具: {tool_name}"

# ==================== V3 反思版智能体 ====================

REACT_SYSTEM_PROMPT_V3 = """你是一个善于深度思考的AI助手，可以查天气、算数学、搜新闻、查汇率。

你必须严格按照 ReAct 模式工作：
1. 每次回复先用【思考】说明推理过程
2. 然后决定是调用工具还是回答

注意：
- 每轮必须有【思考】
- 如果工具返回异常，在思考中说明调整策略
- 最终回答前要有总结性思考
"""

REFLECTION_PROMPT = """请你以"严格质检员"的身份，审视以下AI助手的回答。

用户的原始问题是：
{user_input}

AI助手的回答是：
{agent_answer}

请从以下维度检查，并输出审查报告：

1.【完整性】用户问了几个问题/子任务？是否每个都回答了？有没有遗漏？
2.【准确性】数据和计算是否正确？有没有张冠李戴或数字错误？
3.【逻辑性】推理过程是否合理？结论是否从证据中自然得出？有没有逻辑跳跃？
4.【实用性】给出的建议是否切实可行？有没有空话套话？
5.【潜在风险】有没有需要提醒用户但被忽略的重要信息？

最后给出：
- 综合评分：A（优秀）/ B（良好，有小瑕疵）/ C（及格，有明显遗漏）/ D（不及格，需要重做）
- 如果评分是 C 或 D，请明确指出需要补充或修正的内容
"""

REVISION_PROMPT = """你之前对用户的问题给出了一个回答，但质检发现了一些问题。

用户原始问题：
{user_input}

你之前的回答：
{agent_answer}

质检报告：
{reflection}

请根据质检报告的反馈，修正和完善你的回答。要求：
- 保留原回答中正确的部分
- 补充遗漏的内容
- 修正错误的数据或逻辑
- 输出一个更完整、更准确的最终回答
"""

def run_agent_v3(user_input: str):
    """V3 反思版：带自我审查和纠错的智能体"""
    print(f"\n{'='*60}")
    print(f"  V3 反思版智能体")
    print(f"  用户: {user_input}")
    print(f"{'='*60}")

    # ===== 阶段一：ReAct 执行，收集信息并生成初版回答 =====
    print(f"\n{'─'*40}")
    print("📝 阶段一：执行任务，生成初版回答")
    print(f"{'─'*40}")

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT_V3},
        {"role": "user", "content": user_input}
    ]

    agent_answer = ""
    for i in range(10):
        print(f"\n--- 第 {i+1} 轮 ---")
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )
        msg = response.choices[0].message

        if msg.content:
            print(f"\n💭 {msg.content}")

        if not msg.tool_calls:
            agent_answer = msg.content
            break

        messages.append(msg)
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  🔧 调用工具: {name}({args})")
            result = execute_tool(name, args)
            print(f"  📋 工具返回: {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    if not agent_answer:
        return "执行阶段未能生成回答"

    # ===== 阶段二：反思，对初版回答进行自我审查 =====
    print(f"\n{'─'*40}")
    print("🔍 阶段二：自我反思，审查初版回答")
    print(f"{'─'*40}")

    reflection_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": REFLECTION_PROMPT.format(
                user_input=user_input,
                agent_answer=agent_answer
            )
        }]
    )
    reflection = reflection_response.choices[0].message.content
    print(f"\n🔍 质检报告:\n{reflection}")

    # ===== 阶段三：如果质检不通过，修正回答 =====
    needs_revision = any(grade in reflection for grade in ["评分：C", "评分：D", "评分: C", "评分: D"])

    if needs_revision:
        print(f"\n{'─'*40}")
        print("🔄 阶段三：修正回答")
        print(f"{'─'*40}")

        revision_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": REVISION_PROMPT.format(
                    user_input=user_input,
                    agent_answer=agent_answer,
                    reflection=reflection
                )
            }]
        )
        final_answer = revision_response.choices[0].message.content
        print(f"\n✅ 修正后的最终回答:\n{final_answer}")
        return final_answer
    else:
        print(f"\n✅ 质检通过！初版回答质量合格，无需修正。")
        return agent_answer

# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试案例：复杂的出行规划任务
    run_agent_v3(
        "帮我做个出行规划：我从北京出发，想去一个天气好的城市玩两天。"
        "帮我对比一下上海、广州、杭州的天气，推荐一个最合适的目的地。"
        "另外查一下AI相关的新闻，我想在路上看看。"
        "最后算一下如果高铁票800元、酒店每晚350元、吃饭每天200元，两天总共要花多少钱。"
    )
