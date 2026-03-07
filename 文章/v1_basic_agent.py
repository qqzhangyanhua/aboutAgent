"""
V1 基础版智能体：简单的工具调用循环
"""
import json
import os
from openai import OpenAI

# ==================== 环境配置 ====================
os.environ["OPENAI_API_KEY"] = "你的API Key"
# 如果用 DeepSeek 等国产模型，取消下面的注释并修改
# client = OpenAI(base_url="https://api.deepseek.com", api_key="你的Key")
client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-14aaf5be0bcc450e982573fff9ff5328")
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

# ==================== V1 基础版智能体 ====================

def run_agent_v1(user_input: str):
    """V1 基础版：简单的工具调用循环"""
    print(f"\n{'='*60}")
    print(f"  V1 基础版智能体")
    print(f"  用户: {user_input}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，可以查天气、算数学、搜新闻、查汇率。"},
        {"role": "user", "content": user_input}
    ]

    for i in range(10):
        print(f"\n--- 第 {i+1} 轮 ---")
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

        # 调试：打印响应类型和内容
        print(f"DEBUG: response type = {type(response)}")
        print(f"DEBUG: response = {response}")

        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\n✅ 最终回答: {msg.content}")
            return msg.content

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

    return "达到最大轮次"

# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试案例1：简单查询
    run_agent_v1("帮我查一下北京和上海的天气，然后告诉我哪个城市更适合这周末出游")

    # 测试案例2：多步骤任务
    # run_agent_v1("帮我对比北京、上海、杭州的天气，推荐一个适合周末出游的城市，再帮我算一下高铁票650元加两晚酒店每晚380元加两天吃饭每天150元总共多少钱")
