"""
V2 Agent 基类：封装 LLM 调用循环 + 工具分发
对应文章第 17 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python v2_agent_base.py
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


# ==================== V1 工具注册表（复用） ====================

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, description: str, parameters: dict):
        def decorator(func: Callable) -> Callable:
            self._tools[func.__name__] = {
                "function": func,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": description,
                        "parameters": parameters
                    }
                }
            }
            return func
        return decorator

    def get_schemas(self) -> list[dict]:
        return [t["schema"] for t in self._tools.values()]

    def call(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            return f"错误：未知工具 '{name}'"
        return self._tools[name]["function"](**arguments)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ==================== V2 Agent 基类 ====================

class BaseAgent:
    """智能体基类：封装核心循环"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 10
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        tools = self.registry.get_schemas() or None

        for iteration in range(self.max_iterations):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.history,
                tools=tools
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content

            self.history.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            })

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self.registry.call(name, args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return "达到最大迭代次数，任务未完成。"

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


# ==================== 注册工具 ====================

registry = ToolRegistry()

@registry.register(
    description="查询指定城市的天气",
    parameters={"type": "object", "properties": {"city": {"type": "string", "description": "城市名称"}}, "required": ["city"]}
)
def get_weather(city: str) -> str:
    weather_data = {"北京": "晴，28°C", "上海": "多云，25°C", "深圳": "雷阵雨，31°C"}
    return weather_data.get(city, f"{city}：暂无数据")

@registry.register(
    description="数学计算",
    parameters={"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}
)
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "错误：只支持基本数学运算"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


# ==================== 运行 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    agent = BaseAgent(
        name="助手",
        system_prompt="你是一个全能助手，可以查天气和做计算。回答简洁。",
        tool_registry=registry
    )

    questions = [
        "北京今天天气怎么样？",
        "帮我算一下 1024 * 768",
        "深圳天气怎么样？另外算一下 3.14 * 100",
    ]

    for q in questions:
        agent.reset()
        print(f"\n用户：{q}")
        answer = agent.run(q)
        print(f"助手：{answer}")
