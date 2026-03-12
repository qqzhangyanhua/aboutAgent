"""
V1 工具注册表：用装饰器注册工具，自动生成 schema，按名字分发调用
对应文章第 17 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python v1_tool_registry.py
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


class ToolRegistry:
    """工具注册表：用装饰器注册工具，自动生成 schema，按名字分发调用"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, description: str, parameters: dict):
        """装饰器：注册一个工具"""
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


# ==================== 注册示例工具 ====================

registry = ToolRegistry()


@registry.register(
    description="查询指定城市的天气",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名称"}},
        "required": ["city"]
    }
)
def get_weather(city: str) -> str:
    weather_data = {"北京": "晴，28°C", "上海": "多云，25°C", "深圳": "雷阵雨，31°C"}
    return weather_data.get(city, f"{city}：暂无数据")


@registry.register(
    description="搜索新闻",
    parameters={
        "type": "object",
        "properties": {"keyword": {"type": "string", "description": "搜索关键词"}},
        "required": ["keyword"]
    }
)
def search_news(keyword: str) -> str:
    return f"搜索到 3 条关于 '{keyword}' 的新闻：1. {keyword}最新进展 2. {keyword}行业分析 3. {keyword}未来趋势"


@registry.register(
    description="数学计算",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "数学表达式"}},
        "required": ["expression"]
    }
)
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "错误：只支持基本数学运算"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


if __name__ == "__main__":
    print(f"已注册工具：{registry.list_tools()}")
    print(f"\nSchema 示例：")
    print(json.dumps(registry.get_schemas()[0], indent=2, ensure_ascii=False))

    print(f"\n调用测试：")
    print(f"  get_weather('北京') = {registry.call('get_weather', {'city': '北京'})}")
    print(f"  calculate('1024*768') = {registry.call('calculate', {'expression': '1024*768'})}")
    print(f"  search_news('AI') = {registry.call('search_news', {'keyword': 'AI'})}")
