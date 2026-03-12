"""
V3 中间件系统：日志、Token 统计、重试
对应文章第 17 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python v3_middleware.py
"""
import os
import json
import time
from typing import Callable, Any, Protocol
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"


# ==================== V1 ToolRegistry ====================

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, description: str, parameters: dict):
        def decorator(func: Callable) -> Callable:
            self._tools[func.__name__] = {
                "function": func,
                "schema": {"type": "function", "function": {"name": func.__name__, "description": description, "parameters": parameters}}
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


# ==================== V2 BaseAgent ====================

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tool_registry: ToolRegistry | None = None, max_iterations: int = 10):
        self.name = name
        self.system_prompt = system_prompt
        self.registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        tools = self.registry.get_schemas() or None
        for _ in range(self.max_iterations):
            response = client.chat.completions.create(model=MODEL_NAME, messages=self.history, tools=tools)
            msg = response.choices[0].message
            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content
            self.history.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
            for tool_call in msg.tool_calls:
                result = self.registry.call(tool_call.function.name, json.loads(tool_call.function.arguments))
                self.history.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        return "达到最大迭代次数"

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


# ==================== V3 中间件 ====================

class Middleware(Protocol):
    def before_llm_call(self, messages: list[dict]) -> list[dict]: ...
    def after_llm_call(self, response: Any) -> Any: ...


class LoggingMiddleware:
    """日志中间件"""
    def __init__(self):
        self._start_time = 0.0

    def before_llm_call(self, messages: list[dict]) -> list[dict]:
        msg_count = len(messages)
        last_msg = messages[-1]["content"][:50] if messages and messages[-1].get("content") else ""
        print(f"  [LOG] → LLM 调用，{msg_count} 条消息，最后一条：{last_msg}...")
        self._start_time = time.time()
        return messages

    def after_llm_call(self, response: Any) -> Any:
        elapsed = time.time() - self._start_time
        print(f"  [LOG] ← LLM 响应，耗时 {elapsed:.2f}s")
        return response


class TokenCountMiddleware:
    """Token 计费中间件"""
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def before_llm_call(self, messages: list[dict]) -> list[dict]:
        return messages

    def after_llm_call(self, response: Any) -> Any:
        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.call_count += 1
        return response

    def report(self) -> str:
        total = self.total_prompt_tokens + self.total_completion_tokens
        return (
            f"API 调用 {self.call_count} 次 | "
            f"输入 {self.total_prompt_tokens} tokens | "
            f"输出 {self.total_completion_tokens} tokens | "
            f"总计 {total} tokens"
        )


class MiddlewareAgent(BaseAgent):
    """支持中间件的智能体"""

    def __init__(self, middlewares: list | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.middlewares = middlewares or []

    def _call_llm(self, messages: list[dict], tools: list[dict] | None) -> Any:
        current_messages = messages
        for mw in self.middlewares:
            current_messages = mw.before_llm_call(current_messages)
        response = client.chat.completions.create(model=MODEL_NAME, messages=current_messages, tools=tools)
        for mw in reversed(self.middlewares):
            response = mw.after_llm_call(response)
        return response

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        tools = self.registry.get_schemas() or None
        for _ in range(self.max_iterations):
            response = self._call_llm(self.history, tools)
            msg = response.choices[0].message
            if not msg.tool_calls:
                self.history.append({"role": "assistant", "content": msg.content})
                return msg.content
            self.history.append({"role": "assistant", "content": msg.content, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
            for tool_call in msg.tool_calls:
                result = self.registry.call(tool_call.function.name, json.loads(tool_call.function.arguments))
                self.history.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        return "达到最大迭代次数"


# ==================== 注册工具 ====================

registry = ToolRegistry()

@registry.register(description="查询城市天气", parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
def get_weather(city: str) -> str:
    return {"北京": "晴，28°C", "上海": "多云，25°C", "深圳": "雷阵雨，31°C"}.get(city, f"{city}：暂无数据")

@registry.register(description="数学计算", parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "错误：只支持基本运算"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


# ==================== 运行 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    token_counter = TokenCountMiddleware()

    agent = MiddlewareAgent(
        name="带中间件的助手",
        system_prompt="你是一个全能助手，回答简洁。",
        tool_registry=registry,
        middlewares=[LoggingMiddleware(), token_counter]
    )

    print("=== V3 中间件系统演示 ===\n")

    for q in ["深圳天气怎么样？", "帮我算 256 * 512"]:
        agent.reset()
        print(f"用户：{q}")
        answer = agent.run(q)
        print(f"助手：{answer}\n")

    print(f"统计：{token_counter.report()}")
