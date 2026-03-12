"""
V4 完整框架：ToolRegistry + BaseAgent + Middleware + Pipeline + Router
对应文章第 17 篇

运行：
    export DEEPSEEK_API_KEY="your_key"
    python v4_full_framework.py
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


# ==================== 核心组件 ====================

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
            for tc in msg.tool_calls:
                result = self.registry.call(tc.function.name, json.loads(tc.function.arguments))
                self.history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        return "达到最大迭代次数"

    def reset(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


class LoggingMiddleware:
    def __init__(self):
        self._start_time = 0.0

    def before_llm_call(self, messages: list[dict]) -> list[dict]:
        print(f"  [LOG] → LLM 调用，{len(messages)} 条消息")
        self._start_time = time.time()
        return messages

    def after_llm_call(self, response: Any) -> Any:
        print(f"  [LOG] ← 响应，耗时 {time.time() - self._start_time:.2f}s")
        return response


class TokenCountMiddleware:
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
        return f"调用 {self.call_count} 次 | 输入 {self.total_prompt_tokens} | 输出 {self.total_completion_tokens} | 总计 {total} tokens"


class MiddlewareAgent(BaseAgent):
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
            for tc in msg.tool_calls:
                result = self.registry.call(tc.function.name, json.loads(tc.function.arguments))
                self.history.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        return "达到最大迭代次数"


# ==================== 编排组件 ====================

class Pipeline:
    """顺序编排：Agent A 的输出作为 Agent B 的输入"""
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents

    def run(self, initial_input: str) -> str:
        current_input = initial_input
        for agent in self.agents:
            agent.reset()
            output = agent.run(current_input)
            current_input = f"基于以下内容继续工作：\n\n{output}"
            print(f"  [{agent.name}] 完成")
        return output


class Router:
    """路由编排：根据问题类型分发到不同 Agent"""
    def __init__(self, routes: dict[str, BaseAgent], default: BaseAgent | None = None):
        self.routes = routes
        self.default = default

    def run(self, user_input: str) -> str:
        route_names = ", ".join(self.routes.keys())
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"判断以下用户输入应该由哪个专家处理。\n可选专家：{route_names}\n用户输入：{user_input}\n\n只返回专家名称。"
            }],
            temperature=0
        )
        chosen = response.choices[0].message.content.strip()
        agent = self.routes.get(chosen, self.default)
        if agent is None:
            return f"无法路由：{chosen}"
        print(f"  [路由] → {chosen}")
        agent.reset()
        return agent.run(user_input)


# ==================== 迷你框架 ====================

class MiniAgentFramework:
    """迷你智能体框架：整合所有组件"""
    def __init__(self):
        self.registry = ToolRegistry()
        self.agents: dict[str, BaseAgent] = {}

    def tool(self, description: str, parameters: dict):
        return self.registry.register(description, parameters)

    def create_agent(self, name: str, system_prompt: str, middlewares: list | None = None, use_tools: bool = True) -> MiddlewareAgent:
        agent = MiddlewareAgent(
            name=name,
            system_prompt=system_prompt,
            tool_registry=self.registry if use_tools else ToolRegistry(),
            middlewares=middlewares or []
        )
        self.agents[name] = agent
        return agent

    def pipeline(self, agent_names: list[str]) -> Pipeline:
        return Pipeline([self.agents[n] for n in agent_names])

    def router(self, route_map: dict[str, str], default: str | None = None) -> Router:
        routes = {k: self.agents[v] for k, v in route_map.items()}
        return Router(routes, self.agents[default] if default else None)


# ==================== 运行演示 ====================

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY")
        exit(1)

    framework = MiniAgentFramework()

    @framework.tool(description="查询天气", parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    def weather(city: str) -> str:
        return {"北京": "晴 28°C", "上海": "多云 25°C", "深圳": "雷阵雨 31°C"}.get(city, f"{city}：暂无数据")

    @framework.tool(description="数学计算", parameters={"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]})
    def calc(expr: str) -> str:
        try:
            return str(eval(expr))
        except Exception as e:
            return f"错误：{e}"

    token_mw = TokenCountMiddleware()

    framework.create_agent(name="通用助手", system_prompt="你是一个全能助手，能查天气、做计算。回答简洁。", middlewares=[LoggingMiddleware(), token_mw])
    framework.create_agent(name="研究员", system_prompt="你是技术研究员，列出 3-5 个关键要点，简洁。", use_tools=False)
    framework.create_agent(name="写手", system_prompt="你是技术写手，把内容整理成 200 字以内的简短报告。", use_tools=False)

    print("=" * 60)
    print("Demo 1：单 Agent + 工具调用")
    print("=" * 60)
    agent = framework.agents["通用助手"]
    print(f"\n{agent.run('北京天气怎么样？算一下 256*512')}")

    print("\n" + "=" * 60)
    print("Demo 2：Pipeline 流水线编排")
    print("=" * 60)
    pipe = framework.pipeline(["研究员", "写手"])
    report = pipe.run("对比 FastAPI 和 Django 两个框架")
    print(f"\n最终报告：\n{report}")

    print("\n" + "=" * 60)
    print("Demo 3：Router 路由编排")
    print("=" * 60)
    framework.create_agent(name="代码专家", system_prompt="你是 Python 代码专家。回答简洁。", use_tools=False)
    framework.create_agent(name="运维专家", system_prompt="你是运维专家，擅长部署和监控。回答简洁。", use_tools=False)

    router = framework.router(
        route_map={"代码专家": "代码专家", "运维专家": "运维专家", "通用助手": "通用助手"},
        default="通用助手"
    )

    for q in ["帮我写一个冒泡排序", "Nginx 怎么配反向代理？"]:
        print(f"\n用户：{q}")
        answer = router.run(q)
        print(f"回答：{answer[:200]}...")

    print(f"\n\nToken 统计：{token_mw.report()}")
