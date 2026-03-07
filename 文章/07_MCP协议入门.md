# 给智能体插上 USB —— MCP 协议入门：让工具即插即用

---

**封面图提示词：**
> A clean tech illustration showing a friendly robot in the center with multiple USB-style ports on its body. Various tools are being plugged into these ports like USB devices: a calendar icon, a database cylinder, a search magnifying glass, a code terminal, a weather cloud icon. Each "USB plug" has a glowing connection. Cables connect from the tools to the robot with data particles flowing along them. Background: dark navy gradient with subtle grid lines. Accent colors: electric blue for connections, warm orange for the robot's eyes. No text, no watermark. Aspect ratio 2:1.

---

还记得咱们第一篇讲智能体的文章吗？那时候给 AI 加工具，得手写一堆 `function` 定义，再在代码里用 `if-else` 分发：加一个工具就改一次代码，改一次就多一堆分支。就像以前每个手机一个充电口，换设备就得换线。

现在有了 USB-C，一根线搞定。MCP（Model Context Protocol）干的事，就是给智能体的"工具"也搞一套 USB —— 标准协议、即插即用。

---

## 一、MCP 到底解决了什么问题？

### 传统方式的痛点

以前的做法大概是这样的：你写一个智能体，要接日历、数据库、搜索、天气……每个工具都得：

1. 自己定义 function schema
2. 自己写调用逻辑
3. 在分发逻辑里加一个 `elif`

工具一多，代码就变成一坨面条。换一个 LLM、换一个框架，又得重写一遍。这就是**强耦合**：工具和智能体绑死在一起。

### MCP 的思路

MCP 的思路很简单：**定一个标准协议**。工具按协议暴露能力，智能体按协议调用，中间不用互相知道对方怎么实现的。

就像 USB：设备按 USB 规范做接口，电脑按 USB 规范读设备，插上就能用。

### 三个核心角色

| 角色 | 类比 | 职责 |
|------|------|------|
| **Host** | 电脑主板 | 运行智能体，协调 Client 和 LLM |
| **Client** | USB 控制器 | 连接 Server，发现工具，转发调用 |
| **Server** | USB 设备 | 暴露工具（Tools）、资源（Resources）、提示词（Prompts） |

你写代码时，主要打交道的是 Client 和 Server。Host 通常是 Cursor、Claude Desktop 这类应用，它们已经内置了 MCP Client。

---

**配图1 提示词：** Host/Client/Server 架构图。中间是 Host（标注为"智能体运行环境"），左侧是 Client（标注为"MCP Client"），右侧是多个 Server（标注为"笔记 Server"、"文件 Server"、"搜索 Server"）。Client 与各 Server 之间用双向箭头连接，表示通信。简洁扁平风格，深色背景。

---

## 二、MCP 协议长什么样？

### 基于 JSON-RPC 2.0

MCP 底层用的是 JSON-RPC 2.0。请求和响应都是 JSON，有 `id`、`method`、`params`，和普通 RPC 一样。你不用手写这些，SDK 会帮你搞定。

### 通信方式

常见有两种：

- **stdio**：标准输入输出，适合本地进程，Client 启动 Server 子进程，通过管道通信
- **SSE**：Server-Sent Events，适合远程服务，走 HTTP

咱们这篇用 stdio，最简单，也最常用。

### 核心能力

| 能力 | 用途 |
|------|------|
| **Tools** | 可调用的函数，比如 `add_note`、`list_notes` |
| **Resources** | 只读数据，比如文件、API 结果，用 URI 标识 |
| **Prompts** | 预置的提示词模板，可以带参数 |

### 生命周期

Client 连上 Server 后，会发 `initialize`，再发 `initialized`，然后才能 `list_tools`、`call_tool`。SDK 会自动处理这些握手，你只要 `await session.initialize()` 就行。

---

## 三、从零写一个 MCP Server

用官方 Python SDK 的 FastMCP，几行代码就能跑起来。

先装依赖（需要 Python 3.10+）：

```bash
pip install mcp openai
```

写一个笔记管理 Server，提供 `add_note` 和 `list_notes` 两个工具：

```python
# note_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("笔记管理", json_response=True)

# 用内存存笔记，实际项目可以换成数据库
notes: list[dict] = []

@mcp.tool()
def add_note(title: str, content: str) -> str:
    """添加一条笔记。"""
    notes.append({"title": title, "content": content})
    return f"已添加笔记：{title}"

@mcp.tool()
def list_notes() -> str:
    """列出所有笔记。"""
    if not notes:
        return "暂无笔记"
    lines = [f"- {n['title']}: {n['content']}" for n in notes]
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

运行：

```bash
python note_server.py
```

**运行效果：** Server 启动后会挂起，等待 stdin 输入。这是正常的，因为 stdio 模式下它作为子进程被 Client 启动，不会单独在终端"跑起来"。咱们下一步写 Client 时，会由 Client 来启动它。

---

**配图2 提示词：** 三步构建工具箱示意图。第一步：定义工具（add_note、list_notes）；第二步：用 @mcp.tool() 装饰；第三步：mcp.run(transport="stdio")。每步用简洁图标+文字，箭头连接，科技感配色。

---

## 四、写一个 MCP Client 连上去

Client 要做几件事：启动 Server 进程、发现工具、把工具转成 OpenAI function calling 格式、在 LLM 需要时调用工具，再把结果塞回对话。

### 启动 Server 进程

用 `stdio_client` 和 `StdioServerParameters`，指定用 Python 跑咱们的 `note_server.py`：

```python
# mcp_client.py
import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_note_server():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["note_server.py"],
        env=None,
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session, stack
```

### 发现工具

```python
async def list_tools(session):
    resp = await session.list_tools()
    return resp.tools
```

### 转成 OpenAI function calling 格式

OpenAI 的 `tools` 参数要的是这种结构：

```python
def mcp_tools_to_openai(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": getattr(t, "inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]
```

### 调用工具并传回 LLM

当 LLM 返回 `tool_calls` 时，用 `session.call_tool` 执行，再把结果以 `tool` 角色追加到消息里：

```python
async def execute_tool(session, tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")
    result = await session.call_tool(name, args)
    text = result.content[0].text if result.content else ""
    return {"role": "tool", "tool_call_id": tool_call.id, "content": text}
```

---

**配图3 提示词：** User→Agent→MCPServer 交互时序图。从上到下：User 发消息 → Agent（LLM）决定调用工具 → Client 调用 MCP Server → Server 返回结果 → Client 把结果给 Agent → Agent 生成回复给 User。用泳道图或时序图形式，简洁清晰。

---

## 五、完整跑通：AI 自动记笔记

把 Server、Client、OpenAI 串起来，做一个能自动记笔记的小 demo：

```python
# demo.py
import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

async def main():
    # 1. 连接 MCP Server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["note_server.py"],
        env=None,
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # 2. 获取工具并转成 OpenAI 格式
        tools_resp = await session.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": getattr(t, "inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools_resp.tools
        ]

        client = OpenAI()
        messages = [{"role": "user", "content": "帮我记一条笔记：标题是「MCP 学习」，内容是「今天学了 MCP 协议，工具可以即插即用」"}]

        # 3. 第一轮：让 LLM 决定是否调工具
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = await session.call_tool(tc.function.name, args)
                text = result.content[0].text if result.content else ""
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text,
                })

            # 4. 第二轮：让 LLM 根据工具结果生成回复
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            print(resp2.choices[0].message.content)
        else:
            print(msg.content)

if __name__ == "__main__":
    asyncio.run(main())
```

运行前确保 `OPENAI_API_KEY` 已设置，然后：

```bash
python demo.py
```

**模拟运行效果：**

```
好的，我已经帮你添加了笔记「MCP 学习」，内容为「今天学了 MCP 协议，工具可以即插即用」。
```

再试一条「列出我的笔记」：

```python
messages = [{"role": "user", "content": "列出我的笔记"}]
# ... 同上流程
```

**模拟运行效果：**

```
你当前的笔记有：
- MCP 学习: 今天学了 MCP 协议，工具可以即插即用
```

---

**配图4 提示词：** 机器人连接环形工具生态图。中心是一个机器人头像，周围一圈是各种工具图标（日历、文件、数据库、搜索、天气、代码等），用发光的线连成环形。表示 MCP 让智能体可以灵活接入多种工具，形成生态。深色背景，蓝色/橙色点缀。

---

## 六、MCP 生态和实际应用

### 已有 Server 生态

GitHub 上搜 `mcp-server` 能搜到一堆现成的：文件系统、数据库、Git、Slack、Notion……很多都是社区贡献的，直接拿来用就行。

### 谁在支持 MCP

- **Cursor**：内置 MCP，配置一下就能用各种 Server
- **Claude Desktop**：同样支持，在配置里加 Server 路径即可
- **其他 IDE / 应用**：越来越多工具开始接 MCP

### 和其他方案对比

| 维度 | MCP | 手写 Function Calling | LangChain Tools |
|------|-----|------------------------|-----------------|
| 标准化 | 统一协议，跨应用复用 | 每个项目自己定义 | 框架内统一，跨框架不通用 |
| 接入成本 | 写 Server，按协议暴露 | 每个工具手写 schema + 分发 | 用框架封装，学习成本有 |
| 生态 | 官方 + 社区 Server 多 | 无 | LangChain 生态 |
| 适用场景 | 多应用共享工具、即插即用 | 单项目、简单需求 | 复杂 Agent 编排 |

说白了：MCP 的价值在于**标准化**。工具按标准写，就能被任何支持 MCP 的 Host 用，不用重复造轮子。

---

## 七、写在最后

MCP 最大的价值，就是「标准化」这三个字。协议一定，大家按同一套规则玩，工具就能像 USB 设备一样即插即用。

下一篇咱们会聊**多智能体协作**：多个 Agent 怎么分工、怎么通信、怎么一起干活。到时候你会看到，MCP 的 Server 不仅可以给人用的智能体提供工具，也可以给「智能体 A」当「智能体 B」的工具 —— 那才是真正把 USB 插满全身的玩法。

---

## 附录：完整代码文件

### note_server.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("笔记管理", json_response=True)
notes: list[dict] = []

@mcp.tool()
def add_note(title: str, content: str) -> str:
    """添加一条笔记。"""
    notes.append({"title": title, "content": content})
    return f"已添加笔记：{title}"

@mcp.tool()
def list_notes() -> str:
    """列出所有笔记。"""
    if not notes:
        return "暂无笔记"
    lines = [f"- {n['title']}: {n['content']}" for n in notes]
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### demo.py

```python
import asyncio
import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["note_server.py"],
        env=None,
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_resp = await session.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": getattr(t, "inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools_resp.tools
        ]

        client = OpenAI()
        messages = [{"role": "user", "content": "帮我记一条笔记：标题是「MCP 学习」，内容是「今天学了 MCP 协议，工具可以即插即用」"}]

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = await session.call_tool(tc.function.name, args)
                text = result.content[0].text if result.content else ""
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text,
                })
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            print(resp2.choices[0].message.content)
        else:
            print(msg.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 运行方式

```bash
# 需要 Python 3.10+
pip install mcp openai
export OPENAI_API_KEY=your_key
# 确保 note_server.py 与 demo.py 在同一目录
python demo.py
```
