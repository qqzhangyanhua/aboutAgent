# 给智能体插上 USB —— MCP 协议入门：让工具即插即用

## 封面图提示词（2:1 比例，1232×616px）

> A clean tech illustration showing a friendly robot in the center with multiple USB-style ports on its body. Various tools are being plugged into these ports like USB devices: a calendar icon, a database cylinder, a search magnifying glass, a code terminal, a weather cloud icon. Each "USB plug" has a glowing connection. Cables connect from the tools to the robot with data particles flowing along them. Background: dark navy gradient with subtle grid lines. Accent colors: electric blue for connections, warm orange for the robot's eyes. No text, no watermark. Aspect ratio 2:1.

## 一句话定位

搞清楚 MCP 是什么、为什么需要它，然后动手从零写一个 MCP Server 和 Client，让你的智能体即插即用地获得新能力。

---

## 章节大纲

### 开头引入

- 切入点：第一篇智能体文章里，我们给 AI 加工具的方式是手写 function，然后在代码里 if-else 分发。加一个工具就改一次代码。如果你的智能体需要接入 20 个工具呢？
- 类比：以前每个手机都有自己的充电线，后来统一了 USB-C。MCP 就是 AI 工具接入的"USB-C 协议"
- 核心问题：怎么让"写工具的人"和"写智能体的人"不用互相看代码就能对接？

### 一、MCP 到底解决了什么问题？

**要点：**
- 传统方式的痛点：每加一个工具，智能体代码要改一次。工具和智能体强耦合
- MCP 的思路：定义一套标准协议，工具侧实现 Server，智能体侧实现 Client，通过协议通信
- 三个核心角色：
  - **MCP Host**（宿主）：你的 AI 应用，比如 Claude Desktop、你自己的智能体
  - **MCP Client**（客户端）：宿主里负责连接 Server 的模块
  - **MCP Server**（服务端）：工具提供方，可以是本地脚本，也可以是远程服务
- 类比 USB 设备：Server 就是 U 盘，Client 就是 USB 接口，Host 就是电脑

> **正文配图 1 提示词：**
> A clear architecture diagram showing three components connected by arrows. Left: a large computer/app icon labeled as "Host" containing a smaller plug/socket icon labeled as "Client". Right side: three separate tool boxes/devices (database, search engine, file system) each with a USB-style connector, labeled as "Servers". Bidirectional arrows with data packets connect the Client to each Server. Clean technical diagram style, white background, blue for Host/Client, green for Servers. No text labels, icons only. Aspect ratio 16:9.

### 二、MCP 协议长什么样？

**要点：**
- 基于 JSON-RPC 2.0 的消息格式
- 通信方式：stdio（本地进程）或 SSE/HTTP（远程服务）
- 核心能力：Tools（工具）、Resources（资源）、Prompts（提示词模板）
- 生命周期：初始化握手 → 能力发现 → 调用执行 → 返回结果

**核心代码片段（消息格式）：**

```python
# MCP 工具声明（Server 告诉 Client "我有什么能力"）
tool_declaration = {
    "name": "get_weather",
    "description": "查询指定城市的天气",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
}

# 调用请求（Client → Server）
call_request = {
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "get_weather",
        "arguments": {"city": "上海"}
    },
    "id": 1
}

# 返回结果（Server → Client）
call_response = {
    "jsonrpc": "2.0",
    "result": {
        "content": [{"type": "text", "text": "上海今天 23°C，多云"}]
    },
    "id": 1
}
```

### 三、从零写一个 MCP Server

**要点：**
- 用官方 Python SDK `mcp` 来写
- 实现一个最简单的"笔记管理"工具：add_note、list_notes、search_notes
- 用 stdio 方式通信（最简单，先跑通再说）

**核心代码片段：**

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("note-server")
notes = {}

@app.list_tools()
async def list_tools():
    return [
        Tool(name="add_note",
             description="添加一条笔记",
             inputSchema={
                 "type": "object",
                 "properties": {
                     "title": {"type": "string"},
                     "content": {"type": "string"}
                 },
                 "required": ["title", "content"]}),
        Tool(name="list_notes",
             description="列出所有笔记标题",
             inputSchema={"type": "object", "properties": {}}),
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "add_note":
        notes[arguments["title"]] = arguments["content"]
        return [TextContent(type="text", text=f"已保存笔记：{arguments['title']}")]
    elif name == "list_notes":
        titles = list(notes.keys()) or ["暂无笔记"]
        return [TextContent(type="text", text="\n".join(titles))]

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write)
```

> **正文配图 2 提示词：**
> A step-by-step build illustration. Three sequential panels connected by arrows. Panel 1: An empty toolbox being opened. Panel 2: Tools being placed inside - a note icon, a search icon, a list icon being added one by one. Panel 3: The completed toolbox now has a glowing USB connector, ready to be plugged in. Each panel has a numbered step indicator. Clean flat design, light background, green accent color for the tools. No text. Aspect ratio 16:9.

### 四、写一个 MCP Client 连上去

**要点：**
- Client 负责：启动 Server 进程、发现工具、调用工具、把结果传给 LLM
- 把 MCP 工具列表转成 OpenAI function calling 格式
- 结合智能体的 ReAct 循环使用

**核心代码片段：**

```python
from mcp.client.stdio import stdio_client, StdioServerParameters

class MCPAgent:
    def __init__(self):
        self.tools = []
        self.session = None

    async def connect(self, server_script):
        """连接 MCP Server，发现可用工具"""
        params = StdioServerParameters(
            command="python", args=[server_script])
        transport = await stdio_client(params).__aenter__()
        self.session = await ClientSession(*transport).__aenter__()
        await self.session.initialize()

        # 获取工具列表，转成 OpenAI function 格式
        tools_response = await self.session.list_tools()
        self.tools = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            }
        } for t in tools_response.tools]

    async def call_tool(self, name, args):
        """调用 MCP Server 上的工具"""
        result = await self.session.call_tool(name, args)
        return result.content[0].text

    async def chat(self, user_message):
        """结合 LLM 的对话主循环"""
        messages = [{"role": "user", "content": user_message}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=self.tools  # MCP 工具透传给 LLM
        )
        # 如果 LLM 决定调用工具 → 调 MCP Server → 把结果喂回去
        # ... (ReAct 循环逻辑)
```

### 五、完整跑通：AI 自动记笔记

**要点：**
- 演示完整流程：用户说"帮我记一下明天下午3点开周会" → LLM 决定调 add_note → MCP Client 发给 Server → 保存成功 → 返回确认
- 再问"我有哪些笔记" → LLM 调 list_notes → 返回列表
- 关键点：从头到尾 LLM 自己决定要不要调工具、调哪个

**模拟运行效果：**
```
用户：帮我记一下明天下午3点开周会
AI（思考）：用户想保存一条笔记 → 调用 add_note
→ MCP 调用：add_note(title="周会", content="明天下午3点开周会")
→ Server 返回：已保存笔记：周会
AI：好的，已经帮你记下了"周会"——明天下午 3 点。

用户：我目前有哪些笔记？
AI（思考）：用户想查看笔记列表 → 调用 list_notes
→ MCP 调用：list_notes()
→ Server 返回：周会
AI：你目前有 1 条笔记：周会（明天下午3点）
```

> **正文配图 3 提示词：**
> A sequence diagram showing interaction flow between three entities arranged left to right: User (person icon), AI Agent (robot icon), and MCP Server (toolbox icon). Arrows show: 1) User sends message to Agent, 2) Agent thinks and decides to call tool, 3) Agent sends request to Server, 4) Server processes and returns result, 5) Agent formulates response, 6) Agent replies to User. Clean UML-style sequence diagram, with colored lifelines. No text labels, just icons and arrows with numbers. Aspect ratio 16:9.

### 六、MCP 生态和实际应用

**要点：**
- 已有的 MCP Server 生态：文件系统、GitHub、数据库、Slack、Google Drive、Brave Search 等
- Cursor、Claude Desktop、Windsurf 等 IDE/客户端已经原生支持 MCP
- 什么时候自己写 MCP Server：公司内部工具、私有 API、定制化场景
- MCP vs OpenAI Function Calling vs LangChain Tools 的对比

```
| 维度         | MCP                    | Function Calling       | LangChain Tools        |
|-------------|------------------------|------------------------|------------------------|
| 标准化       | 开放协议标准            | OpenAI 私有格式         | 框架内部格式            |
| 复用性       | 跨应用复用              | 绑定 OpenAI API         | 绑定 LangChain          |
| 工具发现     | 动态发现                | 静态声明               | 静态注册                |
| 生态         | 快速增长中              | OpenAI 生态             | LangChain 社区          |
| 适合场景     | 标准化工具分发           | 快速原型               | 复杂 Chain 编排          |
```

### 写在最后

- 衔接语：有了 MCP，给智能体加工具变得标准化了。但一个智能体终究能力有限——如果让多个智能体协作起来呢？下一篇我们来聊聊多智能体协作
- 关键观点：MCP 最大的价值不是技术多先进，而是"标准化"这件事本身。就像 HTTP 让所有浏览器都能访问所有网站一样，MCP 让所有 AI 应用都能接入同一套工具

> **正文配图 4 提示词：**
> A panoramic illustration showing an AI robot in the center connected to a ring of diverse tool icons arranged in a circle around it: calendar, email, database, search engine, file folder, code terminal, weather, calculator, messaging app. All connections use standardized USB-style plugs (same connector shape). One tool is being "hot-swapped" - shown being unplugged and a new one being plugged in, with a green "ready" indicator. Dark background with glowing connections. No text. Aspect ratio 16:9.
