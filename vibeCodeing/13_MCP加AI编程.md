# 让 AI 编程工具不只会写代码——还能查数据库、看设计稿、调接口

你在 Cursor 里写代码，突然想查一下数据库里某张表的结构，或者想看看 GitHub 上某个 issue 的讨论。以前你得：切到终端、敲命令，或者打开浏览器、登录、点来点去。等查完回来，思路早断了。

有了 MCP，这事儿就变了。你直接在 Cursor 里跟 AI 说一句"帮我查一下 users 表有哪些字段"，它自己调数据库、把结果给你。查 GitHub、调 API、读 Figma 设计稿，全在一个窗口里搞定。**MCP 就像给 Cursor 装了一排 USB 接口**——以前它只能改你本地的代码，现在能插上数据库、API、设计工具，真正变成你的"全能编程助手"。

今天我们就来聊聊：MCP 到底是什么、在 Cursor 里怎么配、怎么用，以及怎么自己写一个接公司内部接口的 MCP 服务。

---

一、MCP 快速回顾（一分钟版）

**MCP** 全称 Model Context Protocol，是 Anthropic 牵头搞的一个开放协议。你可以把它理解成"AI 和外部工具之间的普通话"——以前每个 AI 工具、每个外部服务都有自己的接口，互相不认；有了 MCP，大家按同一套规则说话，就能互通了。

协议里有三个核心角色：

**Client（客户端）**：发起请求的一方。在咱们的场景里，就是 Cursor 里的 AI Agent。

**Server（服务端）**：暴露能力的一方。比如 PostgreSQL MCP Server 暴露"查表结构、执行 SQL"的能力，GitHub MCP Server 暴露"查 issue、创建 PR"的能力。

**Tool（工具）**：Server 提供的具体能力。一个 Server 可以有多个 Tool，比如数据库 Server 可能有 `query`、`get_schema`、`execute_sql` 等。

在 AI 编程场景里，这意味着：**Cursor 的 Agent 不再只能读你项目里的代码**，它还能通过 MCP 调用外部工具——查数据库、调 API、读设计稿、操作 GitHub。你不需要切窗口，不需要复制粘贴，一句话就能让 AI 帮你完成这些操作。

---

二、Cursor 里怎么配置 MCP

Cursor 支持两种配置方式：图形界面和配置文件。配置文件更灵活，也方便团队共享，推荐用这个。

**配置文件位置**：

- 全局配置：`~/.cursor/mcp.json`，对所有项目生效
- 项目级配置：项目根目录下的 `.cursor/mcp.json`，只对当前项目生效，可以提交到 Git 让团队共用

两个文件会合并，项目级的会覆盖全局的同名配置。

**配置格式**：

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost:5432/mydb"]
    }
  }
}
```

每个 Server 需要指定 `command` 和 `args`，告诉 Cursor 怎么启动这个 Server。如果是需要环境变量的，比如 API Key，不要写在配置文件里，用**包装脚本**或系统环境变量传进去。

**启用和调试**：

配置好后，在 Cursor 设置里找到 **Features → MCP**，能看到你配置的 Server 列表。绿点表示连接成功，红点表示有问题。注意：**MCP 工具只在 Composer（Agent 模式，按 Cmd+I 打开）里生效**，普通 Chat 用不了。所以测试时一定要进 Composer。

如果连不上，先看 Cursor 的开发者控制台（Help → Toggle Developer Tools），看有没有报错。常见问题：命令路径不对、依赖没装、环境变量没设。

---

## 三、实战 1：连接 PostgreSQL 数据库

**为什么你需要这个？** 开发时经常要查表结构、跑条 SQL 验证数据，以前得切到终端、敲命令。有了数据库 MCP，在 Cursor 里一句话就能搞定，不用打断思路。

PostgreSQL 是一种常用的数据库。有了 PostgreSQL MCP Server，查表、查数据这些都能在 Cursor 里完成。

**配置数据库 MCP Server**：

官方有个 `@modelcontextprotocol/server-postgres`，用 npx 就能跑。在 `mcp.json` 里加一段：

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user:password@localhost:5432/your_db"]
    }
  }
}
```

密码别写死在配置文件里，用环境变量。可以写个 shell 脚本包装一下，比如 `run-mcp-postgres.sh`，里面 `export` 好连接串，再 `exec npx ...`。

**在 Cursor 里直接让 AI 查数据、改表结构**：

进 Composer，直接说："帮我查一下 users 表的结构"、"执行某条查询的前 5 条"、"看看有没有 orders 和 users 的关联关系"。

AI 会调用 MCP 的数据库工具，把结果返回给你。你甚至可以说"根据这个表结构帮我写一个 TypeScript 的接口定义"，它能看到真实表结构，生成的类型更准。

**实际效果**：以前查个表要切终端、敲命令、看结果、再切回来；现在一句话搞定，思路不中断。写代码时突然想验证一下数据长什么样，也不用离开编辑器。

---

## 四、实战 2：连接 GitHub

**为什么你需要这个？** 写代码时经常要查 issue、看 PR、搜 commit，以前得开浏览器、登录、点来点去。接上 GitHub MCP 之后，这些都能在 Cursor 里一句话搞定。

GitHub 官方提供了 MCP Server，接上之后，查 issue、创建 PR、看 commit 历史，全在 Cursor 里完成。

**配置 GitHub MCP Server**：

先去 GitHub 生成一个 **Personal Access Token**（个人访问令牌，相当于用密码登录的凭证）：Settings → Developer settings → Personal access tokens，勾上 repo、read:org 等需要的权限。然后在 `mcp.json` 里配置：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "你的 token"
      }
    }
  }
}
```

同样，token 别提交到 Git，用环境变量或本地覆盖。

**让 AI 帮你查 issue、创建 PR、看 commit 历史**：

进 Composer，可以说："查一下 xxx 仓库的 issue #123 的讨论"、"帮我创建一个 PR，把 feature/login 分支合并到 main，标题是'登录功能'"、"看看这个仓库最近一周的 commit 都有哪些"。

AI 会调用 GitHub 的 MCP 工具，把结果直接给你。你甚至可以说"根据 issue #456 的描述，帮我写一个修复方案"，它能把 issue 内容拉过来当上下文。

---

## 五、实战 3：连接 Figma

**为什么你需要这个？** 设计和开发之间最大的鸿沟，就是设计稿和代码之间的转换。以前你得照着设计稿一点点手写样式。Figma 是常用的设计工具，接上 Figma MCP 之后，**设计稿可以直接变代码**，省掉大量重复劳动。

**设计稿直接变代码**：

在 Composer 里，你可以 @ 一个 Figma 的链接，或者说"把这个 Figma 设计实现成 React 组件"。AI 会通过 MCP 读取 Figma 的设计数据——布局、颜色、字体、间距——然后生成对应的代码。虽然不是 100% 像素级还原，但能省掉大量"照着设计稿手写"的重复劳动。

**配置方式**：

Figma MCP 需要 Figma 的 access token。在 Figma 设置里生成 token，然后配置到 `mcp.json`。具体参数可以查 Figma MCP 的官方文档，配置逻辑和 GitHub 类似：command + args + env。

---

## 六、实战 4：自己写一个 MCP Server

现成的 MCP Server 够用很多场景，但如果你要接**公司内部接口**——比如用户中心、订单系统、权限服务——就得自己写一个。

**为什么你需要这个？** 公司内部往往有一套自己的接口（API），查用户、查订单、查权限，写代码时经常要调。以前你得查文档、复制接口地址、自己写调用代码。自己写一个 MCP Server，把内部接口"插"进 Cursor，AI 就能直接帮你调，不用你再查文档。

**场景**：比如你们有个内部接口"根据用户 ID 查用户详情"，你想在写代码时让 AI 直接调这个接口，而不是自己查文档、复制粘贴。

**用 Python 写一个最简 MCP Server**：

Python 有个 `fastmcp` 库，写起来很快。下面是一个最小示例：

```python
from fastmcp import FastMCP

mcp = FastMCP("internal-api")

@mcp.tool()
def get_user_detail(user_id: str) -> str:
    """根据用户ID查询用户详情，调用公司内部用户中心API"""
    import requests
    resp = requests.get(f"https://api.yourcompany.com/users/{user_id}")
    return resp.json()

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

保存为 `mcp_internal_server.py`，先 `pip install fastmcp requests` 装依赖，然后用 `python mcp_internal_server.py` 跑起来。关键是：**MCP Server 通过标准输入输出和 Cursor 通信**，所以 Cursor 会启动它、通过命令行传数据。

**在 Cursor 里使用**：

在 `mcp.json` 里加一段：

```json
{
  "mcpServers": {
    "internal-api": {
      "command": "python",
      "args": ["/path/to/mcp_internal_server.py"]
    }
  }
}
```

或者如果你用 `fastmcp run`，就改成对应的 command。配置好后，进 Composer，说"帮我查一下用户 12345 的详情"，AI 会调用你的 `get_user_detail` 工具，把结果返回给你。

你可以把多个内部 API 封装成多个 `@mcp.tool`，一个 Server 暴露一堆能力。这样 AI 在写代码时，就能直接"知道"你们内部接口长什么样，不用你再去查文档了。

---

## 七、MCP 在 AI 编程中的最佳实践

**哪些工具值得接入**：

优先接你**高频使用**的：数据库（开发、调试必备）、GitHub（查 issue、PR、commit）、Figma（有设计稿的项目）。其次接**团队特有**的：内部接口、内部文档、内部工单系统。别接一堆"可能用得上"的，接太多会**拖慢性能**——每个 MCP 都会占资源，AI 理解这些工具也要花更多算力。

**安全注意事项**：

不要在任何配置文件里写死 API Key、密码、token。用环境变量，或者用包装脚本。如果 MCP Server 要访问生产数据库，**强烈建议只读权限**，或者用单独的只读账号。AI 执行 MCP 工具前，Cursor 默认会问你确认，别为了省事全关掉——尤其是涉及写操作的。

**性能考虑**：

MCP 工具调用会走网络、会延迟。如果接了很多 Server，AI 在"选择用哪个工具"时也会花更多 token。所以：**按需接入，按项目隔离**。这个项目用 PostgreSQL，就只在项目级 `mcp.json` 里配；全局配置里只放真正通用的，比如 GitHub。

---

## 八、MCP 的未来想象空间

MCP 才刚起步，但方向已经很明显：**AI 编程工具会从"只会改代码"变成"能操作你整个工作流"**。

以后可能会有：接 Jira/Linear 的 MCP，让 AI 根据 ticket 描述直接开写；接监控系统的 MCP，让 AI 根据告警自动分析、定位、甚至写修复方案；接文档系统的 MCP，让 AI 在写代码时自动参考你们最新的 wiki。每个团队都能把自己的工具链暴露给 AI，AI 就真正变成你的"数字同事"——不仅会写代码，还会查数据、调接口、看设计、跟你的协作工具对话。

---

## 九、写在最后

MCP 不是什么高深概念，本质就是**给 AI 装了一排 USB 口**。以前 Cursor 只能看你项目里的代码；现在它能接数据库、接 GitHub、接 Figma、接你公司自己的接口。你不需要切窗口、不需要复制粘贴，一句话就能让 AI 帮你完成这些操作。

从配置现成的 PostgreSQL、GitHub、Figma，到用 Python 写一个接内部接口的 MCP Server，门槛都不高。花半小时配好，以后写代码时查数据、查 issue、看设计稿，都能在一个窗口里搞定。这才是 AI 编程工具该有的样子——不是替代你写代码，而是把你从重复的、琐碎的跨工具操作里解放出来，让你把精力放在真正需要思考的地方。

觉得有用的话，点赞、转发、评论区聊聊你接了什么 MCP。下一篇咱们聊：**AI 写的代码能上生产吗**——从"能跑"到"敢用"，还差几步。
