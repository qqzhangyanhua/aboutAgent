"""
MCP 协议入门 Demo：AI 自动记笔记
运行前：pip install mcp openai，设置 OPENAI_API_KEY
"""
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
