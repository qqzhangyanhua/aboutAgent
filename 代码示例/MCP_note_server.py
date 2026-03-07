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
