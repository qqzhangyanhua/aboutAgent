# 智能体工作流编排 —— LangGraph 实战：让 AI 学会走迷宫

## 封面图提示词（2:1 比例，1232×616px）

> A tech illustration showing a robot navigating through a flowchart/graph structure that looks like a maze. The graph has colored nodes (circles) connected by directional arrows, with some paths branching into conditions (diamond shapes) and some paths looping back. The robot stands at a "START" node and looks toward an "END" node, with a glowing trail showing the path it's choosing. Some nodes glow green (completed), some yellow (current), some gray (not yet reached). Background: dark blueprint/grid style. No text, no watermark. Aspect ratio 2:1.

---

前面 Planning 那篇，咱们搞了 Plan-and-Execute 模式——先列计划再执行。但那个实现有个问题：流程是线性的。如果执行到一半需要回到某个步骤重试呢？如果有些步骤需要根据结果走不同分支呢？这就需要「图」来编排工作流了。LangGraph 就是干这个的——LangChain 团队出的，把智能体流程建模成一张「有向图」，节点是操作，边是条件跳转。

---

## 一、LangGraph 的核心概念

用生活类比来理解这几个概念，比直接啃文档轻松多了。

**State（状态）**：一个字典，存着当前所有信息，流程里每一步都能读写它。类比：一张白纸传来传去，每个人在上面写点东西。

**Node（节点）**：一个 Python 函数，接收 State，做点事，返回更新后的 State。类比：流水线上的工位。

**Edge（边）**：节点之间的连线，决定下一步走哪。类比：流水线上的传送带。

**Conditional Edge（条件边）**：根据 State 的内容决定走哪条路。类比：分岔路口的交通灯。

和普通函数调用的区别在哪？普通调用是 A→B→C 一条线走到底；LangGraph 的图可以有状态、有分支、能循环、还能导出成图可视化。说白了，你是在画流程图，而不是写一堆 if-else。

> **正文配图 1 提示词：** A clear diagram showing LangGraph components. Center: a graph with 4 nodes (colored circles) connected by arrows. One node has a "START" flag, one has "END". Between two nodes, a diamond-shaped decision point with two outgoing arrows (Yes/No paths). Next to the graph: a clipboard labeled "State" showing key-value pairs being updated. Clean technical diagram, white background, colorful nodes. No text labels, just shapes and icons. Aspect ratio 16:9.

---

## 二、V1：最简单的 LangGraph —— 顺序执行

先来一个三步流程：收集信息 → 分析 → 生成报告。没有分支，没有循环，纯粹练手。

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. 定义 State
class ReportState(TypedDict):
    topic: str
    raw_info: str
    analysis: str
    report: str

# 2. 定义三个节点
def collect_info(state: ReportState) -> dict:
    """模拟收集信息"""
    topic = state["topic"]
    return {"raw_info": f"[模拟检索] 关于{topic}的若干资料..."}

def analyze(state: ReportState) -> dict:
    """模拟分析"""
    return {"analysis": f"对 {state['raw_info'][:50]}... 的分析结论"}

def generate_report(state: ReportState) -> dict:
    """生成报告"""
    return {"report": f"【报告】主题：{state['topic']}\n分析：{state['analysis']}\n--- 报告完毕"}

# 3. 建图
builder = StateGraph(ReportState)
builder.add_node("collect", collect_info)
builder.add_node("analyze", analyze)
builder.add_node("report", generate_report)

builder.add_edge(START, "collect")
builder.add_edge("collect", "analyze")
builder.add_edge("analyze", "report")
builder.add_edge("report", END)

graph = builder.compile()

# 4. 跑一下
result = graph.invoke({"topic": "Python 异步编程"})
print(result["report"])
```

**模拟运行效果：**

```
【报告】主题：Python 异步编程
分析：对 [模拟检索] 关于Python 异步编程的若干资料...... 的分析结论
--- 报告完毕
```

流程就是 collect → analyze → report，一条线走完。这就是 LangGraph 最基础的用法。

---

## 三、V2：加上条件分支 —— 让 AI 决定走哪条路

场景：用户提问 → 判断是简单问题还是复杂问题 → 简单问题直接回答，复杂问题走 RAG 检索后回答。

关键在 `add_conditional_edges`：路由函数根据 State 返回下一个节点的名字。

```python
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class QAState(TypedDict):
    question: str
    complexity: str  # "simple" | "complex"
    direct_answer: str
    retrieved_context: str
    final_answer: str

def route_question(state: QAState) -> Literal["direct_answer", "rag_answer"]:
    """根据问题复杂度决定走哪条路"""
    q = state.get("question", "")
    # 简单规则：超过 20 字或包含「对比」「分析」等词算复杂
    if len(q) > 20 or any(w in q for w in ["对比", "分析", "详细"]):
        return "rag_answer"
    return "direct_answer"

def classify_and_route(state: QAState) -> dict:
    """分类问题并写入 state"""
    q = state["question"]
    complexity = "complex" if len(q) > 20 or any(w in q for w in ["对比", "分析", "详细"]) else "simple"
    return {"complexity": complexity}

def direct_answer(state: QAState) -> dict:
    """简单问题直接答"""
    resp = llm.invoke(f"简短回答：{state['question']}")
    return {"direct_answer": resp.content, "final_answer": resp.content}

def rag_answer(state: QAState) -> dict:
    """复杂问题：模拟 RAG 检索后回答"""
    # 模拟检索
    context = f"[模拟检索到 3 条相关文档] 关于 {state['question']} 的详细资料..."
    resp = llm.invoke(f"基于以下资料回答：\n{context}\n\n问题：{state['question']}")
    return {"retrieved_context": context, "final_answer": resp.content}

builder = StateGraph(QAState)
builder.add_node("classify", classify_and_route)
builder.add_node("direct_answer", direct_answer)
builder.add_node("rag_answer", rag_answer)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_question)
builder.add_edge("direct_answer", END)
builder.add_edge("rag_answer", END)

graph = builder.compile()

# 简单问题
r1 = graph.invoke({"question": "Python 是什么？"})
print("简单:", r1["final_answer"][:80] + "...")

# 复杂问题
r2 = graph.invoke({"question": "请对比分析 Django、Flask、FastAPI 三个框架的优缺点"})
print("复杂:", r2.get("retrieved_context", "")[:60] + "...")
```

**模拟运行效果：**

```
简单: Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建...

复杂: [模拟检索到 3 条相关文档] 关于 请对比分析 Django、Flask、FastAPI...
```

路由函数 `route_question` 的返回值直接对应下一个节点名，这就是条件边的用法。

> **正文配图 2 提示词：** A road fork illustration. A robot arrives at a junction where the road splits into two paths. Left path (green, simple): short and straight, leading to a quick answer bubble. Right path (blue, complex): longer with a library/database stop along the way, leading to a detailed answer bubble. A signpost at the junction shows a question mark deciding which way. Clean illustrated style. No text. Aspect ratio 16:9.

---

## 四、V3：加上循环 —— 让 AI 自我纠错

场景：生成回答 → 质检 → 不合格？回到「重新生成」→ 再质检 → 合格则输出。最多循环 3 次，防止无限循环。

关键点：State 里加一个 `retry_count` 字段，条件边根据「是否合格」和「是否超次数」决定继续循环还是结束。

```python
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GenState(TypedDict):
    prompt: str
    draft: str
    retry_count: int
    passed: bool

def generate(state: GenState) -> dict:
    """生成初稿"""
    cnt = state.get("retry_count", 0)
    hint = f"（第 {cnt + 1} 次尝试）" if cnt > 0 else ""
    resp = llm.invoke(f"{state['prompt']}{hint}\n请生成一段 100 字以内的回答。")
    return {"draft": resp.content, "retry_count": cnt + 1}

def quality_check(state: GenState) -> dict:
    """质检：检查是否包含敏感词、是否过短等"""
    draft = state["draft"]
    # 简单规则：少于 30 字或包含「不知道」算不合格
    passed = len(draft) >= 30 and "不知道" not in draft
    return {"passed": passed}

def route_after_check(state: GenState) -> Literal["generate", "end"]:
    """质检后：合格则结束，不合格且未超 3 次则重试"""
    if state.get("passed", False):
        return "end"
    if state.get("retry_count", 0) >= 3:
        return "end"  # 超次数也结束，避免死循环
    return "generate"

builder = StateGraph(GenState)
builder.add_node("generate", generate)
builder.add_node("quality_check", quality_check)

builder.add_edge(START, "generate")
builder.add_edge("generate", "quality_check")
builder.add_conditional_edges("quality_check", route_after_check, {"generate": "generate", "end": END})

graph = builder.compile()

result = graph.invoke({"prompt": "介绍一下 LangGraph"})
print("最终输出:", result["draft"])
print("重试次数:", result["retry_count"])
```

**模拟运行效果：**

```
最终输出: LangGraph 是 LangChain 团队推出的工作流编排框架，将智能体流程建模为有向图...
重试次数: 1
```

如果第一次生成的回答太短，`route_after_check` 会返回 `"generate"`，图就会从 `quality_check` 再连回 `generate`，形成循环。

> **正文配图 3 提示词：** A quality control loop illustration. A conveyor belt carries a document through: Step 1 (robot writing) → Step 2 (inspector robot checking with magnifying glass) → if rejected (red X), document loops back to Step 1 via a return belt. If approved (green check), document exits to a "Done" bin. A counter shows "Attempt: 2/3". Clean factory/industrial style. No text. Aspect ratio 16:9.

---

## 五、V4：完整实战 —— 带工具调用的智能体工作流

完整流程：接收用户消息 → 判断是否需要工具 → 需要则调用工具 → 把工具结果喂回 LLM → 判断是否还需要调工具 → 不需要则输出最终答案。这其实就是 ReAct 模式的图化实现。

LangGraph 提供了 `create_react_agent`，但咱们手写一版，把图的结构看清楚。

```python
import os
from typing import TypedDict, Annotated, Literal, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

os.environ["OPENAI_API_KEY"] = "你的API Key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 工具定义
def get_weather(city: str) -> str:
    """查询指定城市的天气。city: 城市名"""
    return f"{city}：晴，25°C"

def search_web(query: str) -> str:
    """搜索网络。query: 搜索关键词"""
    return f"[模拟搜索结果] 关于「{query}」的相关内容..."

tools = [get_weather, search_web]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent_node(state: AgentState) -> dict:
    """LLM 推理：决定是调用工具还是直接回答"""
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}

def tools_node(state: AgentState) -> dict:
    """执行工具调用"""
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    tool_msgs = []
    for tc in tool_calls:
        name, args = tc["name"], tc["args"]
        fn = next((t for t in tools if t.__name__ == name), None)
        result = fn(**args) if fn else "未知工具"
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": tool_msgs}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """判断下一步：有 tool_calls 则调工具，否则结束"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")  # 工具结果喂回 agent，形成循环

graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="北京今天天气怎么样？")]})
print(result["messages"][-1].content)
```

**模拟运行效果：**

```
北京：晴，25°C
```

流程是：agent → 有 tool_calls → tools → 结果追加到 messages → 回到 agent → 无 tool_calls → END。这就是 ReAct 的图化版。

---

## 六、LangGraph 的杀手锏功能

除了图结构，还有几个值得了解的能力。

**Human-in-the-Loop**：在某个节点暂停，等人类确认后再继续。用 `interrupt()` 即可：

```python
from langgraph.types import interrupt

def human_review_node(state):
    answer = interrupt("请确认是否继续？")
    return {"approved": answer}
```

恢复时用 `Command(resume="yes")` 传入 `invoke`。

**Checkpointing**：保存中间状态，断点续跑。编译时传入 checkpointer：

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())
# 需要 config 里带 thread_id
result = graph.invoke(input, config={"configurable": {"thread_id": "user-123"}})
```

**Streaming**：逐步输出，不用等全部跑完：

```python
for chunk in graph.stream({"topic": "AI"}, stream_mode="updates"):
    print(chunk)  # 每个节点执行完就 yield 一次
```

> **正文配图 4 提示词：** A comparison showing three approaches side by side. Left: a simple straight arrow from A to B (bare Python). Middle: a chain of linked blocks in a line (LangChain). Right: a complex but organized graph/flowchart with nodes, branches, and loops (LangGraph). Below each: a small complexity meter showing increasing sophistication. Clean flat design, three columns. No text. Aspect ratio 16:9.

---

## 七、LangGraph vs 其他方案对比

| 维度 | 裸写 Python | LangChain Chain | LangGraph |
|------|------------|----------------|-----------|
| 流程控制 | if-else 手写 | 线性链 | 图+条件+循环 |
| 状态管理 | 手动维护变量 | 有限 | 内置 State |
| 可视化 | 无 | 无 | 支持导出图 |
| 断点续跑 | 自己实现 | 不支持 | 内置 Checkpoint |
| 学习成本 | 最低 | 中等 | 较高 |
| 适合场景 | 简单流程 | 线性流程 | 复杂有分支的流程 |

---

## 八、写在最后

LangGraph 本质上是在给智能体画流程图。如果你的智能体流程是一条直线（A→B→C），用不着 LangGraph，裸写就行。但如果你的流程有分支、有循环、有重试、需要人工确认——那 LangGraph 能帮你省掉大量「流程胶水代码」。

下一篇咱们回到 RAG 本身，聊一个被严重低估的话题：分块策略。
