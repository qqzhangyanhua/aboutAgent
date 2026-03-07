# 让 AI 先想再干 —— Planning 机制：从 ReAct 到 Plan-and-Execute

## 封面图提示词（2:1 比例）

> A split-screen tech illustration. Left side (slightly chaotic): a robot running in circles, juggling multiple tools, with tangled thought bubbles - representing ReAct's step-by-step reactivity. Right side (organized): a robot sitting at a desk with a clear whiteboard showing a numbered plan/checklist, then calmly executing each step with checkmarks appearing - representing Plan-and-Execute's structured approach. A dividing line in the center with a "VS" or evolution arrow from left to right. Background: left side has warm/orange tones (urgency), right side has cool/blue tones (calm). No text, no watermark. Aspect ratio 2:1.

---

上篇文章咱们用 ReAct 模式造了个会"先想后做"的智能体——AI 想一步、做一步、观察一步，循环往复直到任务完成。简单任务没问题，但你要是丢给它一个"调研 Python Web 框架并输出对比报告"这种活儿，它就容易走着走着忘了自己要干嘛。

打个比方：你接到一个新需求，是先写代码还是先列个 TODO 清单？聪明人一般先想清楚再动手，AI 也一样。今天咱们就来聊聊怎么让智能体学会"先规划，再执行，中途还能调整计划"。

---

## 一、ReAct 的局限在哪？

先快速回顾一下 ReAct 的套路：**Thought（思考）→ Action（行动）→ Observation（观察）**，三步一循环，直到任务结束。

听起来挺合理对吧？但问题就出在"反应式"这三个字上——它每一步只看眼前，没有全局视角。

### 局限一：没有全局视角

ReAct 不知道任务总共要几步，也不知道自己走到哪了。就像你在迷宫里走，只能看见脚下的路，看不见出口在哪。每一步都是"根据当前信息决定下一步"，至于后面还有多少步、整体进度如何，它心里没数。

### 局限二：容易陷入循环

因为没有全局 checklist，AI 可能反复做同样的事。比如搜完 Django 又搜一次 Django 的性能数据，搜完 Flask 又去搜 Flask 的部署方式，来回打转，不知道跳出来。

### 局限三：长任务容易跑偏

任务一长（10 步以上），早期定下的目标很容易被遗忘。执行到第 7 步的时候，第 1 步想干啥可能已经模糊了，结果就是输出不完整。

**反面案例：**

```
任务：帮我调研 Python Web 框架，对比 Django/Flask/FastAPI，输出一篇对比报告

ReAct 模式的执行：
Step 1: 搜索 Django 信息 ✅
Step 2: 搜索 Flask 信息 ✅
Step 3: 搜索 FastAPI 信息 ✅
Step 4: 又搜索了一次 Django 的性能数据（重复了）🔄
Step 5: 开始写报告…只写了 Django 部分
Step 6: 忘了 Flask 还没写完，又去搜 FastAPI 的部署方式 😵
Step 7: 输出了一篇只有 Django 和 FastAPI 的不完整报告 ❌
```

说白了，ReAct 是"走一步看一步"，适合简单任务；复杂任务需要"先想清楚再动手"。

> **正文配图 1 提示词：** A maze illustration from top-down view. A small robot is inside the maze, taking a winding, inefficient path with many dead ends and backtracking (shown with dotted footprints going back and forth). The correct shortest path through the maze is subtly highlighted in green but the robot hasn't found it. Outside the maze, a bird's-eye view camera/drone icon suggests the solution is obvious from above. Clean flat design, white maze walls, light background. No text. Aspect ratio 16:9.

---

## 二、Plan-and-Execute：先列清单再干活

思路很简单：把任务拆成两阶段——**先列计划，再按计划执行**。

### 两阶段流程

1. **Plan 阶段**：用 LLM 把任务拆解成有序的步骤列表，每步写清楚要干啥、用啥工具、预期产出是啥
2. **Execute 阶段**：按计划一步步执行，每步有明确目标，执行完对照检查

和 ReAct 的关键区别：有全局计划，执行时心里有数，不会走着走着忘了要干嘛。

### 完整代码

先复用上篇文章的工具函数和工具定义，然后实现 PlanAndExecuteAgent：

```python
import json
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()

# ===== 工具函数（与上篇相同）=====

def get_weather(city: str) -> str:
    fake_weather = {
        "北京": "晴，25°C，微风，湿度40%",
        "上海": "多云，22°C，东南风3级，湿度65%",
        "广州": "雷阵雨，30°C，闷热，湿度85%",
        "深圳": "阴转小雨，28°C，湿度78%",
        "杭州": "晴，24°C，西北风2级，湿度50%",
    }
    return fake_weather.get(city, f"{city}：暂无天气数据")

def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

def search_news(keyword: str) -> str:
    fake_news = {
        "AI": "【最新】OpenAI发布GPT-5，推理能力大幅提升；Google推出Gemini 2.5 Pro；国内多家大模型厂商宣布降价。",
        "天气": "【预警】未来三天华南地区将迎来强降雨，气象台发布暴雨黄色预警。",
        "股市": "【财经】A股三大指数集体高开，AI概念股领涨，成交额突破万亿。",
    }
    for key, news in fake_news.items():
        if key in keyword:
            return news
    return f"未找到关于'{keyword}'的相关新闻"

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    rates = {
        ("USD", "CNY"): 7.24, ("EUR", "CNY"): 7.86,
        ("JPY", "CNY"): 0.048, ("GBP", "CNY"): 9.15, ("CNY", "USD"): 0.138,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate:
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"
    return f"暂不支持 {from_currency} → {to_currency} 的汇率查询"

# 模拟搜索（用于调研类任务）
def search_web(query: str) -> str:
    fake_data = {
        "Django": "Django：全栈框架，ORM强大，内置Admin，适合中大型项目，学习曲线较陡。",
        "Flask": "Flask：微框架，轻量灵活，适合小型项目和API，扩展丰富。",
        "FastAPI": "FastAPI：异步框架，性能高，自动生成OpenAPI文档，适合高性能API开发。",
    }
    for key, val in fake_data.items():
        if key.lower() in query.lower():
            return val
    return f"未找到关于'{query}'的详细信息"

available_tools = {
    "get_weather": get_weather, "calculate": calculate,
    "search_news": search_news, "get_exchange_rate": get_exchange_rate,
    "search_web": search_web,
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name in available_tools:
        return available_tools[tool_name](**arguments)
    return f"未知工具: {tool_name}"

# ===== PLAN_PROMPT 设计 =====

PLAN_PROMPT = """你是一个任务规划专家。请把以下任务拆解成 3-8 个具体的执行步骤。

要求：
1. 每个步骤要具体、可执行
2. 步骤之间有逻辑先后顺序
3. 每步标注可能用到的工具：get_weather/calculate/search_news/get_exchange_rate/search_web

任务：{task}

请输出 JSON 数组格式，不要其他文字：
[{{"step": 1, "action": "具体操作描述", "tool": "工具名", "expected_output": "预期产出"}}]
"""

# ===== PlanAndExecuteAgent =====

class PlanAndExecuteAgent:
    def __init__(self):
        self.plan = []
        self.results = {}

    def plan_task(self, task: str):
        """第一阶段：制定计划"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PLAN_PROMPT.format(task=task)}]
        )
        content = response.choices[0].message.content.strip()
        # 提取 JSON（可能被 markdown 包裹）
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        self.plan = json.loads(content)
        return self.plan

    def _execute_step(self, step: dict) -> str:
        """执行单步：用 LLM 决定如何调用工具"""
        action = step.get("action", "")
        tool_hint = step.get("tool", "")
        exec_prompt = f"""当前步骤：{action}
建议工具：{tool_hint}

可用工具：get_weather(city), calculate(expression), search_news(keyword), get_exchange_rate(from_currency, to_currency), search_web(query)

请根据步骤内容，输出 JSON 格式的 tool_call：{{"tool": "工具名", "args": {{参数}}}}。如果不需要调用工具，输出 {{"tool": null, "reason": "原因"}}"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": exec_prompt}]
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        try:
            call = json.loads(content)
            if call.get("tool"):
                result = execute_tool(call["tool"], call.get("args", {}))
                return result
            return call.get("reason", "无需调用工具")
        except:
            return content

    def execute_plan(self, task: str) -> str:
        """第二阶段：按计划执行"""
        for step in self.plan:
            s = step.get("step", 0)
            action = step.get("action", "")
            print(f"\n🔧 执行 Step {s}: {action}")
            result = self._execute_step(step)
            self.results[s] = result
            print(f"  📋 结果: {result[:80]}..." if len(result) > 80 else f"  📋 结果: {result}")
        # 汇总输出
        synth_prompt = f"""任务：{task}
执行结果：{json.dumps(self.results, ensure_ascii=False, indent=2)}

请根据以上结果，生成一份完整、结构化的最终回答。"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": synth_prompt}]
        )
        return resp.choices[0].message.content

    def run(self, task: str) -> str:
        print(f"\n{'='*60}\n  Plan-and-Execute 智能体\n  任务: {task}\n{'='*60}")
        self.plan_task(task)
        print(f"\n📋 计划制定完成，共 {len(self.plan)} 步：")
        for step in self.plan:
            print(f"  Step {step['step']}: {step['action']}")
        return self.execute_plan(task)
```

**模拟运行效果：**

```
任务：调研 Python Web 框架，对比 Django/Flask/FastAPI，输出对比报告

📋 计划制定完成，共 5 步：
  Step 1: 搜索 Django 的核心特性、优缺点、适用场景
  Step 2: 搜索 Flask 的核心特性、优缺点、适用场景
  Step 3: 搜索 FastAPI 的核心特性、优缺点、适用场景
  Step 4: 整理三个框架的对比维度（性能/生态/学习曲线/部署）
  Step 5: 生成结构化的对比报告

🔧 执行 Step 1: 搜索 Django 的核心特性...
  📋 结果: Django：全栈框架，ORM强大，内置Admin，适合中大型项目...

🔧 执行 Step 2: 搜索 Flask 的核心特性...
  📋 结果: Flask：微框架，轻量灵活，适合小型项目和API...

🔧 执行 Step 3: 搜索 FastAPI 的核心特性...
  📋 结果: FastAPI：异步框架，性能高，自动生成OpenAPI文档...

🔧 执行 Step 4: 整理三个框架的对比维度...
  📋 结果: 无需调用工具

🔧 执行 Step 5: 生成结构化的对比报告...
  📋 结果: 无需调用工具

✅ 最终输出：完整的三框架对比报告（Django/Flask/FastAPI 均有覆盖）
```

对比 ReAct 的反面案例，Plan-and-Execute 因为有全局 checklist，不会漏掉 Flask，也不会重复搜 Django。

---

## 三、动态重规划：计划赶不上变化怎么办？

纯 Plan-and-Execute 有个问题：计划是一次性生成的，执行中要是发现信息不足、方向有误，咋办？

就像你开车用 GPS 导航，原定路线前面突然施工封路了，导航会重新规划路线。智能体也一样——每步执行完后，评估一下"当前进展是否偏离计划"，必要时调整后续步骤。

### 三种调整策略

- **跳过**：某步已有结果，不需要重复执行
- **插入**：发现需要额外步骤，插到后面
- **修改**：调整后续步骤的方向或内容

### AdaptivePlanAgent 代码

```python
REPLAN_PROMPT = """你是任务监督员。请评估当前执行进度，决定是否需要调整计划。

原始任务：{task}
原始计划：{plan}
已完成步骤及结果：{completed}
当前步骤（Step {step_num}）执行结果：{current_result}

请回答（JSON 格式）：
1. "on_track": true/false —— 当前进展是否符合预期？
2. "need_replan": true/false —— 是否需要调整后续计划？
3. "new_remaining_plan": [...] —— 如果需要调整，输出新的剩余步骤列表（格式同原计划），否则输出空数组 []
"""

class AdaptivePlanAgent(PlanAndExecuteAgent):
    def __init__(self):
        super().__init__()
        self.task = ""

    def _should_replan(self, step: dict, result: str) -> bool:
        """判断是否需要重规划"""
        completed = {k: str(v)[:100] for k, v in self.results.items()}
        plan_str = json.dumps(self.plan, ensure_ascii=False, indent=2)
        prompt = REPLAN_PROMPT.format(
            task=self.task,
            plan=plan_str,
            completed=json.dumps(completed, ensure_ascii=False),
            step_num=step.get("step", 0),
            current_result=result[:200]
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        try:
            data = json.loads(content)
            return data.get("need_replan", False)
        except:
            return False

    def _replan(self, step: dict, result: str):
        """获取新的剩余计划"""
        completed = {k: str(v)[:100] for k, v in self.results.items()}
        plan_str = json.dumps(self.plan, ensure_ascii=False, indent=2)
        prompt = REPLAN_PROMPT.format(
            task=self.task,
            plan=plan_str,
            completed=json.dumps(completed, ensure_ascii=False),
            step_num=step.get("step", 0),
            current_result=result[:200]
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        try:
            data = json.loads(content)
            new_plan = data.get("new_remaining_plan", [])
            if new_plan:
                # 重新编号
                for i, p in enumerate(new_plan):
                    p["step"] = len(self.results) + i + 1
            return new_plan
        except:
            return self.plan[len(self.results):]

    def execute_plan(self, task: str) -> str:
        """带动态重规划的执行"""
        self.task = task
        step_idx = 0
        while step_idx < len(self.plan):
            step = self.plan[step_idx]
            s = step.get("step", step_idx + 1)
            action = step.get("action", "")
            print(f"\n🔧 执行 Step {s}: {action}")
            result = self._execute_step(step)
            self.results[s] = result
            print(f"  📋 结果: {result[:80]}..." if len(result) > 80 else f"  📋 结果: {result}")

            if self._should_replan(step, result):
                new_remaining = self._replan(step, result)
                old_remaining = self.plan[step_idx + 1:]
                if new_remaining != old_remaining:
                    print(f"  🔄 计划调整！原剩余 {len(old_remaining)} 步 → 新计划 {len(new_remaining)} 步")
                    self.plan = self.plan[: step_idx + 1] + new_remaining

            step_idx += 1

        synth_prompt = f"""任务：{task}
执行结果：{json.dumps(self.results, ensure_ascii=False, indent=2)}

请根据以上结果，生成一份完整、结构化的最终回答。"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": synth_prompt}]
        )
        return resp.choices[0].message.content

    def run(self, task: str) -> str:
        print(f"\n{'='*60}\n  自适应规划智能体\n  任务: {task}\n{'='*60}")
        self.plan_task(task)
        print(f"\n📋 计划制定完成，共 {len(self.plan)} 步：")
        for step in self.plan:
            print(f"  Step {step['step']}: {step['action']}")
        return self.execute_plan(task)
```

GPS 导航的类比很贴切：原计划是那条蓝线，遇到路障就动态重算一条绿线，照样能到终点。

> **正文配图 2 提示词：** A GPS navigation analogy illustration. A road map showing a planned route (solid blue line) from point A to point B. Along the way, a roadblock/construction sign appears. The route dynamically reroutes (dashed green line) around the obstacle to still reach point B. A small robot car is on the road at the detour point, with a recalculating GPS screen showing above it. Clean illustrated style, top-down map view. No text. Aspect ratio 16:9.

---

## 四、三种模式对比

| 维度 | ReAct | Plan-and-Execute | 自适应规划 |
|------|-------|-------------------|------------|
| 思考方式 | 走一步看一步 | 先想后做 | 先想后做 + 动态调整 |
| 全局视角 | 无 | 有（一次性） | 有（持续更新） |
| 适合任务 | 1-5 步简单任务 | 5-15 步结构化任务 | 复杂、不确定性高的任务 |
| 容错能力 | 差（容易循环） | 中（计划可能过时） | 强（自动纠偏） |
| LLM 调用次数 | 较少 | 中等 | 较多 |
| 实现复杂度 | 低 | 中 | 高 |

> **正文配图 3 提示词：** A three-panel evolution illustration. Panel 1: A robot running frantically, reacting to things as they appear (ReAct - chaotic). Panel 2: The same robot sitting at a desk first writing a plan on paper, then executing step by step with a checklist (Plan-and-Execute - organized). Panel 3: The robot has both the checklist AND a radar/monitor showing real-time adjustments, updating the plan while executing (Adaptive - smart). An arrow shows progression from left to right. Clean flat design, three distinct sections. No text. Aspect ratio 16:9.

---

## 五、实战：给你的智能体加上规划能力

把三种模式整合成一个 SmartAgent，可以手动指定模式，也可以让 AI 根据任务复杂度自动选：

```python
MODE_SELECT_PROMPT = """请评估以下任务的复杂度，用 1-10 的数字回答（1=极简单如查天气，10=极复杂如多步骤调研报告）。
只输出一个数字，不要其他文字。

任务：{task}
"""

class SmartAgent:
    def __init__(self, mode: str = "auto"):
        self.mode = mode  # "react" | "plan_execute" | "adaptive" | "auto"

    def _select_mode(self, task: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": MODE_SELECT_PROMPT.format(task=task)}]
        )
        try:
            score = int(resp.choices[0].message.content.strip())
            if score <= 3:
                return "react"
            elif score <= 6:
                return "plan_execute"
            else:
                return "adaptive"
        except:
            return "plan_execute"

    def _react_loop(self, task: str) -> str:
        """ReAct 模式（简化版，复用上篇文章逻辑）"""
        tools_schema = [
            {"type": "function", "function": {"name": "get_weather", "description": "查天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "calculate", "description": "计算", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
            {"type": "function", "function": {"name": "search_news", "description": "搜新闻", "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}},
            {"type": "function", "function": {"name": "get_exchange_rate", "description": "查汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
        ]
        messages = [
            {"role": "system", "content": "你按 ReAct 模式工作：先用【思考】说明推理，再决定调用工具或回答。"},
            {"role": "user", "content": task}
        ]
        for _ in range(8):
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools_schema)
            msg = resp.choices[0].message
            if msg.content:
                print(f"💭 {msg.content[:200]}...")
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = execute_tool(name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "达到最大轮次"

    def run(self, task: str) -> str:
        mode = self._select_mode(task) if self.mode == "auto" else self.mode
        print(f"\n📌 选用模式: {mode}")

        if mode == "react":
            return self._react_loop(task)
        elif mode == "plan_execute":
            agent = PlanAndExecuteAgent()
            return agent.run(task)
        else:
            agent = AdaptivePlanAgent()
            return agent.run(task)
```

**使用示例：**

```python
# 自动选模式
agent = SmartAgent(mode="auto")
agent.run("帮我查北京天气，再算 100*7.24 等于多少")

# 强制用 Plan-and-Execute
agent = SmartAgent(mode="plan_execute")
agent.run("调研 Django/Flask/FastAPI，输出对比报告")
```

---

## 六、什么时候该用哪种？

| 场景 | 推荐模式 |
|------|----------|
| 查天气、算个数、单步问答 | ReAct 够用 |
| 写报告、做调研、多步骤操作 | Plan-and-Execute 的甜蜜点 |
| 不确定性高、需要探索的任务 | 自适应规划 |
| 拿不准 | 用 SmartAgent(mode="auto") 让 AI 自己判断 |

实际建议：别一上来就上最重的。简单任务用 ReAct 省 token；结构化任务用 Plan-and-Execute；只有那种"走着走着可能发现要改方向"的任务，才值得上自适应规划。

> **正文配图 4 提示词：** A spectrum/slider illustration. A horizontal slider bar going from "Pure Reactive" (left, shown as a lightning bolt icon) to "Pure Planning" (right, shown as a detailed blueprint icon). Three markers on the slider: ReAct at 20% position, Plan-and-Execute at 60%, Adaptive at 80%. Above the slider, small scenario cards float near their recommended position: simple tasks near reactive end, complex tasks near planning end. Clean infographic style, gradient background from orange (left) to blue (right). No text. Aspect ratio 16:9.

---

## 写在最后

Planning 让智能体学会了"先想后做"，但对于知识密集型任务，光靠向量检索有时候不够——知识之间有关系，不是孤立的文档片段。下一篇咱们来聊 **Graph RAG**，用知识图谱增强检索。

最后说一句：好的 Planning 不是计划越详细越好，而是在"规划粒度"和"执行灵活性"之间找平衡。过度规划本身就是一种浪费，该反应的时候还得反应。

---

## 完整代码（可直接运行）

```python
import json
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "你的API Key"
client = OpenAI()

# ==================== 工具定义 ====================

def get_weather(city: str) -> str:
    fake_weather = {
        "北京": "晴，25°C，微风，湿度40%",
        "上海": "多云，22°C，东南风3级，湿度65%",
        "广州": "雷阵雨，30°C，闷热，湿度85%",
        "杭州": "晴，24°C，西北风2级，湿度50%",
    }
    return fake_weather.get(city, f"{city}：暂无天气数据")

def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {e}"

def search_news(keyword: str) -> str:
    fake_news = {
        "AI": "【最新】OpenAI发布GPT-5；Google推出Gemini 2.5 Pro；国内大模型厂商宣布降价。",
        "天气": "【预警】未来三天华南地区将迎来强降雨，气象台发布暴雨黄色预警。",
    }
    for key, news in fake_news.items():
        if key in keyword:
            return news
    return f"未找到关于'{keyword}'的相关新闻"

def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    rates = {("USD", "CNY"): 7.24, ("EUR", "CNY"): 7.86, ("CNY", "USD"): 0.138}
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate:
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"
    return f"暂不支持 {from_currency} → {to_currency} 的汇率查询"

def search_web(query: str) -> str:
    fake_data = {
        "Django": "Django：全栈框架，ORM强大，内置Admin，适合中大型项目。",
        "Flask": "Flask：微框架，轻量灵活，适合小型项目和API。",
        "FastAPI": "FastAPI：异步框架，性能高，自动生成OpenAPI文档。",
    }
    for key, val in fake_data.items():
        if key.lower() in query.lower():
            return val
    return f"未找到关于'{query}'的详细信息"

available_tools = {
    "get_weather": get_weather, "calculate": calculate,
    "search_news": search_news, "get_exchange_rate": get_exchange_rate,
    "search_web": search_web,
}

def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name in available_tools:
        return available_tools[tool_name](**arguments)
    return f"未知工具: {tool_name}"

# ==================== PlanAndExecuteAgent ====================

PLAN_PROMPT = """你是一个任务规划专家。请把以下任务拆解成 3-8 个具体的执行步骤。
每步标注可能用到的工具：get_weather/calculate/search_news/get_exchange_rate/search_web
任务：{task}
输出 JSON 数组：[{{"step": 1, "action": "操作", "tool": "工具名", "expected_output": "预期"}}]"""

class PlanAndExecuteAgent:
    def __init__(self):
        self.plan = []
        self.results = {}

    def plan_task(self, task: str):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PLAN_PROMPT.format(task=task)}]
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        self.plan = json.loads(content)
        return self.plan

    def _execute_step(self, step: dict) -> str:
        action = step.get("action", "")
        tool_hint = step.get("tool", "")
        exec_prompt = f"""步骤：{action}，建议工具：{tool_hint}
可用：get_weather(city), calculate(expression), search_news(keyword), get_exchange_rate(from_currency, to_currency), search_web(query)
输出 JSON：{{"tool": "工具名", "args": {{}}}} 或 {{"tool": null}}"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": exec_prompt}]
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        try:
            call = json.loads(content)
            if call.get("tool"):
                return execute_tool(call["tool"], call.get("args", {}))
            return "无需调用工具"
        except:
            return content

    def execute_plan(self, task: str) -> str:
        for step in self.plan:
            s = step.get("step", 0)
            self.results[s] = self._execute_step(step)
        synth = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"任务：{task}\n结果：{json.dumps(self.results, ensure_ascii=False)}\n请生成完整回答。"}]
        )
        return synth.choices[0].message.content

    def run(self, task: str) -> str:
        print(f"\nPlan-and-Execute | 任务: {task}")
        self.plan_task(task)
        print(f"计划: {len(self.plan)} 步")
        return self.execute_plan(task)

# ==================== 运行 ====================

if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    result = agent.run("帮我查北京和上海天气，对比哪个更适合周末出游，再算 500 美元能换多少人民币")
    print(f"\n✅ 最终回答:\n{result}")
```

---

**依赖**：`pip install openai`，基于 OpenAI API（gpt-4o-mini）。
