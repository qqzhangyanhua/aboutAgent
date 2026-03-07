# 让 AI 先想再干 —— Planning 机制：从 ReAct 到 Plan-and-Execute

## 封面图提示词（2:1 比例，1232×616px）

> A split-screen tech illustration. Left side (slightly chaotic): a robot running in circles, juggling multiple tools, with tangled thought bubbles - representing ReAct's step-by-step reactivity. Right side (organized): a robot sitting at a desk with a clear whiteboard showing a numbered plan/checklist, then calmly executing each step with checkmarks appearing - representing Plan-and-Execute's structured approach. A dividing line in the center with a "VS" or evolution arrow from left to right. Background: left side has warm/orange tones (urgency), right side has cool/blue tones (calm). No text, no watermark. Aspect ratio 2:1.

## 一句话定位

分析 ReAct 的局限，引入 Plan-and-Execute 模式，实现一个能"先列计划再执行、遇到问题还能动态调整"的智能体。

---

## 章节大纲

### 开头引入

- 切入点：第一篇智能体文章里我们用的是 ReAct 模式——AI 想一步做一步。大部分简单任务没问题，但遇到复杂任务就容易"走着走着忘了自己要干嘛"
- 类比：你接到一个新需求，是先写代码还是先列个 TODO 清单？聪明人先想清楚再动手，AI 也一样
- 核心问题：怎么让智能体学会"先规划，再执行，中途还能调整计划"

### 一、ReAct 的局限在哪？

**要点：**
- ReAct 回顾：Thought → Action → Observation 循环
- 局限一：没有全局视角——每一步只看眼前，不知道还剩多少步
- 局限二：容易陷入循环——反复做同样的事，不知道跳出来
- 局限三：长任务容易跑偏——10 步以上的任务，早期目标被遗忘
- 根本原因：ReAct 是"反应式"的，缺乏"规划式"的预判

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

> **正文配图 1 提示词：**
> A maze illustration from top-down view. A small robot is inside the maze, taking a winding, inefficient path with many dead ends and backtracking (shown with dotted footprints going back and forth). The correct shortest path through the maze is subtly highlighted in green but the robot hasn't found it. Outside the maze, a bird's-eye view camera/drone icon suggests the solution is obvious from above. Clean flat design, white maze walls, light background. No text. Aspect ratio 16:9.

### 二、Plan-and-Execute：先列清单再干活

**要点：**
- 核心思想：把任务拆成两阶段——
  1. **Plan 阶段**：用 LLM 把任务拆解成有序的步骤列表
  2. **Execute 阶段**：按计划逐步执行，每步有明确目标
- 和 ReAct 的关键区别：有全局计划，执行时对照检查
- 参考论文：Wang et al. "Plan-and-Solve Prompting" (2023)

**核心代码片段：**

```python
PLAN_PROMPT = """你是一个任务规划专家。请把以下任务拆解成 3-8 个具体的执行步骤。
要求：
1. 每个步骤要具体、可执行
2. 步骤之间有逻辑先后顺序
3. 标注每步需要用到的工具（搜索/计算/写作等）

任务：{task}

请输出 JSON 格式：
[{{"step": 1, "action": "具体操作", "tool": "需要的工具", "expected_output": "预期产出"}}]
"""

class PlanAndExecuteAgent:
    def __init__(self, tools):
        self.tools = tools
        self.plan = []
        self.results = {}

    def plan_task(self, task):
        """第一阶段：制定计划"""
        response = ask_llm(PLAN_PROMPT.format(task=task))
        self.plan = json.loads(response)
        print(f"📋 计划制定完成，共 {len(self.plan)} 步：")
        for step in self.plan:
            print(f"  Step {step['step']}: {step['action']}")
        return self.plan

    def execute_plan(self):
        """第二阶段：按计划执行"""
        for step in self.plan:
            print(f"\n🔧 执行 Step {step['step']}: {step['action']}")
            result = self._execute_step(step)
            self.results[step["step"]] = result
            print(f"  ✅ 完成: {result[:100]}...")
        return self._synthesize_results()
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

🔧 执行 Step 1: ... ✅
🔧 执行 Step 2: ... ✅
🔧 执行 Step 3: ... ✅
🔧 执行 Step 4: ... ✅
🔧 执行 Step 5: ... ✅

✅ 最终输出：完整的三框架对比报告
```

### 三、动态重规划：计划赶不上变化怎么办？

**要点：**
- 纯 Plan-and-Execute 的问题：计划是一次性生成的，执行中发现信息不足或方向有误怎么办？
- 解决方案：每步执行完后，让 LLM 评估"当前进展是否偏离计划"，必要时重新规划
- 三种调整策略：
  - 跳过（某步已有结果，不需要重复）
  - 插入（发现需要额外步骤）
  - 修改（调整后续步骤的方向）

**核心代码片段：**

```python
REPLAN_PROMPT = """你是任务监督员。请评估当前执行进度，决定是否需要调整计划。

原始任务：{task}
原始计划：{plan}
已完成步骤和结果：{completed}
当前步骤执行结果：{current_result}

请回答：
1. 当前进展是否符合预期？（是/否）
2. 是否需要调整后续计划？（是/否）
3. 如果需要，输出调整后的剩余计划（JSON 格式）
"""

class AdaptivePlanAgent(PlanAndExecuteAgent):
    def execute_plan(self):
        step_idx = 0
        while step_idx < len(self.plan):
            step = self.plan[step_idx]
            result = self._execute_step(step)
            self.results[step["step"]] = result

            # 每步执行后评估是否需要重规划
            if self._should_replan(step, result):
                new_plan = self._replan(step, result)
                remaining = self.plan[step_idx + 1:]
                if new_plan != remaining:
                    print(f"🔄 计划调整！原剩余 {len(remaining)} 步 → 新计划 {len(new_plan)} 步")
                    self.plan = self.plan[:step_idx + 1] + new_plan

            step_idx += 1
```

> **正文配图 2 提示词：**
> A GPS navigation analogy illustration. A road map showing a planned route (solid blue line) from point A to point B. Along the way, a roadblock/construction sign appears. The route dynamically reroutes (dashed green line) around the obstacle to still reach point B. A small robot car is on the road at the detour point, with a recalculating GPS screen showing above it. Clean illustrated style, top-down map view. No text. Aspect ratio 16:9.

### 四、ReAct vs Plan-and-Execute vs 自适应规划

```
| 维度         | ReAct           | Plan-and-Execute   | 自适应规划           |
|-------------|-----------------|--------------------|--------------------|
| 思考方式     | 走一步看一步      | 先想后做            | 先想后做 + 动态调整   |
| 全局视角     | 无               | 有（一次性）        | 有（持续更新）       |
| 适合任务     | 1-5 步简单任务    | 5-15 步结构化任务   | 复杂、不确定性高的任务 |
| 容错能力     | 差（容易循环）    | 中（计划可能过时）   | 强（自动纠偏）       |
| LLM 调用次数 | 较少             | 中等               | 较多               |
| 实现复杂度   | 低               | 中                 | 高                 |
```

### 五、实战：给你的智能体加上规划能力

**要点：**
- 把第一篇文章的 ReAct Agent 升级成 Plan-and-Execute Agent
- 对比同一个复杂任务在两种模式下的执行效果
- 代码结构：Agent 类增加 plan() 和 replan() 方法

**核心代码片段（整合）：**

```python
class SmartAgent:
    def __init__(self, tools, mode="adaptive"):
        self.tools = tools
        self.mode = mode  # "react" | "plan_execute" | "adaptive"

    def run(self, task):
        if self.mode == "react":
            return self._react_loop(task)
        elif self.mode == "plan_execute":
            plan = self.plan_task(task)
            return self.execute_plan(plan)
        elif self.mode == "adaptive":
            plan = self.plan_task(task)
            return self.adaptive_execute(plan, task)

    def _select_mode(self, task):
        """根据任务复杂度自动选择模式"""
        complexity = ask_llm(f"评估任务复杂度(1-10)：{task}")
        score = int(complexity)
        if score <= 3:
            return "react"
        elif score <= 7:
            return "plan_execute"
        else:
            return "adaptive"
```

> **正文配图 3 提示词：**
> A three-panel evolution illustration. Panel 1: A robot running frantically, reacting to things as they appear (ReAct - chaotic). Panel 2: The same robot sitting at a desk first writing a plan on paper, then executing step by step with a checklist (Plan-and-Execute - organized). Panel 3: The robot has both the checklist AND a radar/monitor showing real-time adjustments, updating the plan while executing (Adaptive - smart). An arrow shows progression from left to right. Clean flat design, three distinct sections. No text. Aspect ratio 16:9.

### 六、什么时候该用哪种？

**要点：**
- ReAct 够用的场景：查个天气、算个数、单步问答
- Plan-and-Execute 的甜蜜点：写报告、做调研、多步骤操作
- 自适应规划的场景：不确定性高、需要探索的任务
- 实际建议：可以让 AI 自己判断用哪种模式（Meta-Planning）

### 写在最后

- 衔接语：Planning 让智能体学会了"先想后做"，但对于知识密集型任务，光靠向量检索有时候不够——知识之间有关系，不是孤立的文档片段。下一篇我们来聊 Graph RAG，用知识图谱增强检索
- 关键观点：好的 Planning 不是计划越详细越好，而是在"规划粒度"和"执行灵活性"之间找到平衡。过度规划本身就是一种浪费

> **正文配图 4 提示词：**
> A spectrum/slider illustration. A horizontal slider bar going from "Pure Reactive" (left, shown as a lightning bolt icon) to "Pure Planning" (right, shown as a detailed blueprint icon). Three markers on the slider: ReAct at 20% position, Plan-and-Execute at 60%, Adaptive at 80%. Above the slider, small scenario cards float near their recommended position: simple tasks near reactive end, complex tasks near planning end. Clean infographic style, gradient background from orange (left) to blue (right). No text. Aspect ratio 16:9.
