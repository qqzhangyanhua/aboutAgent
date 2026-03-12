# 一个 Agent 够用吗？—— 多步骤任务与多 Agent 的拆分决策

---

**封面图提示词（2:1 比例，1232×616px）**

> A tech illustration showing a crossroads/fork in the road. On the left path: a single robot confidently walking forward with a checklist, handling multiple tasks sequentially (representing single agent with multi-step). On the right path: three different colored robots walking together in a team formation, each carrying a different tool (representing multi-agent). A developer figure stands at the fork, scratching their head, looking at both paths. A signpost at the fork shows decision criteria icons. Background: clean gradient. No text, no watermark. Aspect ratio 2:1.

---

第 8 篇咱们实现了三种多智能体协作模式——主从、辩论、流水线。第 10 篇搞了 Plan-and-Execute，让单个 Agent 处理多步骤任务。

但有个关键问题这两篇都没回答：**什么时候一个 Agent 搞多步骤就够了，什么时候必须拆成多个 Agent？**

这不是技术问题，是**设计决策**。拆错了比不拆更糟——多 Agent 意味着更高的延迟、更多的 token 消耗、更复杂的调试。但该拆的时候不拆，一个 Agent 承担太多角色，prompt 就会"精神分裂"，输出质量反而下降。

这篇文章给你一套**决策框架**——面对一个需求，5 分钟内判断该用单 Agent 还是多 Agent。

---

## 一、先定义清楚两个概念

### 单 Agent 多步骤

一个 Agent，一个 System Prompt，通过 Planning 机制把任务拆成多步依次执行。

```
用户："帮我调研 Python Web 框架，写一份对比报告"

┌─────────────────────────────────┐
│          单个 Agent              │
│  System Prompt: 技术调研助手     │
│                                  │
│  Step 1: 搜集 Django 信息        │
│  Step 2: 搜集 Flask 信息         │
│  Step 3: 搜集 FastAPI 信息       │
│  Step 4: 对比分析                │
│  Step 5: 撰写报告                │
└─────────────────────────────────┘
```

### 多 Agent 协作

多个 Agent，每个有独立的 System Prompt 和角色，通过消息传递协作。

```
用户："帮我调研 Python Web 框架，写一份对比报告"

┌──────────┐    ┌──────────┐    ┌──────────┐
│ 研究员    │───→│ 分析师    │───→│ 写手     │
│ Agent     │    │ Agent     │    │ Agent    │
│ 搜集资料  │    │ 对比分析  │    │ 撰写报告 │
└──────────┘    └──────────┘    └──────────┘
```

---

## 二、拆不拆的五条判断标准

### 标准 1：角色冲突

**如果一个任务需要两种互相矛盾的角色，必须拆。**

```
需要拆的例子：
  "写一篇文章，然后严格审核它"
  → 写作需要创造力（温度高），审核需要严谨（温度低）
  → 同一个 Agent 很难同时"放飞自我"和"严格审查"

不需要拆的例子：
  "查三个框架的信息，然后对比分析"
  → 搜集和分析是同一种角色（研究员），不矛盾
```

```python
def check_role_conflict(tasks: list[str]) -> bool:
    """判断任务列表中是否存在角色冲突"""
    role_pairs_that_conflict = [
        ("创作", "审核"),
        ("提出方案", "挑毛病"),
        ("乐观评估", "风险分析"),
        ("用户代言", "系统维护"),
    ]
    # 简化判断：如果任务描述暗示了冲突角色对，建议拆
    task_text = " ".join(tasks)
    for role_a, role_b in role_pairs_that_conflict:
        if role_a in task_text and role_b in task_text:
            return True
    return False
```

### 标准 2：上下文窗口压力

**如果一个 Agent 需要同时装下太多信息，拆。**

```
需要拆的例子：
  搜集 5 个框架的详细信息（每个 3000 token）
  + 10 轮对话历史（4000 token）
  + 分析对比的中间结果（5000 token）
  = 24000 token → 上下文里装不下，前面的信息会被遗忘

不需要拆的例子：
  查一个订单 + 查物流 + 生成回复
  = 大约 2000 token → 绰绰有余
```

### 标准 3：专业度要求

**如果子任务需要完全不同的专业 prompt 才能做好，拆。**

```
需要拆的例子：
  "把这段 Python 代码翻译成 Rust，然后评估两者的性能差异"
  → Python→Rust 翻译需要精通两种语言的 prompt
  → 性能评估需要基准测试专家的 prompt
  → 两套 prompt 塞在同一个 System Prompt 里会互相干扰

不需要拆的例子：
  "查三个城市的天气，生成一份旅行建议"
  → 查天气和生成建议是同一种"旅行助手"角色
```

### 标准 4：并行能力

**如果子任务之间没有依赖，可以并行执行，拆。**

```
需要拆（可并行）：
  同时搜集 Django、Flask、FastAPI 三个框架的信息
  → 三个搜集任务互不依赖，可以并行
  → 用三个 Agent 同时搜，耗时 = max(三者) 而非 sum(三者)

不需要拆（必须串行）：
  先查订单 → 根据订单状态决定能否退款 → 提交退款
  → 每步依赖上一步的结果，串行就行
```

### 标准 5：复杂度 vs 成本

**多 Agent 的隐性成本很高，简单任务不值得拆。**

```
多 Agent 的隐性成本：
  - 每个 Agent 都要一份 System Prompt → 多几倍的 prompt token
  - Agent 之间传消息 → 额外的 LLM 调用
  - 调度逻辑 → 额外的代码复杂度
  - 调试难度 → 出了 bug 要看多个 Agent 的日志

规则：
  如果任务用一个 Agent + 3-5 步就能完成，不要拆。
  如果任务需要 10+ 步或 2 种以上角色，考虑拆。
```

---

**配图1（放在本节后，16:9）**

> A decision flowchart with diamond decision nodes and rectangular action nodes. Starting node: "New Task". First decision: "Role Conflict?" → Yes → "Split". No → Second decision: "Context Overflow?" → Yes → "Split". No → Third decision: "Need Parallelism?" → Yes → "Split". No → Fourth decision: "Steps > 10?" → Yes → "Consider Split". No → "Single Agent". Clean flowchart style with green/red arrows for yes/no. No text. Aspect ratio 16:9.

---

## 三、决策流程图（代码版）

```python
from dataclasses import dataclass


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    description: str
    subtasks: list[str]
    has_role_conflict: bool
    estimated_context_tokens: int
    can_parallelize: bool
    num_distinct_expertise: int
    total_steps: int


def decide_single_or_multi(analysis: TaskAnalysis) -> dict:
    """决策：用单 Agent 还是多 Agent"""
    reasons_to_split: list[str] = []
    reasons_to_keep: list[str] = []

    # 标准 1：角色冲突
    if analysis.has_role_conflict:
        reasons_to_split.append("存在角色冲突（如创作+审核），拆开避免人格分裂")
    else:
        reasons_to_keep.append("角色统一，一个 prompt 能搞定")

    # 标准 2：上下文压力
    if analysis.estimated_context_tokens > 15000:
        reasons_to_split.append(f"预估上下文 {analysis.estimated_context_tokens} token，超过安全线")
    else:
        reasons_to_keep.append(f"上下文约 {analysis.estimated_context_tokens} token，没有压力")

    # 标准 3：专业度
    if analysis.num_distinct_expertise >= 3:
        reasons_to_split.append(f"需要 {analysis.num_distinct_expertise} 种不同专业能力")
    else:
        reasons_to_keep.append("专业需求单一，一个 Agent 够用")

    # 标准 4：并行
    if analysis.can_parallelize:
        reasons_to_split.append("子任务可以并行，拆开能提速")

    # 标准 5：步骤数
    if analysis.total_steps > 10:
        reasons_to_split.append(f"步骤数 {analysis.total_steps}，长任务容易跑偏")
    else:
        reasons_to_keep.append(f"只有 {analysis.total_steps} 步，不复杂")

    # 决策
    decision = "multi_agent" if len(reasons_to_split) >= 2 else "single_agent"

    return {
        "decision": decision,
        "recommendation": "多 Agent" if decision == "multi_agent" else "单 Agent 多步骤",
        "reasons_to_split": reasons_to_split,
        "reasons_to_keep": reasons_to_keep
    }


# 测试场景 1：写文章
case1 = TaskAnalysis(
    description="帮我调研 Python Web 框架并写一份对比报告",
    subtasks=["搜集 Django 信息", "搜集 Flask 信息", "搜集 FastAPI 信息", "对比分析", "撰写报告"],
    has_role_conflict=False,
    estimated_context_tokens=8000,
    can_parallelize=True,
    num_distinct_expertise=1,
    total_steps=5
)

# 测试场景 2：代码审查流水线
case2 = TaskAnalysis(
    description="写代码、做 Code Review、写测试、做安全审计",
    subtasks=["编写功能代码", "代码审查", "编写测试用例", "安全审计"],
    has_role_conflict=True,
    estimated_context_tokens=20000,
    can_parallelize=False,
    num_distinct_expertise=4,
    total_steps=12
)

# 测试场景 3：简单客服
case3 = TaskAnalysis(
    description="查订单、查物流、回答用户",
    subtasks=["查询订单状态", "查询物流信息", "组织回复"],
    has_role_conflict=False,
    estimated_context_tokens=3000,
    can_parallelize=False,
    num_distinct_expertise=1,
    total_steps=3
)

for i, case in enumerate([case1, case2, case3], 1):
    result = decide_single_or_multi(case)
    print(f"\n场景 {i}：{case.description}")
    print(f"  决策：{result['recommendation']}")
    if result["reasons_to_split"]:
        print(f"  拆的理由：{'; '.join(result['reasons_to_split'])}")
    if result["reasons_to_keep"]:
        print(f"  不拆的理由：{'; '.join(result['reasons_to_keep'])}")
```

---

## 四、拆成多 Agent 后怎么编排？

如果决定拆，下一步是选编排模式。第 8 篇讲了三种，这里补充完整：

### 4.1 五种编排模式

| 模式 | 适用场景 | 信息流 |
|------|---------|--------|
| **流水线** | 步骤固定、串行依赖 | A → B → C |
| **主从** | 任务可拆分、子任务独立 | Master → [Worker₁, Worker₂, ...] → Master |
| **辩论** | 需要多视角、互相校验 | A ↔ B（多轮） → Judge |
| **路由** | 不同类型的问题交给不同专家 | Router → Expert_X |
| **混合** | 复杂场景，多种模式组合 | 按需组合 |

### 4.2 模式选择速查表

```
任务是线性的、每步依赖上一步？ → 流水线
任务可以拆成独立子任务？         → 主从
任务需要从多个角度验证？         → 辩论
任务类型多、需要专家分诊？       → 路由
以上都不满足或都满足？           → 混合
```

### 4.3 实战案例：代码审查流水线

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)
MODEL_NAME = "deepseek-chat"


class SimpleAgent:
    """简化版 Agent"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, input_text: str) -> str:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content


# 用单 Agent 做的版本（对比用）
solo_agent = SimpleAgent(
    name="全能开发者",
    system_prompt=(
        "你是一个全能开发者。根据需求写代码，然后自己审查，找出问题，最后修复。"
        "按顺序输出：代码 → 审查意见 → 修复后的代码。"
    )
)

# 用多 Agent 做的版本
coder = SimpleAgent(
    name="编码专家",
    system_prompt="你是 Python 编码专家。根据需求写出高质量的代码。只输出代码，不要其他内容。"
)

reviewer = SimpleAgent(
    name="代码审查员",
    system_prompt=(
        "你是严格的代码审查员。审查以下代码，指出：\n"
        "1. Bug 和逻辑错误\n"
        "2. 安全隐患\n"
        "3. 性能问题\n"
        "4. 可维护性改进建议\n"
        "只输出审查意见，不要改代码。"
    )
)

fixer = SimpleAgent(
    name="代码修复专家",
    system_prompt="你是代码修复专家。根据审查意见修复代码。输出修复后的完整代码。"
)


def pipeline_review(requirement: str) -> dict:
    """流水线式代码审查"""
    print(f"[1/3] 编码专家正在写代码...")
    code = coder.run(requirement)

    print(f"[2/3] 审查员正在 Review...")
    review = reviewer.run(f"请审查以下代码：\n\n{code}")

    print(f"[3/3] 修复专家正在修复...")
    fixed = fixer.run(f"原始代码：\n{code}\n\n审查意见：\n{review}\n\n请根据审查意见修复代码。")

    return {"code": code, "review": review, "fixed_code": fixed}


# 运行
requirement = "写一个 Python 函数，从 URL 下载文件并保存到本地，支持断点续传和超时重试"
result = pipeline_review(requirement)

print("\n=== 审查意见 ===")
print(result["review"][:500])
```

拆成三个 Agent 的好处：编码 Agent 的 prompt 鼓励创造力，审查 Agent 的 prompt 鼓励严谨——两种矛盾的风格在不同 Agent 里和谐共存。

---

## 五、常见误区

### 误区 1："多 Agent 一定比单 Agent 好"

不一定。多 Agent 多了调度成本、通信成本和调试成本。简单任务用多 Agent 是大炮打蚊子。

### 误区 2："Agent 越多越好"

三个 Agent 协作比一个强，但十个 Agent 协作可能比三个差——因为信息在 Agent 之间传递时会失真，就像传话游戏。

### 误区 3："拆 Agent 可以解决 prompt 太长的问题"

拆 Agent 能降低单个 Agent 的上下文压力，但 Agent 之间传递信息时也需要上下文。如果传递的信息没有经过压缩，问题只是换了个地方。

---

## 六、我的实际做法

说实话，我现在做项目的默认姿势是：**先用单 Agent 把功能跑通，然后等它"疼"了再拆。**

什么叫"疼"了？就是你发现一个 Agent 的 System Prompt 里写着"你既是代码审查专家又是产品经理又是测试工程师"，三种角色的指令互相打架——审查说"这段代码风格不好要改"，产品说"用户要的就是这个功能赶紧上"，谁赢取决于大模型当天的心情。这时候才该拆。

另一种"疼"的信号是上下文爆了——你发现每次调用都塞了 3 万 token，其中 2 万是上一步的中间结果，大模型根本没看就开始回答了。

提前拆只有一种情况值得：你**非常确定**两个步骤可以并行跑，而且串行会让用户等太久。比如同时做翻译和格式校验，这种天然并行的场景提前拆是合理的。

除此之外，别预判。预判错了，你花三天搭多 Agent 编排框架，结果发现一个 Agent 配上好的 Prompt 就够用了。
