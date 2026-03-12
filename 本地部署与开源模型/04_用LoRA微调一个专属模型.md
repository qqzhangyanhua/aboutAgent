# 用 LoRA 微调一个专属模型：从数据准备到部署

你拿 Qwen、Llama 这些开源模型聊技术、写代码、翻译文档，样样都行。但一问你公司的产品逻辑、你们行业的黑话、你们客服那套固定话术——它就开始瞎编了。不是模型笨，是它压根没学过你这摊事儿。就像请了个博学多才的顾问，啥都懂，就是不懂你们公司。

**微调**能解决这个问题。用你自己的数据，把模型"培训"一遍，让它真正懂你的业务、说你的话、按你的规矩办事。今天这篇，就把 LoRA 微调的完整流程讲透：从数据准备到部署上线，每一步怎么走，坑在哪，怎么避。

---

## 一、什么是微调，什么是 LoRA

先捋清楚两个概念。

**微调（Fine-tuning）**：在已经预训练好的大模型基础上，用你的数据再训练一轮。模型原有的"通用能力"保留，同时学会你的领域知识、输出风格、特定行为。类比一下：预训练模型是个通才，微调就是送他去你们公司上培训班，培训完还是那个通才，但多了你们公司的"肌肉记忆"。

**全量微调**和 **LoRA** 是两种完全不同的玩法。

全量微调：把模型里几十亿个参数全部更新一遍。相当于把整栋楼重新装修——效果好，但成本吓人。70B 模型全量微调，没几十张 A100 根本玩不转，普通人想都别想。

**LoRA**（Low-Rank Adaptation，低秩适配）：不碰原模型的主体，只在关键层旁边加几层"小补丁"（适配器），只训练这些补丁。相当于整栋楼不动，只改几个房间——成本低、速度快、效果还不差。一张 24GB 的 4090，就能微调 7B 模型；4B 模型甚至 10GB 显存就够。所以现在大家说的"微调"，十有八九指的是 LoRA。

LoRA 的核心思想是：大模型的权重更新其实可以用低秩矩阵近似。与其动几十亿参数，不如学两个小矩阵 A 和 B，相乘之后"模拟"出更新效果。参数量从几十亿降到几百万，训练快、显存省、还能随时插拔——想用就加载适配器，不想用就恢复原模型。

---

## 二、什么时候需要微调

不是所有问题都要上微调。先搞清楚：**Prompt** 和 **RAG** 搞不定的，才是微调的战场。

**Prompt 能搞定的**：任务简单、规则清晰、改改指令就能对齐。比如"请用 JSON 格式输出"、"每段话不超过 50 字"。这类需求，写清楚 Prompt 就行，没必要微调。

**RAG 能搞定的**：知识在外部文档里，需要检索后拼进上下文。比如企业知识库问答、政策法规查询。知识更新频繁的话，RAG 比微调更合适——改文档就行，不用重新训练。

**微调才合适的场景**，大致有四类：

1. **输出风格**：你们客服习惯用"亲"、"哈"、"呢"，模型却一本正经；你们要求必须引用条款编号，模型却只给结论。这种"说话方式"的问题，Prompt 很难稳定控制，微调能直接改掉。

2. **固定格式**：必须按某种模板输出，比如工单、报告、产品描述。格式一复杂，Prompt 容易翻车，微调能让模型"刻进骨子里"。

3. **专业术语**：你们行业有一堆黑话，模型要么不认识，要么乱用。微调可以教会模型正确理解和生成这些术语。

4. **特定行为**：比如"遇到不确定的必须说不知道"、"禁止输出某些敏感内容"。这类行为约束，单靠 Prompt 容易绕过，微调更可靠。

一句话：**风格、格式、术语、行为**——这四个里沾上俩，就可以认真考虑微调了。

---

## 三、数据准备——微调最重要的一步

微调效果，八成看数据。模型再强、框架再牛，数据烂了，白搭。

### 数据格式

主流格式是 **instruction / input / output** 三段式，对应"任务说明 / 输入 / 期望输出"。很多框架也支持简化为 **text** 一列，把三段拼成一段对话。以 Alpaca 格式为例：

```json
{
  "instruction": "将以下产品描述改写成客服话术风格",
  "input": "产品A，售价299元，支持7天无理由退货",
  "output": "亲，产品A现在特价299元哦～支持7天无理由退货，有任何问题随时找我们哈～"
}
```

如果用 **text** 列（很多框架要求），需要按模型的对话模板拼起来。以 Qwen 为例：

```python
def format_instruction(instruction, input_text, output):
    return f"""<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
```

数据存成 JSONL，一行一条，方便流式读取和大规模处理。

### 数据质量比数量重要

宁可 500 条高质量，也不要 5000 条凑数。低质量数据会"教坏"模型：格式不统一、答案有错、风格混乱，模型全学进去，越训越差。

高质量的标准：**任务一致**（都是同一类任务）、**格式统一**（输出结构一致）、**答案正确**（不能有事实错误）、**风格一致**（说话方式统一）。建议人工抽检至少 5%～10%，发现问题立刻修。

### 多少条数据够用

没有绝对答案，看任务复杂度。经验值：

- **简单任务**（风格迁移、格式转换）：200～500 条能见效
- **中等任务**（领域问答、专业术语）：500～2000 条
- **复杂任务**（多步推理、强约束）：2000～5000 条甚至更多

可以先从 300～500 条起步，训一版看效果。如果欠拟合（模型学不会），加数据；如果过拟合（训练集完美、新问题拉胯），减数据或加正则。

---

## 四、实战：用 Unsloth 微调一个 Qwen 模型

Unsloth 是当前最省显存、速度最快的 LoRA 微调框架之一。同样配置下，比原生 Hugging Face 方案快 1.5～2 倍，显存省一半。下面用 Qwen2.5-4B 走一遍完整流程（4B 模型约需 10GB 显存，大多数单卡能跑）。

### 环境准备

```bash
pip install --upgrade unsloth unsloth_zoo
pip install "transformers>=4.40.0" "datasets" "trl" "peft" "accelerate"
```

建议 Python 3.10+，CUDA 11.8 或 12.x。有条件的可以用 Colab 免费 T4，4B 模型能跑。

### 加载模型

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # 先设小一点，省显存；确认能跑再往上加

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-4B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,   # 4B 用 bf16 即可，QLoRA 可选
    load_in_16bit=True,   # bf16 训练，显存友好
    full_finetuning=False,
)
```

### 配置 LoRA 参数

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA 秩，越大表达能力越强，但易过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,           # 一般 alpha = r
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 省显存
    random_state=3407,
    max_seq_length=max_seq_length,
)
```

**r** 是 LoRA 的秩，越大可调参数越多，效果可能更好，但也更容易过拟合。8～32 是常见范围，16 是个稳妥起点。

### 准备数据并训练

假设你已经有了 `train.jsonl`，每行是一个 `{"text": "..."}` 对象：

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files={"train": "train.jsonl"}, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,           # 先少跑几步试水
        logging_steps=1,
        output_dir="outputs_qwen",
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer.train()
```

`max_steps` 根据数据量调整。几百条数据，50～100 步通常够用；数据多可以加到 200～500。注意观察 loss，掉不下去可能是数据或学习率有问题。

### 测试效果

```python
FastLanguageModel.for_inference(model)

prompt = "将以下产品描述改写成客服话术：产品A，售价299元，支持7天无理由退货"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

对比微调前后的输出，看风格、格式是否符合预期。

---

## 五、部署微调后的模型

训完了，怎么用？常见三条路。

### 导出 GGUF，用 Ollama 加载

GGUF 是 llama.cpp 系的通用格式，Ollama、LM Studio 都支持。Unsloth 直接支持导出：

```python
model.save_pretrained_gguf("qwen_finetuned_gguf", tokenizer, quantization_method="q4_k_m")
```

`q4_k_m` 是 4-bit 量化，体积小、显存省；追求效果可以用 `q8_0` 或 `f16`。导出后得到 `.gguf` 文件，放到 Ollama 的模型目录，或通过 `ollama create` 写个 Modelfile 加载。

### 用 Ollama 跑

```bash
# 若已有 Modelfile
ollama create my-qwen -f Modelfile

# 运行
ollama run my-qwen
```

Ollama 会提供本地 API，很多应用改个 `base_url` 就能连。

### 合并权重做 API 服务

如果要做生产级 API，可以合并 LoRA 到基座，导出完整模型，再用 vLLM、TGI 等部署：

```python
model.save_pretrained_merged("qwen_finetuned_merged", tokenizer, save_method="merged_16bit")
```

合并后的模型就是标准 Hugging Face 格式，按常规流程部署即可。

---

## 六、微调的常见坑

### 过拟合

表现：训练集上对答如流，新问题一塌糊涂。原因多半是数据太少、训太久、或 **r** 设太大。

应对：控制 `max_steps`，加 `lora_dropout`（如 0.05～0.1），适当减小 **r**。有验证集的话，看验证 loss，开始上升就停。

### 数据泄露

训练数据里混进了测试用例，或者和业务数据高度重复，会导致"虚假的高分"。上线后一遇新数据就崩。

应对：严格划分训练/测试集，训练集不要包含任何你打算用来评估的样本。数据清洗时检查是否有重复、是否有未来信息。

### 灾难性遗忘

微调把模型"训偏"了，通用能力下降。比如原来会写代码，微调后代码能力变差。

应对：在训练数据里混入一定比例的通用任务（如通用问答、代码生成），比例不用高，10%～20% 往往能缓解。或者用更小的学习率、更少的步数，别训太狠。

---

## 七、微调 vs Prompt vs RAG：最终决策

三条路不互斥，而是分层配合。

**第一层：Prompt**。能说清楚的，先写 Prompt。零成本、秒生效。

**第二层：RAG**。知识在文档里、需要检索的，上 RAG。知识更新方便，不用动模型。

**第三层：微调**。风格、格式、术语、行为——Prompt 和 RAG 搞不定的，再上微调。成本高一点，但效果最稳。

实际项目里，很多是 **RAG + 微调**：RAG 负责"查知识"，微调负责"说话方式"和"输出格式"。两者结合，既能有领域知识，又能有统一风格。

---

## 结尾

LoRA 微调把"专属模型"的门槛拉低了一大截。一张消费级显卡、几百条高质量数据、一个下午的时间，就能训出一个懂你业务、说你的话的模型。关键是数据要干净、任务要想清楚、别一上来就全量微调。

从数据准备到部署，这条链路跑通一次，后面就是复制粘贴、调参优化的事了。祝你的专属模型早日上线。
