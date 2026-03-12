# Ollama 从入门到实战：本地跑大模型的最简方案

想在自己电脑上跑大模型？不用配 CUDA，不用写 Docker，一行命令搞定。对，你没听错，就是这么简单。

以前折腾过本地部署的人都知道，装个环境能劝退一大半人：Python 版本、CUDA 驱动、显存不够、依赖冲突……一套流程走下来，模型还没跑起来，人先麻了。Ollama 的出现，相当于把"开飞机"变成了"骑电动车"——门槛直接砍到脚踝。

---

## 一、为什么要在本地跑大模型

先说清楚一件事：**不是所有场景都需要本地部署**。ChatGPT、Claude 这些云端服务，用着省心，能力也强。但本地跑模型，有四个云端替代不了的刚需。

**隐私**。公司内部文档、个人隐私数据、还没公开的创意，你愿意扔到别人的服务器上吗？本地跑，数据不出你的电脑，这是底线。

**免费**。API 按 token 计费，用量一大，账单看着肉疼。本地模型跑起来之后，电费之外没有边际成本，想怎么调就怎么调，不用盯着余额发愁。

**离线可用**。出差、通勤、网络不稳的时候，云端 API 一断就歇菜。本地模型插上电就能跑，不依赖网络，适合对稳定性要求高的场景。

**学习**。想搞懂大模型怎么工作、Prompt 怎么调、RAG 怎么搭，光看文档不够，得亲手玩。本地跑起来，随时实验，成本可控，这是最好的学习方式。

---

## 二、Ollama 是什么

一句话：**Ollama 是"大模型界的 Docker"**。

Docker 把应用和依赖打包成镜像，一条命令就能在任何机器上跑起来，不用管底层环境。Ollama 干的是同样的事——把大模型打包成可执行格式，一条命令拉下来、跑起来，不用你操心 CUDA、量化、推理框架这些破事。

它帮你做了三件事：**模型下载和管理**（类似 `docker pull`）、**本地推理服务**（模型加载到内存、提供 API）、**统一接口**（不管底层是 Llama 还是 Mistral，调用方式一样）。

官方支持 Mac、Linux、Windows，**Mac 上尤其友好**——Apple Silicon 芯片原生优化，M1/M2/M3 跑 7B 模型流畅得很。没有独显的笔记本也能玩，靠 CPU 和内存硬跑，慢一点但能跑。

---

## 三、安装和第一次运行

安装比你想的还简单。

**Mac**：去官网 https://ollama.com 下载安装包，拖进应用程序文件夹，完事。或者用 Homebrew：

```bash
brew install ollama
```

**Linux**：一行脚本搞定：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**：下载 `OllamaSetup.exe` 双击安装，不需要管理员权限，装完会自动加到系统托盘。

---

装好之后，打开终端，输入：

```bash
ollama run llama3.2
```

第一次运行会自动下载模型（Llama 3.2 3B 大约 2GB），下完直接进入对话界面。你打字，它回复，跟 ChatGPT 聊天一样。输入 `/bye` 退出。

就这么简单。**没有配置文件，没有环境变量，没有"请先安装 CUDA 12.3"**。一行命令，从零到跑起来。

---

## 四、模型选择指南

Ollama 支持几十个模型，选哪个取决于你的**显存/内存**。下面按硬件分档，照着选就行。

**8GB 内存（或 8GB 显存）**：老老实实用小模型。**Llama 3.2 3B**、**Phi-3 mini**、**Qwen2.5:0.5B** 都能流畅跑，适合聊天、简单问答、轻量代码辅助。别指望复杂推理，但日常够用。

**16GB 内存**：这是最常见的配置。**Llama 3.2**（7B/8B 量化版）、**Mistral 7B**、**Qwen2.5 7B**、**DeepSeek-Coder 6.7B** 都能跑。7B 模型是性价比之王，中英文都不错，代码能力也够用。显存紧张就用 `:7b-q4_0` 这类量化版本，体积砍半，效果差不了多少。

**24GB 及以上**：可以上 **Llama 3.1 70B**、**Qwen2.5 32B** 这类大家伙。推理质量明显提升，但速度会慢，适合对质量要求高、不赶时间的场景。

**实用建议**：先拉一个小模型试水，确认能跑起来，再按需升级。拉模型用 `ollama pull <模型名>`，比如 `ollama pull qwen2.5:7b`。查看已安装的模型用 `ollama list`。

---

## 五、API 调用——像用 OpenAI 一样用本地模型

Ollama 默认在 `http://localhost:11434` 起一个 HTTP 服务，**兼容 OpenAI 的 API 格式**。也就是说，你原来用 OpenAI SDK 写的代码，改个 `base_url` 就能切到本地模型。

用 curl 直接调：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "用一句话解释什么是递归"}]
  }'
```

返回的是标准 JSON，`choices[0].message.content` 里就是模型回复。

**关键点**：Ollama 的 API 路径是 `/v1/chat/completions`，和 OpenAI 一模一样。所以任何支持 OpenAI 格式的库——**OpenAI Python SDK**、**LangChain**、**LlamaIndex**——都能无缝对接，只需要把 `base_url` 改成 `http://localhost:11434/v1`，`api_key` 随便填个字符串（比如 `ollama`）就行，本地不校验。

---

## 六、实战：本地模型接入 Python 项目

下面是一段最小可用的 Python 代码，用 **OpenAI 官方 SDK** 调 Ollama：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 本地不校验，随便填
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "写一个 Python 函数，计算斐波那契数列第 n 项"}]
)

print(response.choices[0].message.content)
```

先 `pip install openai`，确保 Ollama 在后台跑着（`ollama serve` 或直接 `ollama run` 过一次就会常驻），然后执行这段代码，就能拿到模型输出。

**流式输出**也支持，加个 `stream=True`：

```python
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "讲个冷笑话"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

这样回复会一个字一个字蹦出来，体验跟 ChatGPT 一样。你的项目如果本来就用 OpenAI SDK，切到 Ollama 就是改两行配置的事。

---

## 七、Ollama 进阶玩法

基础用熟了，可以玩点花的。

**自定义 Modelfile**。Modelfile 相当于模型的"配方"，可以基于现有模型改参数、加 System Prompt、调 temperature。比如你想让模型永远用第二人称"你"来回答，可以写：

```
FROM llama3.2
SYSTEM "你是一个友好的助手，回答时始终使用'你'来称呼用户。"
PARAMETER temperature 0.7
```

保存为 `Modelfile`，然后执行：

```bash
ollama create my-llama -f Modelfile
```

就得到了一个叫 `my-llama` 的专属模型，用 `ollama run my-llama` 启动。

**多模型切换**。Ollama 支持同时拉多个模型，用 `ollama run <模型名>` 切换。比如聊天用 `llama3.2`，写代码用 `deepseek-coder`，看需求随时换。内存够的话，可以预加载几个常用模型，切换几乎无感。

**模型分享**。自定义的 Modelfile 可以推到 GitHub，别人 `ollama create` 一下就能复现。也可以把模型推到 Ollama 官方 Hub，类似 Docker Hub，团队内部共享很方便。

---

## 八、性能调优和常见问题

**跑得慢怎么办**？优先检查：是不是模型太大了（换小一档的量化版）、后台有没有其他吃内存的程序、Mac 上有没有插电源（省电模式会降频）。`ollama ps` 可以看当前加载的模型和占用。

**显存/内存不足**？用量化版本，比如 `llama3.2:7b-q4_0`，体积小一半，效果损失不大。8GB 显存建议别超过 7B 模型，16GB 可以试试 13B 的量化版。

**连不上 localhost:11434**？确认 Ollama 在跑：Mac 看菜单栏有没有 Ollama 图标，Linux/Windows 看任务栏。没有就手动执行 `ollama serve`。端口被占用的话，可以改环境变量 `OLLAMA_HOST` 换端口。

**模型下载慢**？默认从官方拉，国内可能慢。可以配置镜像，或者用 `ollama pull` 时指定带镜像的地址，具体看官方文档的镜像配置说明。

---

## 结尾

Ollama 的价值，不是替代云端大模型，而是**把本地跑模型的门槛打穿**。以前只有搞 ML 的人能玩的东西，现在普通人一条命令就能上手。隐私、免费、离线、学习——这四个需求但凡沾一个，都值得试试。

装好 Ollama，拉一个小模型，跑通第一段 API 调用代码，你就入门了。剩下的，就是按需换模型、调参数、接自己的项目。大模型没那么神秘，动手玩一玩，比看十篇教程都管用。
