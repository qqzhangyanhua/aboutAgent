# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese tutorial series on AI Agents and RAG (Retrieval-Augmented Generation) systems. The repository contains 15 articles with corresponding code examples demonstrating progressively advanced concepts from basic agents to production-ready systems.

**Structure:**
- `文章/` - 15 tutorial articles (numbered 01-15)
- `大纲/` - Outlines for articles 05-12
- `代码示例/` - Standalone runnable code examples

**Target Audience:** Chinese-speaking developers learning AI agent development and RAG systems from scratch.

## Technology Stack

- **Language:** Python 3.10+
- **LLM Provider:** DeepSeek API (OpenAI-compatible, can be swapped with OpenAI/Qwen/etc.)
- **Vector Databases:** ChromaDB (basic), FAISS, Qdrant (advanced)
- **Frameworks:** LangChain, LlamaIndex, LangGraph, CrewAI, AutoGen (used selectively)
- **Graph Processing:** NetworkX (for GraphRAG)
- **Evaluation:** RAGAS framework
- **Protocol:** MCP SDK (Model Context Protocol)

## Code Examples Architecture

### Agent Examples (`代码示例/01_从零开发一个AI智能体/`)
Three progressive versions demonstrating agent evolution:
- `v1_basic_agent.py` - Basic tool-calling loop
- `v2_react_agent.py` - ReAct pattern (Reasoning + Acting)
- `v3_reflection_agent.py` - Self-reflection mechanism

**Common Pattern:**
```python
# All examples use DeepSeek API with OpenAI SDK
client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
MODEL_NAME = "deepseek-chat"
```

### RAG Examples (`代码示例/02_从零实现最简RAG/`)
Three progressive versions showing RAG improvements:
- `v1_basic_rag.py` - Naive chunking + vector search
- `v2_improved_rag.py` - Overlapping chunks + similarity filtering
- `v3_rerank_rag.py` - Adds reranking for better relevance

**Dependencies:** `openai`, `chromadb`

### Advanced Examples (Root of `代码示例/`)
- `graph_rag_demo.py` - Complete GraphRAG implementation with entity extraction and knowledge graph
- `MCP_note_server.py` - MCP protocol server example
- `MCP_demo.py` - MCP client integration with OpenAI

## Running Code Examples

### Setup Environment
```bash
# Install dependencies for specific example
cd 代码示例/02_从零实现最简RAG
pip install -r requirements.txt

# Set API key (required for all examples)
export DEEPSEEK_API_KEY="your_key_here"
```

### Run Examples
```bash
# Agent examples
python 代码示例/01_从零开发一个AI智能体/v1_basic_agent.py

# RAG examples
python 代码示例/02_从零实现最简RAG/v1_basic_rag.py

# GraphRAG (requires additional dependencies)
pip install -r 代码示例/requirements_graphrag.txt
python 代码示例/graph_rag_demo.py
```

## Key Design Principles

### API Key Management
All examples read API keys from environment variables, never hardcoded:
```python
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
```

### Model Swapping
Code is designed to easily swap LLM providers by changing three parameters:
- `base_url` - API endpoint
- `api_key` - Authentication
- `MODEL_NAME` - Model identifier

### Minimal Dependencies
Each example uses the minimum required dependencies. No unnecessary framework bloat.

### Educational Focus
- Code is heavily commented in Chinese
- Each version builds incrementally on the previous
- Simulated APIs (weather, news) to avoid external dependencies during learning
- Clear separation between "what the agent thinks" and "what actually executes"

## Article Structure

Articles follow a consistent pattern:
1. **Problem Statement** - Why this technique matters
2. **Core Concepts** - Fundamental principles explained simply
3. **Progressive Implementation** - V1 → V2 → V3 with clear improvements
4. **Code Walkthrough** - Line-by-line explanation
5. **Pitfalls & Solutions** - Common mistakes and fixes

## Development Guidelines

### When Adding New Examples
1. Create versioned files (v1, v2, v3) showing progression
2. Include docstring with dependencies and run instructions
3. Use DeepSeek API as default (cost-effective for Chinese users)
4. Add corresponding `requirements.txt` if dependencies differ
5. Keep examples under 300 lines for readability

### When Modifying Existing Code
1. Maintain backward compatibility with article content
2. Preserve the educational progression (don't make v1 too complex)
3. Keep Chinese comments intact
4. Test with `DEEPSEEK_API_KEY` environment variable

### Code Style
- Use descriptive Chinese variable names for domain concepts (e.g., `工具箱`, `记忆`)
- Use English for technical terms (e.g., `client`, `embedding`)
- Keep functions short and focused
- Avoid abstractions that obscure the learning point

## Common Tasks

### Test a Code Example
```bash
cd 代码示例/02_从零实现最简RAG
export DEEPSEEK_API_KEY="sk-xxx"
python v1_basic_rag.py
```

### Add a New Article Code Example
```bash
# Create directory with article number
mkdir -p 代码示例/XX_新主题
cd 代码示例/XX_新主题

# Create versioned examples
touch v1_basic.py v2_improved.py v3_advanced.py
echo "openai\nchromadb" > requirements.txt
```

### Verify All Examples Run
```bash
# Check all Python files for syntax errors
find 代码示例 -name "*.py" -exec python -m py_compile {} \;
```

## Important Notes

- **No Git Hooks:** This is a documentation/tutorial repo, not a production codebase
- **No Tests:** Examples are meant to be run interactively, not unit tested
- **Chinese-First:** All documentation, comments, and variable names prioritize Chinese readability
- **API Costs:** Examples use DeepSeek to minimize costs for learners (¥1/million tokens vs OpenAI's $15/million)
- **Standalone Examples:** Each code file should run independently without importing from other examples
