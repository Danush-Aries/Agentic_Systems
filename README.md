# Agentic Systems

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-lightgrey)

A lightweight, zero-dependency Python framework for building **multi-agent AI systems**. Implement the ReAct (Reasoning + Acting) pattern, compose agents into pipelines, register custom tools, and persist agent memory — all with clean, readable code.

---

## What it does

Agentic Systems gives you the building blocks to create autonomous AI agents that:

1. **Reason** about a task step-by-step (Thought).
2. **Act** by calling tools (Action / Action Input).
3. **Observe** results and continue until a final answer is reached.
4. **Remember** past interactions across runs via a memory store.
5. **Orchestrate** multiple agents in a sequential pipeline.

It is designed to work out of the box — no API keys required. Drop in any LLM client (OpenAI, Anthropic, local model) to replace the built-in mock.

---

## Features

- **ReAct agent** — full Thought/Action/Observation loop with configurable max steps
- **Tool registry** — register any Python callable as a named tool (decorator or direct call)
- **Built-in tools** — safe math calculator, text summariser, datetime tool, mock web search
- **Memory store** — in-process or file-backed memory with keyword-based retrieval
- **Agent pipeline** — chain agents sequentially; each output feeds the next agent
- **Fail-fast orchestration** — pipelines abort immediately on agent failure
- **Zero dependencies** — runs on the standard library alone (Python 3.9+)
- **Fully tested** — 37 unit tests covering every core component
- **CI/CD ready** — GitHub Actions workflow included

---

## Installation

```bash
git clone https://github.com/Dhanush-Aries/Agentic_Systems.git
cd Agentic_Systems

# No pip install needed — zero external dependencies.
# For development / running tests:
pip install -r requirements-dev.txt
```

---

## Usage

### Quick start — run the demo

```bash
python main.py demo
```

### Interactive chat session

```bash
python main.py chat
```

### Two-agent pipeline demo

```bash
python main.py pipeline
```

### Quick calculator

```bash
python main.py calc "2**10 + sqrt(16)"
# 1040
```

---

### Code: single ReAct agent

```python
from agentic_systems import ReActAgent, ToolRegistry
from agentic_systems.tools.builtin_tools import calculator, datetime_tool

# Register tools
registry = ToolRegistry()
registry.register(calculator, description="Evaluate a math expression.")
registry.register(datetime_tool, description="Get the current date/time.")

# Create and run an agent (uses built-in mock LLM by default)
agent = ReActAgent(
    name="MyAgent",
    tool_registry=registry,
    max_steps=8,
)

result = agent.run("What is 2 ** 16?")
print(result.output)   # 65536
print(result.steps)    # number of Thought/Action cycles used
print(result.success)  # True
```

### Code: connect a real LLM (OpenAI example)

```python
from openai import OpenAI
from agentic_systems import ReActAgent, ToolRegistry

client = OpenAI()

def openai_llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content

registry = ToolRegistry()
# ... register your tools ...

agent = ReActAgent(
    name="GPT4Agent",
    tool_registry=registry,
    llm_client=openai_llm,
)
result = agent.run("What is the capital of France?")
print(result.output)
```

### Code: register a custom tool

```python
from agentic_systems import ReActAgent, ToolRegistry

registry = ToolRegistry()

@registry.register(description="Get the weather for a city.")
def get_weather(city: str) -> str:
    # Replace with a real API call
    return f"Sunny, 22 degrees C in {city}."

agent = ReActAgent("WeatherBot", tool_registry=registry)
result = agent.run("What is the weather in Paris?")
```

### Code: multi-agent pipeline

```python
from agentic_systems import ReActAgent, AgentPipeline, ToolRegistry
from agentic_systems.orchestrator.pipeline import PipelineStep
from agentic_systems.tools.builtin_tools import web_search_mock, text_summarizer

registry = ToolRegistry()
registry.register(web_search_mock, description="Search the web.")
registry.register(text_summarizer, description="Summarise text.")

researcher = ReActAgent("Researcher", tool_registry=registry)
editor     = ReActAgent("Editor",     tool_registry=registry)

pipeline = AgentPipeline()
pipeline.add_step(PipelineStep(agent=researcher, task_template="Search for: {input}"))
pipeline.add_step(PipelineStep(agent=editor,     task_template="Summarise: {input}"))

result = pipeline.run("agentic AI systems")
print(result.final_output)
print(f"Completed in {result.total_time:.2f}s over {result.steps_run} steps")
```

### Code: persistent memory

```python
from agentic_systems import ReActAgent, MemoryStore, ToolRegistry

# Memory persists to disk between runs
memory = MemoryStore(persist_path="agent_memory.json")

agent = ReActAgent(
    name="MemoryAgent",
    tool_registry=ToolRegistry(),
    memory=memory,
)

agent.run("Remember that my favourite colour is blue.")
# On a later run the agent will recall this context when relevant
```

---

## Project structure

```
Agentic_Systems/
├── agentic_systems/
│   ├── agents/
│   │   ├── base_agent.py      # Abstract BaseAgent + AgentResult
│   │   └── react_agent.py     # ReAct loop implementation
│   ├── tools/
│   │   ├── registry.py        # ToolRegistry — register & call tools
│   │   └── builtin_tools.py   # calculator, web_search_mock, summariser, datetime
│   ├── memory/
│   │   └── memory_store.py    # In-memory / file-backed MemoryStore
│   └── orchestrator/
│       └── pipeline.py        # AgentPipeline + PipelineStep
├── examples/
│   ├── custom_tool_example.py
│   └── pipeline_example.py
├── tests/
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_tools.py
├── main.py                    # CLI entry point
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Agent pattern | ReAct (Reasoning + Acting) |
| LLM interface | Pluggable callable — works with any provider |
| Dependencies | Standard library only (zero pip installs for runtime) |
| Tests | pytest |
| CI | GitHub Actions |

---

## Extending the framework

- **Custom LLM**: pass any `(messages: list[dict]) -> str` callable as `llm_client`.
- **Custom tools**: decorate any function with `@registry.register(description="...")`.
- **Custom agents**: subclass `BaseAgent` and implement `run()`.
- **Parallel pipelines**: wrap `AgentPipeline` runs with `concurrent.futures` for fan-out patterns.

---

## License

MIT License
