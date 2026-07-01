"""
Playground adapter around the Agentic Systems framework.

This wraps the existing ``ReActAgent`` (and friends) as a streaming, event-
producing iterator suitable for a WebSocket feed. It intentionally does not
modify the underlying framework — it observes it and re-emits its steps as
structured JSON events.

If wiring the real agent is not desired for a demo (e.g. no LLM key), the
``iter_events`` function will fall back to a deterministic mocked trace.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

# Import the framework lazily so the playground still boots even if the
# parent package layout changes.
try:
    from agentic_systems import ReActAgent, ToolRegistry  # type: ignore
    from agentic_systems.tools.builtin_tools import (  # type: ignore
        calculator,
        datetime_tool,
        web_search_mock,
        text_summarizer,
    )
    _FRAMEWORK_AVAILABLE = True
except Exception:  # noqa: BLE001
    _FRAMEWORK_AVAILABLE = False


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "math_solver": {
        "label": "Math Solver",
        "description": "A single ReAct agent equipped with the safe calculator tool.",
        "tools": ["calculator", "datetime_tool"],
        "example_task": "What is 2 ** 16 divided by 4?",
    },
    "researcher": {
        "label": "Researcher",
        "description": "Searches the (mock) web then summarises what it finds.",
        "tools": ["web_search_mock", "text_summarizer"],
        "example_task": "Give me a short summary of agentic AI systems.",
    },
    "assistant": {
        "label": "General Assistant",
        "description": "All built-in tools available. Best for open-ended tasks.",
        "tools": ["calculator", "datetime_tool", "web_search_mock", "text_summarizer"],
        "example_task": "What time is it right now?",
    },
}


@dataclass
class Event:
    """A single event emitted to the WebSocket client."""

    type: str  # "thought" | "action" | "observation" | "final" | "error" | "info"
    content: str
    ts: float

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "content": self.content, "ts": self.ts}


def list_templates() -> List[Dict[str, Any]]:
    return [{"id": tid, **tpl} for tid, tpl in TEMPLATES.items()]


async def iter_events(template_id: str, task: str) -> AsyncIterator[Event]:
    """
    Yield ReAct events for the given task.

    TODO: wire real agent — currently emits a mocked but plausible trace so
    the WebSocket contract can be exercised end-to-end without an LLM key.
    When ``_FRAMEWORK_AVAILABLE`` is True and a real ``llm_client`` is
    configured, replace the mocked block below with a callback-based hook
    into ``ReActAgent.run`` (e.g. by monkey-patching ``add_message`` to
    push events into an ``asyncio.Queue``).
    """
    template = TEMPLATES.get(template_id, TEMPLATES["assistant"])
    tools_str = ", ".join(template["tools"])

    yield Event("info", f"Loaded template '{template['label']}' with tools: {tools_str}", time.time())
    await asyncio.sleep(0.15)

    yield Event("thought", f"I received the task: {task!r}. Let me plan my approach.", time.time())
    await asyncio.sleep(0.4)

    primary_tool = template["tools"][0]
    yield Event("action", f"{primary_tool}(input={task!r})", time.time())
    await asyncio.sleep(0.4)

    yield Event(
        "observation",
        f"[mock] tool '{primary_tool}' returned a plausible result for task: {task}",
        time.time(),
    )
    await asyncio.sleep(0.3)

    yield Event(
        "thought",
        "I now have enough information from the tool call to answer confidently.",
        time.time(),
    )
    await asyncio.sleep(0.3)

    yield Event(
        "final",
        f"(mocked) Answer to: {task}\n\nThis is a demo trace. Wire a real LLM client "
        "into playground/backend/adapters.py::iter_events to see live agent output.",
        time.time(),
    )


def build_real_agent(template_id: str) -> Optional[Any]:
    """
    Build a real ReActAgent for the given template. Returned for callers that
    want to move beyond the mocked trace once an LLM client is available.
    """
    if not _FRAMEWORK_AVAILABLE:
        return None
    template = TEMPLATES.get(template_id, TEMPLATES["assistant"])
    registry = ToolRegistry()
    tool_map = {
        "calculator": calculator,
        "datetime_tool": datetime_tool,
        "web_search_mock": web_search_mock,
        "text_summarizer": text_summarizer,
    }
    for tname in template["tools"]:
        fn = tool_map.get(tname)
        if fn is not None:
            registry.register(fn, description=fn.__doc__ or tname)
    return ReActAgent(name=template["label"], tool_registry=registry, max_steps=8)
