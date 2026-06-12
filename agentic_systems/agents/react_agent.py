"""
ReAct (Reasoning + Acting) agent implementation.

The agent loops through:
  Thought -> Action -> Observation -> ... -> Final Answer
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .base_agent import AgentResult, BaseAgent
from ..tools.registry import ToolRegistry
from ..memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a capable AI assistant operating in a ReAct loop.
For every task you MUST follow this exact format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with tool arguments>

When you have gathered enough information to answer, respond with:
Thought: I now know the final answer.
Final Answer: <your complete answer to the original task>

Available tools:
{tools}

Rules:
- Always reason before acting.
- Use only the tools listed above.
- Pass tool arguments as valid JSON.
- Never skip the Thought step.
"""


class ReActAgent(BaseAgent):
    """
    A ReAct-style agent that iterates Thought/Action/Observation cycles.

    This implementation uses an LLM client callable for generation. If no
    client is provided it falls back to a simple rule-based mock so the
    framework can be tested without API keys.

    Parameters
    ----------
    name:
        Human-readable agent name.
    tool_registry:
        A ToolRegistry instance containing available tools.
    llm_client:
        A callable ``(messages: list[dict]) -> str`` that returns the
        assistant's next turn. Defaults to a built-in mock.
    memory:
        Optional MemoryStore for cross-run persistence.
    max_steps:
        Maximum Thought/Action cycles before giving up.
    verbose:
        Whether to emit step-level log output.
    """

    def __init__(
        self,
        name: str = "ReActAgent",
        tool_registry: Optional[ToolRegistry] = None,
        llm_client: Optional[Callable[[List[Dict]], str]] = None,
        memory: Optional[MemoryStore] = None,
        max_steps: int = 10,
        verbose: bool = False,
    ):
        super().__init__(name=name, max_steps=max_steps, verbose=verbose)
        self.tools = tool_registry or ToolRegistry()
        self.llm = llm_client or _mock_llm
        self.memory = memory or MemoryStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Run the ReAct loop for the given task."""
        self.reset()
        context = context or {}

        # Recall relevant memories
        recalled = self.memory.search(task, top_k=3)
        context_text = ""
        if recalled:
            context_text = "Relevant context from memory:\n" + "\n".join(
                f"- {m}" for m in recalled
            )

        system_prompt = _SYSTEM_PROMPT.format(
            tools=self._format_tools()
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if context_text:
            messages.append({"role": "system", "content": context_text})

        user_content = task
        if context:
            user_content += "\n\nAdditional context: " + json.dumps(context)
        messages.append({"role": "user", "content": user_content})

        self.add_message("user", task)

        for step in range(self.max_steps):
            logger.debug("Step %d/%d", step + 1, self.max_steps)

            response = self.llm(messages)
            messages.append({"role": "assistant", "content": response})
            self.add_message("assistant", response)

            # Check for Final Answer
            final = _extract_final_answer(response)
            if final is not None:
                self.memory.add(f"Task: {task}\nAnswer: {final}")
                return AgentResult(
                    output=final,
                    success=True,
                    steps=step + 1,
                    messages=self.history,
                )

            # Parse action
            action_name, action_input = _extract_action(response)
            if action_name is None:
                # No valid action found — treat the whole response as answer
                return AgentResult(
                    output=response,
                    success=True,
                    steps=step + 1,
                    messages=self.history,
                )

            # Execute tool
            observation = self._execute_tool(action_name, action_input)
            obs_message = f"Observation: {observation}"
            messages.append({"role": "user", "content": obs_message})
            self.add_message("tool", obs_message, tool=action_name)

        # Exceeded max steps
        return AgentResult(
            output="Max steps reached without a final answer.",
            success=False,
            steps=self.max_steps,
            messages=self.history,
            error="max_steps_exceeded",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_tools(self) -> str:
        lines = []
        for tool in self.tools.list_tools():
            lines.append(f"- {tool['name']}: {tool['description']}")
            if tool.get("parameters"):
                lines.append(f"  Parameters: {json.dumps(tool['parameters'])}")
        return "\n".join(lines) if lines else "No tools available."

    def _execute_tool(self, name: str, input_data: Dict[str, Any]) -> str:
        try:
            result = self.tools.call(name, **input_data)
            return str(result)
        except KeyError:
            return f"Error: Tool '{name}' not found."
        except Exception as exc:  # noqa: BLE001
            return f"Error executing '{name}': {exc}"


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_action(text: str) -> tuple[Optional[str], Dict[str, Any]]:
    action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE)

    if not action_match:
        return None, {}

    action_name = action_match.group(1).strip()
    action_input: Dict[str, Any] = {}
    if input_match:
        try:
            action_input = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            action_input = {}

    return action_name, action_input


# ------------------------------------------------------------------
# Mock LLM (used when no real client is provided)
# ------------------------------------------------------------------

def _mock_llm(messages: List[Dict[str, Any]]) -> str:
    """
    A deterministic mock LLM for demos and tests.
    Reads the last user message and produces a canned ReAct response.
    """
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )

    # If this is an observation message, wrap up
    if last_user.startswith("Observation:"):
        observation = last_user[len("Observation:"):].strip()
        return (
            f"Thought: I have the result from the tool.\n"
            f"Final Answer: {observation}"
        )

    # Otherwise emit a default tool call
    return (
        "Thought: I should use the calculator to handle this request.\n"
        "Action: calculator\n"
        'Action Input: {"expression": "1 + 1"}'
    )
