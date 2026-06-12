"""
Agent pipeline — run a sequence of agents where each agent's output
feeds into the next agent's task.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """
    A single step in an AgentPipeline.

    Parameters
    ----------
    agent:
        The agent to execute.
    task_template:
        A string that may contain ``{input}`` to receive the previous
        step's output, or a static task string.
    transform:
        Optional callable to post-process the agent result before
        passing it to the next step.
    """

    agent: BaseAgent
    task_template: str = "{input}"
    transform: Optional[Callable[[AgentResult], str]] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = self.name or self.agent.name


@dataclass
class PipelineResult:
    """Aggregated result from an entire pipeline run."""

    final_output: str
    success: bool
    steps_run: int
    total_time: float
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class AgentPipeline:
    """
    Orchestrates a sequential chain of agents.

    Each agent processes a task derived from the previous agent's output.
    The pipeline fails fast if any step is unsuccessful and
    ``fail_fast=True``.

    Parameters
    ----------
    steps:
        Ordered list of PipelineStep objects.
    fail_fast:
        Abort the pipeline on the first agent failure (default True).
    verbose:
        Emit step-level log messages.
    """

    def __init__(
        self,
        steps: Optional[List[PipelineStep]] = None,
        fail_fast: bool = True,
        verbose: bool = False,
    ) -> None:
        self.steps: List[PipelineStep] = steps or []
        self.fail_fast = fail_fast
        self.verbose = verbose

    def add_step(self, step: PipelineStep) -> "AgentPipeline":
        """Append a step and return self (fluent API)."""
        self.steps.append(step)
        return self

    def run(self, initial_input: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Execute all steps sequentially."""
        start = time.time()
        current_input = initial_input
        step_results: List[Dict[str, Any]] = []
        context = context or {}

        for i, step in enumerate(self.steps):
            task = step.task_template.format(input=current_input)
            logger.info(
                "[Pipeline] Step %d/%d — '%s': %s",
                i + 1, len(self.steps), step.name, task[:120],
            )

            result = step.agent.run(task, context=context)
            step_results.append({
                "step": step.name,
                "task": task,
                "output": result.output,
                "success": result.success,
                "agent_steps": result.steps,
            })

            if not result.success and self.fail_fast:
                logger.error("Pipeline aborted at step '%s': %s", step.name, result.error)
                return PipelineResult(
                    final_output=result.output,
                    success=False,
                    steps_run=i + 1,
                    total_time=time.time() - start,
                    step_results=step_results,
                    error=result.error,
                )

            # Apply optional transform; otherwise use raw output
            if step.transform:
                current_input = step.transform(result)
            else:
                current_input = result.output

        return PipelineResult(
            final_output=current_input,
            success=True,
            steps_run=len(self.steps),
            total_time=time.time() - start,
            step_results=step_results,
        )
