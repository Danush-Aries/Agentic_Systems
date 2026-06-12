"""Tests for ReActAgent and AgentPipeline."""

import pytest

from agentic_systems import ReActAgent, AgentPipeline, ToolRegistry
from agentic_systems.orchestrator.pipeline import PipelineStep
from agentic_systems.tools.builtin_tools import calculator, datetime_tool


def _make_agent(steps: int = 5, verbose: bool = False) -> ReActAgent:
    registry = ToolRegistry()
    registry.register(calculator, description="Evaluate math.")
    registry.register(datetime_tool, description="Get date/time.")
    return ReActAgent(
        name="TestAgent",
        tool_registry=registry,
        max_steps=steps,
        verbose=verbose,
    )


# ------------------------------------------------------------------
# ReActAgent
# ------------------------------------------------------------------

class TestReActAgent:
    def test_returns_agent_result(self):
        agent = _make_agent()
        result = agent.run("What is 2 + 2?")
        assert result is not None
        assert isinstance(result.output, str)
        assert result.steps >= 1

    def test_history_populated(self):
        agent = _make_agent()
        agent.run("What is 5 * 5?")
        assert len(agent.history) >= 2  # at least user + assistant

    def test_reset_clears_history(self):
        agent = _make_agent()
        agent.run("Compute something.")
        agent.reset()
        assert len(agent.history) == 0

    def test_custom_llm_client(self):
        """Replace the LLM with a simple stub that always answers directly."""
        def stub_llm(messages):
            return "Final Answer: 42"

        registry = ToolRegistry()
        registry.register(calculator, description="Math.")
        agent = ReActAgent(
            name="StubAgent",
            tool_registry=registry,
            llm_client=stub_llm,
        )
        result = agent.run("What is the meaning of life?")
        assert result.success is True
        assert result.output == "42"
        assert result.steps == 1

    def test_max_steps_exceeded(self):
        """An LLM that never produces a Final Answer should hit max_steps."""
        def looping_llm(messages):
            return (
                "Thought: I keep thinking.\n"
                "Action: calculator\n"
                'Action Input: {"expression": "1+1"}'
            )

        registry = ToolRegistry()
        registry.register(calculator, description="Math.")
        agent = ReActAgent(
            name="LoopAgent",
            tool_registry=registry,
            llm_client=looping_llm,
            max_steps=3,
        )
        result = agent.run("Loop forever.")
        assert result.success is False
        assert result.error == "max_steps_exceeded"
        assert result.steps == 3

    def test_unknown_tool_returns_error_observation(self):
        def bad_tool_llm(messages):
            if any("Observation:" in m.get("content", "") for m in messages):
                return "Final Answer: handled gracefully"
            return (
                "Thought: Use a nonexistent tool.\n"
                "Action: nonexistent_tool\n"
                'Action Input: {}'
            )

        registry = ToolRegistry()
        agent = ReActAgent(
            name="BadToolAgent",
            tool_registry=registry,
            llm_client=bad_tool_llm,
        )
        result = agent.run("Try a bad tool.")
        assert result.success is True  # agent recovered gracefully


# ------------------------------------------------------------------
# AgentPipeline
# ------------------------------------------------------------------

class TestAgentPipeline:
    def _make_pipeline(self) -> AgentPipeline:
        def answer_llm(messages):
            return "Final Answer: pipeline output"

        registry = ToolRegistry()
        a1 = ReActAgent("A1", tool_registry=registry, llm_client=answer_llm)
        a2 = ReActAgent("A2", tool_registry=registry, llm_client=answer_llm)

        pipeline = AgentPipeline()
        pipeline.add_step(PipelineStep(agent=a1, task_template="Step 1: {input}"))
        pipeline.add_step(PipelineStep(agent=a2, task_template="Step 2: {input}"))
        return pipeline

    def test_pipeline_runs_all_steps(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("start")
        assert result.success is True
        assert result.steps_run == 2

    def test_pipeline_fail_fast(self):
        """When an agent returns success=False, fail_fast pipeline stops early."""
        second_step_called = {"flag": False}

        def looping_llm(messages):
            # Always emits a tool call so max_steps is reached (not the
            # early-exit path) and AgentResult.success becomes False.
            return (
                "Thought: keep calling.\n"
                "Action: calculator\n"
                'Action Input: {"expression": "1+1"}'
            )

        def good_llm(messages):
            second_step_called["flag"] = True
            return "Final Answer: second step ran"

        registry = ToolRegistry()
        registry.register(calculator, description="Math.")

        failing_agent = ReActAgent(
            "Fail", tool_registry=registry,
            llm_client=looping_llm, max_steps=2,
        )
        good_agent = ReActAgent(
            "Good", tool_registry=registry,
            llm_client=good_llm,
        )

        pipeline = AgentPipeline(fail_fast=True)
        pipeline.add_step(PipelineStep(agent=failing_agent, task_template="{input}"))
        pipeline.add_step(PipelineStep(agent=good_agent, task_template="{input}"))

        result = pipeline.run("test")
        # failing_agent hits max_steps -> success=False -> pipeline stops
        assert result.success is False
        assert result.steps_run == 1
        assert not second_step_called["flag"]

    def test_pipeline_collects_step_results(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("hello")
        assert len(result.step_results) == 2
        for sr in result.step_results:
            assert "step" in sr
            assert "output" in sr
