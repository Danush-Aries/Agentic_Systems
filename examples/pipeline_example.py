"""
Example: Multi-agent pipeline — research then summarise.
"""

from agentic_systems import ReActAgent, AgentPipeline, ToolRegistry
from agentic_systems.orchestrator.pipeline import PipelineStep
from agentic_systems.tools.builtin_tools import web_search_mock, text_summarizer, datetime_tool


def main() -> None:
    registry = ToolRegistry()
    registry.register(web_search_mock, description="Search the web.")
    registry.register(text_summarizer, description="Summarise text.")
    registry.register(datetime_tool, description="Get current date/time.")

    researcher = ReActAgent("Researcher", tool_registry=registry)
    editor = ReActAgent("Editor", tool_registry=registry)

    pipeline = AgentPipeline(verbose=True)
    pipeline.add_step(PipelineStep(
        agent=researcher,
        task_template="Search for information about: {input}",
        name="Research",
    ))
    pipeline.add_step(PipelineStep(
        agent=editor,
        task_template="Summarise this into 2 clear sentences: {input}",
        name="Summarise",
    ))

    result = pipeline.run("multi-agent AI systems")
    print(f"\nFinal answer:\n{result.final_output}")
    print(f"Total time: {result.total_time:.2f}s | Steps: {result.steps_run}")


if __name__ == "__main__":
    main()
