#!/usr/bin/env python3
"""
Agentic Systems — CLI entry point.

Run built-in demos or interact with a ReAct agent from the terminal.

Usage
-----
    python main.py demo          # Run the built-in demonstration
    python main.py chat          # Start an interactive ReAct session
    python main.py pipeline      # Run a two-agent pipeline demo
    python main.py calc "2**10"  # Quick calculator tool call
"""

from __future__ import annotations

import argparse
import logging
import sys

from agentic_systems import ReActAgent, AgentPipeline, MemoryStore, ToolRegistry
from agentic_systems.orchestrator.pipeline import PipelineStep
from agentic_systems.tools.builtin_tools import (
    calculator,
    web_search_mock,
    text_summarizer,
    datetime_tool,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------
# Helper: build a standard registry
# ------------------------------------------------------------------

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(calculator, description="Evaluate a mathematical expression safely.")
    registry.register(web_search_mock, description="Search the web and return results.")
    registry.register(text_summarizer, description="Summarise a block of text.")
    registry.register(datetime_tool, description="Get the current date and time.")
    return registry


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

def run_demo() -> None:
    print("=" * 60)
    print("  Agentic Systems — Built-in Demo")
    print("=" * 60)

    registry = build_registry()
    agent = ReActAgent(
        name="DemoAgent",
        tool_registry=registry,
        verbose=True,
        max_steps=5,
    )

    tasks = [
        "What is 2 to the power of 16?",
        "What is the current date and time?",
        "Search the web for information about agentic systems.",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 40)
        result = agent.run(task)
        print(f"Answer: {result.output}")
        print(f"Steps:  {result.steps}  |  Success: {result.success}")
        agent.reset()


# ------------------------------------------------------------------
# Interactive chat
# ------------------------------------------------------------------

def run_chat() -> None:
    print("=" * 60)
    print("  Agentic Systems — Interactive ReAct Chat")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    registry = build_registry()
    memory = MemoryStore()
    agent = ReActAgent(
        name="ChatAgent",
        tool_registry=registry,
        memory=memory,
        verbose=False,
        max_steps=8,
    )

    while True:
        try:
            task = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not task:
            continue
        if task.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        result = agent.run(task)
        print(f"Agent: {result.output}")
        if not result.success:
            print(f"  [Warning: {result.error}]")


# ------------------------------------------------------------------
# Pipeline demo
# ------------------------------------------------------------------

def run_pipeline() -> None:
    print("=" * 60)
    print("  Agentic Systems — Two-Agent Pipeline Demo")
    print("=" * 60)

    registry = build_registry()

    search_agent = ReActAgent(
        name="Researcher",
        tool_registry=registry,
        verbose=False,
        max_steps=4,
    )
    summary_agent = ReActAgent(
        name="Summariser",
        tool_registry=registry,
        verbose=False,
        max_steps=4,
    )

    pipeline = AgentPipeline(verbose=True)
    pipeline.add_step(PipelineStep(
        agent=search_agent,
        task_template="Search the web for: {input}",
        name="Search",
    ))
    pipeline.add_step(PipelineStep(
        agent=summary_agent,
        task_template="Summarise the following in 2 sentences: {input}",
        name="Summarise",
    ))

    result = pipeline.run("agentic AI systems and their use cases")

    print(f"\nFinal Output:\n{result.final_output}")
    print(f"\nPipeline ran {result.steps_run} step(s) in {result.total_time:.2f}s")


# ------------------------------------------------------------------
# Quick calculator
# ------------------------------------------------------------------

def run_calc(expression: str) -> None:
    try:
        answer = calculator(expression)
        print(f"{expression} = {answer}")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Systems CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run the built-in demonstration")
    sub.add_parser("chat", help="Start an interactive ReAct session")
    sub.add_parser("pipeline", help="Run a two-agent pipeline demo")

    calc_p = sub.add_parser("calc", help="Evaluate a mathematical expression")
    calc_p.add_argument("expression", help='e.g. "2**10 + sqrt(16)"')

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "chat":
        run_chat()
    elif args.command == "pipeline":
        run_pipeline()
    elif args.command == "calc":
        run_calc(args.expression)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
