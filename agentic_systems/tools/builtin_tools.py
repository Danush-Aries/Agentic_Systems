"""
Built-in tools that ship with the framework.

These can be used directly or serve as examples for writing custom tools.
"""

from __future__ import annotations

import ast
import datetime
import math
import operator
from typing import Any


# ------------------------------------------------------------------
# Calculator
# ------------------------------------------------------------------

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_FUNCS:
        return _SAFE_FUNCS[node.id]  # type: ignore[return-value]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name in _SAFE_FUNCS and callable(_SAFE_FUNCS[func_name]):
            args = [_safe_eval(a) for a in node.args]
            return _SAFE_FUNCS[func_name](*args)
    raise ValueError(f"Unsafe expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Parameters
    ----------
    expression:
        A math expression string, e.g. ``"2 ** 10 + sqrt(16)"``.

    Returns
    -------
    str
        The numeric result as a string.

    Raises
    ------
    ValueError
        If the expression contains unsafe or invalid syntax.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        # Round to avoid floating-point noise in display
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 10))
    except Exception as exc:
        raise ValueError(f"Cannot evaluate '{expression}': {exc}") from exc


# ------------------------------------------------------------------
# Mock web search
# ------------------------------------------------------------------

_MOCK_RESULTS = {
    "default": [
        "Result 1: Agentic systems are autonomous AI architectures where agents plan and execute tasks.",
        "Result 2: Multi-agent systems coordinate specialised agents to solve complex problems.",
        "Result 3: ReAct agents interleave reasoning steps with tool-use actions.",
    ]
}


def web_search_mock(query: str, num_results: int = 3) -> str:
    """
    Return mock web search results for a query (demo / offline use).

    Parameters
    ----------
    query:
        The search query string.
    num_results:
        How many results to return (max 5).
    """
    results = _MOCK_RESULTS.get(query.lower(), _MOCK_RESULTS["default"])
    results = results[: min(num_results, 5)]
    return "\n".join(results)


# ------------------------------------------------------------------
# Text summarizer
# ------------------------------------------------------------------

def text_summarizer(text: str, max_sentences: int = 3) -> str:
    """
    Produce a very simple extractive summary of ``text``.

    Splits the input into sentences and returns the first
    ``max_sentences`` as the summary.

    Parameters
    ----------
    text:
        The text to summarize.
    max_sentences:
        Number of sentences to keep.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:max_sentences])
    return summary or text[:300]


# ------------------------------------------------------------------
# Datetime tool
# ------------------------------------------------------------------

def datetime_tool(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Return the current date and time formatted as a string.

    Parameters
    ----------
    format:
        A ``strftime``-compatible format string.
    """
    return datetime.datetime.now().strftime(format)
