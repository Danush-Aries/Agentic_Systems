"""Tests for built-in tools and the tool registry."""

import pytest

from agentic_systems.tools.registry import ToolRegistry
from agentic_systems.tools.builtin_tools import (
    calculator,
    text_summarizer,
    datetime_tool,
    web_search_mock,
)


# ------------------------------------------------------------------
# Calculator
# ------------------------------------------------------------------

class TestCalculator:
    def test_addition(self):
        assert calculator("1 + 1") == "2"

    def test_multiplication(self):
        assert calculator("6 * 7") == "42"

    def test_power(self):
        assert calculator("2 ** 10") == "1024"

    def test_sqrt(self):
        assert calculator("sqrt(16)") == "4"

    def test_float_result(self):
        result = float(calculator("1 / 3"))
        assert abs(result - 0.333333) < 0.001

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            calculator("__import__('os').system('ls')")

    def test_modulo(self):
        assert calculator("10 % 3") == "1"


# ------------------------------------------------------------------
# Text summarizer
# ------------------------------------------------------------------

class TestTextSummarizer:
    def test_basic(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = text_summarizer(text, max_sentences=2)
        assert "Sentence one" in result
        assert "Sentence two" in result
        # Third sentence should be excluded
        assert "Sentence three" not in result

    def test_empty(self):
        result = text_summarizer("", max_sentences=3)
        assert result == ""

    def test_single_sentence(self):
        result = text_summarizer("Only one sentence.", max_sentences=5)
        assert result == "Only one sentence."


# ------------------------------------------------------------------
# datetime_tool
# ------------------------------------------------------------------

class TestDatetimeTool:
    def test_returns_string(self):
        result = datetime_tool()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_format(self):
        result = datetime_tool(format="%Y")
        assert result.isdigit()
        assert len(result) == 4


# ------------------------------------------------------------------
# Web search mock
# ------------------------------------------------------------------

class TestWebSearchMock:
    def test_returns_results(self):
        result = web_search_mock("agentic systems")
        assert len(result) > 0

    def test_num_results(self):
        result = web_search_mock("test", num_results=2)
        lines = result.strip().splitlines()
        assert len(lines) <= 2


# ------------------------------------------------------------------
# ToolRegistry
# ------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_call(self):
        registry = ToolRegistry()
        registry.register(calculator, description="Evaluate math.")
        result = registry.call("calculator", expression="3 + 3")
        assert result == "6"

    def test_decorator_syntax(self):
        registry = ToolRegistry()

        @registry.register(description="Double a number.")
        def double(x: int) -> int:
            return x * 2

        assert registry.call("double", x=5) == 10

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(calculator, description="Calc")
        tools = registry.list_tools()
        assert any(t["name"] == "calculator" for t in tools)

    def test_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.call("nonexistent")

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(calculator)
        assert "calculator" in registry
        assert "nonexistent" not in registry

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(calculator)
        assert len(registry) == 1
