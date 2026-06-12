"""Tool registry — register and call tools by name."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A simple registry that maps tool names to callable functions.

    Usage
    -----
    >>> registry = ToolRegistry()
    >>> @registry.register(description="Add two numbers")
    ... def add(a: int, b: int) -> int:
    ...     return a + b
    >>> registry.call("add", a=2, b=3)
    5
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Register a callable as a tool.  Can be used as a plain call or
        as a decorator (with or without arguments).

        Examples
        --------
        registry.register(my_func, description="Does X")

        @registry.register(description="Does X")
        def my_func(x: int) -> int: ...
        """

        def _wrap(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            tool_params = parameters or _infer_parameters(fn)
            self._tools[tool_name] = {
                "name": tool_name,
                "description": description or (fn.__doc__ or "").strip(),
                "func": fn,
                "parameters": tool_params,
            }
            logger.debug("Registered tool '%s'.", tool_name)
            return fn

        if func is not None:
            # Called as registry.register(fn, ...)
            return _wrap(func)
        # Called as @registry.register(...) decorator
        return _wrap

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a registered tool by name."""
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' is not registered.")
        fn = self._tools[tool_name]["func"]
        logger.debug("Calling tool '%s' with %s", tool_name, kwargs)
        return fn(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered tools (excluding callables)."""
        return [
            {k: v for k, v in tool.items() if k != "func"}
            for tool in self._tools.values()
        ]

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# ------------------------------------------------------------------
# Parameter inference helper
# ------------------------------------------------------------------

def _infer_parameters(fn: Callable) -> Dict[str, str]:
    """Build a simple {param_name: annotation_name} dict from a function."""
    sig = inspect.signature(fn)
    params: Dict[str, str] = {}
    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            params[param_name] = "any"
        else:
            params[param_name] = getattr(annotation, "__name__", str(annotation))
    return params
