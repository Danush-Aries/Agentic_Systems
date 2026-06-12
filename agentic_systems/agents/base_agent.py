"""Base agent interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Represents a message in the agent's conversation history."""
    role: str  # "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResult:
    """Result returned from an agent run."""
    output: str
    success: bool
    steps: int
    messages: List[AgentMessage] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclasses must implement the `run` method, which receives a task string
    and returns an AgentResult.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        max_steps: int = 10,
        verbose: bool = False,
    ):
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.verbose = verbose
        self._history: List[AgentMessage] = []

    @abstractmethod
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute a task and return a result."""

    def reset(self) -> None:
        """Clear the agent's conversation history."""
        self._history.clear()
        logger.debug("Agent '%s' history cleared.", self.name)

    def add_message(self, role: str, content: str, **metadata) -> None:
        msg = AgentMessage(role=role, content=content, metadata=metadata)
        self._history.append(msg)
        if self.verbose:
            logger.info("[%s] %s: %s", self.name, role.upper(), content[:200])

    @property
    def history(self) -> List[AgentMessage]:
        return list(self._history)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
