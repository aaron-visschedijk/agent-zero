"""
Tool system for the agentic AI framework.

Provides Tool and ToolRegistry classes for managing callable tools.
"""

from typing import Callable
from dataclasses import dataclass
import logging

from pydantic import BaseModel

from agent_zero.tool.exceptions import ToolConfigurationError, ToolExecutionError

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Represents a tool that can be called by the agent."""

    name: str
    description: str
    func: Callable[..., str | BaseModel]

    def execute(self, *args, **kwargs):
        """Execute the tool function."""
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logger.error("Error executing tool %s: %s", self.name, e)
            raise ToolExecutionError(f"Error executing tool {self.name}: {e}") from e


def function_tool(
    func: Callable[..., str | BaseModel],
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Create a tool from a function."""
    name = name or func.__name__
    description = description or func.__doc__
    if not description:
        raise ToolConfigurationError(f"Tool {func.__name__} has no description")
    return Tool(name=name, description=description, func=func)
