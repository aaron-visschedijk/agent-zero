"""
Tool system for the agentic AI framework.

Provides Tool and ToolRegistry classes for managing callable tools.
"""

from typing import Callable, get_type_hints
from dataclasses import dataclass
import logging

from pydantic import BaseModel


from agent_zero.tool.exceptions import ToolExecutionError

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

    def get_tool_prompt(self) -> str:
        """Get the prompt for the tool."""
        tool_parameters = {
            param: type
            for param, type in get_type_hints(self.func).items()
            if param != "return"
        }
        return f"""
        tool_name: {self.name}
        tool_parameters: {tool_parameters}
        tool_description: {self.description}
        """


def function_tool(func=None, *, tool_name=None, tool_description=None):
    """
    Decorator to wrap a function into a Tool.
    Can be used with or without parameters.
    """

    def _wrap(f):
        name = tool_name if tool_name is not None else f.__name__
        description = tool_description if tool_description is not None else (f.__doc__ or "")
        return Tool(name=name, description=description, func=f)

    if func is None:
        # Used as @function_tool(...)
        return _wrap
    else:
        # Used as @function_tool
        return _wrap(func)
