"""Exceptions for the agent module."""

class AgentException(Exception):
    """Base exception for the agent module."""

class ToolNotAvailableError(AgentException):
    """Exception raised when a tool is not available."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name} is not available.")
