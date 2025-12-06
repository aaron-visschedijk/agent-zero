"""Tool exceptions."""


class ToolError(Exception):
    """Base class for tool errors."""

class ToolConfigurationError(ToolError):
    """Error raised when a tool is configured incorrectly."""

class ToolExecutionError(ToolError):
    """Error raised when a tool execution fails."""
