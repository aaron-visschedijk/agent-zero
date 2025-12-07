"""Types for the agent"""

from typing import Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    """LLM output that indicates a tool call."""

    tool_name: str
    tool_parameters: dict[str, Any]


IntermediateAgentOutput = ToolCall | BaseModel | str
AgentOutput = BaseModel | str
