"""LLM client types."""

from enum import StrEnum

from pydantic import BaseModel


class Role(StrEnum):
    """Enum for conversation roles. Based on OpenAI's chat completion message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message for LLM clients."""

    role: str
    content: str
