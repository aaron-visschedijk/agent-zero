"""LLM client base class."""

from abc import ABC, abstractmethod
from typing import Self

from pydantic import BaseModel

from agent_zero.llm_client.types import ChatMessage


class LLMClientBase(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def __init__(self, api_key: str):
        """Initialize the LLM client."""

    @classmethod
    @abstractmethod
    def from_env(cls) -> Self:
        """Create an LLM client from environment variables."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        structured_output: type[BaseModel] | type[str] = str,
    ) -> str | None:
        """Chat with the LLM."""
