"""LLM client base class."""

from abc import ABC, abstractmethod
from typing import Self

from llm_client.types import ChatMessage


class LLMClientBase(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def __init__(self, api_key: str):
        """Initialize the LLM client."""

    @abstractmethod
    @classmethod
    def from_env(cls) -> Self:
        """Create an LLM client from environment variables."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage]) -> str | None:
        """Chat with the LLM."""
