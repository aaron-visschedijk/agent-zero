"""OpenAI LLM client."""

import os
from typing import Self

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from llm_client.base import LLMClientBase
from llm_client.types import ChatMessage, Role


def _to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    """Convert a ChatMessage to an OpenAI chat completion message parameter."""
    if message.role == Role.USER:
        return ChatCompletionUserMessageParam(role="user", content=message.content)
    elif message.role == Role.ASSISTANT:
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=message.content
        )
    elif message.role == Role.SYSTEM:
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    raise ValueError(f"Invalid role: {message.role}")


class OpenAILLMClient(LLMClientBase):
    """OpenAI LLM client."""

    def __init__(self, api_key: str):
        """Initialize the OpenAI LLM client."""
        self.client = OpenAI(api_key=api_key)

    @classmethod
    def from_env(cls) -> Self:
        """Create an OpenAI LLM client from environment variables."""
        return cls(api_key=os.environ["OPENAI_API_KEY"])

    def chat(self, messages: list[ChatMessage]) -> str | None:
        """Chat with the LLM."""
        openai_messages = [_to_openai_message(message) for message in messages]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=openai_messages
        )
        return response.choices[0].message.content
