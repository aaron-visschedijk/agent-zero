"""Unit tests for OpenAI LLM client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_zero.llm_client.openai import OpenAILLMClient, _to_openai_message
from agent_zero.llm_client.types import ChatMessage, Role


class TestToOpenAIMessage:
    """Tests for _to_openai_message helper function."""

    def test_user_role(self):
        """Test conversion of user role message."""
        message = ChatMessage(role=Role.USER, content="Hello")
        result = _to_openai_message(message)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_assistant_role(self):
        """Test conversion of assistant role message."""
        message = ChatMessage(role=Role.ASSISTANT, content="Hi there")
        result = _to_openai_message(message)
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"

    def test_system_role(self):
        """Test conversion of system role message."""
        message = ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant")
        result = _to_openai_message(message)
        assert result["role"] == "system"
        assert result["content"] == "You are a helpful assistant"

    def test_invalid_role(self):
        """Test that invalid role raises ValueError."""
        message = ChatMessage(role="invalid", content="test")
        with pytest.raises(ValueError, match="Invalid role: invalid"):
            _to_openai_message(message)


class TestOpenAILLMClient:
    """Tests for OpenAILLMClient class."""

    def test_init(self):
        """Test client initialization."""
        client = OpenAILLMClient(api_key="test-key")
        assert client.client is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"})
    def test_from_env_success(self):
        """Test creating client from environment variable."""
        client = OpenAILLMClient.from_env()
        assert client.client is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_key(self):
        """Test that missing environment variable raises KeyError."""
        with pytest.raises(KeyError, match="OPENAI_API_KEY"):
            OpenAILLMClient.from_env()

    @patch("agent_zero.llm_client.openai.OpenAI")
    def test_chat_success(self, mock_openai_class):
        """Test successful chat completion."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, how can I help?"

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        # Test
        client = OpenAILLMClient(api_key="test-key")
        messages = [ChatMessage(role=Role.USER, content="Hello")]
        result = client.chat(messages)

        # Verify
        assert result == "Hello, how can I help?"
        mock_client_instance.chat.completions.create.assert_called_once()
        call_args = mock_client_instance.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["role"] == "user"
        assert call_args.kwargs["messages"][0]["content"] == "Hello"

    @patch("agent_zero.llm_client.openai.OpenAI")
    def test_chat_none_content(self, mock_openai_class):
        """Test chat when response content is None."""
        # Setup mock response with None content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        # Test
        client = OpenAILLMClient(api_key="test-key")
        messages = [ChatMessage(role=Role.USER, content="Hello")]
        result = client.chat(messages)

        # Verify
        assert result is None

    @patch("agent_zero.llm_client.openai.OpenAI")
    def test_chat_multiple_messages(self, mock_openai_class):
        """Test chat with multiple messages."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client_instance

        # Test
        client = OpenAILLMClient(api_key="test-key")
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are helpful"),
            ChatMessage(role=Role.USER, content="Hello"),
            ChatMessage(role=Role.ASSISTANT, content="Hi"),
        ]
        result = client.chat(messages)

        # Verify
        assert result == "Response"
        call_args = mock_client_instance.chat.completions.create.call_args
        assert len(call_args.kwargs["messages"]) == 3
        assert call_args.kwargs["messages"][0]["role"] == "system"
        assert call_args.kwargs["messages"][1]["role"] == "user"
        assert call_args.kwargs["messages"][2]["role"] == "assistant"
