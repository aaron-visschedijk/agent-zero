"""Quick test of the llm client"""

from dotenv import load_dotenv

from agent_zero.llm_client.openai import OpenAILLMClient
from agent_zero.llm_client.types import ChatMessage

# Load environment variables from .env file
load_dotenv()

client = OpenAILLMClient.from_env()

response = client.chat(
    [
        ChatMessage(role="user", content="Hello, how are you?"),
    ]
)
print(response)
