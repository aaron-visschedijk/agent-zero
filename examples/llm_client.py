"""Quick test of the llm client"""

import json
from dotenv import load_dotenv
from pydantic import BaseModel

from agent_zero.llm_client.openai import OpenAILLMClient
from agent_zero.llm_client.types import ChatMessage

# Load environment variables from .env file
load_dotenv()

client = OpenAILLMClient.from_env()


class MyResponse(BaseModel):
    """My response"""

    name: str
    age: int


response = client.chat(
    [
        ChatMessage(
            role="user",
            content="Who is the president of the United States and how old is he?",
        ),
    ],
    structured_output=MyResponse,
)
output_dict = json.loads(response)

print(output_dict)
print(output_dict["name"])
print(output_dict["age"])