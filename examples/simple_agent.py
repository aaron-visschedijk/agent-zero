"""Simple agent example"""


from dotenv import load_dotenv
from agent_zero.agent import Agent
from agent_zero.llm_client.openai import OpenAILLMClient
from agent_zero.tool import function_tool


load_dotenv()

@function_tool
def add(a: float, b: float) -> str:
    """Add two numbers together"""
    print(f"Adding {a} and {b}")
    return str(a + b)

agent = Agent(
    name="simple_agent",
    llm_client=OpenAILLMClient.from_env(),
    system_prompt="You are a simple agent that can answer questions.",
    tools=[add],
    output_type=str,
    max_iterations=10,
)

response = agent.run("what is 13765765 + 2976987676?")
print(response)
