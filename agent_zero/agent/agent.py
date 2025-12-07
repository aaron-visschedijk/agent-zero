"""Agent class"""

from pydantic import BaseModel
from agent_zero.agent.exceptions import ToolNotAvailableError
from agent_zero.agent.types import AgentOutput, ToolCall
from agent_zero.llm_client.base import LLMClientBase
from agent_zero.llm_client.types import ChatMessage, Role
from agent_zero.tool import Tool
from agent_zero.utils import try_parse_to_model


class Agent:
    """Agent class"""

    def __init__(
        self,
        name: str,
        llm_client: LLMClientBase,
        system_prompt: str,
        tools: list[Tool] | None = None,
        output_type: type[AgentOutput] = str,
        max_iterations: int = 10,
    ):
        """Initialize the agent"""
        self.name = name
        self.llm_client = llm_client
        self.agent_system_prompt = system_prompt
        self.tools: dict[str, Tool] = (
            {tool.name: tool for tool in tools} if tools else {}
        )
        self.structured_output = output_type
        self.messages: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system_prompt)
        ]
        self.max_iterations = max_iterations
        self.uses_structured_output = output_type is not str

    @property
    def system_prompt(self) -> str:
        """Get the complete system prompt for the agent."""

        tool_prompt = ""
        if self.tools:
            tool_prompt = f"""
            \n\n
            You can use the following tools:
            {'\n\n'.join(tool.get_tool_prompt() for tool in self.tools.values())}

            If you need to call a tool, return the tool name and parameter values as a JSON object.     
            """

        return f"""
        You are {self.name}. Your task is to adhere to the following instructions:
        {self.agent_system_prompt}{tool_prompt}
        """

    def run(self, query: str) -> AgentOutput:
        """Run the agent."""

        self.messages.append(ChatMessage(role=Role.USER, content=query))
        for _ in range(self.max_iterations):
            response = self.llm_client.chat(self.messages, self.structured_output)
            if response is None:
                raise ValueError("LLM returned None")
            if tool_call := try_parse_to_model(response, ToolCall):
                if tool_call.tool_name not in self.tools:
                    raise ToolNotAvailableError(tool_call.tool_name)
                tool = self.tools[tool_call.tool_name]
                tool_response = tool.execute(**tool_call.tool_parameters)
                self.messages.append(
                    ChatMessage(
                        role=Role.ASSISTANT,
                        content=f"Tool {tool_call.tool_name} called with parameters "
                        f"{tool_call.tool_parameters} returned {tool_response}",
                    )
                )
            elif (
                isinstance(self.structured_output, type)
                and issubclass(self.structured_output, BaseModel)
                and (
                    structured_response := try_parse_to_model(
                        response, self.structured_output
                    )
                )
            ):
                return structured_response
            else:
                return response
        raise ValueError(f"Agent reached max iterations: {self.max_iterations}")
