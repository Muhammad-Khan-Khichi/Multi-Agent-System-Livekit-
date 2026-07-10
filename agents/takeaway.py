from typing import Annotated

from livekit.agents import Agent, RunContext, inference
from livekit.agents.llm import function_tool
from pydantic import Field

from agents.base import BaseAgent
from config import VOICES
from tools import search_knowledge, to_greeter
from userData import UserData

RunContext_T = RunContext[UserData]


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are a takeaway agent that takes orders from the customer.\n"
                "Use the search_knowledge tool to look up menu items, prices, "
                "ingredients, or dietary options — do not guess or make up items.\n"
                "Clarify special requests and confirm the order with the customer."
            ),
            tools=[search_knowledge, to_greeter],
            tts=inference.TTS(model="cartesia/sonic-3", voice=VOICES["takeaway"]),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user create or update their order."""
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)