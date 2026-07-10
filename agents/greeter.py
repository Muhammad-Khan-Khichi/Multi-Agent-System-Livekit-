from livekit.agents import Agent, RunContext, inference
from livekit.agents.llm import function_tool

from agents.base import BaseAgent
from config import VOICES
from tools import search_knowledge
from userData import UserData

RunContext_T = RunContext[UserData]


class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are a friendly restaurant receptionist.\n"
                "If the caller asks about menu items, prices, ingredients, or dietary "
                "options, use the search_knowledge tool to look it up — do not guess.\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            llm=inference.LLM(
                model="openai/gpt-4.1-mini", extra_kwargs={"parallel_tool_calls": False}
            ),
            tools=[search_knowledge],
            tts=inference.TTS(model="cartesia/sonic-3", voice=VOICES["greeter"]),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)