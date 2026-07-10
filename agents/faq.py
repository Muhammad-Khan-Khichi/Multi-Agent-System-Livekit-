from __future__ import annotations

import logging

from livekit.agents import Agent, RunContext, inference
from livekit.agents.llm import function_tool

from agents.base import BaseAgent
from config import VOICES
from rag import search
from userData import UserData

logger = logging.getLogger("restaurant-example")

RunContext_T = RunContext[UserData]


class FAQAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the FAQ Agent for a restaurant.\n"
                "Your job is to answer general questions about the restaurant such as:\n"
                "- Hours, location, parking\n"
                "- Delivery info, policies (cancellation, refund, payment)\n"
                "- Allergen information\n"
                "- Vegetarian/vegan options\n"
                "Use the search_faq_knowledge tool to look up answers — do not guess.\n"
                "Keep answers short and friendly — 1 to 3 sentences.\n"
                "If the user wants to make a reservation, place an order, or checkout,\n"
                "use the to_greeter tool to transfer them back."
            ),
            llm=inference.LLM(
                model="openai/gpt-4.1-mini",
                extra_kwargs={"parallel_tool_calls": False},
            ),
            tools=[search_faq_knowledge],
            tts=inference.TTS(model="cartesia/sonic-3", voice=VOICES["faq"]),
        )

    @function_tool()
    async def to_greeter(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to make a reservation, place a takeaway
        order, or proceed to checkout. Transfers the user back to the greeter
        who will route them to the correct agent."""
        return await self._transfer_to_agent("greeter", context)


@function_tool()
async def search_faq_knowledge(context: RunContext_T, query: str) -> str:
    """Search the restaurant's FAQ, policies, and allergen database for
    information. Use this when the user asks about hours, location, parking,
    delivery, policies, allergens, or any general restaurant question.

    Args:
        query: The user's question in natural language.
    """
    results = search.search_all(query, k=3)

    if not results:
        return "No relevant information found."

    parts = []
    for r in results:
        source = r.get("source", "unknown")
        content = r.get("content", "")
        parts.append(f"[{source}] {content}")

    # Also check for allergen-specific queries
    query_lower = query.lower()
    if "allergen" in query_lower or "allergy" in query_lower or "gluten" in query_lower:
        for word in query.split():
            info = search.search_allergens(word.strip("?,.!"))
            if "contains" in info or "no known allergens" in info:
                parts.append(f"[allergens] {info}")
                break

    return "\n".join(parts) if parts else "No relevant information found."