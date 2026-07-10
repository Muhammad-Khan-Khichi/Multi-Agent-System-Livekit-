"""
FAQ Agent — answers general questions about the restaurant using RAG.
Handles questions about hours, location, policies, allergens, etc.
"""

import logging

from agents import Agent, RunContext
from rag import search

logger = logging.getLogger("agents.faq")

FAQ_INSTRUCTIONS = """
You are the FAQ Agent for a restaurant chatbot.

Your job:
- Answer general questions about the restaurant (hours, location, parking,
  delivery, policies, allergens, etc.).
- Use the search results provided to you to give accurate answers.
- If you don't find relevant info, say you're not sure and offer to connect
  the user to a human.

Tone:
- Friendly, concise, and helpful.
- Keep answers short —1-3 sentences.

Handing back:
- If the user wants to make a reservation, order takeaway, or checkout,
  say "Let me connect you to the right agent." and return control to the
  greeter agent.
"""


async def faq_instructions(run_context: RunContext) -> str:
    """Dynamically inject search results into the FAQ agent's instructions."""
    user_msg = run_context.history[-1]["content"] if run_context.history else ""

    # Search across all knowledge sources
    results = search.search_all(user_msg, k=3)

    # Also check for allergen-specific queries
    allergen_info = ""
    if "allergen" in user_msg.lower() or "allergy" in user_msg.lower():
        # Try to extract item name from the message
        for word in user_msg.split():
            info = search.search_allergens(word.strip("?,.!"))
            if "contains" in info or "no known allergens" in info:
                allergen_info = info
                break

    # Build context from search results
    context_parts = []
    for r in results:
        source = r.get("source", "unknown")
        content = r.get("content", "")
        context_parts.append(f"[{source}] {content}")

    knowledge_context = "\n".join(context_parts) if context_parts else "No relevant information found."

    if allergen_info:
        knowledge_context += f"\n\n[allergens] {allergen_info}"

    return f"""{FAQ_INSTRUCTIONS}

--- Knowledge Base Results ---
{knowledge_context}
--- End Knowledge Base ---

Use the above information to answer the user's question. If the information
isn't relevant, say you're not sure.
"""


faq_agent = Agent(
    name="FAQ Agent",
    instructions=faq_instructions,
    model="gpt-4o-mini",
)