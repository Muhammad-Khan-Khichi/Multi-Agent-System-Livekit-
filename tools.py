from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from livekit.agents import Agent, RunContext
from livekit.agents.llm import function_tool
from pydantic import Field

from rag.search import search_menu as _search_menu
from rag.search import search_all as _search_all
from rag.search import search_allergens as _search_allergens
from userData import UserData
from utils import validate_phone, validate_name

if TYPE_CHECKING:
    from agents.base import BaseAgent

logger = logging.getLogger("restaurant-example")

RunContext_T = RunContext[UserData]


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    if not validate_name(name):
        return "That doesn't look like a valid name. Please provide your name again."

    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    if not validate_phone(phone):
        return "That doesn't look like a valid phone number. Please try again."

    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> tuple[Agent, str]:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: "BaseAgent" = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


def _format_result(result: dict) -> str:
    """Format a single search result into a readable string,
    handling different source types (menu, faq, policies)."""
    source = result.get("source", "")
    meta = result.get("metadata", {})

    if source == "menu":
        name = meta.get("name", "Unknown")
        price = meta.get("price", 0)
        category = meta.get("category", "")
        description = meta.get("description", "")
        tags = meta.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"{name} (${price:.2f}, {category}) — {description}{tag_str}"

    elif source == "faq":
        question = meta.get("question", "")
        answer = meta.get("answer", "")
        return f"Q: {question}\nA: {answer}"

    elif source == "policies":
        topic = meta.get("topic", "")
        content = meta.get("content", "")
        return f"Policy — {topic}: {content}"

    else:
        # Fallback: just return the content field
        return result.get("content", "")


@function_tool()
async def search_knowledge(
    query: Annotated[
        str,
        Field(
            description=(
                "Natural-language question about the menu, e.g. "
                "'do you have anything vegetarian', 'what desserts do you have', "
                "'something under $6'"
            )
        ),
    ],
    context: RunContext_T,
) -> str:
    """Called whenever the user asks about menu items, ingredients, prices,
    categories, or dietary options. Use this instead of guessing from memory."""
    try:
        results = _search_menu(query, k=3)
    except Exception as e:
        logger.error(f"Menu search failed: {e}")
        return "Sorry, I can't look up the menu right now."

    if not results:
        return "I couldn't find anything matching that on the menu."

    lines = [_format_result(item) for item in results]
    return "Here's what I found:\n" + "\n".join(lines)


@function_tool()
async def search_faq_knowledge(
    query: Annotated[
        str,
        Field(
            description=(
                "Natural-language question about restaurant policies, "
                "FAQ, hours, parking, cancellation, refund, etc."
            )
        ),
    ],
    context: RunContext_T,
) -> str:
    """Called whenever the user asks about restaurant policies, FAQ,
    hours, parking, cancellation, refund, or other general questions.
    Use this instead of guessing from memory."""
    try:
        results = _search_all(query, k=5)
    except Exception as e:
        logger.error(f"FAQ search failed: {e}")
        return "Sorry, I can't look up that information right now."

    if not results:
        return "I couldn't find any information about that."

    lines = [_format_result(item) for item in results]
    return "Here's what I found:\n" + "\n".join(lines)


@function_tool()
async def search_allergen_info(
    item_name: Annotated[
        str,
        Field(description="The name of the menu item to check allergens for"),
    ],
    context: RunContext_T,
) -> str:
    """Called when the user asks about allergens, dietary restrictions,
    or ingredients in a specific menu item."""
    try:
        result = _search_allergens(item_name)
        return result
    except Exception as e:
        logger.error(f"Allergen search failed: {e}")
        return "Sorry, I can't look up allergen information right now."