from typing import Annotated

from livekit.agents import Agent, RunContext, inference
from livekit.agents.llm import function_tool
from pydantic import Field

from agents.base import BaseAgent
from config import VOICES
from tools import to_greeter, update_name, update_phone
from userData import UserData
from utils import validate_credit_card, validate_expiry, validate_cvv

RunContext_T = RunContext[UserData]


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a checkout agent at a restaurant. The menu is: {menu}\n"
                "Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=inference.TTS(model="cartesia/sonic-3", voice=VOICES["checkout"]),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        if expense <= 0:
            return "The expense must be greater than zero. Please confirm the correct amount."

        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be ${expense:.2f}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card in MM/YY format")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        # Validate credit card number
        if not validate_credit_card(number):
            return (
                "That doesn't look like a valid credit card number. "
                "Please provide a valid card number."
            )

        # Validate expiry date
        if not validate_expiry(expiry):
            return (
                "The expiry date format is invalid. "
                "Please provide it in MM/YY format, e.g. 03/27."
            )

        # Validate CVV
        if not validate_cvv(cvv):
            return (
                "The CVV is invalid. It should be 3 or 4 digits. "
                "Please try again."
            )

        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card ending in {number[-4:]} has been updated successfully."

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)