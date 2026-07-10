from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from livekit.agents import Agent


@dataclass
class UserData:
    customer_name: str | None = None
    customer_phone: str | None = None

    reservation_time: str | None = None

    order: list[str] | None = None

    customer_credit_card: str | None = None
    customer_credit_card_expiry: str | None = None
    customer_credit_card_cvv: str | None = None

    expense: float | None = None
    checked_out: bool | None = None

    agents: dict[str, "Agent"] = field(default_factory=dict)
    prev_agent: "Agent | None" = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)