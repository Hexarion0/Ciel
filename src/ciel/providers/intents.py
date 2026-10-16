"""Exact local clock intents; compound requests still go to the model."""

import re
from datetime import datetime


def answer_clock_query(text: str, now: datetime | None = None) -> str | None:
    """Answer standalone time/date questions without a provider request."""
    clean = text.casefold().replace("’", "'").strip(" .!?")
    clean = re.sub(r"^(?:so[, ]+|please\s+)", "", clean)
    clean = re.sub(r"\s+(?:please|right now)$", "", clean)
    clock = now or datetime.now()
    if re.fullmatch(
        r"(?:what(?:'s| is) (?:the )?(?:current )?time(?: now)?|"
        r"what time is it(?: now)?|tell me (?:the )?time)",
        clean,
    ):
        return f"It's {clock.strftime('%I:%M %p').lstrip('0')}."
    if re.fullmatch(
        r"(?:what(?:'s| is) (?:the )?(?:current )?date(?: today)?|"
        r"what day is it(?: today)?|what(?:'s| is) today's date)",
        clean,
    ):
        return f"It's {clock.strftime('%A, %B')} {clock.day}, {clock.year}."
    return None
