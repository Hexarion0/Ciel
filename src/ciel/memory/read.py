"""Read-only answers for memory capabilities, confirmed timelines, and name corrections."""

import re
from datetime import date, datetime
from typing import Literal

from ciel.memory.store import MemoryStore
from ciel.providers.base import ChatMessage

_CAPABILITY = re.compile(r"^do you (?:have|use) (?:a )?(?:memory|memory system)[.!? ]*$", re.I)
_OVERVIEW = re.compile(
    r"^(?:what do you (?:remember|know) about me|"
    r"tell me what you (?:remember|know) about me|what have you saved about me|"
    r"what personal details have you saved about me)[.!? ]*$", re.I,
)
_PROFILE_KEYS = (
    "preferred_name", "favorite_game", "current_project", "goal", "occupation",
    "home_city", "timezone", "preferred_language", "response_style", "favorite_food",
    "favorite_color", "favorite_drink", "favorite_movie", "favorite_language",
)
_PROFILE_SOURCES = {"user_explicit", "user_confirmed", "user_statement"}
_PAUSED = "Saved-memory recall is paused. Say 'resume memory recall' to use saved facts."
_CONTROL = re.compile(
    r"^(?:please\s+)?(?:pause|disable|resume|enable)\s+(?:saved[- ]memory|memory)\s+"
    r"recall[.!? ]*$", re.I,
)
_STATUS = re.compile(
    r"^(?:is (?:saved[- ]memory|memory) recall (?:paused|enabled|disabled|on|off)|"
    r"(?:what(?:'s| is) (?:my |your |the )?)?memory status)[.!? ]*$", re.I,
)
_PROFILE_READ = re.compile(
    r"^(?:(?:what(?:'s| is)|which is|do you (?:remember|know))\s+my "
    r"(?:current project|active project|goal|occupation|job|home city|time\s?zone|"
    r"preferred language|response style|name|"
    r"(?:favou?rite|fav) (?:game|food|color|colour|drink|movie|language))|"
    r"what do you call me)[.!? ]*$",
    re.I,
)
_TIMELINE = re.compile(
    r"^(?:what development day (?:is it|are we on)|"
    r"what day (?:is it|are we on) (?:of |in )?(?:the )?development|"
    r"(?:what(?:'s| is)\s+)?(?:the |our |your )?development (?:day|date)|"
    r"(?:(?:no|actually)[, ]+)*i meant\s+(?:the |our |your )?development (?:day|date))"
    r"[.!? ]*$", re.I,
)
_MEMORY_SOURCE = re.compile(
    r"^(?:it(?:'s| is) in your (?:saved )?memor(?:y|ies)|"
    r"(?:please )?(?:check|look in|use) your (?:saved )?memor(?:y|ies))[.!? ]*$", re.I,
)
_NAME = r"[^\W\d_][\w'-]*(?:\s+[^\W\d_][\w'-]*){0,3}"
_RENAMES = (
    re.compile(
        rf"^(?:i meant\s+)?(?:it|your name)\s+was\s+(?P<old>{_NAME})"
        r"(?:,\s*your name)?[,]?\s+but\s+i\s+(?:changed|renamed)\s+"
        r"(?:it|you|your name)\s+to\s+ciel[.! ]*$", re.I,
    ),
    re.compile(
        rf"^i\s+(?:renamed you|changed your name)\s+from\s+(?P<old>{_NAME})"
        r"\s+to\s+ciel[.! ]*$", re.I,
    ),
)
_FORMER_NAME = re.compile(
    r"^(?:(?:yes|yeah|no|actually)(?:,?\s+it is)?[, ]+)?(?:but\s+)?"
    rf"(?:it|your name)\s+(?:was|used to be)\s+(?P<old>{_NAME})"
    r"(?:,?\s+okay)?[.!? ]*$", re.I,
)


def _explicit_rename(text: str) -> re.Match[str] | None:
    return next((match for pattern in _RENAMES if (match := pattern.fullmatch(text))), None)


def _previous_name(text: str, recent_messages: list[ChatMessage]) -> str | None:
    """Accept a current declarative correction, resolving 'it' only from the latest user turn."""
    text = text.replace("’", "'")
    match = _explicit_rename(text)
    if match is None:
        previous_user = next(
            (message.content.strip().replace("’", "'") for message in reversed(recent_messages)
             if message.role == "user"), "",
        )
        if _explicit_rename(previous_user) is not None:
            match = _FORMER_NAME.fullmatch(text)
    return match.group("old").strip() if match is not None else None


def memory_read_intent(
    text: str, recent_messages: list[ChatMessage] | None = None,
) -> Literal["capability", "timeline", "name_history", "recall", "control", "status",
             "overview"] | None:
    """Recognize complete read-only turns without interpreting commands or compound requests."""
    clean = text.strip().replace("’", "'")
    if _CONTROL.fullmatch(clean):
        return "control"
    if _STATUS.fullmatch(clean):
        return "status"
    if _OVERVIEW.fullmatch(clean):
        return "overview"
    if _PROFILE_READ.fullmatch(clean):
        return "recall"
    if _MEMORY_SOURCE.fullmatch(clean):
        # A source correction refers to the latest user topic, never an assistant
        # claim or an older command. Only a known read can be resolved locally.
        for message in reversed((recent_messages or [])[-8:]):
            if message.role != "user":
                continue
            previous = message.content.strip().replace("’", "'")
            if _MEMORY_SOURCE.fullmatch(previous):
                continue
            return "timeline" if _TIMELINE.fullmatch(previous) else "recall"
        return "recall"
    if _CAPABILITY.fullmatch(clean):
        return "capability"
    if _TIMELINE.fullmatch(clean):
        return "timeline"
    if _previous_name(clean, recent_messages or []) is not None:
        return "name_history"
    return None


def _profile_overview(store: MemoryStore) -> str:
    """Read bounded current profile values without summaries, old revisions, or model guesses."""
    prefix = "Here are some current saved details about you: "
    details = []
    size = len(prefix)
    for key in _PROFILE_KEYS:
        item = store.get_fact(f"user:{key}")
        if item is None or item.source not in _PROFILE_SOURCES:
            continue
        content = store.redact_forgotten(item.content).strip()
        if not content or len(content) > 240 or size + len(content) + 1 > 1500:
            continue
        details.append(content)
        size += len(content) + 1
        if len(details) == 8:
            break
    if not details:
        return "I don't have a saved personal profile detail to list yet."
    return prefix + " ".join(details)


def answer_memory_read(
    store: MemoryStore, text: str, now: datetime | None = None,
    *, recent_messages: list[ChatMessage] | None = None,
    recall_enabled: bool = True,
) -> str | None:
    """Answer from application evidence or supplied user dialogue, without saving a fact."""
    kind = memory_read_intent(text, recent_messages)
    enabled = store.get_setting("recall_enabled", recall_enabled) is not False
    if kind == "status":
        return "Saved-memory recall is enabled." if enabled else (
            "Saved-memory recall is paused. Your saved facts are kept; "
            "recent conversation remains active."
        )
    if kind == "capability":
        persistence = " across restarts" if not store._is_in_memory else " during this session"
        return f"Yes. I keep conversation history and saved facts{persistence}."
    if kind == "name_history":
        name = _previous_name(text.strip(), recent_messages or [])
        return f"Right, you called me {name} before renaming me CIEL."
    if kind == "overview":
        if not enabled:
            return _PAUSED
        reply = _profile_overview(store)
        return _PAUSED if store.get_setting("recall_enabled", recall_enabled) is False else reply
    if kind != "timeline":
        return None
    if not enabled:
        return _PAUSED
    start = store.get_fact("ciel:project_start_date")
    try:
        start_date = date.fromisoformat(str(start.metadata["value"])) if start else None
    except (KeyError, ValueError):
        start_date = None
    if start_date is None:
        return "I don't have a confirmed project start date saved."
    value = str(start.metadata["value"])
    if store.redact_forgotten(value) != value:
        return "I don't have a usable confirmed project start date for that calculation."
    elapsed = ((now or datetime.now()).date() - start_date).days
    if elapsed < 0:
        return "The saved project start date is in the future, so I can't calculate that yet."
    return (
        f"It's development day {elapsed + 1}, counting "
        f"{start_date.strftime('%B')} {start_date.day}, {start_date.year} as day one."
    )
