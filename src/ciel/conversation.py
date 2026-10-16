"""Small intent checks shared by the voice conversation handler and its tests."""

import re

INTERNAL_REPLY_NOTES = (
    "[Playback was interrupted.]",
    "[Generation ended early because the connection failed.]",
)
_INTERNAL_REPLY_NOTE = re.compile(
    "|".join(re.escape(note) for note in INTERNAL_REPLY_NOTES), re.IGNORECASE,
)


def strip_internal_reply_notes(text: str) -> str:
    """Remove reserved playback annotations from generated or replayed assistant text."""
    return _INTERNAL_REPLY_NOTE.sub("", text).strip()


_CIEL_NAMES = r"(?:ciel|siel|ciell)"
_ADDRESS_FILLERS = re.compile(
    r"^(?:(?:so|well|uh|um|oh|okay|ok|alright|and)[,\s]+){1,3}", re.IGNORECASE
)

_FAREWELL = re.compile(
    r"(?:(?:ok(?:ay)?|alright|thanks|thank you)[,!.\s]+)?"
    r"(?:bye(?:[ -]bye)?|goodbye|good\s*night|see you(?: (?:later|soon|around|tomorrow))?"
    r"|take care|farewell)"
    r"(?:[,\s]+" + _CIEL_NAMES + r")?[.!\s]*",
    re.IGNORECASE,
)


def is_farewell(text: str) -> bool:
    """Recognize a short, deliberate sign-off, rather than words inside a question."""
    return _FAREWELL.fullmatch(text.strip()) is not None


def strip_wake_phrase(text: str, wake_phrase: str = "hey ciel") -> str:
    """Remove leading wake phrases and their punctuation without changing the query."""
    configured = r"[\s,]+".join(re.escape(word) for word in wake_phrase.split())
    variants = [configured] if configured else []
    if re.search(r"\b" + _CIEL_NAMES + r"\b", wake_phrase, re.IGNORECASE):
        variants.append(
            r"(?:(?:hey|hi|hello|yo|okay|ok)[\s,]+)?" + _CIEL_NAMES
        )
    if not variants:
        return text.strip()
    prefix = re.compile(r"^(?:" + "|".join(variants) + r")\b[,.!?;:\s—-]*", re.IGNORECASE)
    query = text.strip()
    while True:
        candidate = query
        match = prefix.match(candidate)
        if match is None:
            candidate = _ADDRESS_FILLERS.sub("", query, count=1)
            match = prefix.match(candidate)
        if match is None:
            break
        query = candidate[match.end() :].lstrip()
    return query


def is_direct_address(text: str, wake_phrase: str = "hey ciel") -> bool:
    """Recognize leading addresses and short questions/greetings ending in the name."""
    text = text.strip()
    candidate = _ADDRESS_FILLERS.sub("", text, count=1)
    if re.match(r"^" + _CIEL_NAMES + r"\s+(?:is|was|has|said|told)\b", candidate, re.I):
        return False
    if strip_wake_phrase(text, wake_phrase) != text:
        return True
    if not re.search(r"\b" + _CIEL_NAMES + r"\b", wake_phrase, re.I):
        return False
    return bool(
        re.match(
            r"^(?:what|how|why|when|where|can|could|would|will|do|tell|help|please|hey|hi|hello)\b",
            candidate, re.I,
        )
        and re.search(r"[,\s]+" + _CIEL_NAMES + r"[.!?]*$", candidate, re.I)
        # A name as the object of a sentence is not a vocative address.
        and not re.search(r"\b(?:about|of|with|to|named|called|is|was)\s+" + _CIEL_NAMES,
                          candidate, re.I)
    )


def interpret_clock_address(
    text: str, wake_phrase: str = "hey ciel", *, explicitly_addressed: bool = False
) -> str | None:
    """Interpret a bounded clock question ending in the assistant's known name.

    A missing 'it' before the trailing name is a possible STT error, not an
    explicit address. Callers must keep the original transcript and must not
    use this interpretation to authorize memory writes or other commands.
    """
    if not re.search(r"\b" + _CIEL_NAMES + r"\b", wake_phrase, re.I):
        return None
    candidate = _ADDRESS_FILLERS.sub("", text.strip().replace("’", "'"), count=1)
    if explicitly_addressed:
        candidate = strip_wake_phrase(candidate, wake_phrase).strip(" .!?;:").casefold()
        if re.fullmatch(r"what(?:'s| is) the name right now", candidate, re.I):
            # A common small.en substitution in an explicit time query. Keep this
            # correction narrow and require the caller's independent address signal.
            return "What's the time right now?"
    match = re.fullmatch(
        r"(?P<question>what time is(?: it(?: (?:now|right now))?)?|"
        r"what(?:'s| is) (?:the )?(?:current )?time(?: (?:now|right now))?|"
        r"tell me (?:the )?time)[,\s]+" + _CIEL_NAMES + r"[.!?]*",
        candidate, re.I,
    )
    if match is None:
        return None
    question = match.group("question")
    if question.casefold() == "what time is":
        question = "What time is it"
    return question + "?"
