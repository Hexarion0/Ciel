"""Small delivery router, independent of any synthesis engine."""

from dataclasses import dataclass
from typing import Literal

VoiceStyle = Literal["neutral", "warm", "amused", "firm", "focused", "soft"]
VOICE_STYLES = ("neutral", "warm", "amused", "firm", "focused", "soft")


def select_voice_style(text: str, explicit: str | None = None) -> VoiceStyle:
    """Honor metadata first; reuse deliberate amusement cues; otherwise stay composed.

    Do not guess the user's emotions or classify arbitrary technical/sensitive text.
    Callers can supply conversation context as an explicit style in the future.
    """
    if explicit is not None:
        if explicit not in VOICE_STYLES:
            raise ValueError(f"Unknown voice style: {explicit}")
        return explicit
    if any(cue in text.casefold() for cue in ("[chuckle]", "[laugh]", "[giggle]")):
        return "amused"
    return "neutral"


@dataclass(frozen=True)
class TTSRequest:
    """Words and delivery metadata; style is never spoken as an instruction."""

    text: str
    style: VoiceStyle = "neutral"

    def __post_init__(self) -> None:
        select_voice_style(self.text, self.style)
