"""Shared Unicode word normalization for indexed memory writes and recall queries."""

import re
import unicodedata

STOP_WORDS = set(
    "a an the is are was were do does did i you my your me we our it this that "
    "what which who when how about please can could would should tell know remember "
    "forget of for to in on at and or actually no yes right now".split()
)
GROUPS = (
    {"favorite", "favourite", "preferred", "prefer", "likes", "like", "love", "enjoy"},
    {"game", "games", "gaming", "play", "playing"},
    {"start", "started", "begin", "began", "beginning", "origin"},
    {"development", "developing", "build", "building", "project"},
    {"date", "day", "days", "timeline"},
    {"name", "named", "call", "called", "nickname"},
    {"color", "colour", "colors", "colours"},
    {"occupation", "job", "profession"},
    {"birthday", "birthdate"},
)
ALIASES = {word: min(group) for group in GROUPS for word in group}


def terms(text: str) -> set[str]:
    """Return normalized words, retaining Unicode names and treating punctuation literally."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"[^\W_]+", normalized)
    return {ALIASES.get(word, word) for word in words if word not in STOP_WORDS}
