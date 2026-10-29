"""Explicit subject corrections for an active search, without rewriting transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ciel.providers.base import ChatMessage

_NAME = re.compile(r"[\w][\w .+#&'’-]*", re.UNICODE)
_SEARCH = re.compile(
    r"\b(?:search(?:\s+(?:the\s+)?(?:web|internet|online))?(?:\s+for)?|"
    r"look\s+(?:it\s+)?up|look\s+online\s+for|find)\s+", re.I,
)
_SUBJECT_LEAD = re.compile(
    r"^(?:(?:hey|hi|hello)[, ]+ciel[,.!? ]+|(?:hey|hi|hello)[,.!?]\s*|ciel[,!:]\s*)?"
    r"(?:(?:do|did)\s+(?:you|u)\s+know(?:\s+about)?\s+|"
    r"have\s+you\s+heard\s+(?:of|about)\s+|"
    r"(?:(?:can|could|would|will)\s+(?:you|u)\s+)?tell\s+me\s+about\s+)?",
    re.I,
)
_TASK_WORDS = set(
    "a an the for about of to from with on in where what which when how why who "
    "does do did is are was were "
    "can could you me please anime manga novel light season episode part chapter volume "
    "continuation continue continues end ends release date price pricing version notes "
    "weather current latest official guide information meaning explain describe summarize "
    "tell show help".split()
)
_QUOTED = re.compile(r'''["“][^"”]*["”]|(?<!\w)['‘][^'’]*['’](?!\w)''')
_NO_SEARCH = re.compile(
    r"\b(?:cancel|never\s*mind)\b|\b(?:don't|do\s+not|never|avoid|without|stop)\b"
    r"[^.!?]{0,80}\b(?:search|searching|browse|browsing|look\s+up)\b|"
    r"\bno\s+(?:more\s+)?(?:search(?:es|ing)?|brows(?:e|ing)|lookups?)\b", re.I,
)
_PLATFORMS = {"linux", "windows", "macos", "android", "ios", "ipados", "freebsd"}
_CAPABILITIES = {
    "offline", "online", "support", "supported", "compatibility", "compatible",
    "installation", "install", "requirements", "availability", "available",
}
_PLATFORM_NAMES = {
    "arch linux", "alpine linux", "amazon linux", "kali linux", "oracle linux",
    "rocky linux", "red hat enterprise linux", "gentoo linux",
}


def _unquoted(text: str) -> str:
    """Mask quoted speech without moving the offsets of actual request words."""
    return _QUOTED.sub(lambda match: " " * len(match[0]), text)


def subject_correction(text: str, *, contextual: bool = False) -> str:
    """Read a short explicit name correction, never infer one from a guessed spelling."""
    if "\n" in text or "\r" in text:
        return ""
    text = text.strip()
    if _NO_SEARCH.search(_unquoted(text)) or text.startswith(('"', "'", "“", "‘")):
        return ""
    if text.count("*") == 1 and (text.startswith("*") or text.endswith("*")):
        name = text.strip("*").strip()
    else:
        tense = r"(?:mean|meant)" if contextual else "meant"
        match = re.fullmatch(r"(?:no[, ]+)?i " + tense + r"\s+(.+?)[.!]?", text, re.I)
        if match is None and contextual:
            match = re.fullmatch(r"(?:no|just)[, ]+(.+?)[.!]?", text, re.I)
        name = match[1].strip() if match else ""
    return name if (1 <= len(name) <= 64 and len(name.split()) <= 6
                    and _NAME.fullmatch(name) and any(char.isalpha() for char in name)) else ""


@dataclass(frozen=True)
class SearchCorrection:
    """An application projection of the owner's corrected search subject."""

    subject: str
    original: str
    query: str


def has_search_request(text: str) -> bool:
    """Recognize an affirmative request even after a separate conversational greeting."""
    from ciel.controller import is_explicit_web_request

    visible = _unquoted(text.replace("’", "'"))
    if any(mark in visible for mark in ('"', '“', '”')) or _NO_SEARCH.search(visible):
        return False
    return any(is_explicit_web_request(part.strip())
               for part in re.split(r"[.!?]\s+", visible))


def search_request_body(request: str) -> str:
    """Extract question data after a search verb, without granting lookup permission."""
    match = _SEARCH.search(_unquoted(request))
    body = request[match.end():] if match else request
    return re.sub(r"\s+(?:for me|please)[.!?]*$", "", body, flags=re.I).strip(" .!?")


def replace_search_subject(request: str, subject: str) -> str:
    """Replace the explicit subject without changing its surrounding requested outcome.

    This is question projection only. The caller establishes whether a correction
    actually belongs to an existing task and whether a lookup is authorized.
    """
    match = _SEARCH.search(_unquoted(request))
    # Skip conversational framing by span, rather than treating names such as
    # "Ciel" or "Know" as generic subject words everywhere.
    lead = _SUBJECT_LEAD.match(request)
    offset = match.end() if match else lead.end() if lead else 0
    body = request[offset:]
    words = list(re.finditer(r"[\w][\w.+#'’-]*", body))
    start = next((i for i, word in enumerate(words)
                  if word[0].casefold() not in _TASK_WORDS), None)
    if start is None:
        return request[:offset] + subject + " " + body.strip()
    end = start + 1
    while words[start][0][0].isupper() and end < len(words) and end - start < 6:
        word = words[end][0]
        if word.casefold() in _TASK_WORDS or not word[0].isupper():
            break
        # A platform followed by capability details usually constrains the named
        # product. Keep it unless the correction or a known OS name owns that word.
        prefix = " ".join(item[0].casefold() for item in words[start:end + 1])
        if (word.casefold() in _PLATFORMS
                and word.casefold() not in subject.casefold().split()
                and prefix not in _PLATFORM_NAMES
                and any(item[0].casefold() in _CAPABILITIES for item in words[end + 1:end + 4])):
            break
        end += 1
    return (request[:offset] + body[:words[start].start()] + subject
            + body[words[end - 1].end():])


def corrected_search(text: str, messages: list[ChatMessage]) -> SearchCorrection | None:
    """Bind an explicit correction to the nearby search, stopping at unrelated turns."""
    from ciel.task_context import resolve_task_context

    task = resolve_task_context(text, messages)
    if (task.kind != "correction" or task.task_type != "web" or not task.corrected_subject
            or task.anchor_index is None or task.anchor_index >= len(messages)):
        return None
    original = messages[task.anchor_index].content.strip()
    query = search_request_body(task.request)
    return SearchCorrection(task.corrected_subject, original, query[:1000])
