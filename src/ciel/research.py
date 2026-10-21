"""Explicit, scoped research observations; never personal facts or remote instructions."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from ciel.internet import (
    InternetResult,
    InternetService,
    InternetSource,
    _article_url,
    _clean,
    _date_score,
    _topic_terms,
)

_OBSERVATION = re.compile(
    r"^(?:(?:please|so|okay|ok)[, ]+)?(?:"
    r"(?:tell|show)(?: me)? (?:what (?:you|u) (?:saw|seen|found|read|retrieved)|"
    r"(?:the |your )?(?:findings|results|sources)|more(?: about (?:it|that|this))?)|"
    r"what (?:did (?:you|u) (?:see|find|read)|(?:have (?:you|u)|you|u) (?:seen|found|read))|"
    r"(?:summarize|explain) (?:that|it|those|the (?:results|findings|page))|"
    r"(?:read|open|explain|summarize|tell me about) (?:the )?"
    r"(?:first|second|third|fourth|fifth|[1-5])(?:st|nd|rd|th)? (?:source|link|article|story)"
    r")[.?! ]*$", re.I,
)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


class ResearchCancellation(threading.Event):
    """Combine local interruption with a refreshed cross-channel permission check."""

    def __init__(self, parent: threading.Event | None, allowed: Callable[[], bool]) -> None:
        super().__init__()
        self.parent, self.allowed = parent, allowed
        self._next_check = 0.0

    def is_set(self) -> bool:
        if super().is_set() or (self.parent is not None and self.parent.is_set()):
            return True
        now = time.monotonic()
        if now >= self._next_check:
            self._next_check = now + 0.1
            if not self.allowed():
                self.set()
        return super().is_set()

    def refresh(self) -> bool:
        """Check permissions without the polling cache before generation or publication."""
        self._next_check = 0.0
        return self.is_set()


def evidence_followup(text: str) -> bool:
    """Recognize references to observed evidence without reviving unrelated old research."""
    return bool(_OBSERVATION.fullmatch(re.sub(r"^/(?:local|fast|strong|deep)\s+", "",
                                              text.strip(), flags=re.I)))


def evidence_payload(lookup: InternetResult | None) -> dict:
    """Serialize bounded observations; MemoryStore validates and redacts them on save."""
    if lookup is None or not lookup.attempted:
        return {}
    fields = ("title", "url", "snippet", "content", "published_at", "fetched_at",
              "publisher", "date_kind")
    return {
        "query": lookup.query, "kind": getattr(lookup, "kind", "web"),
        "retrieved_at": lookup.retrieved_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        ),
        "sources": [{key: getattr(source, key, "") for key in fields}
                    for source in lookup.sources[:10]],
    }


def previous_observations(text: str, conversation) -> InternetResult | None:
    """Read actual evidence through the manager's owner-scoped, fresh database view."""
    if not evidence_followup(text):
        return None
    scope = conversation.context_session_ids()
    data = conversation.latest_web_evidence()
    if not data:
        return None
    # A source reference can continue an investigation, but an unrelated intervening
    # topic cannot silently resurrect it. Ignore only the currently submitted turn
    # and brief acknowledgments when checking the chronological conversation.
    for turn in reversed(conversation.get_recent_turns(limit=16)):
        if turn.role == "assistant" and turn.web_evidence:
            break
        if turn.role == "user" and turn.content.strip() != text.strip():
            if not evidence_followup(turn.content) and not re.fullmatch(
                r"(?:ok(?:ay)?|thanks|thank you|yes|sure)[.! ]*", turn.content.strip(), re.I,
            ):
                return None
    else:
        return None
    if scope != conversation.context_session_ids():
        return None
    sources = []
    for source in data.get("sources", []):
        sources.append(InternetSource(**source))
    if not sources:
        return InternetResult(attempted=True, query=data["query"],
                              retrieved_at=data["retrieved_at"], reply=(
            "My last lookup didn't retrieve usable evidence. I haven't established the answer."
        ))
    ordinal = re.search(r"\b(first|second|third|fourth|fifth|[1-5])(?:st|nd|rd|th)?\b", text,
                        re.I)
    if ordinal:
        word = ordinal.group(1).casefold()
        index = _ORDINALS[word] if word in _ORDINALS else int(word)
        if index > len(sources):
            return InternetResult(reply="That source wasn't in the results I retained.")
        sources = [sources[index - 1]]
    result = InternetResult(attempted=True, query=data["query"], sources=sources)
    result.retrieved_at = data["retrieved_at"]
    if hasattr(result, "kind"):
        result.kind = data.get("kind", "web")
    result.context = InternetService._context(sources, query=result.query)
    result.context += (
        "\nThese are previously retrieved observations, not a fresh lookup. "
        f"Retrieval recorded at {data.get('retrieved_at', 'unknown time')}. "
        "Explain what was actually observed; do not call it current without a fresh check."
    )
    return result


def observed_findings(lookup: InternetResult, *, channel: str = "voice") -> str:
    """Offer attributable observations when a model cannot produce a valid answer."""
    if not lookup.sources:
        return "I couldn't retrieve usable evidence. I haven't established the answer."
    parts = []
    for index, source in enumerate(lookup.sources[:3], 1):
        # Quotes describe retrieved bytes, rather than inventing a semantic conclusion.
        excerpt = " ".join((source.content or source.snippet).split()[:22])
        if not excerpt:
            continue
        label = "page text" if source.content else "search preview"
        title = _clean(source.title, 120)
        reference = f" [{index}]" if channel == "text" else ""
        parts.append(f'From “{title}”{reference}, the {label} included: “{excerpt}”.')
    if not parts:
        return "I found links, but couldn't retrieve enough text to establish the answer."
    return "\n\n".join(parts) + " I couldn't verify the remaining details from this evidence."


def news_headlines(lookup: InternetResult, *, channel: str, minimum: int = 1) -> str:
    """Give dated, attributable English headlines when synthesis cannot be supported.

    Headlines are observations, not an independently established account of events.
    Keep only recent article-shaped results and visibly label the date provenance.
    """
    now = time.time()
    lines, topics = [], []
    for index, source in enumerate(lookup.sources, 1):
        if (not _article_url(source.url) or not source.title
                or not now - 7 * 86400 <= _date_score(source.published_at) <= now
                or re.search(r"[\u0400-\u04ff]", source.title)):
            continue
        terms = _topic_terms(source.title) - {"mongolia", "mongolian", "news", "latest"}
        if any(len(terms & prior) / max(1, min(len(terms), len(prior))) > 0.6
               for prior in topics):
            continue
        topics.append(terms)
        title = _clean(source.title, 250)
        date = _clean(source.published_at, 100)
        label = "Published" if source.date_kind == "published" else "Source-reported date"
        lines.append(f"{title} [{index}] ({label}: {date})" if channel == "text" else title)
        if len(lines) == 5:
            break
    if len(lines) < minimum:
        return ""
    intro = "Here are the recent article headlines I found: "
    return intro + ("\n\n" if channel == "text" else " ") + (
        "\n\n".join(lines) if channel == "text" else ". ".join(lines)
    ) + " I couldn't verify a fuller summary of every story."
