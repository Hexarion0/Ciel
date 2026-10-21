"""Data models for CIEL's memory and conversation persistence system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from ciel.memory.web_evidence import normalize_web_evidence

MAX_REPLY_SOURCES = 20
MAX_SOURCE_URL_CHARS = 2048
MAX_SOURCE_TITLE_CHARS = 300


def normalize_sources(sources: Any) -> list[dict[str, str]]:
    """Copy a bounded list of source labels and safe absolute HTTP(S) links."""
    if not isinstance(sources, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = source.get("url")
        title = source.get("title", "")
        if (
            not isinstance(url, str)
            or not isinstance(title, str)
            or not url
            or len(url) > MAX_SOURCE_URL_CHARS
            or any(char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159 for char in url)
            or "\\" in url
            or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in title)
        ):
            continue
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme.lower() not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                continue
            # Accessing port validates malformed port strings and their range.
            parsed.port
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append({
            "title": title.strip()[:MAX_SOURCE_TITLE_CHARS] or parsed.hostname,
            "url": url,
        })
        if len(result) >= MAX_REPLY_SOURCES:
            break
    return result


class MemoryType(str, Enum):
    """Classification of stored memory items."""

    FACT = "fact"
    PREFERENCE = "preference"
    PROJECT = "project"
    CONVERSATION = "conversation"
    REMINDER = "reminder"


@dataclass
class MemoryItem:
    """A single persistent memory record stored in SQLite."""

    content: str
    memory_type: MemoryType = MemoryType.FACT
    source: str = "user_explicit"
    confidence: float = 1.0
    id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory item to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value
            if isinstance(self.memory_type, MemoryType)
            else str(self.memory_type),
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """A single message turn in a multi-turn conversation session."""

    role: str
    content: str
    session_id: str = "default"
    id: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = ""
    model: str = ""
    tokens: int = 0
    latency: float = 0.0
    sources: list[dict[str, str]] = field(default_factory=list)
    web_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep citation metadata safe and separate from conversation text."""
        self.sources = normalize_sources(self.sources)
        self.web_evidence = (
            normalize_web_evidence(self.web_evidence) if self.role == "assistant" else {}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert conversation turn to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens": self.tokens,
            "latency": self.latency,
            "sources": normalize_sources(self.sources),
            "web_evidence": normalize_web_evidence(self.web_evidence),
        }
