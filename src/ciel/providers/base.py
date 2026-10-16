"""Abstract base interface and data models for AI language model providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """Represents a chat prompt or response message."""

    role: str  # "system", "user", "assistant"
    content: str
    local_only: bool = False  # Application policy; never serialized to a provider.

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Structured response returned by an LLM provider."""

    content: str
    model: str
    provider: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0

    def __str__(self) -> str:
        return self.content


@dataclass
class LLMStreamChunk:
    """A streaming text delta emitted by an LLM provider."""

    delta: str
    model: str
    provider: str
    is_final: bool = False


class LLMProvider(ABC):
    """Abstract interface implemented by all LLM clients."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required API keys or host configs are present."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider endpoint is reachable and responsive."""
        pass

    @abstractmethod
    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat messages synchronously to the LLM and return the response."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream response chunks from the LLM."""
        pass

    @staticmethod
    def normalize_messages(messages: list[ChatMessage] | str) -> list[ChatMessage]:
        """Convert a plain string prompt into a ChatMessage list."""
        if isinstance(messages, str):
            return [ChatMessage(role="user", content=messages)]
        return messages
