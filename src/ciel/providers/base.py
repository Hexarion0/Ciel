"""Abstract base interface and data models for AI language model providers."""

import base64
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChatImage:
    """An ephemeral in-memory image, excluded from representations and transcript text."""

    data: bytes = field(repr=False)
    mime_type: Literal["image/png", "image/jpeg", "image/webp"] = "image/png"
    detail: Literal["auto", "low", "high"] = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not 0 < len(self.data) <= 8 * 1024 * 1024:
            raise ValueError("An image must contain 1 byte to 8 MiB of encoded image data")
        if self.mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("Image type must be PNG, JPEG, or WebP")
        if self.detail not in {"auto", "low", "high"}:
            raise ValueError("Image detail must be auto, low, or high")

    def base64_data(self) -> str:
        """Encode only while constructing a transient provider request."""
        return base64.b64encode(self.data).decode("ascii")

    def to_content_part(self) -> dict[str, Any]:
        """Serialize an OpenAI-compatible image content part."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{self.base64_data()}",
                "detail": self.detail,
            },
        }


@dataclass
class ChatMessage:
    """Represents a chat prompt or response message."""

    role: str  # "system", "user", "assistant"
    content: str
    local_only: bool = False  # Application policy; never serialized to a provider.
    images: tuple[ChatImage, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.images and self.role != "user":
            raise ValueError("Image attachments are supported only on user messages")
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ChatImage) for image in self.images
        ):
            raise ValueError("Image attachments must be a tuple of ChatImage objects")

    def to_dict(self) -> dict[str, Any]:
        """Preserve text-only requests and serialize images only as typed content parts."""
        if not self.images:
            return {"role": self.role, "content": self.content}
        return {
            "role": self.role,
            "content": [
                {"type": "text", "text": self.content},
                *(image.to_content_part() for image in self.images),
            ],
        }

    def to_ollama_dict(self) -> dict[str, Any]:
        """Use Ollama's native images field rather than data-URI content parts."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            result["images"] = [image.base64_data() for image in self.images]
        return result


def require_text_messages(messages: list[ChatMessage] | str) -> None:
    """Reject image attachments before ordinary text routing can drop or reroute them."""
    if not isinstance(messages, str) and any(message.images for message in messages):
        raise ValueError(
            "Image attachments require VisionClient; text routing cannot accept images.",
        )


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
