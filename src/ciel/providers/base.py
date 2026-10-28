"""Abstract base interface and data models for AI language model providers."""

import base64
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

MAX_TOOL_CALLS = 128
MAX_TOOL_ARGUMENTS = 256 * 1024
_TOOL_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")


def _validate_tool_identifier(value: str, label: str, limit: int, *, partial: bool = False) -> None:
    """Keep native tool identifiers bounded and safe for provider request fields."""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Tool {label} must be a string of at most {limit} characters")
    if not value and partial:
        return
    if _TOOL_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"Tool {label} must be a nonempty ASCII identifier")


def _validate_tool_arguments(arguments: str) -> None:
    """Bound opaque arguments without executing or interpreting a requested action."""
    if not isinstance(arguments, str) or len(arguments.encode("utf-8")) > MAX_TOOL_ARGUMENTS:
        raise ValueError("Tool arguments must be a string of at most 256 KiB")


@dataclass(frozen=True)
class ToolCall:
    """A native model request; the application validates arguments before execution."""

    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        _validate_tool_identifier(self.id, "call ID", 256)
        _validate_tool_identifier(self.name, "name", 128)
        _validate_tool_arguments(self.arguments)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the OpenAI-compatible native function-call wire shape."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ToolCallDelta:
    """One indexed native tool-call fragment, kept separate from speech text."""

    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""

    def __post_init__(self) -> None:
        if type(self.index) is not int or not 0 <= self.index < MAX_TOOL_CALLS:
            raise ValueError("Tool call index must be an integer from 0 to 127")
        _validate_tool_identifier(self.id, "call ID fragment", 256, partial=True)
        _validate_tool_identifier(self.name, "name fragment", 128, partial=True)
        _validate_tool_arguments(self.arguments)


class ToolCallAccumulator:
    """Collect bounded native call fragments without adding them to answer text."""

    def __init__(self) -> None:
        self._calls: dict[int, ToolCallDelta] = {}

    def add(self, deltas: tuple[ToolCallDelta, ...]) -> None:
        """Append stream fragments by call index, including fragmented names and IDs."""
        if not isinstance(deltas, tuple) or len(deltas) > MAX_TOOL_CALLS:
            raise ValueError("Tool call fragments must be a tuple of at most 128 items")
        for delta in deltas:
            if not isinstance(delta, ToolCallDelta):
                raise ValueError("Tool call fragments must contain ToolCallDelta objects")
            previous = self._calls.get(delta.index, ToolCallDelta(delta.index))
            self._calls[delta.index] = ToolCallDelta(
                delta.index,
                previous.id + delta.id,
                previous.name + delta.name,
                previous.arguments + delta.arguments,
            )

    def finish(self) -> tuple[ToolCall, ...]:
        """Return complete calls in provider index order, rejecting missing identifiers."""
        calls = tuple(
            ToolCall(delta.id, delta.name, delta.arguments)
            for _, delta in sorted(self._calls.items())
        )
        if len({call.id for call in calls}) != len(calls):
            raise ValueError("Native tool call IDs must be unique within a response")
        return calls


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

    role: str  # "system", "user", "assistant", "tool"
    content: str
    local_only: bool = False  # Application policy; never serialized to a provider.
    images: tuple[ChatImage, ...] = field(default=(), repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        if self.images and self.role != "user":
            raise ValueError("Image attachments are supported only on user messages")
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ChatImage) for image in self.images
        ):
            raise ValueError("Image attachments must be a tuple of ChatImage objects")
        if (
            not isinstance(self.tool_calls, tuple)
            or len(self.tool_calls) > MAX_TOOL_CALLS
            or not all(isinstance(call, ToolCall) for call in self.tool_calls)
        ):
            raise ValueError("Tool calls must be a tuple of at most 128 ToolCall objects")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Native tool calls are supported only on assistant messages")
        if len({call.id for call in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("Native tool call IDs must be unique within a message")
        if self.role == "tool":
            _validate_tool_identifier(self.tool_call_id, "result call ID", 256)
        elif self.tool_call_id:
            raise ValueError("Tool call IDs are supported only on tool result messages")

    def to_dict(self) -> dict[str, Any]:
        """Preserve text-only requests and serialize images only as typed content parts."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            result["content"] = [
                {"type": "text", "text": self.content},
                *(image.to_content_part() for image in self.images),
            ]
        if self.tool_calls:
            result["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_ollama_dict(self) -> dict[str, Any]:
        """Use Ollama's native images field rather than data-URI content parts."""
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            result["images"] = [image.base64_data() for image in self.images]
        if self.tool_calls:
            result["tool_calls"] = []
            for call in self.tool_calls:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Ollama tool arguments must be a JSON object")
                result["tool_calls"].append({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments},
                })
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
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
    tool_calls: tuple[ToolCall, ...] = ()

    def __str__(self) -> str:
        return self.content


@dataclass
class LLMStreamChunk:
    """Separate streaming text and native tool-call deltas emitted by a provider."""

    delta: str
    model: str
    provider: str
    is_final: bool = False
    tool_calls: tuple[ToolCallDelta, ...] = ()


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
