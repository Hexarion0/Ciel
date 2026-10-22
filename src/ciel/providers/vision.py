"""Explicit, cancellable image inference without text-model or provider fallback."""

import re
import struct
import threading
import time
import zlib
from typing import Any

import httpx

from ciel.config import Settings
from ciel.providers.base import ChatImage, ChatMessage, LLMResponse
from ciel.providers.router import ModelRouter
from ciel.providers.streaming import streaming_client


class VisionError(RuntimeError):
    """An actionable image-connection failure with no prompt or payload in its text."""


class VisionCancelled(VisionError):
    """The owner cancelled image verification or inference."""


def synthetic_test_image() -> ChatImage:
    """Build a tiny red/blue image entirely in memory, without desktop access."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data)) + kind + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\x00" + b"\xff\x00\x00" * 64 + b"\x00\x00\xff" * 64 for _ in range(64)
    )
    return ChatImage(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", 128, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""),
    )


class VisionClient:
    """Use one selected model and prove image input works before personal pixels."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.screen_provider
        self.model = settings.screen_model.strip()
        self._verified = False
        self._verification_lock = threading.Lock()

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise VisionCancelled("Screen request cancelled.")

    def _connection(self) -> tuple[str, dict[str, str]]:
        if not self.settings.screen_enabled:
            raise VisionError("Screen awareness is disabled. Enable it in Ciel setup.")
        if not self.model:
            raise VisionError("Choose an image-capable model in Ciel setup.")
        headers = {"Content-Type": "application/json"}
        if self.provider == "ollama":
            host = self.settings.ollama_host.strip().rstrip("/")
            if not host:
                raise VisionError("Configure the selected Ollama connection in Ciel setup.")
            return f"{host}/api/chat", headers
        endpoints = {
            "nim": "https://integrate.api.nvidia.com/v1/chat/completions",
            "groq": "https://api.groq.com/openai/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        }
        key = getattr(self.settings, f"{self.provider}_api_key", None)
        if key is None or not key.get_secret_value().strip():
            raise VisionError(
                f"The selected {self.provider} screen connection needs its API key in Ciel setup."
            )
        headers["Authorization"] = f"Bearer {key.get_secret_value()}"
        return endpoints[self.provider], headers

    def _request(
        self, messages: list[ChatMessage], cancel_event: threading.Event | None,
        *, max_tokens: int = 1200,
    ) -> LLMResponse:
        """Make exactly one bounded request; close blocked HTTP on cancellation."""
        self._check_cancel(cancel_event)
        url, headers = self._connection()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.to_ollama_dict() if self.provider == "ollama" else message.to_dict()
                for message in messages
            ],
            "stream": False,
        }
        if self.provider == "ollama":
            payload["options"] = {"temperature": 0.2, "num_predict": max_tokens}
        else:
            payload.update(temperature=0.2, max_tokens=max_tokens)
        started = time.monotonic()
        timed_out = threading.Event()
        data: Any = None
        try:
            with streaming_client(self.settings.screen_timeout_seconds, cancel_event) as client:
                def expire() -> None:
                    timed_out.set()
                    try:
                        client.close()
                    except Exception:
                        pass  # The owning context closes it when the read unwinds.

                deadline = threading.Timer(self.settings.screen_timeout_seconds, expire)
                deadline.daemon = True
                deadline.start()
                try:
                    self._check_cancel(cancel_event)
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                finally:
                    deadline.cancel()
        except httpx.HTTPStatusError as error:
            raise VisionError(
                f"The selected screen model returned HTTP {error.response.status_code}. "
                "Check the image model and its availability in Ciel setup."
            ) from None
        except httpx.TimeoutException:
            raise VisionError("The selected screen model timed out. Please try again.") from None
        except (httpx.RequestError, ValueError):
            if timed_out.is_set():
                raise VisionError(
                    "The selected screen model timed out. Please try again.",
                ) from None
            raise VisionError("The selected screen connection failed. Please try again.") from None
        self._check_cancel(cancel_event)
        if timed_out.is_set():
            raise VisionError("The selected screen model timed out. Please try again.")
        if not isinstance(data, dict) or data.get("error"):
            raise VisionError("The selected screen model did not return a valid answer.")
        if self.provider == "ollama":
            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
        else:
            choices = data.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(choice, dict) and choice.get("finish_reason") in {"error", "length"}:
                raise VisionError("The selected screen model did not finish its answer.")
        if not isinstance(content, str) or not content.strip():
            raise VisionError("The selected screen model returned no visible answer.")
        return LLMResponse(
            content=content.strip(), model=data.get("model") or self.model, provider=self.provider,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            latency=time.monotonic() - started,
        )

    def verify(self, cancel_event: threading.Event | None = None) -> None:
        """Verify once per client with synthetic pixels, never personal screenshots."""
        self._check_cancel(cancel_event)
        while not self._verification_lock.acquire(timeout=0.05):
            self._check_cancel(cancel_event)
        try:
            if self._verified:
                return
            result = self._request([
                ChatMessage(
                    "user", "What color is each half of this image? "
                    "Reply exactly LEFT=<color>; RIGHT=<color>.",
                    images=(synthetic_test_image(),),
                ),
            ], cancel_event, max_tokens=64)
            if not re.search(
                r"left\s*=\s*red\s*;\s*right\s*=\s*blue", result.content, re.IGNORECASE,
            ):
                raise VisionError(
                    "The selected model did not pass the synthetic image check. "
                    "Choose an image-capable model in Ciel setup."
                )
            self._check_cancel(cancel_event)
            self._verified = True
        finally:
            self._verification_lock.release()

    def send(
        self, messages: list[ChatMessage], cancel_event: threading.Event | None = None,
    ) -> LLMResponse:
        """Answer images only after local-history policy and synthetic verification pass."""
        self._check_cancel(cancel_event)
        if ModelRouter.requires_local(messages) and self.provider != "ollama":
            raise VisionError(
                "This conversation contains protected local history. "
                "Screen analysis needs a selected local image model in Ciel setup."
            )
        image_count = sum(len(message.images) for message in messages)
        if not image_count:
            raise VisionError("Screen analysis needs an available image.")
        if image_count != 1:
            raise VisionError(
                "Combine monitor and comparison views into one image for the selected screen model."
            )
        self._connection()
        self.verify(cancel_event)
        self._check_cancel(cancel_event)
        return self._request(messages, cancel_event)
