"""Local Ollama LLM provider for zero-cost offline fallback."""

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ciel.config import get_settings
from ciel.logging import get_logger
from ciel.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk
from ciel.providers.streaming import streaming_client

logger = get_logger("providers.ollama")


class OllamaProvider(LLMProvider):
    """Client for local Ollama server."""

    name: str = "ollama"
    default_model: str = "llama3.2"

    def __init__(self, host: str | None = None, timeout: float = 60.0) -> None:
        settings = get_settings()
        self.host = (host or settings.providers.ollama_host).rstrip("/")
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Ollama is considered configured if a host URL is provided."""
        return bool(self.host and len(self.host.strip()) > 0)

    def health_check(self) -> bool:
        """Check if local Ollama daemon is running and reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.host}/api/tags")
                return res.status_code == 200
        except Exception as err:
            logger.debug("Ollama health check failed: %s", err)
            return False

    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat request to Ollama."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
            **kwargs,
        }

        start_t = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.host}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        latency = time.time() - start_t
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.name,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            },
            latency=latency,
        )

    def stream(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream chat tokens from Ollama."""
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True,
            **kwargs,
        }

        with streaming_client(self.timeout, cancel_event) as client:
            with client.stream(
                "POST",
                f"{self.host}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk_json = json.loads(line)
                        delta = chunk_json.get("message", {}).get("content", "")
                        is_done = chunk_json.get("done", False)
                        yield LLMStreamChunk(
                            delta=delta,
                            model=target_model,
                            provider=self.name,
                            is_final=is_done,
                        )
                        if is_done:
                            break
                    except json.JSONDecodeError:
                        continue
