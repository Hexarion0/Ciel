"""OpenRouter LLM provider implementation."""

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ciel.config import ProviderConfig, get_settings
from ciel.logging import get_logger
from ciel.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk
from ciel.providers.streaming import streaming_client

logger = get_logger("providers.openrouter")


class OpenRouterProvider(LLMProvider):
    """Client for OpenRouter API (unified multi-model gateway)."""

    name: str = "openrouter"
    default_model: str = "openrouter/free"
    base_url: str = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        config: ProviderConfig | None = None,
    ) -> None:
        providers = config if config is not None else get_settings().providers
        if api_key:
            self.api_key = api_key
        elif providers.openrouter_api_key:
            self.api_key = providers.openrouter_api_key.get_secret_value()
        else:
            self.api_key = None
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Return True if OpenRouter key is set."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def _get_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError("OpenRouter API key is not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Hexarion0/Ciel",
            "X-Title": "CIEL Assistant",
            "Content-Type": "application/json",
        }

    def health_check(self) -> bool:
        """Check API connectivity."""
        if not self.is_configured():
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/auth/key", headers=self._get_headers())
                return res.status_code == 200
        except Exception as err:
            logger.debug("OpenRouter health check failed: %s", err)
            return False

    def send(
        self,
        messages: list[ChatMessage] | str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat messages to OpenRouter."""
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        start_t = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        latency = time.time() - start_t
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model") or target_model,
            provider=self.name,
            usage=usage,
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
        """Stream response chunks from OpenRouter."""
        cancel_event = kwargs.pop("cancel_event", None)
        if cancel_event is not None and cancel_event.is_set():
            return
        msgs = self.normalize_messages(messages)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": [m.to_dict() for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }

        response_model = target_model
        with streaming_client(self.timeout, cancel_event) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        yield LLMStreamChunk(
                            delta="",
                            model=response_model,
                            provider=self.name,
                            is_final=True,
                        )
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        if chunk_json.get("error") is not None:
                            raise RuntimeError("OpenRouter stream reported an error")
                        response_model = chunk_json.get("model") or response_model
                        choices = chunk_json.get("choices") or []
                        if not choices:
                            continue  # Usage-only events contain no reply text.
                        if choices[0].get("finish_reason") == "error":
                            raise RuntimeError("OpenRouter stream reported an error")
                        delta = (choices[0].get("delta") or {}).get("content", "")
                        if delta:
                            yield LLMStreamChunk(
                                delta=delta,
                                model=response_model,
                                provider=self.name,
                                is_final=False,
                            )
                    except json.JSONDecodeError:
                        continue
