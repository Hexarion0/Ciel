"""Local Ollama LLM provider for zero-cost offline fallback."""

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ciel.config import get_settings
from ciel.logging import get_logger
from ciel.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    ToolCallAccumulator,
    require_text_messages,
)
from ciel.providers.streaming import streaming_client
from ciel.providers.tool_calls import (
    ollama_tool_calls,
    ollama_tool_deltas,
    reply_content,
    response_object,
)

logger = get_logger("providers.ollama")


class OllamaProvider(LLMProvider):
    """Client for local Ollama server."""

    name: str = "ollama"
    default_model: str = "llama3.2"

    def __init__(self, host: str | None = None, timeout: float = 60.0) -> None:
        settings = get_settings()
        self.host = (host or settings.providers.ollama_host).rstrip("/")
        self.default_model = getattr(settings.providers, 'ollama_model', type(self).default_model)
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

    @staticmethod
    def _messages_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Associate native tool results with their declared Ollama function names."""
        names: dict[str, str] = {}
        payload = []
        for message in messages:
            serialized = message.to_ollama_dict()
            names.update((call.id, call.name) for call in message.tool_calls)
            if message.role == "tool" and message.tool_call_id in names:
                serialized["tool_name"] = names[message.tool_call_id]
            payload.append(serialized)
        return payload

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
        require_text_messages(msgs)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": self._messages_payload(msgs),
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
        data = response_object(data, "Ollama response")
        message = response_object(data.get("message", {}), "Ollama message")
        content = reply_content(message.get("content"))
        tool_calls = ollama_tool_calls(message.get("tool_calls"))

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.name,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            },
            latency=latency,
            tool_calls=tool_calls,
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
        require_text_messages(msgs)
        target_model = model or self.default_model

        payload = {
            "model": target_model,
            "messages": self._messages_payload(msgs),
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True,
            **kwargs,
        }

        calls = ToolCallAccumulator()
        next_tool_index = 0
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
                        chunk_json = response_object(json.loads(line), "Ollama event")
                        message = response_object(chunk_json.get("message", {}), "Ollama message")
                        delta = reply_content(message.get("content"))
                        tool_calls = ollama_tool_deltas(message.get("tool_calls"), next_tool_index)
                        calls.add(tool_calls)
                        if tool_calls:
                            next_tool_index = max(next_tool_index, max(
                                call.index for call in tool_calls
                            ) + 1)
                        is_done = chunk_json.get("done", False)
                        if not isinstance(is_done, bool):
                            raise ValueError("Ollama done field must be boolean")
                        if is_done:
                            calls.finish()
                        yield LLMStreamChunk(
                            delta=delta,
                            model=target_model,
                            provider=self.name,
                            is_final=is_done,
                            tool_calls=tool_calls,
                        )
                        if is_done:
                            break
                    except json.JSONDecodeError:
                        continue
                calls.finish()
