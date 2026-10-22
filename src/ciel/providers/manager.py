"""Multi-provider coordinator with automated fallback chain and health tracking."""

from collections.abc import Iterator
from threading import Event
from typing import Any

from ciel.logging import get_logger
from ciel.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
    require_text_messages,
)
from ciel.providers.groq import GroqProvider
from ciel.providers.nim import NimProvider
from ciel.providers.ollama import OllamaProvider
from ciel.providers.openrouter import OpenRouterProvider

logger = get_logger("providers.manager")


class ProviderManager:
    """Manages AI provider clients, tracks their health, and executes the fallback chain."""

    # Default fallback sequence: NIM -> Groq -> OpenRouter -> Local Ollama
    DEFAULT_FALLBACK_CHAIN = ["nim", "groq", "openrouter", "ollama"]

    def __init__(self, fallback_chain: list[str] | None = None) -> None:
        self.fallback_chain = fallback_chain or self.DEFAULT_FALLBACK_CHAIN
        self.providers: dict[str, LLMProvider] = {
            "nim": NimProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
            "ollama": OllamaProvider(),
        }

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register or override a provider client."""
        self.providers[name.lower()] = provider

    def get_provider(self, name: str) -> LLMProvider | None:
        """Get provider by name."""
        return self.providers.get(name.lower())

    @property
    def nim(self) -> LLMProvider:
        """NVIDIA NIM provider instance."""
        return self.providers["nim"]

    @property
    def groq(self) -> LLMProvider:
        """Groq provider instance."""
        return self.providers["groq"]

    @property
    def openrouter(self) -> LLMProvider:
        """OpenRouter provider instance."""
        return self.providers["openrouter"]

    @property
    def ollama(self) -> LLMProvider:
        """Local Ollama provider instance."""
        return self.providers["ollama"]

    def get_provider_statuses(self) -> dict[str, str]:
        """Check and return status for all registered providers."""
        statuses: dict[str, str] = {}
        for name, provider in self.providers.items():
            if not provider.is_configured():
                statuses[name] = "not configured"
            elif provider.health_check():
                statuses[name] = "available"
            else:
                statuses[name] = "offline / unreachable"
        return statuses

    def get_first_available_provider(self) -> LLMProvider | None:
        """Find the first configured and reachable provider in the fallback chain."""
        for name in self.fallback_chain:
            p = self.get_provider(name)
            if p and p.is_configured():
                return p
        return None

    def send_with_fallback(
        self,
        messages: list[ChatMessage] | str,
        preferred_provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        allowed_providers: tuple[str, ...] | None = None,
        allow_model_fallback: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send prompt attempting preferred provider first with automatic fallback."""
        require_text_messages(messages)
        preferred = preferred_provider.lower() if preferred_provider else None
        order: list[str] = []
        if preferred in self.providers:
            order.append(preferred)
        for name in self.fallback_chain:
            if name not in order and name in self.providers:
                order.append(name)

        last_error: Exception | None = None

        for provider_name in order:
            if allowed_providers is not None and provider_name not in allowed_providers:
                continue
            provider = self.providers[provider_name]
            if not provider.is_configured():
                logger.debug("Skipping provider '%s' (not configured).", provider_name)
                continue

            try:
                logger.debug("Attempting inference with provider '%s'...", provider_name)
                response = provider.send(
                    messages=messages,
                    model=model if preferred == provider_name else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **({"allow_model_fallback": False}
                       if provider_name == "nim" and not allow_model_fallback else {}),
                    **kwargs,
                )
                if not isinstance(response.content, str) or not response.content.strip():
                    raise RuntimeError("Provider returned no reply text")
                logger.info(
                    "Successfully generated response via '%s' (model: %s, latency: %.2fs)",
                    provider_name,
                    response.model,
                    response.latency,
                )
                return response
            except Exception as err:
                last_error = err
                logger.warning(
                    "Provider '%s' failed (%s). Triggering fallback chain.",
                    provider_name,
                    err,
                )

        error_msg = f"All AI providers in fallback chain failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error

    def stream_with_fallback(
        self,
        messages: list[ChatMessage] | str,
        preferred_provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        cancel_event: Event | None = None,
        allowed_providers: tuple[str, ...] | None = None,
        allow_model_fallback: bool = True,
        **kwargs: Any,
    ) -> Iterator[LLMStreamChunk]:
        """Stream response from the first functioning provider."""
        require_text_messages(messages)
        preferred = preferred_provider.lower() if preferred_provider else None
        order: list[str] = []
        if preferred in self.providers:
            order.append(preferred)
        for name in self.fallback_chain:
            if name not in order and name in self.providers:
                order.append(name)

        for provider_name in order:
            if allowed_providers is not None and provider_name not in allowed_providers:
                continue
            if cancel_event is not None and cancel_event.is_set():
                return
            provider = self.providers[provider_name]
            if not provider.is_configured():
                continue

            emitted = False
            try:
                logger.debug("Attempting stream with provider '%s'...", provider_name)
                for chunk in provider.stream(
                    messages=messages,
                    model=model if preferred == provider_name else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cancel_event=cancel_event,
                    **({"allow_model_fallback": False}
                       if provider_name == "nim" and not allow_model_fallback else {}),
                    **kwargs,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    emitted |= bool(chunk.delta.strip())
                    yield chunk
                if emitted or (cancel_event is not None and cancel_event.is_set()):
                    return
                raise RuntimeError("Provider stream returned no reply text")
            except Exception as err:
                if cancel_event is not None and cancel_event.is_set():
                    return
                if emitted:
                    # Switching after speech started would splice unrelated answers.
                    raise RuntimeError("Provider stream failed after a partial reply") from err
                logger.warning(
                    "Provider '%s' streaming failed: %s. Attempting fallback.",
                    provider_name,
                    err,
                )

        raise RuntimeError("All streaming providers in fallback chain failed.")
