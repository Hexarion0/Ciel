"""AI provider clients, base interfaces, and fallback manager."""

from functools import lru_cache

from ciel.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk
from ciel.providers.groq import GroqProvider
from ciel.providers.manager import ProviderManager
from ciel.providers.nim import NimProvider
from ciel.providers.ollama import OllamaProvider
from ciel.providers.openrouter import OpenRouterProvider
from ciel.providers.router import ComplexityLevel, ModelRouter, RoutingDecision


@lru_cache(maxsize=1)
def get_provider_manager() -> ProviderManager:
    """Return the global cached ProviderManager instance."""
    return ProviderManager()


@lru_cache(maxsize=1)
def get_model_router() -> ModelRouter:
    """Return the global cached ModelRouter instance."""
    return ModelRouter()


__all__ = [
    "ChatMessage",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMProvider",
    "NimProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "OllamaProvider",
    "ProviderManager",
    "get_provider_manager",
    "ComplexityLevel",
    "RoutingDecision",
    "ModelRouter",
    "get_model_router",
]
