"""Unit tests for ModelRouter complexity classification and provider routing."""

from unittest.mock import MagicMock

import pytest

from ciel.providers.base import ChatMessage, LLMResponse
from ciel.providers.manager import ProviderManager
from ciel.providers.router import ComplexityLevel, ModelRouter


@pytest.fixture
def mock_manager():
    """Mocked ProviderManager with simulated provider configs."""
    manager = MagicMock(spec=ProviderManager)
    manager.nim = MagicMock()
    manager.nim.is_configured.return_value = True
    manager.groq = MagicMock()
    manager.groq.is_configured.return_value = True
    manager.openrouter = MagicMock()
    manager.openrouter.is_configured.return_value = False
    manager.ollama = MagicMock()
    manager.ollama.is_configured.return_value = True

    manager.send_with_fallback = MagicMock(
        return_value=LLMResponse(
            content="Mocked response",
            provider="groq",
            model="llama-3.1-8b-instant",
        )
    )
    return manager


def test_classify_simple_greetings(mock_manager):
    router = ModelRouter(manager=mock_manager)
    level, reason, override = router.classify_complexity("Hello Ciel, what time is it?")
    assert level == ComplexityLevel.SIMPLE
    assert override is None


def test_classify_complex_coding_query(mock_manager):
    router = ModelRouter(manager=mock_manager)
    level, reason, override = router.classify_complexity(
        "Can you refactor this Python function to optimize recursion and fix the bug?"
    )
    assert level == ComplexityLevel.COMPLEX
    assert "code" in reason.lower() or "keyword" in reason.lower()


def test_classify_manual_overrides(mock_manager):
    router = ModelRouter(manager=mock_manager)

    level, _, override = router.classify_complexity("/fast explain quantum mechanics in one word")
    assert level == ComplexityLevel.SIMPLE
    assert override == "fast"

    level, _, override = router.classify_complexity("/strong write a complete microservices plan")
    assert level == ComplexityLevel.COMPLEX
    assert override == "strong"

    level, _, override = router.classify_complexity("/local how much disk space is left?")
    assert level == ComplexityLevel.SIMPLE
    assert override == "local"


def test_routing_decisions(mock_manager):
    router = ModelRouter(manager=mock_manager)

    # Simple with Groq configured -> groq
    decision_simple = router.route("Hello")
    assert decision_simple.complexity == ComplexityLevel.SIMPLE
    assert decision_simple.provider_name == "groq"

    # Complex -> NIM 70B
    decision_complex = router.route("Design a high-concurrency database architecture with sharding")
    assert decision_complex.complexity == ComplexityLevel.COMPLEX
    assert decision_complex.provider_name == "nim"


def test_router_send(mock_manager):
    router = ModelRouter(manager=mock_manager)
    messages = [
        ChatMessage(role="system", content="You are CIEL."),
        ChatMessage(role="user", content="/fast What is the capital of France?"),
    ]

    response = router.send(messages)
    assert response.content == "Mocked response"
    mock_manager.send_with_fallback.assert_called_once()


@pytest.mark.parametrize(
    "prompt",
    ["So what's the time right now?", "What's today's date?", "What is 2 + 2?"],
)
def test_clock_variants_and_math_are_simple(mock_manager, prompt):
    assert ModelRouter(mock_manager).classify_complexity(prompt)[0] == ComplexityLevel.SIMPLE


@pytest.mark.parametrize(
    "prompt",
    [
        "What is Python?",
        "Explain what an API does",
        "Compare tea and coffee",
        " ".join(["Tell me about your day"] * 15),
    ],
)
def test_keywords_and_length_alone_do_not_select_ultra(mock_manager, prompt):
    assert ModelRouter(mock_manager).classify_complexity(prompt)[0] == ComplexityLevel.MEDIUM


@pytest.mark.parametrize("purpose", ["summary", "speech_gate"])
def test_background_tasks_ignore_code_history_and_strong_override(mock_manager, purpose):
    router = ModelRouter(mock_manager)
    prompt = "/strong Previous dialogue: refactor this Python architecture " * 40
    decision = router.route(prompt, purpose=purpose)
    assert decision.complexity == ComplexityLevel.SIMPLE
    assert decision.override_applied is None
    router.send([ChatMessage("user", prompt)], purpose=purpose)
    assert mock_manager.stream_with_fallback.call_args.kwargs["model"] != (
        "nvidia/nemotron-3-ultra-550b-a55b"
    )


def test_continue_preserves_complex_request_route(mock_manager):
    router = ModelRouter(mock_manager)
    router.send(
        [
            ChatMessage("user", "Refactor this Python function"),
            ChatMessage("assistant", "Here is the first part."),
            ChatMessage("user", "Continue."),
        ]
    )
    assert mock_manager.send_with_fallback.call_args.kwargs["preferred_provider"] == "nim"


def test_greeting_does_not_hide_complex_request(mock_manager):
    router = ModelRouter(mock_manager)
    assert (
        router.classify_complexity("Hello Ciel, design a distributed architecture")[0]
        == ComplexityLevel.COMPLEX
    )


def test_override_prefix_requires_whole_command(mock_manager):
    router = ModelRouter(mock_manager)
    assert router.classify_complexity("/fastfood ideas")[2] is None
    assert router._clean_override_prefixes([ChatMessage("user", "/fastfood ideas")])[0].content == (
        "/fastfood ideas"
    )


def test_repeated_continuations_and_stream_keep_original_route(mock_manager):
    router = ModelRouter(mock_manager)
    messages = [
        ChatMessage("user", "Refactor this Python function"),
        ChatMessage("assistant", "Part one."),
        ChatMessage("user", "Continue."),
        ChatMessage("assistant", "Part two."),
        ChatMessage("user", "Go on."),
    ]
    router.send(messages)
    assert mock_manager.send_with_fallback.call_args.kwargs["preferred_provider"] == "nim"
    list(router.stream(messages))
    assert mock_manager.stream_with_fallback.call_args.kwargs["preferred_provider"] == "nim"


@pytest.mark.parametrize("private", [False, True])
def test_controller_uses_only_dedicated_nim_or_private_local_model(
    mock_manager, monkeypatch, private,
):
    from threading import Event

    from ciel.config import Settings

    settings = Settings(_env_file=None, controller_model="configured-light-model")
    monkeypatch.setattr("ciel.providers.router.get_settings", lambda: settings)
    router = ModelRouter(mock_manager)
    router.send([
        ChatMessage("system", "Short controller prompt", local_only=private),
        ChatMessage("user", "/strong Refactor complicated code"),
    ], purpose="controller", cancel_event=Event())
    options = mock_manager.stream_with_fallback.call_args.kwargs
    if private:
        assert options["preferred_provider"] == "ollama"
        assert options["allowed_providers"] == ("ollama",)
    else:
        assert options["preferred_provider"] == "nim"
        assert options["model"] == "configured-light-model"
        assert options["allowed_providers"] == ("nim",)
        assert options["allow_model_fallback"] is False
