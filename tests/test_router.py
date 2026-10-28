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

    level, _, override = router.classify_complexity("/deep explain distributed architecture")
    assert level == ComplexityLevel.COMPLEX
    assert override == "strong"


def test_retired_local_command_is_not_a_supported_manual_override(mock_manager):
    router = ModelRouter(manager=mock_manager)
    level, _, override = router.classify_complexity("/local explain the ocean")
    assert level == ComplexityLevel.MEDIUM
    assert override is None


def test_routing_decisions(mock_manager):
    router = ModelRouter(manager=mock_manager)

    # Conversation always prefers the configured Super route, including greetings.
    decision_simple = router.route("Hello")
    assert decision_simple.complexity == ComplexityLevel.SIMPLE
    assert decision_simple.provider_name == "nim"

    # Complex -> NIM Ultra
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
def test_controller_keeps_dedicated_nim_model_or_private_local_route(
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
        assert "allowed_providers" not in options
        assert options["allow_model_fallback"] is False


@pytest.mark.parametrize("provider", ["groq", "openrouter", "ollama"])
def test_controller_routes_to_available_provider_without_nim(mock_manager, provider):
    for name in ("nim", "groq", "openrouter", "ollama"):
        candidate = getattr(mock_manager, name)
        candidate.is_configured.return_value = name == provider
        candidate.default_model = f"configured-{name}"
    router = ModelRouter(mock_manager)
    decision = router.route("Choose an action", purpose="controller")
    assert decision.provider_name == provider
    assert decision.model_name == f"configured-{provider}"


@pytest.mark.parametrize("provider", ["groq", "openrouter", "ollama"])
@pytest.mark.parametrize(
    "prompt", ["Hello", "Tell me about the ocean", "Refactor this Python code"],
)
def test_routes_respect_provider_default_models(mock_manager, provider, prompt):
    for name in ("nim", "groq", "openrouter", "ollama"):
        candidate = getattr(mock_manager, name)
        candidate.is_configured.return_value = name == provider
        candidate.default_model = f"selected-{name}-model"
    decision = ModelRouter(mock_manager).route(prompt)
    assert decision.provider_name == provider
    assert decision.model_name == f"selected-{provider}-model"


def test_local_context_uses_local_provider_selected_model(mock_manager):
    mock_manager.ollama.default_model = "installed-local-model"
    decision = ModelRouter(mock_manager).route("Continue", local_only=True)
    assert decision.provider_name == "ollama"
    assert decision.model_name == "installed-local-model"


@pytest.mark.parametrize("purpose", ["conversation", "controller", "summary"])
def test_legacy_private_history_stays_local_after_retiring_command(mock_manager, purpose):
    from threading import Event

    router = ModelRouter(mock_manager)
    router.send(
        [
            ChatMessage("user", "/local My old private conversation"),
            ChatMessage("assistant", "Private reply"),
            ChatMessage("user", "What did I tell you?"),
        ],
        purpose=purpose,
        cancel_event=Event(),
    )
    options = mock_manager.stream_with_fallback.call_args.kwargs
    assert options["preferred_provider"] == "ollama"
    assert options["allowed_providers"] == ("ollama",)
    delivered = mock_manager.stream_with_fallback.call_args.args[0]
    assert delivered[0].content == "My old private conversation"


def test_legacy_private_history_stays_local_when_streamed(mock_manager):
    router = ModelRouter(mock_manager)
    list(router.stream([
        ChatMessage("user", "/local My old private conversation"),
        ChatMessage("user", "Continue"),
    ]))
    options = mock_manager.stream_with_fallback.call_args.kwargs
    assert options["preferred_provider"] == "ollama"
    assert options["allowed_providers"] == ("ollama",)


@pytest.mark.parametrize("followup", ["Can you fix it?", "Optimize that.", "Please review this."])
def test_referring_work_followup_preserves_recent_code_route(mock_manager, followup):
    router = ModelRouter(mock_manager)
    messages = [
        ChatMessage("user", "Refactor this Python function"),
        ChatMessage("assistant", "Here is the implementation."),
        ChatMessage("user", followup),
    ]
    assert router.classify_complexity(router._routing_text(messages))[0] == ComplexityLevel.COMPLEX


@pytest.mark.parametrize("latest", ["What is Python?", "Hey", "/fast Can you fix it?"])
def test_new_topic_and_current_override_do_not_inherit_old_complexity(mock_manager, latest):
    router = ModelRouter(mock_manager)
    messages = [
        ChatMessage("user", "Refactor this Python function"),
        ChatMessage("assistant", "Here is the implementation."),
        ChatMessage("user", latest),
    ]
    expected = ComplexityLevel.MEDIUM if latest == "What is Python?" else ComplexityLevel.SIMPLE
    assert router.classify_complexity(router._routing_text(messages))[0] == expected


def test_continuation_route_does_not_scan_unbounded_history(mock_manager):
    router = ModelRouter(mock_manager)
    messages = [ChatMessage("user", "Refactor this Python function")]
    messages.extend(ChatMessage("user", "Continue.") for _ in range(8))
    assert router.classify_complexity(router._routing_text(messages))[0] == ComplexityLevel.MEDIUM


@pytest.mark.parametrize("purpose", ["controller", "conversation"])
def test_private_provider_failure_never_reaches_cloud(purpose):
    from threading import Event

    manager = ProviderManager()
    for name in ("nim", "groq", "openrouter", "ollama"):
        provider = MagicMock()
        provider.is_configured.return_value = True
        provider.default_model = f"selected-{name}"
        provider.stream.side_effect = RuntimeError("Unavailable")
        manager.register_provider(name, provider)
    with pytest.raises(RuntimeError, match="All streaming providers"):
        ModelRouter(manager).send(
            [ChatMessage("user", "A private question", local_only=True)],
            purpose=purpose, cancel_event=Event(),
        )
    manager.ollama.stream.assert_called_once()
    for name in ("nim", "groq", "openrouter"):
        getattr(manager, name).stream.assert_not_called()
