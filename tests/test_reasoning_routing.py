"""Conversation model choice and private reasoning stay separate from visible replies."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from ciel.config import (
    DEFAULT_NIM_COMPLEX_MODEL,
    DEFAULT_NIM_FALLBACK_MODEL,
    DEFAULT_NIM_MODEL,
    ProviderConfig,
    Settings,
)
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk, ToolCall
from ciel.providers.manager import ProviderManager
from ciel.providers.nim import NimProvider, _FinalAnswerStream
from ciel.providers.router import ModelRouter


@pytest.fixture
def router(monkeypatch):
    """Use explicit provider settings so a developer's dotenv cannot affect routing."""
    settings = Settings(_env_file=None, nim_reasoning="auto", nim_model=DEFAULT_NIM_MODEL,
                        nim_complex_model=DEFAULT_NIM_COMPLEX_MODEL)
    monkeypatch.setattr("ciel.providers.router.get_settings", lambda: settings)
    manager = MagicMock(spec=ProviderManager)
    for name in ("nim", "groq", "openrouter", "ollama"):
        provider = getattr(manager, name)
        provider.is_configured.return_value = True
        provider.default_model = f"{name}-default"
    manager.send_with_fallback.return_value = LLMResponse("Final answer.", "test", "nim")
    manager.stream_with_fallback.return_value = [LLMStreamChunk("Final answer.", "test", "nim")]
    return ModelRouter(manager)


@pytest.mark.parametrize("prompt,model,thinking", [
    ("Hello", DEFAULT_NIM_MODEL, False),
    ("Tell me about your day", DEFAULT_NIM_MODEL, False),
    ("Implement this Python function", DEFAULT_NIM_COMPLEX_MODEL, True),
    ("/deep Compare the tradeoffs", DEFAULT_NIM_COMPLEX_MODEL, True),
])
def test_conversation_prefers_super_or_ultra_with_auto_reasoning(router, prompt, model, thinking):
    decision = router.route(prompt)
    assert (decision.provider_name, decision.model_name) == ("nim", model)
    assert decision.reasoning_enabled is thinking


@pytest.mark.parametrize("mode,thinking", [("on", True), ("off", False)])
@pytest.mark.parametrize("streaming", [False, True])
def test_session_override_is_passed_to_both_request_paths(router, mode, thinking, streaming):
    router.set_reasoning_mode(mode)
    messages = [ChatMessage("user", "Implement a Python function")]
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    if streaming:
        list(router.stream(messages, tools=tools, tool_choice="auto"))
        options = router.manager.stream_with_fallback.call_args.kwargs
    else:
        router.send(messages, tools=tools, tool_choice="auto")
        options = router.manager.send_with_fallback.call_args.kwargs
    assert router.reasoning_mode == mode
    assert options["nim_reasoning"] is thinking
    assert options["model"] == DEFAULT_NIM_COMPLEX_MODEL
    assert options["tools"] == tools and options["tool_choice"] == "auto"


def test_request_keeps_its_reasoning_snapshot_when_session_changes(router, monkeypatch):
    original_clean = router._clean_override_prefixes

    def change_mode_after_routing(messages):
        router.set_reasoning_mode("off")
        return original_clean(messages)

    monkeypatch.setattr(router, "_clean_override_prefixes", change_mode_after_routing)
    router.send([ChatMessage("user", "Implement this Python function")])
    assert router.reasoning_mode == "off"
    assert router.manager.send_with_fallback.call_args.kwargs["nim_reasoning"] is True


@pytest.mark.parametrize("purpose", ["controller", "tool_progress", "summary", "speech_gate"])
def test_background_requests_do_not_inherit_conversation_reasoning(router, purpose):
    router.set_reasoning_mode("on")
    assert router.route("/deep Implement this code", purpose=purpose).reasoning_enabled is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("private", [False, True])
def test_verification_uses_normal_model_without_reasoning_or_source_complexity(
    router, streaming, private,
):
    router.set_reasoning_mode("on")
    messages = [ChatMessage("user", json.dumps({
        "source_context": "Implement a distributed Python function and debug this code.",
    }), local_only=private)]
    if streaming:
        list(router.stream(messages, purpose="verification", max_tokens=128))
        options = router.manager.stream_with_fallback.call_args.kwargs
    else:
        router.send(messages, purpose="verification", max_tokens=128)
        options = router.manager.send_with_fallback.call_args.kwargs
    assert router.reasoning_mode == "on"
    assert options["preferred_provider"] == ("ollama" if private else "nim")
    assert options["model"] == ("ollama-default" if private else DEFAULT_NIM_MODEL)
    assert options["nim_reasoning"] is False
    assert options["max_tokens"] == 128
    if private:
        assert options["allowed_providers"] == ("ollama",)
    assert router.route("Implement Python code").reasoning_enabled is True


def test_reasoning_does_not_override_private_local_routing(router):
    router.set_reasoning_mode("on")
    router.send([ChatMessage("user", "Refactor this Python code", local_only=True)])
    options = router.manager.send_with_fallback.call_args.kwargs
    assert options["preferred_provider"] == "ollama"
    assert options["allowed_providers"] == ("ollama",)


def test_invalid_reasoning_mode_leaves_current_preference_intact(router):
    with pytest.raises(ValueError, match="auto, on, or off"):
        router.set_reasoning_mode("sometimes")
    assert router.reasoning_mode == "auto"
    with pytest.raises(ValidationError):
        ProviderConfig(nim_reasoning="sometimes")


@pytest.mark.parametrize("streaming", [False, True])
def test_nim_reasoning_snapshot_never_leaks_into_other_provider_options(streaming):
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    nim, groq = MagicMock(), MagicMock()
    nim.is_configured.return_value = groq.is_configured.return_value = True
    manager.register_provider("nim", nim)
    manager.register_provider("groq", groq)
    if streaming:
        nim.stream.side_effect = RuntimeError("unavailable")
        groq.stream.return_value = [LLMStreamChunk("Final.", "groq", "groq")]
        list(manager.stream_with_fallback("Hello", preferred_provider="nim", nim_reasoning=True))
        primary, backup = nim.stream.call_args.kwargs, groq.stream.call_args.kwargs
    else:
        nim.send.side_effect = RuntimeError("unavailable")
        groq.send.return_value = LLMResponse("Final.", "groq", "groq")
        manager.send_with_fallback("Hello", preferred_provider="nim", nim_reasoning=True)
        primary, backup = nim.send.call_args.kwargs, groq.send.call_args.kwargs
    assert primary["reasoning_enabled"] is True
    assert "reasoning_enabled" not in backup and "nim_reasoning" not in backup


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_nim_wire_options_and_final_answer_only(monkeypatch, streaming, thinking):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        hidden = {"reasoning_content": "Private reasoning", "reasoning": "Private thoughts"}
        if streaming:
            events = [
                {"choices": [{"delta": hidden}]},
                {"choices": [{"delta": {"content": "Visible answer."},
                              "finish_reason": "stop"}]},
            ]
            return httpx.Response(200, text="\n\n".join(
                "data: " + json.dumps(event) for event in events
            ))
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "Visible answer.", **hidden,
        }}]})

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    provider = NimProvider(api_key="test", config=ProviderConfig())
    if streaming:
        answer = "".join(chunk.delta for chunk in provider.stream(
            "Hello", reasoning_enabled=thinking,
        ))
    else:
        answer = provider.send("Hello", reasoning_enabled=thinking).content
    assert answer == "Visible answer."
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is thinking
    assert "reasoning_enabled" not in requests[0]


def test_explicit_routing_setting_overrides_legacy_template_without_mutating_it():
    template = {"enable_thinking": True, "low_effort": True}
    payload = {"model": DEFAULT_NIM_MODEL, "chat_template_kwargs": template,
               "reasoning_enabled": False}
    request = NimProvider._model_payload(payload, DEFAULT_NIM_MODEL)
    assert request["chat_template_kwargs"] == {"enable_thinking": False, "low_effort": True}
    assert template["enable_thinking"] is True
    fallback = NimProvider._model_payload(payload, "meta/backup")
    assert "chat_template_kwargs" not in fallback and "reasoning_enabled" not in fallback


@pytest.mark.parametrize("thinking,expected", [(False, 400), (True, 8192)])
def test_voice_completion_budget_reserves_hidden_reasoning_only_when_enabled(thinking, expected):
    payload = {"model": DEFAULT_NIM_MODEL, "max_tokens": 400,
               "reasoning_enabled": thinking}
    request = NimProvider._model_payload(payload, DEFAULT_NIM_MODEL)
    assert request["max_tokens"] == expected
    assert payload["max_tokens"] == 400
    assert NimProvider._model_payload(payload, "meta/backup")["max_tokens"] == 400


def test_reasoning_enabled_completion_keeps_native_tool_calls(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value.json.return_value = {"choices": [{"message": {
        "content": None, "reasoning_content": "Private plan", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "web_search", "arguments": '{"query":"weather"}'},
        }],
    }}]}
    monkeypatch.setattr("httpx.Client", lambda **_kw: client)
    reply = NimProvider(api_key="test", config=ProviderConfig()).send(
        "Search", reasoning_enabled=True,
    )
    assert reply.content == ""
    assert reply.tool_calls == (ToolCall("call_1", "web_search", '{"query":"weather"}'),)
    client.post.assert_called_once()


@pytest.mark.parametrize("pieces,expected", [
    (["<thi", "nk>Private", " reasoning</th", "ink>", "Visible answer."], "Visible answer."),
    (["<think>Private unclosed reasoning"], ""),
    (["<reflection>Private</reflection><think>More", "</think>Final."], "Final."),
    (['{"example":"<think>literal', '</think>"}'], '{"example":"<think>literal</think>"}'),
    (["Plain ", "answer."], "Plain answer."),
])
def test_streamed_thinking_tags_never_become_spoken_text(pieces, expected):
    filtering = _FinalAnswerStream()
    assert "".join(filtering.feed(piece) for piece in pieces) == expected


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tool_only", [False, True])
def test_unavailable_super_uses_ultra_without_downgrading_or_exposing_reasoning(
    monkeypatch, streaming, thinking, tool_only,
):
    requests = []
    call = {"id": "call_1", "type": "function", "function": {
        "name": "web_search", "arguments": '{"query":"weather"}',
    }}

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["model"] == DEFAULT_NIM_MODEL:
            return httpx.Response(404)
        hidden = {"reasoning_content": "Private plan"}
        visible = {"tool_calls": [call]} if tool_only else {"content": "Visible answer."}
        if streaming:
            if tool_only:
                visible["tool_calls"][0] = {"index": 0, **call}
            events = [
                {"choices": [{"delta": hidden}]},
                {"choices": [{"delta": visible, "finish_reason": "stop"}]},
            ]
            return httpx.Response(200, text="\n\n".join(
                "data: " + json.dumps(event) for event in events
            ))
        return httpx.Response(200, json={"choices": [{"message": {**hidden, **visible}}]})

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    provider = NimProvider(api_key="test", config=ProviderConfig())
    if streaming:
        chunks = list(provider.stream("Hello", reasoning_enabled=thinking))
        assert all(chunk.model == DEFAULT_NIM_FALLBACK_MODEL for chunk in chunks)
        content = "".join(chunk.delta for chunk in chunks)
        assert bool([call for chunk in chunks for call in chunk.tool_calls]) is tool_only
    else:
        reply = provider.send("Hello", reasoning_enabled=thinking)
        assert reply.model == DEFAULT_NIM_FALLBACK_MODEL
        content = reply.content
        assert bool(reply.tool_calls) is tool_only
    assert content == ("" if tool_only else "Visible answer.")
    assert [payload["model"] for payload in requests] == [
        DEFAULT_NIM_MODEL, DEFAULT_NIM_FALLBACK_MODEL,
    ]
    assert all(payload["chat_template_kwargs"]["enable_thinking"] is thinking
               for payload in requests)
