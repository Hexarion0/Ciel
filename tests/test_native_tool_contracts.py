"""Native tool calls survive provider fallback, routing, and transcript replay."""

import json
from dataclasses import FrozenInstanceError
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import Settings
from ciel.providers.base import (
    MAX_TOOL_ARGUMENTS,
    ChatImage,
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    ToolCallAccumulator,
    ToolCallDelta,
)
from ciel.providers.manager import ProviderManager
from ciel.providers.router import ModelRouter


@pytest.fixture(autouse=True)
def isolated_router_settings(monkeypatch):
    """Tests use explicit defaults without opening private local configuration."""
    settings = Settings(_env_file=None)
    monkeypatch.setattr("ciel.providers.router.get_settings", lambda: settings)


def make_manager(*names):
    """Build a coordinator using only fake providers and no provider configuration I/O."""
    manager = ProviderManager.__new__(ProviderManager)
    manager.fallback_chain = list(names)
    manager.providers = {}
    for name in ("nim", "groq", "openrouter", "ollama"):
        provider = MagicMock()
        provider.is_configured.return_value = name in names
        provider.default_model = f"selected-{name}"
        manager.providers[name] = provider
    return manager


def test_tool_call_wire_roundtrip_and_result_identifier():
    call = ToolCall("call_1", "memory_search", '{"query":"tea"}')
    assistant = ChatMessage("assistant", "", False, (), (call,))
    wire = json.loads(json.dumps(assistant.to_dict()))
    assert wire == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "memory_search", "arguments": '{"query":"tea"}'},
        }],
    }
    native = wire["tool_calls"][0]
    restored = ToolCall(native["id"], **native["function"])
    assert restored == call
    result = ChatMessage("tool", '{"matches":[]}', True, (), (), call.id)
    assert result.to_dict() == {
        "role": "tool", "content": '{"matches":[]}', "tool_call_id": "call_1",
    }
    assert "local_only" not in result.to_dict()
    assert assistant.to_ollama_dict()["tool_calls"][0]["function"] == {
        "name": "memory_search", "arguments": {"query": "tea"},
    }
    assert result.to_ollama_dict()["tool_call_id"] == "call_1"
    with pytest.raises(FrozenInstanceError):
        call.name = "changed"


def test_existing_positional_text_and_image_contracts_are_preserved():
    image = ChatImage(b"encoded-image")
    message = ChatMessage("user", "Describe this", True, (image,))
    wire = message.to_dict()
    assert wire["content"][0] == {"type": "text", "text": "Describe this"}
    assert wire["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert message.to_ollama_dict()["images"] == [image.base64_data()]
    assert ChatMessage("user", "Hello").to_dict() == {"role": "user", "content": "Hello"}
    assert LLMResponse("Answer", "model", "provider", {"tokens": 1}, 0.1).tool_calls == ()
    assert LLMStreamChunk("Answer", "model", "provider", True).tool_calls == ()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ToolCall("", "search", "{}"),
        lambda: ToolCall("call\n1", "search", "{}"),
        lambda: ToolCall("call_1", "unsafe name", "{}"),
        lambda: ToolCall("call_1", "search", "x" * (MAX_TOOL_ARGUMENTS + 1)),
        lambda: ToolCallDelta(True),
        lambda: ToolCallDelta(-1),
        lambda: ToolCallDelta(128),
        lambda: ChatMessage("tool", "Missing ID"),
        lambda: ChatMessage("assistant", "", tool_call_id="call_1"),
        lambda: ChatMessage("user", "", tool_calls=(ToolCall("call_1", "search", "{}"),)),
        lambda: ChatMessage("assistant", "", tool_calls=[]),
    ],
)
def test_native_tool_fields_reject_malformed_or_unbounded_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_accumulator_assembles_interleaved_calls_and_keeps_arguments_opaque():
    collector = ToolCallAccumulator()
    collector.add((
        ToolCallDelta(1, "call_2", "clock", "{}"),
        ToolCallDelta(0, "call_", "memory_", '{"query":'),
    ))
    collector.add((ToolCallDelta(0, "1", "search", '"tea"}'),))
    assert collector.finish() == (
        ToolCall("call_1", "memory_search", '{"query":"tea"}'),
        ToolCall("call_2", "clock", "{}"),
    )
    invalid_json = ToolCall("call_bad", "search", "not JSON")
    assert invalid_json.arguments == "not JSON"
    with pytest.raises(ValueError):
        ChatMessage("assistant", "", tool_calls=(invalid_json,)).to_ollama_dict()


def test_accumulator_rejects_incomplete_duplicate_and_oversized_calls():
    incomplete = ToolCallAccumulator()
    incomplete.add((ToolCallDelta(0, arguments="{}"),))
    with pytest.raises(ValueError):
        incomplete.finish()
    duplicate = ToolCallAccumulator()
    duplicate.add((
        ToolCallDelta(0, "call_1", "clock", "{}"),
        ToolCallDelta(1, "call_1", "search", "{}"),
    ))
    with pytest.raises(ValueError, match="unique"):
        duplicate.finish()
    oversized = ToolCallAccumulator()
    oversized.add((ToolCallDelta(0, "call_1", "search", "é" * (MAX_TOOL_ARGUMENTS // 2)),))
    with pytest.raises(ValueError, match="256 KiB"):
        oversized.add((ToolCallDelta(0, arguments="x"),))


def test_tool_only_send_success_preserves_tools_on_provider_fallback():
    manager = make_manager("nim", "groq")
    manager.nim.send.side_effect = RuntimeError("unavailable")
    expected = LLMResponse("", "selected-groq", "groq", tool_calls=(
        ToolCall("call_1", "clock", "{}"),
    ))
    manager.groq.send.return_value = expected
    tools = [{"type": "function", "function": {"name": "clock"}}]
    tool_choice = {"type": "function", "function": {"name": "clock"}}
    response = manager.send_with_fallback(
        "What time is it?", preferred_provider="NIM", model="selected-nim",
        tools=tools, tool_choice=tool_choice, response_format={"type": "json_object"},
    )
    assert response is expected
    assert response.content == ""
    for provider in (manager.nim, manager.groq):
        options = provider.send.call_args.kwargs
        assert options["tools"] is tools
        assert options["tool_choice"] is tool_choice
        assert options["response_format"] == {"type": "json_object"}
    assert manager.nim.send.call_args.kwargs["model"] == "selected-nim"
    assert manager.groq.send.call_args.kwargs["model"] is None


def test_tool_only_stream_is_success_and_does_not_trigger_fallback():
    manager = make_manager("nim", "groq")
    chunks = [
        LLMStreamChunk("", "model", "nim", tool_calls=(
            ToolCallDelta(0, "call_1", "clock", "{}"),
        )),
        LLMStreamChunk("", "model", "nim", True),
    ]
    manager.nim.stream.return_value = iter(chunks)
    assert list(manager.stream_with_fallback("What time is it?")) == chunks
    manager.groq.stream.assert_not_called()


def test_stream_never_switches_provider_after_partial_tool_fragment():
    manager = make_manager("nim", "groq")

    def broken(**kwargs):
        yield LLMStreamChunk("", "model", "nim", tool_calls=(
            ToolCallDelta(0, "call_1", "memory_search", '{"query":'),
        ))
        raise RuntimeError("connection closed")

    manager.nim.stream.side_effect = broken
    stream = manager.stream_with_fallback("Recall tea")
    assert next(stream).tool_calls[0].arguments == '{"query":'
    with pytest.raises(RuntimeError, match="after a partial reply"):
        next(stream)
    manager.groq.stream.assert_not_called()


def test_empty_stream_fallback_does_not_publish_whitespace_from_failed_provider():
    manager = make_manager("nim", "groq")
    manager.nim.stream.return_value = iter([LLMStreamChunk(" \n", "blank", "nim")])
    expected = LLMStreamChunk("", "model", "groq", tool_calls=(
        ToolCallDelta(0, "call_1", "clock", "{}"),
    ))
    manager.groq.stream.return_value = iter([expected])
    assert list(manager.stream_with_fallback("What time is it?")) == [expected]


@pytest.mark.parametrize("private", [False, True])
def test_router_preserves_native_history_tools_overrides_and_private_policy(private):
    manager = make_manager("nim", "groq", "ollama")
    provider = manager.ollama if private else manager.nim
    call = ToolCall("call_1", "memory_search", '{"query":"tea"}')
    response = LLMResponse("", "model", "provider", tool_calls=(call,))
    provider.send.return_value = response
    tools = [{"type": "function", "function": {"name": "memory_search"}}]
    messages = [
        ChatMessage("user", "/strong Recall tea", private),
        ChatMessage("assistant", "", tool_calls=(call,)),
        ChatMessage("tool", "A useful memory", private, tool_call_id=call.id),
    ]
    assert ModelRouter(manager).send(messages, tools=tools, tool_choice="auto") is response
    delivered = provider.send.call_args.kwargs
    assert delivered["messages"][0].content == "Recall tea"
    assert delivered["messages"][0].local_only is private
    assert delivered["messages"][1].tool_calls == (call,)
    assert delivered["messages"][2].tool_call_id == call.id
    assert delivered["tools"] is tools
    assert delivered["tool_choice"] == "auto"
    assert messages[0].content == "/strong Recall tea"
    if private:
        manager.nim.send.assert_not_called()
        manager.groq.send.assert_not_called()


def test_router_collects_tool_deltas_without_turning_arguments_into_answer_text():
    manager = make_manager("nim", "ollama")
    manager.ollama.stream.return_value = iter([
        LLMStreamChunk("Let me check.", "local-model", "ollama"),
        LLMStreamChunk("", "local-model", "ollama", tool_calls=(
            ToolCallDelta(0, "call_", "memory_", '{"query":'),
        )),
        LLMStreamChunk("", "local-model", "ollama", True, (
            ToolCallDelta(0, "1", "search", '"tea"}'),
        )),
    ])
    tools = [{"type": "function", "function": {"name": "memory_search"}}]
    response = ModelRouter(manager).send(
        [ChatMessage("user", "Recall tea", True)], cancel_event=Event(),
        tools=tools, tool_choice="auto",
    )
    assert response.content == "Let me check."
    assert response.tool_calls == (ToolCall("call_1", "memory_search", '{"query":"tea"}'),)
    assert response.model == "local-model"
    assert response.provider == "ollama"
    assert manager.ollama.stream.call_args.kwargs["tools"] is tools
    manager.nim.stream.assert_not_called()


def test_router_stream_forwards_native_calls_and_prevents_private_cloud_fallback():
    manager = make_manager("nim", "ollama")
    tool_delta = LLMStreamChunk("", "local-model", "ollama", tool_calls=(
        ToolCallDelta(0, "call_1", "clock", "{}"),
    ))
    manager.ollama.stream.return_value = iter([tool_delta])
    tools = [{"type": "function", "function": {"name": "clock"}}]
    router = ModelRouter(manager)
    assert list(router.stream(
        [ChatMessage("user", "/fast Time?", True)], tools=tools, tool_choice="required",
    )) == [tool_delta]
    options = manager.ollama.stream.call_args.kwargs
    assert options["messages"][0].content == "Time?"
    assert options["tools"] is tools
    assert options["tool_choice"] == "required"
    manager.nim.stream.assert_not_called()
    manager.ollama.stream.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="All streaming providers"):
        list(router.stream([ChatMessage("user", "Continue", True)], tools=tools))
    manager.nim.stream.assert_not_called()
