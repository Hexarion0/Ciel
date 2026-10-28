"""Native tool requests remain structured throughout provider HTTP adapters."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import ProviderConfig
from ciel.providers.base import ChatImage, ChatMessage, ToolCall, ToolCallAccumulator
from ciel.providers.groq import GroqProvider
from ciel.providers.nim import NimProvider
from ciel.providers.ollama import OllamaProvider
from ciel.providers.openrouter import OpenRouterProvider

OPENAI_PROVIDERS = [NimProvider, GroqProvider, OpenRouterProvider]
ALL_PROVIDERS = [*OPENAI_PROVIDERS, OllamaProvider]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search public pages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}]
CHOICE = {"type": "function", "function": {"name": "web_search"}}


def _provider(provider_class, monkeypatch):
    if provider_class is OllamaProvider:
        monkeypatch.setattr("ciel.providers.ollama.get_settings", lambda: SimpleNamespace(
            providers=SimpleNamespace(ollama_host="http://local.test"),
        ))
        return provider_class(host="http://local.test")
    return provider_class(api_key="synthetic-test-key", config=ProviderConfig())


def _client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: client)
    return client


def _call(arguments='{"query": "weather"}', **changes):
    result = {
        "id": "call_7",
        "type": "function",
        "function": {"name": "web_search", "arguments": arguments},
    }
    result.update(changes)
    return result


def _sse_events(client, events):
    response = client.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([
        "data: " + json.dumps(event) for event in events
    ] + ["data: [DONE]"])


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
def test_openai_tool_only_completion_preserves_json_and_model(provider_class, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    arguments = '{ "query" : "a <think>tag</think> and Final answer:" }'
    client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": None, "tool_calls": [_call(arguments)]}}],
    }
    reply = provider.send("Search", model="primary", tools=TOOLS, tool_choice=CHOICE)
    assert reply.content == ""
    assert reply.model == "primary"
    assert reply.tool_calls == (ToolCall("call_7", "web_search", arguments),)
    client.post.assert_called_once()
    payload = client.post.call_args.kwargs["json"]
    assert payload["tools"] == TOOLS
    assert payload["tool_choice"] == CHOICE


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
def test_openai_stream_accumulates_split_json_without_speech(provider_class, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    _sse_events(client, [
        {"choices": [{"delta": {"content": "Checking. ", "tool_calls": [{
            "index": 0, "id": "call_", "type": "function",
            "function": {"name": "web_", "arguments": '{"query":'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "7", "function": {"name": "search", "arguments": ' "local'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ' weather"}'},
        }]}}]},
        {"choices": [], "usage": {"total_tokens": 5}},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    chunks = list(provider.stream("Search", model="primary", tools=TOOLS, tool_choice="auto"))
    assert "".join(chunk.delta for chunk in chunks) == "Checking. "
    calls = ToolCallAccumulator()
    for chunk in chunks:
        calls.add(chunk.tool_calls)
    assert calls.finish() == (ToolCall("call_7", "web_search", '{"query": "local weather"}'),)
    assert chunks[-1].is_final
    client.stream.assert_called_once()
    payload = client.stream.call_args.kwargs["json"]
    assert payload["tools"] == TOOLS
    assert payload["tool_choice"] == "auto"


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
def test_openai_tool_only_stream_does_not_use_backup(provider_class, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    _sse_events(client, [{"choices": [{"delta": {"tool_calls": [{
        "index": 0, **_call(),
    }]}}]}])
    chunks = list(provider.stream("Search", model="primary", tools=TOOLS))
    assert chunks[0].tool_calls[0].name == "web_search"
    assert all(chunk.delta == "" for chunk in chunks)
    assert chunks[-1].is_final
    client.stream.assert_called_once()


@pytest.mark.parametrize("provider_class", ALL_PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
def test_tool_history_and_choice_are_sent_to_native_api(provider_class, stream, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    call = ToolCall("call_7", "web_search", '{"query":"weather"}')
    history = [
        ChatMessage("user", "Search"),
        ChatMessage("assistant", "", tool_calls=(call,)),
        ChatMessage("tool", '{"result":"sunny"}', tool_call_id="call_7"),
    ]
    if provider_class is OllamaProvider:
        data = {"message": {"content": "Sunny."}, "done": True}
        response = client.stream.return_value.__enter__.return_value
        response.iter_lines.return_value = iter([json.dumps(data)])
    else:
        data = {"choices": [{"message": {"content": "Sunny."}}]}
        _sse_events(client, [{"choices": [{"delta": {"content": "Sunny."}}]}])
    client.post.return_value.json.return_value = data
    if stream:
        list(provider.stream(history, tools=TOOLS, tool_choice="none"))
        payload = client.stream.call_args.kwargs["json"]
    else:
        provider.send(history, tools=TOOLS, tool_choice="none")
        payload = client.post.call_args.kwargs["json"]
    assert payload["tools"] == TOOLS
    assert payload["tool_choice"] == "none"
    assert payload["messages"][2]["tool_call_id"] == "call_7"
    if provider_class is OllamaProvider:
        assert payload["messages"][2]["tool_name"] == "web_search"
    argument_value = payload["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert argument_value == ({"query": "weather"} if provider_class is OllamaProvider
                              else call.arguments)


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
@pytest.mark.parametrize("bad_calls", [
    {}, [None], [_call(arguments={"query": "unsafe coercion"})],
    [_call(id=None)], [_call(function={"name": ["bad"], "arguments": "{}"})],
    [_call(arguments="x" * (256 * 1024 + 1))],
    [_call()] * 129,
    [_call(), _call()],
])
def test_openai_rejects_malformed_complete_calls(provider_class, bad_calls, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": None, "tool_calls": bad_calls}}],
    }
    with pytest.raises(ValueError):
        provider.send("Search", model="primary")
    client.post.assert_called_once()


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
@pytest.mark.parametrize("bad_delta", [
    {"index": True}, {"index": -1}, {"index": 128}, {"index": 0, "function": []},
    {"index": 0, "function": {"arguments": {}}}, {"index": 0, "id": ["bad"]},
])
def test_openai_rejects_malformed_stream_calls(provider_class, bad_delta, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    _sse_events(client, [{"choices": [{"delta": {"tool_calls": [bad_delta]}}]}])
    with pytest.raises(ValueError):
        list(provider.stream("Search", model="primary"))
    client.stream.assert_called_once()


@pytest.mark.parametrize("stream", [False, True])
def test_ollama_native_dict_arguments_receive_stable_ids(stream, monkeypatch):
    provider = _provider(OllamaProvider, monkeypatch)
    client = _client(monkeypatch)
    data = {"message": {
        "content": "Checking.",
        "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "snow"}}}],
    }, "done": True}
    client.post.return_value.json.return_value = data
    response = client.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([json.dumps(data)])
    if stream:
        chunks = list(provider.stream("Search", tools=TOOLS))
        assert "".join(chunk.delta for chunk in chunks) == "Checking."
        accumulator = ToolCallAccumulator()
        for chunk in chunks:
            accumulator.add(chunk.tool_calls)
        calls = accumulator.finish()
        assert chunks[-1].is_final
    else:
        reply = provider.send("Search", tools=TOOLS)
        assert reply.content == "Checking."
        calls = reply.tool_calls
    assert calls[0].id == "ollama_call_0"
    assert calls[0].name == "web_search"
    assert json.loads(calls[0].arguments) == {"query": "snow"}


@pytest.mark.parametrize("arguments", [[], None, {1: "non-string key"}, {"bad": object()}])
def test_ollama_rejects_unsafe_argument_types(arguments, monkeypatch):
    provider = _provider(OllamaProvider, monkeypatch)
    client = _client(monkeypatch)
    client.post.return_value.json.return_value = {"message": {"tool_calls": [{
        "function": {"name": "web_search", "arguments": arguments},
    }]}}
    with pytest.raises(ValueError):
        provider.send("Search")


@pytest.mark.parametrize("provider_class", OPENAI_PROVIDERS)
def test_stream_argument_bound_applies_across_fragments(provider_class, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    fragment = "x" * (128 * 1024 + 1)
    _sse_events(client, [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, **_call(fragment)}]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": fragment},
        }]}}]},
    ])
    with pytest.raises(ValueError, match="256 KiB"):
        list(provider.stream("Search", model="primary"))
    client.stream.assert_called_once()


@pytest.mark.parametrize("provider_class", ALL_PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
def test_text_adapters_reject_image_messages_before_http(provider_class, stream, monkeypatch):
    provider = _provider(provider_class, monkeypatch)
    client = _client(monkeypatch)
    messages = [ChatMessage("user", "Look", images=(ChatImage(b"synthetic"),))]
    with pytest.raises(ValueError, match="VisionClient"):
        if stream:
            list(provider.stream(messages, tools=TOOLS))
        else:
            provider.send(messages, tools=TOOLS)
    client.post.assert_not_called()
    client.stream.assert_not_called()
