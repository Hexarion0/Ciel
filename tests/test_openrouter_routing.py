"""Free routing preserves explicit model choices and records the model actually used."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.providers.openrouter import OpenRouterProvider


def _client(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    monkeypatch.setattr("ciel.providers.openrouter.httpx.Client", lambda **_kwargs: client)
    return client


def test_default_uses_free_router_and_reports_selected_model(monkeypatch):
    client = _client(monkeypatch)
    client.post.return_value.json.return_value = {
        "model": "example/available-model:free",
        "choices": [{"message": {"content": "Hello."}}],
    }
    reply = OpenRouterProvider(api_key="synthetic").send("Hello")
    assert client.post.call_args.kwargs["json"]["model"] == "openrouter/free"
    assert reply.model == "example/available-model:free"
    assert reply.provider == "openrouter"


def test_explicit_model_is_preserved_when_response_has_no_model(monkeypatch):
    client = _client(monkeypatch)
    client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "Hello."}}],
    }
    reply = OpenRouterProvider(api_key="synthetic").send("Hello", model="chosen/model")
    assert client.post.call_args.kwargs["json"]["model"] == "chosen/model"
    assert reply.model == "chosen/model"


def test_stream_retains_actual_model_through_usage_event_and_final_marker(monkeypatch):
    client = _client(monkeypatch)
    response = client.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([
        "data: " + json.dumps({"model": "example/selected:free", "choices": [
            {"delta": {"content": "Hello."}},
        ]}),
        "data: " + json.dumps({"choices": [], "usage": {"total_tokens": 5}}),
        "data: [DONE]",
    ])
    chunks = list(OpenRouterProvider(api_key="synthetic").stream("Hello"))
    assert client.stream.call_args.kwargs["json"]["model"] == "openrouter/free"
    assert "".join(chunk.delta for chunk in chunks) == "Hello."
    assert chunks[-1].is_final
    assert all(chunk.model == "example/selected:free" for chunk in chunks)


@pytest.mark.parametrize("event", [
    {"error": {"code": 500, "message": "private provider details"}},
    {"choices": [{"delta": {}, "finish_reason": "error"}]},
])
def test_stream_error_never_marks_a_truncated_reply_successful(monkeypatch, event):
    client = _client(monkeypatch)
    response = client.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([
        'data: {"choices":[{"delta":{"content":"The first step is"}}]}',
        "data: " + json.dumps(event),
        "data: [DONE]",
    ])
    chunks = []
    with pytest.raises(RuntimeError, match="OpenRouter stream reported an error") as error:
        for chunk in OpenRouterProvider(api_key="synthetic").stream("Hello"):
            chunks.append(chunk)
    assert "private provider details" not in str(error.value)
    assert not any(chunk.is_final for chunk in chunks)
