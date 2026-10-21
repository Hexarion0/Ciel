"""Unit tests for AI providers and fallback manager."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ciel.config import DEFAULT_NIM_MODEL, ProviderConfig
from ciel.providers.base import ChatMessage, LLMProvider, LLMResponse, LLMStreamChunk
from ciel.providers.groq import GroqProvider
from ciel.providers.manager import ProviderManager
from ciel.providers.nim import NimProvider
from ciel.providers.ollama import OllamaProvider
from ciel.providers.openrouter import OpenRouterProvider


def test_chat_message_serialization():
    """Verify ChatMessage dictionary conversion and string normalization."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}

    normalized = LLMProvider.normalize_messages("Hello world")
    assert len(normalized) == 1
    assert normalized[0].content == "Hello world"
    assert normalized[0].role == "user"


@patch("httpx.Client.post")
def test_nim_provider_send(mock_post):
    """Verify NIM provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "NIM answer"}}],
        "usage": {"total_tokens": 42},
    }
    mock_post.return_value = mock_resp

    provider = NimProvider(api_key="nvapi-test-key-12345")
    assert provider.is_configured() is True

    res = provider.send("Test prompt")
    assert res.content == "NIM answer"
    assert res.provider == "nim"
    assert res.usage["total_tokens"] == 42


@patch("httpx.Client.post")
def test_groq_provider_send(mock_post):
    """Verify Groq provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Groq fast answer"}}],
        "usage": {"total_tokens": 15},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(api_key="gsk_test-key-12345")
    assert provider.is_configured() is True

    res = provider.send("Quick question")
    assert res.content == "Groq fast answer"
    assert res.provider == "groq"


@patch("httpx.Client.post")
def test_openrouter_provider_send(mock_post):
    """Verify OpenRouter provider parses chat completions accurately."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter answer"}}],
        "usage": {"total_tokens": 30},
    }
    mock_post.return_value = mock_resp

    provider = OpenRouterProvider(api_key="sk-or-v1-test")
    assert provider.is_configured() is True

    res = provider.send("Explain quantum computing")
    assert res.content == "OpenRouter answer"
    assert res.provider == "openrouter"


@patch("httpx.Client.post")
def test_ollama_provider_send(mock_post):
    """Verify local Ollama provider parses chat completions."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Ollama local response"},
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    mock_post.return_value = mock_resp

    provider = OllamaProvider(host="http://localhost:11434")
    assert provider.is_configured() is True

    res = provider.send("Local prompt")
    assert res.content == "Ollama local response"
    assert res.provider == "ollama"


def test_provider_manager_fallback_chain():
    """Verify fallback chain switches to next provider when primary fails."""
    manager = ProviderManager(fallback_chain=["nim", "groq", "ollama"])

    # Mock failing NIM provider
    failing_nim = MagicMock(spec=LLMProvider)
    failing_nim.is_configured.return_value = True
    failing_nim.send.side_effect = Exception("NIM rate limit exceeded")

    # Mock working Groq provider
    working_groq = MagicMock(spec=LLMProvider)
    working_groq.is_configured.return_value = True
    working_groq.send.return_value = LLMResponse(
        content="Fallback response from Groq",
        model="llama-3.1-8b-instant",
        provider="groq",
    )

    manager.register_provider("nim", failing_nim)
    manager.register_provider("groq", working_groq)

    # Should attempt NIM, catch error, and successfully return from Groq
    res = manager.send_with_fallback("Hello assistant", preferred_provider="nim")
    assert res.content == "Fallback response from Groq"
    assert res.provider == "groq"
    failing_nim.send.assert_called_once()
    working_groq.send.assert_called_once()


def test_provider_manager_all_fail():
    """Verify exception is raised when all providers in fallback chain fail."""
    manager = ProviderManager(fallback_chain=["nim"])
    failing_nim = MagicMock(spec=LLMProvider)
    failing_nim.is_configured.return_value = True
    failing_nim.send.side_effect = Exception("API connection timeout")

    manager.register_provider("nim", failing_nim)

    with pytest.raises(RuntimeError, match="All AI providers in fallback chain failed"):
        manager.send_with_fallback("Hello")


def test_stream_falls_back_before_text_but_never_splices_after_partial_reply():
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    primary, backup = MagicMock(), MagicMock()
    primary.is_configured.return_value = backup.is_configured.return_value = True
    primary.stream.side_effect = RuntimeError("offline")
    backup.stream.side_effect = lambda **kwargs: iter([LLMStreamChunk("Backup.", "test", "groq")])
    manager.register_provider("nim", primary)
    manager.register_provider("groq", backup)
    assert [part.delta for part in manager.stream_with_fallback("Hello")] == ["Backup."]
    backup.reset_mock()

    def broken(**kwargs):
        yield LLMStreamChunk("Partial.", "test", "nim")
        raise RuntimeError("connection lost")

    primary.stream.side_effect = broken
    stream = manager.stream_with_fallback("Hello")
    assert next(stream).delta == "Partial."
    with pytest.raises(RuntimeError, match="after a partial reply"):
        next(stream)
    backup.stream.assert_not_called()


def test_canceled_stream_does_not_start_fallback():
    from threading import Event

    canceled = Event()
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    primary, backup = MagicMock(), MagicMock()
    primary.is_configured.return_value = backup.is_configured.return_value = True

    def interrupted(**kwargs):
        canceled.set()
        raise RuntimeError("closed by barge-in")

    primary.stream.side_effect = interrupted
    manager.register_provider("nim", primary)
    manager.register_provider("groq", backup)
    assert list(manager.stream_with_fallback("Hello", cancel_event=canceled)) == []
    backup.stream.assert_not_called()


@pytest.mark.parametrize("provider_class", [NimProvider, GroqProvider, OpenRouterProvider,
                                          OllamaProvider])
def test_stream_cancel_control_is_not_sent_to_api(provider_class, monkeypatch):
    import json
    from threading import Event

    provider = (
        provider_class() if provider_class is OllamaProvider else provider_class(api_key="test")
    )
    response, client = MagicMock(), MagicMock()
    client.__enter__.return_value = client
    client.stream.return_value.__enter__.return_value = response
    data = ({"message": {"content": "Hi."}, "done": True} if provider_class is OllamaProvider
            else {"choices": [{"delta": {"content": "Hi."}}]})
    response.iter_lines.return_value = iter([
        json.dumps(data) if provider_class is OllamaProvider else "data: " + json.dumps(data),
    ])
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    assert [part.delta for part in provider.stream("Hi", cancel_event=Event())] == ["Hi."]
    payload = client.stream.call_args.kwargs["json"]
    assert "cancel_event" not in payload
    json.dumps(payload)  # Must remain JSON serializable.


def test_http_stream_client_closes_when_canceled(monkeypatch):
    from threading import Event

    from ciel.providers.streaming import streaming_client

    canceled, closed = Event(), Event()
    client = MagicMock()
    client.__enter__.return_value = client
    client.close.side_effect = closed.set
    monkeypatch.setattr("httpx.Client", lambda **kwargs: client)
    with streaming_client(5, canceled):
        canceled.set()
        assert closed.wait(0.5)


def completion(content, **kwargs):
    import httpx

    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://example.test/completions"),
        json={"choices": [{"message": {"content": content, **kwargs}}]},
    )


@patch("httpx.Client.post")
@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize("wrapper", ["plain", "fenced", "reasoning", "reasoning_fenced"])
def test_nim_preserves_complete_structured_judgments(mock_post, as_array, wrapper):
    """Gate JSON survives prose markers and literal tags within interpretation data."""
    judgment = {
        "addressed": True,
        "confidence": 0.95,
        "interpretation": "Direct response: Decision: Final answer: <think>literal</think>",
    }
    encoded = json.dumps([judgment] if as_array else judgment)
    visible = f"```json\n{encoded}\n```" if "fenced" in wrapper else encoded
    content = visible
    if "reasoning" in wrapper:
        content = "<think>Private reasoning.</think>\n" + content
    mock_post.return_value = completion(content)
    response = NimProvider(api_key="test-key").send("Judge this speech.")
    assert response.content == visible
    parsed = json.loads(response.content.removeprefix("```json\n").removesuffix("\n```"))
    assert parsed == ([judgment] if as_array else judgment)
    mock_post.assert_called_once()


@pytest.mark.parametrize("content, expected", [
    ("<think>Private.</think>\nHello, hexarion.", "Hello, hexarion."),
    ("Reasoning: private notes.\n\nFinal answer: Hello.", "Hello."),
    ("Final response: \"Hello.\"", "Hello."),
    ("<think>Unfinished private reasoning.", ""),
])
def test_nim_structured_reply_guard_keeps_conversation_cleanup(content, expected):
    assert NimProvider.clean_reasoning(content) == expected


@patch("httpx.Client.post")
def test_nim_null_reply_tries_backup_without_speaking_reasoning(mock_post):
    mock_post.side_effect = [
        completion(None, reasoning_content="private reasoning"),
        completion("I'm here."),
    ]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/llama-3.2-90b-vision-instruct"
    response = provider.send("Are you good?", max_tokens=400)
    assert response.content == "I'm here."
    assert response.model == provider.fallback_model
    initial = mock_post.call_args_list[0].kwargs["json"]
    backup = mock_post.call_args_list[1].kwargs["json"]
    assert initial["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in backup
    assert mock_post.call_count == 2


@patch("httpx.Client.post")
@pytest.mark.parametrize("content", [None, "", "<think>private reasoning</think>"])
def test_nim_empty_replies_fail_clearly_after_one_backup(mock_post, content):
    mock_post.side_effect = [completion(content), completion(content)]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/backup"
    with pytest.raises(RuntimeError, match="no user-visible answer"):
        provider.send("Hello")
    assert mock_post.call_count == 2


@patch("httpx.Client.post")
def test_nim_caller_can_explicitly_enable_reasoning(mock_post):
    mock_post.return_value = completion("visible answer", reasoning_content="private thoughts")
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    template = {"enable_thinking": True, "low_effort": True}
    assert (
        provider.send("A complex query", chat_template_kwargs=template).content == "visible answer"
    )
    assert mock_post.call_args.kwargs["json"]["chat_template_kwargs"] == template
    assert template == {"enable_thinking": True, "low_effort": True}


def test_nim_template_options_are_model_specific_for_streams():
    payload = {"model": "nvidia/nemotron-3-super-120b-a12b", "stream": True}
    assert NimProvider._model_payload(payload, payload["model"])["chat_template_kwargs"] == {
        "enable_thinking": False,
    }
    assert "chat_template_kwargs" not in NimProvider._model_payload(payload, "meta/backup")
    assert "chat_template_kwargs" not in payload


@patch("httpx.Client.post")
@patch("ciel.providers.nim.time.sleep")
def test_nim_transient_http_errors_retry_then_use_backup(mock_sleep, mock_post):
    import httpx

    request = httpx.Request("POST", "https://example.test/completions")
    response = httpx.Response(503, request=request)
    errors = [
        httpx.HTTPStatusError("Unavailable", request=request, response=response) for _ in range(3)
    ]
    mock_post.side_effect = [*errors, completion("backup answer")]
    provider = NimProvider(api_key="test-key", model="nvidia/nemotron-3-super-120b-a12b")
    provider.fallback_model = "meta/backup"
    assert provider.send("Hello").content == "backup answer"
    assert mock_post.call_count == 4
    assert mock_sleep.call_count == 2  # No pointless delay after the last failure.


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [404, 410, 401, 403, 503])
def test_nim_http_fallback_policy(monkeypatch, streaming, status):
    """Retirement skips retries; auth fails fast; transient errors retry twice."""
    import httpx

    requests = []
    provider = NimProvider(api_key="test", model="retired-model")
    provider.fallback_model = "backup-model"

    def handle(request):
        import json

        model = json.loads(request.content)["model"]
        requests.append(model)
        if model == "retired-model":
            return httpx.Response(status)
        if streaming:
            return httpx.Response(200, text=(
                'data: {"choices": []}\n\n'
                'data: {"choices": [{"delta": {"content": "Backup."}}]}\n\n'
                'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            ))
        return completion("Backup.")

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    sleeper = MagicMock()
    monkeypatch.setattr("ciel.providers.nim.time.sleep", sleeper)

    def reply():
        if streaming:
            chunks = list(provider.stream("Hello"))
            assert chunks[-1].is_final
            return "".join(chunk.delta for chunk in chunks)
        return provider.send("Hello").content

    if status in (401, 403):
        with pytest.raises(httpx.HTTPStatusError):
            reply()
        assert requests == ["retired-model"]
    else:
        assert reply() == "Backup."
        assert requests == ["retired-model"] * (3 if status == 503 else 1) + ["backup-model"]
    assert sleeper.call_count == (2 if status == 503 else 0)


def test_nim_failed_stream_backup_is_attempted_only_once(monkeypatch):
    import httpx

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(410 if len(requests) == 1 else 503)

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    provider = NimProvider(api_key="test", model="retired-model")
    provider.fallback_model = "backup-model"
    with pytest.raises(httpx.HTTPStatusError):
        list(provider.stream("Hello"))
    assert len(requests) == 2


def test_nim_canceled_retirement_response_does_not_try_backup(monkeypatch):
    from threading import Event

    import httpx

    cancel = Event()
    requests = []

    def handle(request):
        requests.append(request)
        cancel.set()
        return httpx.Response(410)

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    provider = NimProvider(api_key="test", model="retired-model")
    assert list(provider.stream("Hello", cancel_event=cancel)) == []
    assert len(requests) == 1


def test_nim_stream_stops_at_finish_reason_without_waiting_for_done(monkeypatch):
    client, response = MagicMock(), MagicMock()
    client.__enter__.return_value = client
    client.stream.return_value.__enter__.return_value = response

    def lines():
        yield 'data: {"choices": []}'
        yield 'data: {"choices": [{"delta": {"content": "Hi!"}, "finish_reason": "stop"}]}'
        raise AssertionError("Should not wait for another SSE event after the final choice")

    response.iter_lines.side_effect = lines
    monkeypatch.setattr("httpx.Client", lambda **kw: client)
    chunks = list(NimProvider(api_key="test").stream("Hello"))
    assert "".join(part.delta for part in chunks) == "Hi!"
    assert chunks[-1].is_final


@pytest.mark.parametrize("thinking", [None, True])
def test_lightning_uses_final_answer_mode_by_default(thinking):
    model = "nvidia/nemotron-3.5-lightning-30b-a3b"
    payload = {"model": model}
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    prepared = NimProvider._model_payload(payload, model)
    assert prepared["chat_template_kwargs"]["enable_thinking"] is (thinking or False)


@pytest.mark.parametrize('partial', [False, True])
def test_nim_only_falls_back_before_visible_stream_content(monkeypatch, partial):
    import httpx

    client, primary, backup = MagicMock(), MagicMock(), MagicMock()
    client.__enter__.return_value = client
    client.stream.side_effect = [primary, backup]
    first = primary.__enter__.return_value
    second = backup.__enter__.return_value

    def lines():
        if partial:
            yield 'data: {"choices": [{"delta": {"content": "Partial."}}]}'
            raise httpx.ReadError('Disconnected')
        yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'

    first.iter_lines.side_effect = lines
    second.iter_lines.return_value = iter([
        'data: {"choices": [{"delta": {"content": "Backup."}, "finish_reason": "stop"}]}',
    ])
    monkeypatch.setattr('httpx.Client', lambda **kw: client)
    provider = NimProvider(api_key='test', model='primary')
    provider.fallback_model = 'backup'
    stream = provider.stream('Hello')
    if partial:
        assert next(stream).delta == 'Partial.'
        with pytest.raises(httpx.ReadError):
            next(stream)
        assert client.stream.call_count == 1
    else:
        chunks = list(stream)
        assert [chunk.delta for chunk in chunks] == ['Backup.', '']
        assert chunks[-1].is_final
        assert all(chunk.model == 'backup' for chunk in chunks)
        assert client.stream.call_count == 2


@pytest.mark.parametrize("streaming", [False, True])
def test_controller_nim_never_falls_back_to_heavy_model(monkeypatch, streaming):
    import httpx

    requests = []
    provider = NimProvider(api_key="test", model="nvidia/nemotron-3.5-lightning-30b-a3b")

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(404)

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    with pytest.raises(httpx.HTTPStatusError):
        if streaming:
            list(provider.stream("Decide", allow_model_fallback=False))
        else:
            provider.send("Decide", allow_model_fallback=False)
    assert len(requests) == 1
    assert requests[0]["model"] == provider.default_model
    assert "allow_model_fallback" not in requests[0]
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [404, 410])
def test_nim_default_primary_and_backup_do_not_repeat_missing_model(
    monkeypatch, streaming, status,
):
    """One available default must not become an implicit heavy or duplicate retry."""
    import httpx

    requests = []
    provider = NimProvider(api_key="test", config=ProviderConfig())

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status)

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    with pytest.raises(httpx.HTTPStatusError):
        if streaming:
            list(provider.stream("Hello"))
        else:
            provider.send("Hello")
    assert provider.default_model == provider.complex_model == provider.fallback_model == (
        DEFAULT_NIM_MODEL
    )
    assert [request["model"] for request in requests] == [DEFAULT_NIM_MODEL]


def test_nim_provider_preserves_custom_model_overrides():
    """Provider construction must retain all selected routes and explicit model override."""
    provider = NimProvider(api_key="test", model="explicit-primary", config=ProviderConfig(
        nim_model="configured-primary", nim_complex_model="configured-complex",
        nim_fallback_model="configured-backup",
    ))
    assert provider.default_model == "explicit-primary"
    assert provider.complex_model == "configured-complex"
    assert provider.fallback_model == "configured-backup"


def test_nim_blank_backup_uses_the_available_default():
    provider = NimProvider(api_key="test", config=ProviderConfig(nim_fallback_model=""))
    assert provider.fallback_model == DEFAULT_NIM_MODEL


@pytest.mark.parametrize("content", ["", "   ", None])
def test_blank_synchronous_reply_uses_provider_fallback(content):
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    primary, backup = MagicMock(), MagicMock()
    primary.is_configured.return_value = backup.is_configured.return_value = True
    primary.send.return_value = LLMResponse(content, "missing-reply", "nim")
    backup.send.return_value = LLMResponse("A visible reply.", "backup-model", "groq")
    manager.register_provider("nim", primary)
    manager.register_provider("groq", backup)
    response = manager.send_with_fallback("Hi", preferred_provider="nim")
    assert response.content == "A visible reply."
    assert response.provider == "groq"


@pytest.mark.parametrize("mode", ["send", "stream", "router"])
def test_controller_fallback_uses_provider_default_without_nim_options(monkeypatch, mode):
    import httpx

    requests = []
    decision = json.dumps({"action": "answer", "intent": "chat", "confidence": 1.0})

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["stream"]:
            event = json.dumps({"choices": [{"delta": {"content": decision}}]})
            return httpx.Response(200, text=f"data: {event}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"choices": [{"message": {"content": decision}}]})

    client_class = httpx.Client
    monkeypatch.setattr("httpx.Client", lambda **kw: client_class(
        transport=httpx.MockTransport(handle), **kw,
    ))
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    primary = MagicMock()
    primary.is_configured.return_value = True
    primary.send.side_effect = primary.stream.side_effect = RuntimeError("Unavailable")
    manager.register_provider("nim", primary)
    backup = GroqProvider(api_key="test")
    backup.default_model = "selected-groq-model"
    manager.register_provider("groq", backup)
    if mode == "router":
        from threading import Event

        from ciel.providers.router import ModelRouter

        response = ModelRouter(manager).send(
            [ChatMessage("user", "Choose an action")],
            purpose="controller", cancel_event=Event(),
        )
        assert response.provider == "groq"
        assert response.model == "selected-groq-model"
        assert primary.stream.call_args.kwargs["allow_model_fallback"] is False
    else:
        request = manager.stream_with_fallback if mode == "stream" else manager.send_with_fallback
        response = request(
            "Choose an action", preferred_provider="nim", model="nim-controller-model",
            allow_model_fallback=False,
        )
    if mode == "stream":
        assert json.loads("".join(part.delta for part in response))["action"] == "answer"
    else:
        assert json.loads(response.content)["action"] == "answer"
    assert requests[0]["model"] == "selected-groq-model"
    assert "allow_model_fallback" not in requests[0]


def test_whitespace_only_stream_uses_visible_fallback():
    manager = ProviderManager(fallback_chain=["nim", "groq"])
    primary, backup = MagicMock(), MagicMock()
    primary.is_configured.return_value = backup.is_configured.return_value = True
    primary.stream.return_value = iter([LLMStreamChunk("  \n", "blank", "nim")])
    backup.stream.return_value = iter([LLMStreamChunk("Backup.", "visible", "groq")])
    manager.register_provider("nim", primary)
    manager.register_provider("groq", backup)
    chunks = list(manager.stream_with_fallback("Hi"))
    assert "".join(chunk.delta for chunk in chunks).strip() == "Backup."
    assert chunks[-1].provider == "groq"


@pytest.mark.parametrize("streaming", [False, True])
def test_case_insensitive_provider_preference_preserves_explicit_model(streaming):
    manager = ProviderManager(fallback_chain=["groq"])
    provider = MagicMock()
    provider.is_configured.return_value = True
    provider.send.return_value = LLMResponse("Reply", "explicit-model", "groq")
    provider.stream.return_value = iter([LLMStreamChunk("Reply", "explicit-model", "groq")])
    manager.register_provider("groq", provider)
    if streaming:
        list(manager.stream_with_fallback("Hi", preferred_provider="GROQ", model="explicit-model"))
        assert provider.stream.call_args.kwargs["model"] == "explicit-model"
    else:
        manager.send_with_fallback("Hi", preferred_provider="GROQ", model="explicit-model")
        assert provider.send.call_args.kwargs["model"] == "explicit-model"
