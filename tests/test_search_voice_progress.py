"""Search acknowledgements stay audible for voice requests and typed corrections."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetResult, InternetService, InternetSource
from ciel.providers.base import LLMResponse, LLMStreamChunk, ToolCall, ToolCallDelta
from tests.test_ambient import run_callbacks


def _search_runtime(monkeypatch, *, streaming, events, speak, repeated=True):
    """Script native reads and a supported answer without network or audio hardware."""
    monkeypatch.setenv("INTERNET_ENABLED", "true")
    monkeypatch.setenv("INTERNET_AUTOMATIC", "true")
    monkeypatch.setenv("MEMORY_FLUSH_ENABLED", "false")
    monkeypatch.setenv("MEMORY_CONSOLIDATION_ENABLED", "false")
    monkeypatch.setattr("ciel.tool_progress.ToolProgress.line",
                        lambda *_args, **_kwargs: "I'll check online.")
    quote = "Cedar is a synthetic example."
    source = InternetSource("Cedar", "https://example.com/cedar", content=quote)
    service = MagicMock()
    service.config = InternetConfig()

    def lookup(*_args, **_kwargs):
        events.append("lookup")
        return InternetResult(attempted=True, query="Cedar", sources=[source])

    service.lookup.side_effect = lookup
    factory = MagicMock(return_value=service)
    factory._context = InternetService._context
    monkeypatch.setattr("ciel.internet.InternetService", factory)
    calls = [ToolCall("search", "web_search", '{"query":"Cedar"}')]
    if repeated:
        calls.append(ToolCall("page", "read_page", '{"url":"https://example.com/cedar"}'))
    answer = json.dumps({"claims": [{"text": quote, "source": 1, "quote": quote}]})
    router = MagicMock()

    def relevance(*_args, **_kwargs):
        return LLMResponse('{"relevant_claims":[1]}', "synthetic", "nim")

    rounds = [LLMResponse("", "synthetic", "nim", tool_calls=(call,)) for call in calls]
    rounds.append(LLMResponse(answer, "synthetic", "nim"))

    def respond(messages, **kwargs):
        if messages[0].content.startswith("WEB GOAL CHECK"):
            return relevance(messages, **kwargs)
        return rounds.pop(0)

    def stream(messages, **kwargs):
        response = respond(messages, **kwargs)
        yield LLMStreamChunk(response.content, response.model, response.provider, tool_calls=tuple(
            ToolCallDelta(index, call.id, call.name, call.arguments)
            for index, call in enumerate(response.tool_calls)
        ))

    # Buffered research may use send even when ordinary conversation streams.
    # Both transports describe the same conversation; the assertions below still
    # verify two actual reads, one spoken lead-in, citations, and one spoken answer.
    router.send.side_effect = respond
    router.stream.side_effect = stream
    tts = MagicMock()
    tts.supports_sentence_pipeline = False
    tts.speak.side_effect = speak
    return service, router, tts, quote


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", ["voice", "typed", "correction"])
def test_search_announces_once_before_lookup_and_speaks_the_completed_answer(
    tmp_path, monkeypatch, capsys, streaming, mode,
):
    events = []

    def speak(text, **kwargs):
        events.append("ack" if "lookup" not in events else "final")
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        return True

    service, router, tts, quote = _search_runtime(
        monkeypatch, streaming=streaming, events=events, speak=speak,
    )
    if mode == "voice":
        info, message = {"wake_verified": True}, "Hey Ciel, search Cedar."
    else:
        info = {"input_source": "keyboard", "voice_search_progress": mode == "correction"}
        message = "Cedar*" if mode == "correction" else "Search Cedar."
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(message, info)], router=router, tts=tts,
        streaming=streaming, controller=True,
    )
    try:
        assert events == ["ack", "lookup", "lookup", "final"]
        assert service.lookup.call_count == 2
        assert tts.speak.call_count == 2
        if mode != "voice":
            tts.begin_stream.assert_not_called()
        assert capsys.readouterr().out.count("I'll check online.") == 1
        assert [turn.content for turn in turns if turn.role == "assistant"] == [
            quote if mode == "voice" else quote + " [1]",
        ]
    finally:
        store.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_canceling_search_acknowledgement_prevents_lookup(
    tmp_path, monkeypatch, streaming,
):
    events, canceled = [], Event()

    def speak(_text, **_kwargs):
        events.append("ack")
        canceled.set()
        return False

    service, router, tts, _ = _search_runtime(
        monkeypatch, streaming=streaming, events=events, speak=speak, repeated=False,
    )
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Hey Ciel, search Cedar.", {
            "wake_verified": True, "cancel_event": canceled,
        })], router=router, tts=tts, streaming=streaming, controller=True,
    )
    try:
        assert events == ["ack"]
        service.lookup.assert_not_called()
        assert not [turn for turn in turns if turn.role == "assistant"]
    finally:
        store.close()
