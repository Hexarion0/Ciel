"""Internet evidence reaches the shared runtime without becoming spoken citations."""

import json
from datetime import datetime, timezone
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetResult, InternetService, InternetSource
from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.providers.base import LLMResponse, LLMStreamChunk
from tests.test_ambient import run_callbacks
from tests.test_controller import _web_model_responses


def stub_internet(monkeypatch, result):
    """Replace retrieval only, keeping the real conversation and SQLite paths."""
    service = MagicMock()
    service.config = InternetConfig()
    service.prepare.return_value = result
    factory = MagicMock(return_value=service)
    factory._context = InternetService._context
    monkeypatch.setattr("ciel.internet.InternetService", factory)
    return service


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_voice_and_text_use_evidence_and_persist_sources(
    tmp_path, monkeypatch, capsys, typed, streaming,
):
    evidence = "PUBLIC WEB EVIDENCE: A newly announced product."
    reply = "The company announced a new product."
    source = InternetSource("Product announcement", "https://example.com/announcement", reply)
    service = stub_internet(
        monkeypatch, InternetResult(attempted=True, sources=[source], context=evidence),
    )
    router = MagicMock()
    router.send.return_value = LLMResponse(json.dumps({"claims": [
        {"text": reply, "source": 1, "quote": reply},
    ]}), "test", "test")
    router.send.side_effect = _web_model_responses(
        lambda *_args, **_kwargs: router.send.return_value,
    )
    router.stream.side_effect = lambda *_args, **_kwargs: iter([
        LLMStreamChunk(reply, "test", "test"),
    ])
    info = {"input_source": "keyboard"} if typed else {}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, what's the current product announcement?", info)],
        router=router, streaming=streaming,
    )
    try:
        router.stream.assert_not_called()  # Web claims must be validated before playback.
        report_call = next(call for call in reversed(router.send.call_args_list)
                           if call.kwargs.get("purpose") == "conversation")
        messages = report_call.args[0]
        assert any(message.role == "system" and source.url in message.content
                   and reply in message.content
                   for message in messages)
        assert "Application completed public internet retrieval" in messages[0].content
        assert "source excerpts do not answer" in messages[0].content
        assert "older assistant claims" in messages[0].content
        expected_query = "what's the current product announcement?"
        if typed:
            expected_query = "Ciel, " + expected_query
        assert service.prepare.call_args.args[0] == expected_query
        assert all(message.role != "system"
                   for message in service.prepare.call_args.kwargs["recent_messages"])
        assistant = [turn for turn in turns if turn.role == "assistant"]
        expected_reply = reply + " [1]" if typed else reply
        assert len(assistant) == 1 and assistant[0].content == expected_reply
        assert assistant[0].sources == [{"title": source.title, "url": source.url}]
        if typed:
            tts.speak.assert_called_once()
            assert tts.speak.call_args.args[0] == reply
            tts.begin_stream.assert_not_called()
        assert all(source.url not in call.args[0] for call in tts.speak.call_args_list)
        assert "[system]: Source 1: Product announcement" in capsys.readouterr().out
        assert all(source.url not in message.content
                   for message in store.get_recent_turns(assistant[0].session_id))
    finally:
        store.close()


def test_relevant_saved_memory_is_available_for_query_enrichment(tmp_path, monkeypatch):
    seed = MemoryStore(str(tmp_path / "memory.db"))
    try:
        seed.save_memory(MemoryItem(
            "You prefer budget laptops.", memory_type=MemoryType.PREFERENCE,
        ))
    finally:
        seed.close()
    service = stub_internet(monkeypatch, InternetResult())
    router = MagicMock()
    router.send.return_value = LLMResponse("Here are options.", "test", "test")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, compare laptops for me.", {})], router=router,
    )
    try:
        assert any("budget laptops" in memory
                   for memory in service.prepare.call_args.kwargs["memories"])
    finally:
        store.close()


def test_weather_recalls_an_explicit_saved_city(tmp_path, monkeypatch):
    from ciel.memory.service import MemoryService

    seed = MemoryStore(str(tmp_path / "memory.db"))
    try:
        assert "saved" in MemoryService(seed).handle("Remember I live in Ulaanbaatar.")
    finally:
        seed.close()
    service = stub_internet(monkeypatch, InternetResult())
    router = MagicMock()
    router.send.return_value = LLMResponse("Here is the forecast.", "test", "test")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, what's the weather here?", {})], router=router,
    )
    try:
        assert any("Ulaanbaatar" in memory
                   for memory in service.prepare.call_args.kwargs["memories"])
    finally:
        store.close()


def test_missing_city_is_a_local_clarification_without_provider(tmp_path, monkeypatch):
    reply = "Which city should I check the weather for?"
    stub_internet(monkeypatch, InternetResult(attempted=True, reply=reply))
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, what's the weather here?", {})], router=router,
    )
    try:
        router.send.assert_not_called()
        router.stream.assert_not_called()
        assert tts.speak.call_args.args[0] == reply
        assert turns[-1].provider == "local" and turns[-1].model == "internet"
    finally:
        store.close()


def test_lookup_failure_cannot_generate_a_fabricated_answer(tmp_path, monkeypatch):
    context = "Lookup failed. Current facts are unverified; offer only labeled background."
    stub_internet(monkeypatch, InternetResult(attempted=True, context=context))
    router = MagicMock()
    router.send.return_value = LLMResponse("I cannot search. Made-up current news.", "test", "test")
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, what's today's news?", {})], router=router,
    )
    try:
        assert all(call.kwargs["purpose"] == "controller" for call in router.send.call_args_list)
        router.stream.assert_not_called()
        reply = tts.speak.call_args.args[0]
        assert "couldn't" in reply and "Made-up current news" not in reply
        assert turns[-1].provider == "local" and turns[-1].content == reply
        assert turns[-1].sources == []
    finally:
        store.close()


def test_canceled_lookup_cannot_generate_speak_or_save_an_answer(tmp_path, monkeypatch):
    cancel_event = Event()
    service = stub_internet(monkeypatch, InternetResult())

    def cancel(*_args, **_kwargs):
        cancel_event.set()
        return InternetResult(attempted=True, context="Stale evidence.")

    service.prepare.side_effect = cancel
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, today's news?", {"cancel_event": cancel_event})], router=router,
    )
    try:
        router.send.assert_not_called()
        router.stream.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_local_only_session_skips_retrieval_for_followup(tmp_path, monkeypatch):
    """Legacy private history remains protected after the local command's removal."""
    def initialize(store, session_id):
        store.save_memory(MemoryItem(
            content="A private conversation from an earlier release.",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": session_id, "local_only": True},
        ))

    service = stub_internet(monkeypatch, InternetResult())
    router = MagicMock()
    router.send.return_value = LLMResponse("I can answer locally.", "test", "ollama")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch,
        [("Today's news?", {"input_source": "keyboard"}),
         ("What about tomorrow?", {"input_source": "keyboard"})],
        router=router, initialize_conversation=initialize,
    )
    try:
        service.prepare.assert_not_called()
        assert all(any(message.local_only for message in call.args[0])
                   for call in router.send.call_args_list)
        assert all("Internet retrieval is blocked for this private conversation"
                   in call.args[0][0].content for call in router.send.call_args_list)
    finally:
        store.close()


def test_search_free_conversation_keeps_accurate_internet_capability(tmp_path, monkeypatch):
    stub_internet(monkeypatch, InternetResult())
    router = MagicMock()
    router.send.return_value = LLMResponse("I'm here to help.", "test", "test")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, explain photosynthesis.", {})], router=router,
    )
    try:
        prompt = router.send.call_args.args[0][0].content
        assert "Public internet retrieval is enabled" in prompt
        assert "No lookup was performed for this turn" in prompt
        assert "do not deny the application's internet capability" in prompt
        assert "Do not narrate a search" in prompt
    finally:
        store.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_logged_wrong_topic_results_cannot_reach_generation_or_speech(
    tmp_path, monkeypatch, streaming,
):
    from ciel.memory import ConversationManager

    def initialize(store, session_id):
        conversation = ConversationManager(store, session_id=session_id)
        conversation.add_turn("user", (
            "Can you find like the Teensura anime in last episode and when it ends on the "
            "light normal so I can read it from there?"
        ))
        conversation.add_turn("assistant", (
            "I cannot search the internet. Season 2 ends in volume 8 chapter 54."
        ))

    search = MagicMock(return_value=[{
        "title": "How to Search the Internet", "href": "https://example.com/search-tips",
        "body": "Learn how to search the internet and select a search engine.",
    }])
    monkeypatch.setattr("ciel.internet.InternetService._search", search)
    router = MagicMock()
    router.send.return_value = LLMResponse("Another unsupported answer.", "test", "test")
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Can you search like the last episode? Like search the internet?",
          {"input_source": "keyboard"})], router=router, streaming=streaming,
        initialize_conversation=initialize,
    )
    try:
        assert "teensura" in search.call_args.args[0].casefold()
        assert all(call.kwargs["purpose"] == "controller" for call in router.send.call_args_list)
        router.stream.assert_not_called()
        assert tts.speak.call_count == 2  # Search acknowledgement and the honest fallback.
        assert tts.speak.call_args.args[0] == turns[-1].content
        assert all("Another unsupported answer" not in call.args[0]
                   for call in tts.speak.call_args_list)
        tts.begin_stream.assert_not_called()
        assert "couldn't" in turns[-1].content
        assert "Another unsupported answer" not in turns[-1].content
        assert turns[-1].provider == "local" and turns[-1].sources == []
        assert "volume 8" not in turns[-1].content
    finally:
        store.close()


def test_local_clock_bypasses_retrieval(tmp_path, monkeypatch):
    service = stub_internet(monkeypatch, InternetResult())
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, what time is it?", {})],
    )
    store.close()
    service.prepare.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_web_sources_are_visible_during_playback_and_survive_interruption(
    tmp_path, monkeypatch, streaming,
):
    from ciel import web
    from ciel.audio.tts import TextToSpeech

    reply = "Here is the news from a public announcement."
    source = InternetSource(
        "Public announcement", "https://example.com/news", reply,
        published_at=datetime.now(timezone.utc).isoformat(), date_kind="published",
    )
    stub_internet(
        monkeypatch,
        InternetResult(attempted=True, sources=[source], context="Verified public evidence."),
    )
    router = MagicMock()
    router.send.return_value = LLMResponse(json.dumps({"claims": [
        {"text": reply, "source": 1, "quote": reply},
    ]}), "test", "test")
    router.send.side_effect = _web_model_responses(
        lambda *_args, **_kwargs: router.send.return_value,
    )
    router.stream.side_effect = lambda *_args, **_kwargs: iter([
        LLMStreamChunk(reply, "test", "test"),
    ])
    ui = web.DesktopWebUI("owner", "hey ciel")
    ui.start = MagicMock(return_value="http://127.0.0.1:8765")
    monkeypatch.setattr(web, "DesktopWebUI", lambda *_args, **_kwargs: ui)
    canceled = Event()
    tts = MagicMock()
    tts.clean_text_for_speech.side_effect = TextToSpeech.clean_text_for_speech

    def speak(text, **controls):
        controls["on_start"]()
        assert ui.snapshot()["messages"][-1]["sources"] == [source.to_dict()]
        controls["on_progress"](text[:10], False, False)
        canceled.set()
        controls["on_progress"](text[:10], True, True)
        return False

    tts.speak.side_effect = speak
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, today's news?", {"cancel_event": canceled})],
        router=router, tts=tts, streaming=streaming,
        cli_args=("web", "--no-open-browser"),
    )
    try:
        shown = ui.snapshot()["messages"][-1]
        assert shown["interrupted"] and shown["sources"] == [source.to_dict()]
        assert turns[-1].sources == [source.to_dict()]
        assert shown["text"] == reply[:10]
    finally:
        store.close()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("content", [
    "Start Volume 17. Season 4 adapts volumes 14–16.",
    json.dumps({"claims": [{
        "text": "Start volume 17.", "source": 1,
        "quote": "Season 4 began on April 3, 2026, with two consecutive cours.",
    }]}),
])
def test_nonempty_schedule_sources_cannot_produce_invented_reading_volume(
    tmp_path, monkeypatch, streaming, content,
):
    quote = "Season 4 began on April 3, 2026, with two consecutive cours."
    source = InternetSource("Season 4", "https://example.com/season4", quote)
    stub_internet(monkeypatch, InternetResult(
        attempted=True, query="Tensura season 4 part 2 light novel volume read", sources=[source],
    ))
    router = MagicMock()
    router.send.return_value = LLMResponse(content, "test", "test")
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Where does Tensura season 4 part 2 anime continue in the light novel?",
          {"input_source": "keyboard"})],
        router=router, streaming=streaming,
    )
    try:
        router.stream.assert_not_called()
        tts.speak.assert_called_once()
        assert tts.speak.call_args.args[0] == turns[-1].content
        tts.begin_stream.assert_not_called()
        reply = turns[-1].content
        assert "couldn't verify" in reply and "last episode" in reply
        assert "17" not in reply and "claims" not in reply
        assert "17" not in turns[-1].content
    finally:
        store.close()


def test_canceled_web_answer_validation_cannot_speak_or_save_stale_json(tmp_path, monkeypatch):
    quote = "The company announced a new product."
    stub_internet(monkeypatch, InternetResult(
        attempted=True, query="current product announcement",
        sources=[InternetSource("Announcement", "https://example.com/news", quote)],
    ))
    canceled = Event()
    router = MagicMock()

    def complete(*_args, **_kwargs):
        canceled.set()
        return LLMResponse(json.dumps({"claims": [
            {"text": quote, "source": 1, "quote": quote},
        ]}), "test", "test")

    router.send.side_effect = complete
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("today's product announcement?", {"input_source": "keyboard", "cancel_event": canceled})],
        router=router, streaming=True,
    )
    try:
        router.stream.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()
