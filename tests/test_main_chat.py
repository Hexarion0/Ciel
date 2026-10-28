"""Text chat lets its responding model choose bounded native read tools."""

import json
import threading
import time
from collections import deque
from unittest.mock import MagicMock

import pytest

from ciel.chat.session import ChatSession
from ciel.chat.telegram import ChatReply
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import LLMResponse, LLMStreamChunk, ToolCall, ToolCallDelta
from ciel.providers.vision import synthetic_test_image
from ciel.screen.service import ScreenResult


class NativeRouter:
    """Serve synthetic native completions through both supported response entry points."""

    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []
        self.goal_checks = []

    def _response(self, messages, kwargs):
        if messages[0].content.startswith("WEB GOAL CHECK"):
            assert kwargs.get("purpose") == "verification"
            self.goal_checks.append((list(messages), kwargs))
            return LLMResponse('{"relevant_claims":[1]}', "main-model", "synthetic")
        if kwargs.get("purpose") != "conversation":
            return LLMResponse("{}", "auxiliary", "synthetic")
        self.calls.append((list(messages), kwargs))
        assert self.responses, "Unexpected extra main-model request"
        response = self.responses.popleft()
        return response(messages, kwargs) if callable(response) else response

    def send(self, messages, **kwargs):
        return self._response(messages, kwargs)

    def stream(self, messages, **kwargs):
        response = self._response(messages, kwargs)
        yield LLMStreamChunk(
            response.content, response.model, response.provider,
            tool_calls=tuple(ToolCallDelta(index, call.id, call.name, call.arguments)
                             for index, call in enumerate(response.tool_calls)),
        )
        yield LLMStreamChunk("", response.model, response.provider, is_final=True)


def answer(text):
    return LLMResponse(text, "main-model", "synthetic")


def tool(name, arguments, call_id="call_1"):
    return LLMResponse("", "main-model", "synthetic", tool_calls=(
        ToolCall(call_id, name, json.dumps(arguments)),
    ))


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    monkeypatch.setattr("ciel.chat.session.ToolProgress.warm", lambda *_args, **_kwargs: None)
    created = []

    def create(*responses, internet=False):
        settings = Settings(
            _env_file=None, memory_db_path=str(tmp_path / f"memory-{len(created)}.db"),
            memory_workspace_path=str(tmp_path / f"workspace-{len(created)}"),
            memory_daily_capture_enabled=False, memory_flush_enabled=False,
            memory_consolidation_enabled=False, ciel_persona_file="",
            internet_enabled=internet,
        )
        router = NativeRouter(*responses)
        session = ChatSession(settings, channel="telegram", session_id="synthetic-owner",
                              router=router)
        session.internet.prepare = MagicMock(side_effect=AssertionError("Classifier must not run"))
        session.context.tools.run = MagicMock(side_effect=AssertionError("Controller must not run"))
        session.screen.interpret = MagicMock(side_effect=AssertionError("Classifier must not run"))
        created.append(session)
        return session, router

    yield create
    for session in created:
        session.close()


def test_ordinary_text_uses_main_response_with_full_dialogue(make_session):
    session, router = make_session(answer("Start with a short list."), answer("Pick one task."))
    assert session.reply("How can I plan my day?") == "Start with a short list."
    assert session.reply("Can you make it more concrete?") == "Pick one task."
    assert len(router.calls) == 2
    second_messages, kwargs = router.calls[1]
    assert kwargs["purpose"] == "conversation"
    assert kwargs["tool_choice"] == "auto"
    assert any(message.role == "assistant" and message.content == "Start with a short list."
               for message in second_messages)
    assert any(message.role == "user" and message.content == "Can you make it more concrete?"
               for message in second_messages)
    session.internet.prepare.assert_not_called()
    session.context.tools.run.assert_not_called()
    session.screen.interpret.assert_not_called()


def test_main_response_requests_memory_read_and_receives_structured_history(make_session):
    session, router = make_session(
        tool("memory_search", {"query": "archived project note"}),
        answer("Your note says the next step is to compare the two designs."),
    )
    session.context.tools.search = MagicMock(return_value={
        "results": [{"path": "memory/project.md", "text": "Compare the two designs."}],
    })
    session.context.recall = MagicMock(return_value=[])
    result = session.reply("Can you help me pick up the project discussion?")
    assert "compare the two designs" in result
    session.context.tools.search.assert_called_once_with(
        "archived project note", local_only=False, include_sessions=False,
    )
    continuation = router.calls[1][0]
    assistant = next(message for message in continuation if message.tool_calls)
    returned = next(message for message in continuation if message.role == "tool")
    assert assistant.tool_calls[0].name == "memory_search"
    assert returned.tool_call_id == assistant.tool_calls[0].id
    assert "Compare the two designs" in json.loads(returned.content)["results"][0]["text"]
    assert [turn.role for turn in session.conversation.get_recent_turns()] == ["user", "assistant"]
    session.context.tools.run.assert_not_called()
    session.internet.prepare.assert_not_called()


def test_native_web_lookup_retains_verified_sources_without_a_decision_model(make_session):
    quote = "The example project supports private text chat."
    session, router = make_session(
        tool("web_search", {"query": "example project official guide"}),
        answer(json.dumps({"claims": [{"text": quote, "quote": quote, "source": 1}]})),
        internet=True,
    )
    source = InternetSource("Official project guide", "https://example.org/guide", content=quote)
    session.internet.lookup = MagicMock(return_value=InternetResult(
        attempted=True, query="example project official guide", sources=[source],
    ))
    session.tool_progress.announce = MagicMock(return_value=True)
    result = session.reply("Find the official example project guide")
    assert quote in result
    assert source.url in result
    assert len(router.calls) == 2
    assert router.goal_checks[0][1]["purpose"] == "verification"
    assert all(kwargs["purpose"] == "conversation" for _, kwargs in router.calls)
    session.internet.prepare.assert_not_called()
    session.internet.lookup.assert_called_once()
    assert session.internet.lookup.call_args.args[0] == "example project official guide"
    session.tool_progress.announce.assert_called_once()
    saved = session.conversation.get_recent_turns()[-1]
    assert saved.sources[0]["url"] == source.url
    assert saved.web_evidence["sources"][0]["url"] == source.url


def test_reused_web_evidence_reaches_main_model_without_another_lookup(make_session):
    quote = "The example project supports private text chat."
    session, router = make_session(
        answer(json.dumps({"claims": [{"text": quote, "quote": quote, "source": 1}]})),
        internet=True,
    )
    session.conversation.add_turn("user", "Search the example project")
    session.conversation.add_turn("assistant", quote, web_evidence={
        "query": "example project", "kind": "web", "retrieved_at": "2026-10-27T00:00:00Z",
        "sources": [{"title": "Project guide", "url": "https://example.org/guide",
                     "content": quote}],
    })
    session.internet.lookup = MagicMock(side_effect=AssertionError("No new lookup requested"))
    result = session.reply("Tell me what you found")
    assert quote in result
    assert "https://example.org/guide" in result
    messages = router.calls[0][0]
    assert any(message.role == "assistant" and message.content == quote for message in messages)
    assert any(message.role == "system" and "Required output: JSON only" in message.content
               for message in messages)
    assert any(message.role == "system" and "Project guide" in message.content
               for message in messages)
    session.internet.lookup.assert_not_called()


def test_paused_web_is_not_exposed_or_executed_even_if_model_requests_it(make_session):
    session, router = make_session(tool("web_search", {"query": "blocked"}), internet=True)
    session.internet.lookup = MagicMock(side_effect=AssertionError("Search is paused"))
    assert session.reply("Do not search.") == "Okay, I won't search."
    response = session.reply("Explain the tradeoffs from what you know")
    assert "couldn't safely complete" in response
    schemas = router.calls[0][1]["tools"]
    assert "web_search" not in {schema["function"]["name"] for schema in schemas}
    assert "paused at the user's request" in router.calls[0][0][0].content
    session.internet.lookup.assert_not_called()


def test_canceling_native_lookup_prevents_continuation_and_persistence(make_session):
    session, router = make_session(tool("web_search", {"query": "project guide"}), internet=True)
    canceled = threading.Event()

    def lookup(*_args, **_kwargs):
        canceled.set()
        return InternetResult(attempted=True, query="project guide")

    session.internet.lookup = MagicMock(side_effect=lookup)
    assert session.reply("Find the project guide", canceled) == ""
    assert len(router.calls) == 1
    assert not any(turn.role == "assistant" for turn in session.conversation.get_recent_turns())


def test_active_visual_followup_is_chosen_by_main_model_without_screen_classifier(make_session):
    session, router = make_session(tool("check_screen", {}))
    session.screen.answer = MagicMock(return_value=ScreenResult(
        "The button applies the setting.", "synthetic-vision", "vision-model",
    ))
    assert session.reply("Look at my screen") == "The button applies the setting."
    query = "Could you check that button again?"
    assert session.reply(query) == "The button applies the setting."
    assert len(router.calls) == 1
    assert session.screen.answer.call_count == 2
    request = session.screen.answer.call_args.args[0]
    assert request.query == query
    assert request.original_query == "Look at my screen"
    assert request.send_image is False
    assert session.conversation.get_recent_turns()[-1].provider == "synthetic-vision"
    session.screen.interpret.assert_not_called()


def test_native_screen_progress_survives_its_own_task_revision(make_session):
    session, router = make_session(tool("check_screen", {}))
    session.screen.answer = MagicMock(return_value=ScreenResult("First view."))
    assert session.reply("Look at my screen") == "First view."
    progress = []
    session.progress = progress.append

    def inspect(request, messages, *, current, progress, **kwargs):
        revision = session.screen.begin(request)
        assert revision
        assert current()
        assert progress("screen", request.query)
        return ScreenResult("The button is enabled.", task_revision=revision)

    session.screen.answer.side_effect = inspect
    session.tool_progress.announce = lambda _tool, _topic, emit, **kw: (
        emit("Checking the screen.") or kw["current"]()
    )
    assert session.reply("Could you check that button again?") == "The button is enabled."
    assert len(router.calls) == 1
    assert progress == ["Checking the screen."]


@pytest.mark.parametrize("query", [
    "Do not look at that button", 'He said "check that button".',
    "He wrote 'check that button'.", "Could that button cause the problem?",
])
def test_native_screen_call_requires_an_actual_observation_request(make_session, query):
    session, _router = make_session(tool("check_screen", {}))
    session.screen.answer = MagicMock(return_value=ScreenResult("A settings window."))
    session.reply("Look at my screen")
    session.screen.answer.reset_mock()
    result = session.reply(query)
    assert result == "Did you want me to look at your screen?"
    session.screen.answer.assert_not_called()


def test_direct_screenshot_preserves_temporary_telegram_attachment(make_session):
    session, router = make_session()
    image = synthetic_test_image().data
    session.screen.answer = MagicMock(return_value=ScreenResult(
        "Here is the screenshot.", "local", "screen", "synthetic-capture", 2_000_000_000,
        image,
    ))
    session.screen.still_valid = MagicMock(return_value=True)
    session.screen.image_valid = MagicMock(return_value=True)
    result = session.reply("Send me a screenshot")
    assert isinstance(result, ChatReply)
    assert result.photos[0].data == image
    assert result.photos[0].valid()
    session.screen.image_valid.return_value = False
    assert not result.photos[0].valid()
    assert router.calls == []
    assert all("synthetic-image" not in turn.content
               for turn in session.conversation.get_recent_turns())


def test_failure_after_lookup_does_not_publish_unverified_source_citations(make_session):
    session, _router = make_session(
        tool("web_search", {"query": "project guide"}),
        tool("unsupported_tool", {}), internet=True,
    )
    source = InternetSource("Unverified page", "https://example.org/unverified",
                            content="This page has not supported a checked answer.")
    session.internet.lookup = MagicMock(return_value=InternetResult(
        attempted=True, query="project guide", sources=[source],
    ))
    response = session.reply("Find the project guide")
    assert "couldn't safely complete" in response
    assert source.url not in response
    saved = session.conversation.get_recent_turns()[-1]
    assert saved.sources == []
    assert saved.web_evidence["sources"][0]["url"] == source.url


def test_request_timeout_cancels_only_native_turn(make_session, monkeypatch):
    seen = threading.Event()

    def blocked(_messages, kwargs):
        seen.set()
        while not kwargs["cancel_event"].is_set():
            time.sleep(0.001)
        raise TimeoutError("Synthetic model wait")

    session, _router = make_session(blocked)
    monkeypatch.setattr("ciel.chat.session.REQUEST_TIMEOUT_SECONDS", 0.02)
    service_stop = threading.Event()
    assert "too long" in session.reply("Tell me a story", service_stop)
    assert seen.is_set()
    assert not service_stop.is_set()
