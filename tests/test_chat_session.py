"""Text chat shares durable facts without sharing desktop history or voice startup."""

import json
import subprocess
import sys
import threading

import pytest

from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource
from ciel.memory import MemoryStore
from ciel.memory.bridge import telegram_conversation_id
from ciel.providers.base import LLMResponse, ToolCall


class RecordingRouter:
    def __init__(self):
        self.messages = []
        self.options = []
        self.tool_calls = []
        self.answer = "Hey, what's on your mind?"

    def send(self, messages, **kwargs):
        self.messages.append(messages)
        self.options.append(kwargs)
        if messages[0].content.startswith("WEB GOAL CHECK"):
            assert kwargs["purpose"] == "verification"
            return LLMResponse('{"relevant_claims": [1]}', "goal-check", "test-provider")
        if kwargs.get("tools") and self.tool_calls:
            return LLMResponse("", "test-model", "test-provider",
                               tool_calls=(self.tool_calls.pop(0),))
        return LLMResponse(self.answer, "test-model", "test-provider")


@pytest.fixture
def session(tmp_path):
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "facts.db"),
        ciel_persona_file="", internet_enabled=False,
        chat_services="telegram", chat_state_path=str(tmp_path / "chat.db"),
        telegram_bot_token="123:test-token",
    )
    state = ChatState(settings.chat.state_path)
    state.bind(123, 42)
    chat = ChatSession(settings, channel="telegram", session_id=telegram_conversation_id(123, 42),
                       router=RecordingRouter())
    yield chat
    chat.close()
    state.close()


def test_text_chat_import_and_answer_do_not_import_audio(tmp_path):
    program = """
import sys
from ciel.config import Settings
from ciel.chat.session import ChatSession
from ciel.providers.base import LLMResponse
class Router:
    def send(self, *args, **kwargs):
        return LLMResponse('Hello.', 'test', 'test')
settings = Settings(_env_file=None, memory_db_path=sys.argv[1],
                    ciel_persona_file='', internet_enabled=False)
session = ChatSession(settings, channel='telegram', session_id='test', router=Router())
assert session.reply('Hi Ciel') == 'Hello.'
assert not any(name in sys.modules for name in (
    'ciel.audio.native', 'ciel.audio.stt', 'ciel.audio.tts', 'ciel.platform.linux',
    'sounddevice', 'faster_whisper', 'pocket_tts',
))
session.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "headless.db")],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_confirmed_correction_is_visible_across_channels_and_restart(session):
    assert "saved" in session.reply("Remember my favorite game is Stand of Two").lower()
    proposal = session.reply("Update my favorite game to Standoff 2")
    assert "replace" in proposal.lower()
    assert session.store.get_fact("user:favorite_game").metadata["value"] == "Stand of Two"
    assert "Standoff 2" in session.reply("Yes, update it")
    session.close()

    reopened = MemoryStore(db_path=session.settings.memory.db_path)
    try:
        assert reopened.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
        other = ChatSession(session.settings, channel="future-channel", session_id="other",
                            router=RecordingRouter(), store=reopened)
        assert "Standoff 2" in other.reply("What's my favorite game?")
        assert all("Stand of Two" not in turn.content
                   for turn in other.conversation.get_recent_turns())
    finally:
        reopened.close()


def test_pending_confirmation_and_history_are_separate(session):
    session.reply("Remember my favorite game is Standoff 2")
    session.reply("My favorite game is Portal 2")
    session.conversation.add_turn("assistant", "Desktop-secret-placeholder")
    other = ChatSession(session.settings, channel="other", session_id="other",
                        router=RecordingRouter())
    try:
        assert not other.memory.has_pending_confirmation
        other.reply("Yes, update it")
        assert other.store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
        assert "Desktop-secret-placeholder" not in str(other.router.messages)
        assert session.memory.has_pending_confirmation
    finally:
        other.close()


def test_text_persona_and_memory_write_status_are_accurate(session):
    session.reply("Hey Ciel, what's up?")
    prompt = session.router.messages[-1][0].content
    assert "Text output:" in prompt
    assert "Spoken output:" not in prompt
    assert "transcribed" not in prompt
    assert "No durable memory write was performed for this turn" in prompt
    assert "Those features are unavailable in this channel" in prompt
    assert "disabled in settings" in prompt


def test_recall_pause_and_reset_keep_saved_facts(session):
    session.reply("Remember my favorite game is Standoff 2")
    session.reply("Pause memory recall")
    assert "paused" in session.reply("/memory")
    session.reply("/reset")
    assert session.conversation.get_recent_turns() == []
    assert session.store.get_fact("user:favorite_game") is not None
    assert not session.memory.has_pending_confirmation


def test_legacy_local_history_remains_protected_after_restart(session, monkeypatch):
    # Existing private conversation history retains its original privacy boundary.
    session.conversation.add_turn("user", "/local Tell me a short story")
    session.close()
    session = ChatSession(session.settings, channel="telegram", session_id=session._base_session_id,
                          router=RecordingRouter())

    def forbidden(*args, **kwargs):
        pytest.fail("Private history must not reach public retrieval")

    monkeypatch.setattr(session.internet, "lookup", forbidden)
    try:
        session.reply("Continue")
        session.reply("What happens next?")
        assert len(session.router.messages) == 2
        assert all(messages[0].local_only for messages in session.router.messages)
        assert all(not {"web_search", "read_page"} & {
            tool["function"]["name"] for tool in options.get("tools", [])
        } for options in session.router.options)
    finally:
        session.close()


@pytest.mark.parametrize("text", [
    "/local", "/LOCAL", "/local Remember my name is Secret", " /LoCaL\tSearch for news ",
    "Hello\n/local Tell me a story", "Hello\n  /LOCAL remember my name is Secret",
])
def test_removed_local_command_cannot_generate_or_save_anything(session, monkeypatch, text):
    def forbidden(*args, **kwargs):
        pytest.fail("Removed commands must not reach memory or internet handlers")

    monkeypatch.setattr(session.internet, "prepare", forbidden)
    monkeypatch.setattr(session.context, "memory_reply", forbidden)
    assert "no longer available" in session.reply(text)
    assert session.conversation.get_total_turn_count() == 0
    assert not session.router.messages


def test_removed_local_command_keeps_pending_memory_proposal(session):
    session.reply("Remember my favorite game is Standoff 2")
    session.reply("My favorite game is Portal 2")
    assert session.memory.has_pending_confirmation
    count = session.conversation.get_total_turn_count()
    session.reply("/local Yes, update it")
    assert session.memory.has_pending_confirmation
    assert session.store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
    assert session.conversation.get_total_turn_count() == count


def test_chat_help_does_not_offer_removed_local_command(session):
    assert "/local" not in session.reply("/help")


@pytest.mark.parametrize("text", [
    "/fast Tell me a story", "/strong Tell me a story", "/deep Tell me a story",
    "What did /local do?",
])
def test_other_routes_and_ordinary_command_mentions_still_reach_provider(session, text):
    assert session.reply(text) == session.router.answer
    assert next(message.content for message in reversed(session.router.messages[-1])
                if message.role == "user") == text
    assert session.router.options[-1]["purpose"] == "conversation"


def test_reminders_and_desktop_actions_do_not_reach_provider(session):
    assert "haven't scheduled" in session.reply("Remind me to stretch in ten minutes")
    assert "aren't available" in session.reply("Open Firefox")
    assert not session.router.messages


def test_no_search_preference_survives_opinion_followup_until_explicit_lookup(session, monkeypatch):
    session.settings.internet_enabled = True
    calls = []

    def lookup(*args, **kwargs):
        calls.append(args[0])
        return InternetResult(attempted=True, reply="No usable results were found.")

    monkeypatch.setattr(session.internet, "lookup", lookup)
    assert session.reply("Do not search.") == "Okay, I won't search."
    session.reply("What do you think?")
    session.reply("Don't search for that, instead explain the tradeoffs.")
    assert session.reply("Don't search, just explain it from what you know.") != (
        "Okay, I won't search."
    )
    assert calls == []
    assert "paused at the user's request" in session.router.messages[-1][0].content
    session.router.tool_calls.append(ToolCall(
        "public-search", "web_search",
        json.dumps({"query": "Search for the official project page"}),
    ))
    session.reply("Search for the official project page")
    assert calls == ["Search for the official project page"]


def test_request_deadline_cancels_only_request(session, monkeypatch):
    monkeypatch.setattr("ciel.chat.session.REQUEST_TIMEOUT_SECONDS", 0.01)
    service_stop = threading.Event()
    provider_closed = threading.Event()

    def blocked(messages, *, cancel_event, **kwargs):
        try:
            assert cancel_event.wait(1)
            raise TimeoutError("synthetic timeout")
        finally:
            provider_closed.set()

    monkeypatch.setattr(session.router, "send", blocked)
    assert "too long" in session.reply("Tell me a story", cancel_event=service_stop)
    assert provider_closed.wait(2), "The fake provider must release its bounded worker"
    assert not service_stop.is_set()
    service_stop.set()
    count = session.conversation.get_total_turn_count()
    assert session.reply("Remember my name is Test", cancel_event=service_stop) == ""
    assert session.conversation.get_total_turn_count() == count


def test_verified_web_reply_includes_source_link(session, monkeypatch):
    session.settings.internet_enabled = True
    source = InternetSource(
        title="Official example", url="https://example.org/answer",
        snippet="The example project supports text chat.",
    )
    lookup = InternetResult(attempted=True, query="example project", sources=[source])
    monkeypatch.setattr(session.internet, "lookup", lambda *args, **kwargs: lookup)
    session.router.tool_calls.append(ToolCall(
        "example-search", "web_search", '{"query": "example project"}',
    ))
    session.router.answer = json.dumps({"claims": [{
        "text": "The example project supports text chat.",
        "quote": "The example project supports text chat.", "source": 1,
    }]})
    answer = session.reply("Search for the example project")
    assert "supports text chat" in answer
    assert source.url in answer
    assert session.conversation.get_recent_turns()[-1].sources[0]["url"] == source.url
