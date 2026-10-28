"""Shared research honors fresh owner scope and cancellation in every runtime path."""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.config import Settings
from ciel.internet import InternetResult, InternetSource
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.bridge import (
    conversation_context,
    current_conversation_id,
    desktop_conversation_id,
    telegram_conversation_id,
)
from ciel.providers.base import LLMResponse, ToolCall
from tests.test_ambient import run_callbacks
from tests.test_controller import _web_model_responses
from tests.test_internet_runtime import stub_internet

QUOTE = "The example project supports private text chat."
OWNER_MARKER = "PRIVATE_PAIRED_OWNER_HISTORY_MARKER"


def lookup():
    """Use a factual excerpt, without contacting a public search provider."""
    return InternetResult(attempted=True, query="example project", sources=[
        InternetSource("Example project", "https://example.org/guide", content=QUOTE),
    ])


def response():
    """Generate a valid sourced report so cancellation is the only reason to discard it."""
    return LLMResponse(json.dumps({"claims": [
        {"text": QUOTE, "quote": QUOTE, "source": 1},
    ]}), "test-model", "test-provider")


@pytest.fixture
def paired_context(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_SERVICES", "telegram")
    monkeypatch.setenv("CHAT_STATE_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:synthetic-test")
    state = ChatState(tmp_path / "chat.db")
    state.bind(123, 42)
    store = MemoryStore(tmp_path / "memory.db")
    ConversationManager(store, session_id=telegram_conversation_id(123, 42)).add_turn(
        "user", OWNER_MARKER,
    )
    store.close()
    yield state
    state.close()


def revoke_permission(tmp_path, change):
    """Act through separate connections, as another voice/chat process would."""
    if change == "unpair":
        other_state = ChatState(tmp_path / "chat.db")
        try:
            other_state.unpair()
        finally:
            other_state.close()
    else:
        other_store = MemoryStore(tmp_path / "memory.db")
        try:
            settings = Settings(_env_file=None, memory_db_path=other_store.db_path_str)
            session_id = current_conversation_id(
                settings, telegram_conversation_id(123, 42), store=other_store,
            )
            other_store.set_setting(f"web_pause:{session_id}", {
                "paused": True, "updated_at": time.time_ns(),
            })
        finally:
            other_store.close()


def initialize_paired_conversation(store: MemoryStore, session_id: str) -> None:
    """Seed paired history into the foreground launch's current shared owner scope."""
    settings = Settings(_env_file=None, memory_db_path=store.db_path_str)
    context = conversation_context(settings, session_id, store=store)
    paired_session_id, = [member for member in context.session_ids if member != session_id]
    ConversationManager(store, session_id=paired_session_id).add_turn("user", OWNER_MARKER)


def test_legacy_voice_checks_other_channel_pause_after_initial_retrieval(
    tmp_path, monkeypatch, paired_context,
):
    service = stub_internet(monkeypatch, lookup())

    def retrieve(*args, **kwargs):
        revoke_permission(tmp_path, "pause")
        return lookup()

    service.prepare.side_effect = retrieve
    router = MagicMock()
    router.send.return_value = response()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search the example project", {"input_source": "keyboard"})],
        router=router, controller=False,
        initialize_conversation=initialize_paired_conversation,
    )
    try:
        router.send.assert_not_called()
        service.lookup.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
        assert service.prepare.call_args.kwargs["cancel_event"].is_set()
    finally:
        store.close()


@pytest.mark.parametrize("change", ["pause", "unpair"])
def test_legacy_voice_refreshes_permissions_before_saving_and_speaking_report(
    tmp_path, monkeypatch, paired_context, change,
):
    stub_internet(monkeypatch, lookup())
    router = MagicMock()

    def complete(*args, **kwargs):
        revoke_permission(tmp_path, change)
        return response()

    router.send.side_effect = complete
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search the example project", {"input_source": "keyboard"})],
        router=router, controller=False,
        initialize_conversation=initialize_paired_conversation,
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.kwargs["cancel_event"].is_set()
        assert router.send.call_args.kwargs["purpose"] == "conversation"
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_legacy_voice_captures_scope_before_replaying_owner_history(
    tmp_path, monkeypatch, paired_context,
):
    service = stub_internet(monkeypatch, lookup())
    original = ConversationManager.get_active_messages

    def replay_then_unpair(self, *args, **kwargs):
        messages = original(self, *args, **kwargs)
        assert any(OWNER_MARKER in message.content for message in messages)
        paired_context.unpair()
        return messages

    monkeypatch.setattr(ConversationManager, "get_active_messages", replay_then_unpair)
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search the example project", {"input_source": "keyboard"})],
        router=router, controller=False,
        initialize_conversation=initialize_paired_conversation,
    )
    try:
        service.prepare.assert_not_called()
        router.send.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_default_voice_captures_scope_before_reading_gate_history(
    tmp_path, monkeypatch, paired_context,
):
    service = stub_internet(monkeypatch, lookup())
    original = ConversationManager.get_gate_messages

    def read_gate_then_unpair(self, *args, **kwargs):
        messages = original(self, *args, **kwargs)
        assert any(OWNER_MARKER in message.content for message in messages)
        paired_context.unpair()
        return messages

    monkeypatch.setattr(ConversationManager, "get_gate_messages", read_gate_then_unpair)
    router = MagicMock()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search the example project", {"input_source": "keyboard"})],
        router=router, controller=True,
        initialize_conversation=initialize_paired_conversation,
    )
    try:
        service.lookup.assert_not_called()
        router.send.assert_not_called()
        tts.speak.assert_not_called()
        assert not any(turn.role == "assistant" for turn in turns)
    finally:
        store.close()


def test_legacy_typed_chat_uses_shared_research_beyond_three_progressing_attempts(
    tmp_path, monkeypatch,
):
    initial = InternetResult(attempted=True, query="example project reference 1", sources=[
        InternetSource(
            "Project guide 1", "https://example.org/1", content="Project reference one.",
        ),
    ])
    service = stub_internet(monkeypatch, initial)
    service.lookup.side_effect = [InternetResult(
        attempted=True, query=f"example project reference {index}", sources=[
            InternetSource(f"Project guide {index}", f"https://example.org/{index}", content=(
                QUOTE if index == 5 else f"Project reference number {index}."
            )),
        ],
    ) for index in range(2, 6)]
    replies = []
    for index in range(2, 6):
        replies.extend([
            LLMResponse('{"claims": []}', "model", "provider"),
            LLMResponse(json.dumps({"action": "search", "confidence": 1,
                                    "query": f"example project reference {index}"}),
                        "model", "provider"),
        ])
    replies.append(response())
    router = MagicMock()
    router.send.side_effect = _web_model_responses(replies)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search the example project", {"input_source": "keyboard"})],
        router=router, controller=False,
    )
    try:
        assert service.lookup.call_count == 4
        tts.speak.assert_called_once()
        assert tts.speak.call_args.args[0] == QUOTE
        tts.begin_stream.assert_not_called()
        final = [turn for turn in turns if turn.role == "assistant"][-1]
        assert final.content == QUOTE + " [1]"
        assert len(final.web_evidence["sources"]) == 1
        assert final.web_evidence["sources"][0]["content"] == QUOTE
        assert final.web_evidence["sources"][0]["url"] == "https://example.org/5"
        assert final.sources[0]["url"] == "https://example.org/5"
    finally:
        store.close()


def test_chat_captures_scope_before_replaying_owner_history(tmp_path, monkeypatch, paired_context):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        ciel_persona_file="", internet_enabled=True)
    router = MagicMock()
    session = ChatSession(settings, channel="desktop", session_id=desktop_conversation_id(),
                          router=router)
    original = session.conversation.get_active_messages

    def replay_then_unpair(*args, **kwargs):
        messages = original(*args, **kwargs)
        assert any(OWNER_MARKER in message.content for message in messages)
        paired_context.unpair()
        return messages

    monkeypatch.setattr(session.conversation, "get_active_messages", replay_then_unpair)
    session.internet.prepare = MagicMock(return_value=lookup())
    try:
        assert session.reply("Search the example project") == ""
        session.internet.prepare.assert_not_called()
        router.send.assert_not_called()
        assert not any(turn.role == "assistant" for turn in session.conversation.get_recent_turns())
    finally:
        session.close()


@pytest.mark.parametrize("change", ["pause", "unpair"])
def test_chat_refreshes_permissions_after_report_before_writing(tmp_path, paired_context, change):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        ciel_persona_file="", internet_enabled=True)
    router = MagicMock()

    def complete(messages, **kwargs):
        if not any(message.role == "tool" for message in messages):
            return LLMResponse("", "test-model", "test-provider", tool_calls=(
                ToolCall("public_lookup", "web_search", '{"query":"example project"}'),
            ))
        revoke_permission(tmp_path, change)
        return response()

    router.send.side_effect = complete
    session = ChatSession(settings, channel="desktop", session_id=desktop_conversation_id(),
                          router=router)
    session.internet.lookup = MagicMock(return_value=lookup())
    try:
        assert session.reply("Search the example project") == ""
        assert router.send.call_count == 2
        session.internet.lookup.assert_called_once()
        assert not any(turn.role == "assistant" for turn in session.conversation.get_recent_turns())
    finally:
        session.close()


def test_cancelled_rpc_can_unwind_after_session_close_without_saving_an_answer(tmp_path):
    started, release, finished = (threading.Event() for _ in range(3))
    cancel = threading.Event()
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        ciel_persona_file="", internet_enabled=False)
    router = MagicMock()

    def blocked(*args, **kwargs):
        started.set()
        release.wait(1)
        finished.set()
        return LLMResponse("Late provider reply", "model", "provider")

    router.send.side_effect = blocked
    session = ChatSession(settings, channel="telegram", session_id="test-owner", router=router)
    replies = []
    worker = threading.Thread(
        target=lambda: replies.append(session.reply("Tell me a story", cancel)), daemon=True,
    )
    worker.start()
    try:
        assert started.wait(1)
        cancel.set()
        worker.join(1)
        assert not worker.is_alive()
        assert replies == [""]
        session.close()
        release.set()
        assert finished.wait(1)
        reopened = MemoryStore(settings.memory.db_path)
        try:
            assert not any(
                turn.role == "assistant" for turn in reopened.get_recent_turns("test-owner")
            )
        finally:
            reopened.close()
    finally:
        release.set()
        worker.join(1)
        session.close()
