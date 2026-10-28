"""Fresh foreground launches retain facts and archives with current-run chat continuity."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryItem, MemoryStore
from ciel.memory import bridge as bridge_module
from ciel.memory.bridge import (
    conversation_context,
    current_conversation_id,
    desktop_conversation_id,
    start_desktop_conversation,
    telegram_conversation_id,
)
from ciel.memory.context import recall_context_memories
from ciel.providers.base import LLMResponse
from ciel.screen.service import ScreenRequest
from tests.test_ambient import run_callbacks


@pytest.fixture
def linked(tmp_path):
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        chat_services="telegram", chat_state_path=str(tmp_path / "chat.db"),
        telegram_bot_token=SecretStr("123:test-token"), internet_enabled=False,
    )
    state = ChatState(settings.chat.state_path)
    state.bind(123, 456)
    store = MemoryStore(settings.memory.db_path)
    yield settings, state, store
    store.close()
    state.close()


def manager(settings, store, session_id):
    """Construct the same owner-scoped manager used by foreground and chat sessions."""
    return ConversationManager(
        store, session_id=session_id, max_turns=2, auto_summarize_threshold=2,
        context_resolver=lambda: conversation_context(settings, session_id, store=store),
    )


def test_new_launch_replays_no_prior_channels_or_summary_and_keeps_facts(linked):
    settings, _, store = linked
    first_id = start_desktop_conversation(store)
    first = manager(settings, store, first_id)
    chat_base = telegram_conversation_id(123, 456)
    old_chat_id = current_conversation_id(settings, chat_base, store=store)
    chat = manager(settings, store, old_chat_id)
    for index in range(6):
        first.add_turn("user", f"Previous foreground task {index}")
    chat.add_turn("user", "Previous Telegram task")
    provider = MagicMock()
    provider.send.return_value = LLMResponse("Previous summary", "fake", "fake")
    assert first.summarize_older_turns(provider) == "Previous summary"
    old_summary_id = first.summary_session_id()
    store.upsert_fact("user:favorite_game", MemoryItem(content="My favorite game is Portal."))

    second_id = start_desktop_conversation(store)
    second = manager(settings, store, second_id)

    assert second_id != first_id
    assert second.get_recent_turns() == []
    assert second.get_summary() is None
    assert [message.content for message in second.get_active_messages("Persona")] == ["Persona"]
    assert "Portal" in "\n".join(recall_context_memories(store, "favorite game"))
    assert len(store.get_recent_turns(first_id)) == 6
    assert store.get_recent_turns(old_chat_id)[0].content == "Previous Telegram task"
    assert store.get_session_summary(old_summary_id).content == "Previous summary"


def test_current_run_voice_and_telegram_share_context_across_connections(linked):
    settings, _, store = linked
    desktop_id = start_desktop_conversation(store)
    chat_store = MemoryStore(settings.memory.db_path)
    try:
        chat_id = current_conversation_id(settings, telegram_conversation_id(123, 456))
        desktop = manager(settings, store, desktop_id)
        chat = manager(settings, chat_store, chat_id)
        desktop.add_turn("user", "Current voice task")
        chat.add_turn("assistant", "Current Telegram reply")

        assert desktop.context_session_ids() == [desktop_id, chat_id]
        assert chat.context_session_ids() == desktop.context_session_ids()
        assert chat.summary_session_id() == desktop.summary_session_id()
        assert [turn.content for turn in desktop.get_recent_turns()] == [
            "Current voice task", "Current Telegram reply",
        ]
        assert [turn.content for turn in chat.get_recent_turns()] == [
            "Current voice task", "Current Telegram reply",
        ]
    finally:
        chat_store.close()


def test_new_run_revokes_old_bridge_without_reconnecting_old_managers(linked):
    settings, _, store = linked
    first_id = start_desktop_conversation(store)
    chat_base = telegram_conversation_id(123, 456)
    old_chat_id = current_conversation_id(settings, chat_base, store=store)
    old_desktop = manager(settings, store, first_id)
    old_chat = manager(settings, store, old_chat_id)
    old_scope = old_desktop.context_session_ids()

    second_id = start_desktop_conversation(store)
    new_chat_id = current_conversation_id(settings, old_chat_id, store=store)
    new_chat = manager(settings, store, new_chat_id)
    new_chat.add_turn("user", "New run private discussion")

    assert second_id != first_id
    assert new_chat_id != old_chat_id
    assert old_desktop.context_session_ids() != old_scope
    assert old_desktop.context_session_ids() == [first_id]
    assert old_chat.context_session_ids() == [old_chat_id]
    assert old_desktop.get_recent_turns() == []
    assert old_chat.get_recent_turns() == []


def test_old_owner_cannot_refresh_into_current_run_or_bridge(linked):
    settings, state, store = linked
    desktop_id = start_desktop_conversation(store)
    old_base = telegram_conversation_id(123, 456)
    old_chat_id = current_conversation_id(settings, old_base, store=store)
    state.unpair()
    state.bind(123, 789)
    new_chat_id = current_conversation_id(
        settings, telegram_conversation_id(123, 789), store=store,
    )

    assert current_conversation_id(settings, old_chat_id, store=store) == old_chat_id
    assert conversation_context(settings, old_chat_id, store=store).session_ids == (old_chat_id,)
    assert conversation_context(settings, new_chat_id, store=store).session_ids == (
        desktop_id, new_chat_id,
    )


def test_fresh_launch_pointer_and_facts_survive_database_reopen(linked):
    settings, _, store = linked
    desktop_id = start_desktop_conversation(store)
    store.upsert_fact("user:favorite_game", MemoryItem(content="Favorite game: Portal"))
    reopened = MemoryStore(settings.memory.db_path)
    try:
        chat_id = current_conversation_id(settings, telegram_conversation_id(123, 456))
        assert conversation_context(settings, chat_id, store=reopened).session_ids == (
            desktop_id, chat_id,
        )
        assert reopened.get_fact("user:favorite_game").content == "Favorite game: Portal"
    finally:
        reopened.close()


@pytest.mark.parametrize("invalid", [None, {}, "unrelated archive", "{malformed JSON"])
def test_invalid_active_launch_never_reconnects_legacy_archives(linked, invalid):
    settings, _, store = linked
    first_id = start_desktop_conversation(store)
    if invalid == "{malformed JSON":
        with store._lock, store._get_connection() as connection:
            connection.execute(
                "UPDATE memory_settings SET value_json=? WHERE name=?",
                (invalid, bridge_module._ACTIVE_DESKTOP_KEY),
            )
    else:
        store.set_setting(bridge_module._ACTIVE_DESKTOP_KEY, invalid)

    for identity in (first_id, desktop_conversation_id(), telegram_conversation_id(123, 456)):
        assert conversation_context(settings, identity, store=store).session_ids == (identity,)
        assert conversation_context(settings, identity).session_ids == (identity,)


def test_malformed_or_previous_version_chat_identity_is_never_refreshed(linked):
    settings, _, store = linked
    start_desktop_conversation(store)
    for identity in (
        telegram_conversation_id(123, 456) + ":run:invalid",
        "telegram:123:456:previous-version",
    ):
        assert current_conversation_id(settings, identity, store=store) == identity
        assert conversation_context(settings, identity, store=store).session_ids == (identity,)


def test_persistent_telegram_refreshes_before_reply_without_prior_tasks_or_context(linked):
    settings, _, store = linked
    first_id = start_desktop_conversation(store)
    first_voice = manager(settings, store, first_id)
    first_voice.add_turn("user", "The previous foreground launch discussed telescopes.")
    store.upsert_fact("user:favorite_game", MemoryItem(content="My favorite game is Portal."))
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse("Previous Telegram explanation.", "fake", "fake"),
        LLMResponse("Current Telegram explanation.", "fake", "fake"),
    ]
    session = ChatSession(
        settings, channel="telegram", session_id=telegram_conversation_id(123, 456),
        router=router, store=store,
    )
    try:
        assert session.reply("Explain what a constellation is.") == "Previous Telegram explanation."
        old_chat_id = session.conversation.session_id
        old_search_id = session.internet._parallel_session_id
        session.memory.handle("My favorite food is dumplings.")
        assert session.memory.has_pending_confirmation
        old_screen = session.screen
        old_screen.begin(ScreenRequest("What is on my screen?"))
        assert old_screen.active()
        old_screen.close = MagicMock(wraps=old_screen.close)
        session.context.pause_web(True)

        second_id = start_desktop_conversation(store)
        current_voice = manager(settings, store, second_id)
        current_voice.add_turn("user", "Current foreground note about a new topic.")
        assert session.reply("Discuss my favorite game.") == "Current Telegram explanation."

        messages = router.send.call_args.args[0]
        prompt = "\n".join(message.content for message in messages)
        assert "Current foreground note about a new topic." in prompt
        assert "Portal" in prompt
        assert "telescopes" not in prompt
        assert "constellation" not in prompt
        assert "Previous Telegram explanation." not in prompt
        assert not session.memory.has_pending_confirmation
        assert store.get_fact("user:favorite_food") is None
        old_screen.close.assert_called_once()
        assert session.screen is not old_screen
        assert not session.screen.active()
        assert not session.context.web_paused()
        assert session.conversation.session_id != old_chat_id
        assert session.internet._parallel_session_id != old_search_id
        assert current_voice.context_session_ids() == session.conversation.context_session_ids()
        assert store.get_recent_turns(old_chat_id)[0].content == "Explain what a constellation is."
    finally:
        session.close()


def test_chat_reset_rotates_anonymous_search_session_without_removing_saved_facts(linked):
    settings, _, store = linked
    session = ChatSession(
        settings, channel="telegram", session_id=telegram_conversation_id(123, 456),
        router=MagicMock(), store=store,
    )
    try:
        store.upsert_fact("user:favorite_game", MemoryItem(content="Favorite game: Portal"))
        session.conversation.add_turn("user", "Old search discussion")
        old_search_id = session.internet._parallel_session_id

        assert "cleared" in session.reply("/reset")

        assert session.internet._parallel_session_id != old_search_id
        assert session.conversation.get_recent_turns() == []
        assert store.get_fact("user:favorite_game").content == "Favorite game: Portal"
    finally:
        session.close()


def test_two_foreground_launches_do_not_replay_prior_prompts_but_recall_saved_facts(
    tmp_path, monkeypatch,
):
    from ciel.internet import InternetResult

    monkeypatch.setattr("ciel.internet.InternetService.prepare",
                        lambda *_args, **_kw: InternetResult())
    first_router = MagicMock()
    first_router.send.return_value = LLMResponse("Previous foreground answer.", "fake", "fake")

    def save_fact(store, _session_id):
        store.upsert_fact("user:favorite_game", MemoryItem(content="My favorite game is Portal."))

    _, _, _, first_store, first_turns = run_callbacks(
        tmp_path, monkeypatch, [("Explain nebulae.", {"input_source": "keyboard"})],
        router=first_router, initialize_conversation=save_fact,
    )
    first_store.close()
    second_router = MagicMock()
    second_router.send.return_value = LLMResponse("Current foreground answer.", "fake", "fake")

    _, _, _, second_store, second_turns = run_callbacks(
        tmp_path, monkeypatch, [("Discuss my favorite game.", {"input_source": "keyboard"})],
        router=second_router,
    )
    try:
        prompt = "\n".join(message.content for message in second_router.send.call_args.args[0])
        assert "Explain nebulae." not in prompt
        assert "Previous foreground answer." not in prompt
        assert "Portal" in prompt
        assert first_turns[0].session_id != second_turns[0].session_id
        assert second_store.get_fact("user:favorite_game") is not None
        archived = second_store.get_recent_turns(first_turns[0].session_id)
        assert [turn.content for turn in archived] == [
            "Explain nebulae.", "Previous foreground answer.",
        ]
    finally:
        second_store.close()
