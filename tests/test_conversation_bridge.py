"""Owner-bound chat/voice continuity with channel-local reset and private routing."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from ciel.chat.session import ChatSession
from ciel.chat.state import ChatState
from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryItem, MemoryStore
from ciel.memory.bridge import (
    conversation_context,
    desktop_conversation_id,
    telegram_conversation_id,
)
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter


@pytest.fixture
def connected(tmp_path):
    settings = Settings(
        _env_file=None, memory_db_path=str(tmp_path / "memory.db"),
        chat_services="telegram", chat_state_path=str(tmp_path / "chat.db"),
        telegram_bot_token=SecretStr("123:test-token"), ciel_persona_file="",
        internet_enabled=False,
    )
    state = ChatState(settings.chat.state_path)
    state.bind(123, 456)
    store = MemoryStore(settings.memory.db_path)
    yield settings, state, store
    store.close()
    state.close()


def manager(settings, store, session_id, **kwargs):
    """Use the same fresh-binding context resolver as desktop and headless chat."""
    return ConversationManager(
        store, session_id=session_id,
        context_resolver=lambda: conversation_context(settings, session_id), **kwargs,
    )


def router(answer="We can continue that plan."):
    provider = MagicMock()
    provider.send.return_value = LLMResponse(answer, "test", "test")
    return provider


def test_paired_chat_and_voice_read_one_chronological_context_across_connections(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id())
    desktop.add_turn("user", "I am choosing between two game projects.")
    desktop.add_turn("assistant", "What matters most to you?")
    other_store = MemoryStore(settings.memory.db_path)
    chat = ChatSession(
        settings, channel="telegram", session_id=telegram_conversation_id(123, 456),
        store=other_store, router=router(),
    )
    try:
        chat.reply("The smaller project will be easier to finish.")
        sent = chat.router.send.call_args.args[0]
        assert "I am choosing between two game projects." in [m.content for m in sent]
        turns = desktop.get_recent_turns()
        assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
        assert turns[-2].content == "The smaller project will be easier to finish."
        assert {turn.session_id for turn in turns} == set(desktop.context_session_ids())
        replay = desktop.get_active_messages("Persona")
        origins = "\n".join(m.content for m in replay if m.role == "system")
        assert "Telegram text" in origins and "desktop voice/text" in origins
        assert turns[-2].timestamp.isoformat() in origins
        assert desktop.summary_session_id() == chat.conversation.summary_session_id()
    finally:
        chat.close()


def test_live_unpair_and_repair_never_select_another_owners_chat_archive(connected):
    settings, state, store = connected
    desktop = manager(settings, store, desktop_conversation_id())
    old_chat = manager(settings, store, telegram_conversation_id(123, 456))
    unrelated = ConversationManager(store, session_id=telegram_conversation_id(999, 456))
    old_chat.add_turn("user", "Old owner private chat.")
    unrelated.add_turn("user", "Unrelated bot private chat.")
    desktop.add_turn("user", "Local desktop discussion.")
    assert "Old owner private chat." in [turn.content for turn in desktop.get_recent_turns()]

    state.unpair()
    assert desktop.context_session_ids() == [desktop_conversation_id()]
    assert [turn.content for turn in desktop.get_recent_turns()] == ["Local desktop discussion."]
    state.bind(123, 789)
    current_chat = manager(settings, store, telegram_conversation_id(123, 789))
    assert current_chat.context_session_ids() == desktop.context_session_ids()
    assert "Old owner private chat." not in [t.content for t in desktop.get_recent_turns()]
    assert old_chat.context_session_ids() == [telegram_conversation_id(123, 456)]
    assert store.get_recent_turns(telegram_conversation_id(123, 456))[0].content.startswith("Old")


def test_chat_reset_invalidates_shared_summary_but_preserves_voice_and_facts(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id(), max_turns=2,
                      auto_summarize_threshold=2)
    chat = ChatSession(settings, channel="telegram", session_id=telegram_conversation_id(123, 456),
                       router=router(), store=store)
    store.upsert_fact("user:favorite_game", MemoryItem(content="Favorite game: Standoff 2"))
    for index in range(6):
        desktop.add_turn("user", f"Voice project step {index}")
        chat.conversation.add_turn("user", f"Chat project step {index}")
    summary_id = desktop.summary_session_id()
    assert desktop.summarize_older_turns(router("Voice and chat project decisions."))
    assert chat.conversation.get_summary() == "Voice and chat project decisions."
    generation = store.get_session_generation(summary_id)

    answer = chat.reply("/reset")

    assert "Voice history and saved facts are kept" in answer
    assert store.get_recent_turns(telegram_conversation_id(123, 456)) == []
    assert store.get_session_summary(summary_id) is None
    assert store.get_session_generation(summary_id) == generation + 1
    assert len(desktop.get_recent_turns(limit=20)) == 6
    assert store.get_fact("user:favorite_game") is not None


def test_reset_in_another_channel_rejects_in_flight_shared_summary(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id(), max_turns=2,
                      auto_summarize_threshold=2)
    chat_store = MemoryStore(settings.memory.db_path)
    try:
        chat = manager(settings, chat_store, telegram_conversation_id(123, 456))
        for index in range(6):
            desktop.add_turn("user", f"Voice {index}")
            chat.add_turn("user", f"Chat {index}")
        provider = router("Stale shared summary.")

        def reset_then_reply(*args, **kwargs):
            chat.clear_session()
            return provider.send.return_value

        provider.send.side_effect = reset_then_reply
        assert desktop.summarize_older_turns(provider) is None
        assert desktop.get_summary() is None
        assert len(desktop.get_recent_turns(limit=20)) == 6
    finally:
        chat_store.close()


def test_pending_memory_confirmation_is_never_confirmed_by_the_other_channel(connected):
    settings, _, store = connected
    chat = ChatSession(settings, channel="telegram", session_id=telegram_conversation_id(123, 456),
                       router=router(), store=store)
    desktop = ChatSession(settings, channel="desktop", session_id=desktop_conversation_id(),
                          router=router())
    try:
        chat.reply("Remember my favorite game is Standoff 2")
        chat.reply("My favorite game is Portal 2")
        assert chat.memory.has_pending_confirmation
        assert not desktop.memory.has_pending_confirmation
        desktop.reply("Yes, update it")
        assert store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
        assert chat.memory.has_pending_confirmation
        chat.reply("Yes, update it")
        assert store.get_fact("user:favorite_game").metadata["value"] == "Portal 2"
    finally:
        desktop.close()


def test_private_voice_context_survives_channel_switch_clipping_and_restart(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id())
    desktop.add_turn("user", "/local Private project discussion")
    for index in range(40):
        desktop.add_turn("user", f"Recent discussion {index}")
    chat = manager(settings, store, telegram_conversation_id(123, 456), max_context_chars=100)
    chat.add_turn("user", "Continue from where we were.")

    assert chat.local_only_context()
    assert ModelRouter.requires_local(chat.get_gate_messages())
    replay = chat.get_active_messages("Persona")
    assert replay[-1].content == "Continue from where we were."
    assert replay[0].local_only
    assert sum(len(message.content) for message in replay[1:]) <= 100
    reopened = MemoryStore(settings.memory.db_path)
    try:
        restarted = manager(settings, reopened, telegram_conversation_id(123, 456))
        assert restarted.local_only_context()
        assert len(restarted.context_session_ids()) == 2
    finally:
        reopened.close()


def test_shared_summary_updates_are_visible_without_cross_process_cache(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id(), max_turns=2,
                      auto_summarize_threshold=2)
    chat = manager(settings, store, telegram_conversation_id(123, 456), max_turns=2,
                   auto_summarize_threshold=2)
    for index in range(8):
        (desktop if index % 2 else chat).add_turn("user", f"Project step {index}")
    assert desktop.summarize_older_turns(router("First decisions."))
    assert chat.get_summary() == "First decisions."
    chat.add_turn("user", "Additional decision one")
    chat.add_turn("assistant", "Additional decision two")
    assert chat.summarize_older_turns(router("Updated decisions."))
    assert desktop.get_summary() == "Updated decisions."
    summary = store.get_session_summary(desktop.summary_session_id())
    assert summary.metadata["context_session_ids"] == desktop.context_session_ids()


def test_failed_summary_keeps_unsummarized_recent_turns_within_character_budget(connected):
    settings, _, store = connected
    desktop = manager(settings, store, desktop_conversation_id(), max_turns=2,
                      auto_summarize_threshold=2, max_context_chars=1000)
    for index in range(10):
        desktop.add_turn("user", f"Project detail {index}")
    provider = router()
    provider.send.side_effect = RuntimeError("Provider unavailable")
    assert desktop.summarize_older_turns(provider) is None
    messages = desktop.get_active_messages("Persona")
    assert "Project detail 0" in [message.content for message in messages]
    assert sum(len(message.content) for message in messages[1:]) <= 1000


def test_unpaired_context_resolution_never_creates_chat_state(tmp_path):
    settings = Settings(
        _env_file=None, chat_services="telegram",
        chat_state_path=str(tmp_path / "missing.db"), telegram_bot_token="123:local-test",
    )
    context = conversation_context(settings, desktop_conversation_id())
    assert context.session_ids == (desktop_conversation_id(),)
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.parametrize("bot_id,owner_id", [(123, None), (True, 456), (0, 456)])
def test_history_identity_requires_verified_numeric_owner(bot_id, owner_id):
    with pytest.raises(ValueError):
        telegram_conversation_id(bot_id, owner_id)
