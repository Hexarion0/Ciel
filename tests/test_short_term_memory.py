"""Tests for CIEL short-term memory, sliding context window, and rolling summarization."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ciel.memory import ConversationManager, MemoryStore
from ciel.providers.base import LLMResponse, LLMStreamChunk


@pytest.fixture
def temp_store():
    """Create an isolated SQLite memory store backed by a temp database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name

    store = MemoryStore(db_path=tmp_path)
    yield store
    store.close()
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def test_cross_restart_conversation_persistence(temp_store: MemoryStore):
    """Verify turns saved in session A are preserved and restored upon restarting manager."""
    # 1. First session run
    manager1 = ConversationManager(store=temp_store, session_id="test_session", max_turns=5)
    manager1.add_turn(role="user", content="Hello Ciel, my name is Hexarion")
    manager1.add_turn(role="assistant", content="Hey Hexarion! Great to meet you.")
    manager1.add_turn(role="user", content="What GPU do I have?")
    manager1.add_turn(role="assistant", content="You have a GTX 1660 SUPER.")

    # 2. Simulate assistant restart (new manager instance on same DB)
    manager2 = ConversationManager(store=temp_store, session_id="test_session", max_turns=5)
    turns = manager2.get_recent_turns()
    assert len(turns) == 4
    assert turns[0].role == "user"
    assert turns[0].content == "Hello Ciel, my name is Hexarion"
    assert turns[3].content == "You have a GTX 1660 SUPER."

    # Verify active messages structure
    messages = manager2.get_active_messages(system_prompt="System Persona")
    assert len(messages) == 5  # 1 system + 4 turns
    assert messages[0].role == "system"
    assert messages[0].content == "System Persona"
    assert messages[1].content == "Hello Ciel, my name is Hexarion"


def test_manager_persists_sources_without_injecting_them_into_provider_messages(temp_store):
    """Links remain UI metadata instead of becoming trusted model context."""
    manager = ConversationManager(store=temp_store, session_id="cited")
    sources = [{"title": "Forecast reference", "url": "https://example.com/forecast"}]
    manager.add_turn("assistant", "It is sunny.", sources=sources)
    assert manager.get_recent_turns()[0].sources == sources
    messages = manager.get_active_messages("Persona")
    assert [message.content for message in messages] == ["Persona", "It is sunny."]


def test_archived_interruption_notes_are_metadata_not_assistant_speech(temp_store):
    manager = ConversationManager(store=temp_store, session_id="interrupted")
    manager.add_turn("user", "Explain streaming.")
    manager.add_turn("assistant", "The first part.\n[Playback was interrupted.]")
    manager.add_turn("assistant", "[Playback was interrupted.]")
    manager.add_turn("user", "What does [Playback was interrupted.] mean?")
    restored = ConversationManager(store=temp_store, session_id="interrupted")
    messages = restored.get_active_messages("Be helpful.")
    assistant = [message.content for message in messages if message.role == "assistant"]
    assert assistant == ["The first part."]
    assert any("interrupted" in message.content for message in messages if message.role == "system")
    assert messages[-1].content == "What does [Playback was interrupted.] mean?"
    assert temp_store.get_recent_turns("interrupted")[1].content.endswith(
        "[Playback was interrupted.]"
    )  # Stored history is retained.


def test_summary_does_not_learn_internal_status_as_assistant_dialogue(temp_store):
    manager = ConversationManager(
        store=temp_store, session_id="status-summary", max_turns=2, auto_summarize_threshold=2,
    )
    manager.add_turn("user", "Explain streaming.")
    manager.add_turn("assistant", "The first part.\n[Playback was interrupted.]")
    manager.add_turn("user", "Continue.")
    manager.add_turn("assistant", "The rest.")
    provider = MagicMock()
    provider.send.return_value = LLMResponse("We discussed streaming.", "test", "test")
    manager.summarize_older_turns(provider)
    transcript = provider.send.call_args.args[0][-1].content
    assert "Assistant: The first part." in transcript
    assert "[Playback was interrupted.]" not in transcript


def test_sliding_context_window_trimming(temp_store: MemoryStore):
    """Verify that when turn count exceeds max_turns, only the latest N turns are returned."""
    manager = ConversationManager(store=temp_store, session_id="overflow_session", max_turns=4)

    for i in range(10):
        manager.add_turn(role="user", content=f"User turn {i}")
        manager.add_turn(role="assistant", content=f"Assistant turn {i}")

    # Total 20 turns recorded in SQLite
    assert manager.get_total_turn_count() == 20

    # Active messages should only contain the system prompt + latest 4 turns
    messages = manager.get_active_messages(system_prompt="Base System", limit=4)
    assert len(messages) == 5  # 1 system + 4 turns
    assert messages[1].content == "User turn 8"
    assert messages[2].content == "Assistant turn 8"
    assert messages[3].content == "User turn 9"
    assert messages[4].content == "Assistant turn 9"


def test_rolling_summarization(temp_store: MemoryStore):
    """Verify older turns are summarized and injected as background context."""
    manager = ConversationManager(
        store=temp_store,
        session_id="summarize_session",
        max_turns=2,
        auto_summarize_threshold=4,
    )

    # Add 6 turns (exceeds threshold of 4)
    manager.add_turn(role="user", content="We are planning to build a desktop assistant in Python.")
    manager.add_turn(
        role="assistant", content="Awesome, I recommend using faster-whisper and sounddevice."
    )
    manager.add_turn(role="user", content="Let's make sure it runs on Arch Linux with Hyprland.")
    manager.add_turn(role="assistant", content="Got it, we will configure Linux audio backend.")
    manager.add_turn(role="user", content="What is our current task?")
    manager.add_turn(role="assistant", content="We are implementing the short term memory module.")

    # Mock LLM router for summarizer
    mock_router = MagicMock()
    mock_router.send.return_value = LLMResponse(
        content="User and assistant planned a Python voice assistant on Arch Linux with Hyprland.",
        model="test_model",
        provider="nim",
    )

    summary = manager.summarize_older_turns(router_or_provider=mock_router)
    assert summary is not None
    assert "Arch Linux" in summary

    # Verify summary is injected into active messages
    active_msgs = manager.get_active_messages(system_prompt="Base Persona")
    # Expected: [System Persona, Summary Context, Turn 5, Turn 6]
    assert len(active_msgs) == 4
    assert active_msgs[0].content == "Base Persona"
    assert "PREVIOUS CONVERSATION CONTEXT" in active_msgs[1].content
    assert "Arch Linux" in active_msgs[1].content
    assert active_msgs[2].content == "What is our current task?"
    assert active_msgs[3].content == "We are implementing the short term memory module."


def test_clear_session(temp_store: MemoryStore):
    """Verify clearing session removes turns and summary."""
    manager = ConversationManager(store=temp_store, session_id="clear_me", max_turns=5)
    manager.add_turn(role="user", content="Some test text")
    manager.add_turn(role="assistant", content="Some reply")
    assert manager.get_total_turn_count() == 2

    cleared = manager.clear_session()
    assert cleared == 2
    assert manager.get_total_turn_count() == 0
    assert len(manager.get_active_messages(system_prompt="Base")) == 1


def test_fallback_summary_preserves_claim_sources(temp_store: MemoryStore):
    """An assistant's unsupported claim must stay attributed to the assistant."""
    manager = ConversationManager(
        store=temp_store,
        session_id="sources",
        max_turns=2,
        auto_summarize_threshold=2,
    )
    manager.add_turn(role="user", content="The project started five days ago.")
    manager.add_turn(role="assistant", content="The database proves it started three days ago.")
    manager.add_turn(role="user", content="That is not the project's start date.")
    manager.add_turn(role="assistant", content="Thanks for correcting me.")

    summary = manager.summarize_older_turns()

    assert "User said: The project started five days ago." in summary
    assert "Assistant said: The database proves it started three days ago." in summary
    assert manager.get_summary() == summary


def test_background_summary_does_not_delay_or_duplicate_requests(temp_store):
    import threading

    manager = ConversationManager(
        store=temp_store,
        max_turns=2,
        auto_summarize_threshold=2,
        summary_interval=1,
    )
    for index in range(4):
        manager.add_turn("user", f"Topic {index}")
    entered, release = threading.Event(), threading.Event()
    router = MagicMock()

    def respond(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return LLMResponse(content="summary", model="test", provider="test")

    router.send.side_effect = respond
    try:
        assert manager.schedule_summary(router)
        assert entered.wait(1)
        assert not manager.schedule_summary(router)
    finally:
        release.set()
        manager._summary_thread.join(timeout=2)
    assert not manager.schedule_summary(router)
    assert router.send.call_count == 1


def test_summary_restores_and_clears_only_exact_session(temp_store):
    from ciel.memory.models import MemoryItem, MemoryType

    for session in ("desk", "desk-other"):
        temp_store.save_memory(
            MemoryItem(
                content="Summary without an identity in its text",
                memory_type=MemoryType.CONVERSATION,
                metadata={"session_id": session},
            )
        )
    manager = ConversationManager(store=temp_store, session_id="desk")
    assert manager.get_summary() == "Summary without an identity in its text"
    manager.clear_session()
    temp_store.close()
    restarted = MemoryStore(temp_store.db_path_str)
    try:
        assert ConversationManager(store=restarted, session_id="desk").get_summary() is None
        assert ConversationManager(store=restarted, session_id="desk-other").get_summary()
    finally:
        restarted.close()


def test_summary_is_incremental_bounded_and_resumes_after_restart(temp_store):
    from ciel.memory.manager import SUMMARY_BATCH_TURNS
    from ciel.memory.models import MemoryType

    manager = ConversationManager(store=temp_store, max_turns=2, auto_summarize_threshold=2)
    for index in range(70):
        manager.add_turn("user", f"topic-{index}: " + "x" * 5000)
    router = MagicMock()
    router.send.return_value = LLMResponse(content="Earlier context", model="test", provider="test")
    assert manager.summarize_older_turns(router)
    first_input = router.send.call_args.args[0][1].content
    assert "topic-0:" in first_input
    assert "topic-32:" not in first_input
    assert len(first_input) < SUMMARY_BATCH_TURNS * 1100
    restarted = ConversationManager(store=temp_store, max_turns=2, auto_summarize_threshold=2)
    assert restarted.summarize_older_turns(router)
    next_input = router.send.call_args.args[0][1].content
    assert "Earlier context" in next_input
    assert "topic-32:" in next_input
    assert "topic-0:" not in next_input
    assert restarted.summarize_older_turns(router)
    assert restarted.summarize_older_turns(router) is None
    assert not restarted.schedule_summary(router)
    assert router.send.call_count == 3
    assert temp_store.count_memories(MemoryType.CONVERSATION) == 1
    restarted.add_turn("user", "new question")
    assert restarted.summarize_older_turns(router)
    last_input = router.send.call_args.args[0][1].content
    assert "topic-68:" in last_input
    assert "topic-67:" not in last_input


def test_clearing_during_summary_does_not_recreate_context(temp_store):
    import threading

    manager = ConversationManager(
        store=temp_store,
        max_turns=2,
        auto_summarize_threshold=2,
        summary_interval=1,
    )
    for index in range(4):
        manager.add_turn("user", f"old topic {index}")
    entered, release = threading.Event(), threading.Event()
    router = MagicMock()

    def respond(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return LLMResponse(content="Stale summary", model="test", provider="test")

    router.send.side_effect = respond
    try:
        assert manager.schedule_summary(router)
        assert entered.wait(1)
        assert manager.clear_session() == 4
    finally:
        release.set()
        manager._summary_thread.join(timeout=2)
    assert manager.get_summary() is None
    assert ConversationManager(store=temp_store).get_summary() is None
    for index in range(4):
        manager.add_turn("user", f"new topic {index}")
    router.send.return_value = LLMResponse(content="Fresh summary", model="test", provider="test")
    router.send.side_effect = None
    assert manager.summarize_older_turns(router) == "Fresh summary"


def test_legacy_summary_cursor_prevents_repeated_request(temp_store):
    from ciel.memory.models import MemoryItem, MemoryType

    manager = ConversationManager(store=temp_store, max_turns=2, auto_summarize_threshold=2)
    for index in range(6):
        manager.add_turn("user", f"topic {index}")
    temp_store.save_memory(
        MemoryItem(
            content="Legacy summary",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": "default", "older_turns_count": 4},
        )
    )
    router = MagicMock()
    assert manager.get_summary() == "Legacy summary"
    assert manager.summarize_older_turns(router) is None
    router.send.assert_not_called()


def test_background_summary_with_in_memory_store():
    store = MemoryStore(":memory:")
    try:
        manager = ConversationManager(
            store=store,
            max_turns=2,
            auto_summarize_threshold=2,
            summary_interval=1,
        )
        for index in range(4):
            manager.add_turn("user", f"topic {index}")
        assert manager.schedule_summary()
        manager._summary_thread.join(timeout=2)
        assert manager.get_summary()
        assert manager.get_total_turn_count() == 4
    finally:
        store.close()


def test_background_summary_waits_for_new_turn_batch(temp_store):
    manager = ConversationManager(
        store=temp_store,
        max_turns=2,
        auto_summarize_threshold=2,
        summary_interval=6,
    )
    router = MagicMock()
    router.send.return_value = LLMResponse(content="Summary", model="test", provider="test")
    for index in range(7):
        manager.add_turn("user", f"Turn {index}")
    assert not manager.schedule_summary(router)  # Only five older turns.
    router.send.assert_not_called()
    manager.add_turn("user", "Turn seven")
    assert manager.schedule_summary(router)
    manager._summary_thread.join(timeout=2)
    assert router.send.call_count == 1
    manager.add_turn("user", "One more turn")
    assert not manager.schedule_summary(router)


def test_real_router_uses_summary_purpose_even_for_code_transcript(temp_store):
    from ciel.config import get_settings
    from ciel.providers.manager import ProviderManager
    from ciel.providers.router import ModelRouter

    providers = MagicMock(spec=ProviderManager)
    for name in ("nim", "groq", "openrouter", "ollama"):
        provider = MagicMock()
        provider.is_configured.return_value = name == "nim"
        setattr(providers, name, provider)
    providers.stream_with_fallback.return_value = [LLMStreamChunk(
        delta="Discussed Python code.",
        model="test",
        provider="test",
    )]
    router = ModelRouter(providers)
    manager = ConversationManager(store=temp_store, max_turns=2, auto_summarize_threshold=2)
    for _ in range(4):
        manager.add_turn("user", "Refactor this Python function and debug the SQL error")
    assert manager.summarize_older_turns(router) == "Discussed Python code."
    arguments = providers.stream_with_fallback.call_args.kwargs
    assert arguments["model"] == get_settings().providers.nim_model
    assert arguments["max_tokens"] == 600


def test_pending_turns_remain_available_while_summary_is_batched(temp_store):
    manager = ConversationManager(
        store=temp_store,
        max_turns=4,
        auto_summarize_threshold=4,
        summary_interval=6,
    )
    for index in range(9):
        manager.add_turn("user", f"Pending topic {index}")
    assert not manager.schedule_summary()
    messages = manager.get_active_messages("Persona")
    assert len(messages) == 10
    assert messages[1].content == "Pending topic 0"
    assert len(manager.get_active_messages("Persona", limit=4)) == 5
