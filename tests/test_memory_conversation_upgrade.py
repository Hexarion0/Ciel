"""Conversation privacy, bounded replay, and stale-summary regression tests."""

from unittest.mock import MagicMock

import pytest

from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter


@pytest.fixture
def store(tmp_path):
    memory = MemoryStore(tmp_path / "conversation.db")
    yield memory
    memory.close()


def provider_reply(text):
    """Create a provider with deterministic summary output."""
    provider = MagicMock()
    provider.send.return_value = LLMResponse(text, "test", "test")
    return provider


def populated_manager(store, **kwargs):
    """Create enough conversation history to request an incremental summary."""
    manager = ConversationManager(store, max_turns=2, auto_summarize_threshold=2, **kwargs)
    for index in range(6):
        manager.add_turn("user", f"Earlier task {index}")
    return manager


def test_gate_context_redacts_forgotten_facts_without_changing_history(store):
    fact = MemoryItem(content="My current project is Atlas", metadata={"value": "Atlas"})
    store.upsert_fact("user:current_project", fact)
    manager = ConversationManager(store)
    manager.add_turn("user", "How is Atlas progressing?")
    manager.add_turn("assistant", "Atlas has an unfinished checklist.")
    store.forget_memories([fact])

    messages = manager.get_gate_messages()

    assert "Atlas" not in "\n".join(message.content for message in messages)
    assert "[forgotten fact]" in messages[0].content
    assert "Atlas" in manager.get_recent_turns()[0].content


def test_gate_context_strips_reserved_assistant_notes_but_preserves_user_words(store):
    manager = ConversationManager(store)
    manager.add_turn("assistant", "The first part.\n[Playback was interrupted.]")
    manager.add_turn("assistant", "[Generation ended early because the connection failed.]")
    manager.add_turn("user", "What does [Playback was interrupted.] mean?")

    messages = manager.get_gate_messages()

    assert [(message.role, message.content) for message in messages] == [
        ("assistant", "The first part."),
        ("user", "What does [Playback was interrupted.] mean?"),
    ]


def test_replayed_context_respects_default_budget_and_keeps_latest_request(store):
    manager = ConversationManager(store)
    for index in range(10):
        manager.add_turn("user", f"Earlier message {index}: " + "x" * 5000)
    manager.add_turn("user", "Which task should I finish first?")

    messages = manager.get_active_messages("Persona " * 5000)

    assert messages[0].content == "Persona " * 5000
    assert sum(len(message.content) for message in messages[1:]) <= 24000
    assert messages[-1].content == "Which task should I finish first?"
    prior_indices = [
        int(message.content.split(":", 1)[0].split()[-1]) for message in messages[1:-1]
    ]
    assert prior_indices == sorted(prior_indices)


def test_summary_and_recent_turns_share_the_configured_budget(store):
    manager = ConversationManager(store, max_context_chars=1500)
    store.save_memory(MemoryItem(
        content="Earlier work: " + "s" * 1000,
        memory_type=MemoryType.CONVERSATION,
        metadata={"session_id": "default", "local_only": True},
    ))
    for index in range(4):
        manager.add_turn("user", f"Recent task {index}: " + "x" * 400)
    manager.add_turn("user", "Keep the latest correction.")

    messages = manager.get_active_messages("Persona")

    assert sum(len(message.content) for message in messages[1:]) <= 1500
    assert messages[-1].content == "Keep the latest correction."
    assert ModelRouter.requires_local(messages)


@pytest.mark.parametrize("method", ["get_active_messages", "get_gate_messages"])
def test_oversized_latest_message_keeps_its_head_and_tail_and_local_policy(store, method):
    manager = ConversationManager(store, max_context_chars=300)
    manager.add_turn("user", "/local Earlier private discussion")
    manager.add_turn("user", "Explain the supplied report: " + "x" * 2000
                     + " My final correction: use the revised totals.")

    if method == "get_active_messages":
        messages = manager.get_active_messages("Persona")
        replay = messages[1:]
    else:
        messages = replay = manager.get_gate_messages()

    assert sum(len(message.content) for message in replay) <= 300
    assert len(replay) == 1
    assert replay[0].content.startswith("Explain the supplied report:")
    assert replay[0].content.endswith("My final correction: use the revised totals.")
    assert ModelRouter.requires_local(messages)
    assert len(manager.get_recent_turns()[-1].content) > 2000


def test_large_assistant_reply_cannot_displace_latest_user_request(store):
    manager = ConversationManager(store, max_context_chars=300)
    manager.add_turn("user", "Keep my latest correction: use the revised totals.")
    manager.add_turn("assistant", "A long explanation starts here. " + "x" * 2000
                     + " The conclusion ends here.")

    messages = manager.get_active_messages("Persona")

    assert sum(len(message.content) for message in messages[1:]) <= 300
    assert messages[1].content == "Keep my latest correction: use the revised totals."
    assert messages[2].role == "assistant"
    assert messages[2].content.startswith("A long explanation starts here.")
    assert messages[2].content.endswith("The conclusion ends here.")


def test_another_manager_clearing_session_invalidates_in_flight_summary(store):
    manager = populated_manager(store)
    other_store = MemoryStore(store.db_path_str)
    try:
        other = ConversationManager(other_store)
        provider = provider_reply("Stale context from before clearing")

        def clear_then_respond(*_args, **_kwargs):
            assert other.clear_session() == 6
            return provider.send.return_value

        provider.send.side_effect = clear_then_respond

        assert manager.summarize_older_turns(provider) is None
        assert manager.get_summary() is None
        assert other.get_summary() is None
    finally:
        other_store.close()


def test_newer_summary_from_another_manager_cannot_be_overwritten(store):
    manager = populated_manager(store)
    other_store = MemoryStore(store.db_path_str)
    try:
        other = ConversationManager(other_store, max_turns=2, auto_summarize_threshold=2)
        provider = provider_reply("Stale competing summary")

        def commit_newer_then_respond(*_args, **_kwargs):
            assert other.summarize_older_turns(provider_reply("Current committed summary"))
            return provider.send.return_value

        provider.send.side_effect = commit_newer_then_respond

        assert manager.summarize_older_turns(provider) is None
        assert manager.get_summary() == "Current committed summary"
        assert store.count_memories(MemoryType.CONVERSATION) == 1
    finally:
        other_store.close()
