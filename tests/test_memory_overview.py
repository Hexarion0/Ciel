"""Personal memory overviews report current durable facts independently of model claims."""

from unittest.mock import MagicMock

import pytest

from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from ciel.memory.read import answer_memory_read, memory_read_intent
from ciel.memory.service import MemoryService
from ciel.persona import PERSONA_CONTEXT_VERSION
from tests.test_ambient import run_callbacks


@pytest.fixture
def store():
    database = MemoryStore(":memory:")
    yield database
    database.close()


@pytest.mark.parametrize("query", [
    "What do you remember about me?", "What do you know about me?",
    "Tell me what you remember about me.", "What have you saved about me?",
    "What personal details have you saved about me?",
])
def test_overview_reads_current_values_without_replaying_old_claims(store, query):
    memory = MemoryService(store)
    memory.handle("Remember my favorite game is Sable")
    memory.handle("Remember my favorite game is Hades")
    memory.handle("yes")
    memory.handle("Remember my preferred language is Mongolian")
    history = ConversationManager(store=store)
    history.add_turn("assistant", "Your favorite game is Sable. I updated it.")

    assert memory_read_intent(query) == "overview"
    reply = answer_memory_read(store, query, recent_messages=history.get_recent_turns())

    assert "Your favorite game is Hades." in reply
    assert "Mongolian" in reply
    assert "Sable" not in reply
    assert store.count_memories() == 2


def test_unconfirmed_or_forgotten_values_are_not_listed(store):
    memory = MemoryService(store)
    memory.handle("Remember my favorite game is Sable")
    memory.handle("Actually, my favorite game is Hades")
    query = "What do you remember about me?"
    assert "Sable" in answer_memory_read(store, query)
    assert "Hades" not in answer_memory_read(store, query)
    memory.handle("no")
    memory.handle("Forget my favorite game")
    reply = answer_memory_read(store, query)
    assert "Sable" not in reply and "Hades" not in reply
    assert "don't have" in reply


def test_overview_respects_persisted_pause_and_empty_profile(store):
    query = "What do you remember about me?"
    assert "don't have" in answer_memory_read(store, query)
    MemoryService(store).handle("Remember my favorite game is Hades")
    store.set_setting("recall_enabled", False)
    reply = answer_memory_read(store, query)
    assert "paused" in reply and "Hades" not in reply


@pytest.mark.parametrize("query", [
    "What do you remember about my friend?", "What do you remember about me and delete it?",
    "She said what do you remember about me", '"What do you remember about me?"',
    "What do you know about memory upgrades?", "What do you remember about my favorite game?",
])
def test_overview_does_not_swallow_other_topics_or_compound_requests(query):
    assert memory_read_intent(query) is None


def test_overview_is_bounded_and_does_not_treat_arbitrary_notes_as_personal_profile(store):
    topics = ["preferred_name", "favorite_game", "current_project", "goal", "occupation",
              "home_city", "timezone", "preferred_language", "response_style", "favorite_food"]
    for topic in topics:
        store.upsert_fact(f"user:{topic}", MemoryItem(
            content=f"Your {topic} is " + "x" * 170 + ".", metadata={"value": "x" * 170},
        ))
    store.save_memory(MemoryItem(content="A friend's favorite game is PRIVATE-NOTE."))
    store.save_memory(MemoryItem(content="Outdated assistant claim.",
                                 memory_type=MemoryType.CONVERSATION))

    reply = answer_memory_read(store, "What do you remember about me?")

    assert len(reply) <= 1600
    assert "PRIVATE-NOTE" not in reply and "Outdated" not in reply
    assert "some" in reply.lower()
    assert "favorite_game" in reply


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("info", [
    {"input_source": "keyboard"}, {"wake_verified": True, "stt_confidence": .99},
])
def test_restart_overview_reads_corrected_fact_without_provider(
    tmp_path, monkeypatch, controller, info,
):
    database = MemoryStore(tmp_path / "memory.db")
    memory = MemoryService(database)
    memory.handle("Remember my favorite game is Sable")
    memory.handle("Remember my favorite game is Hades")
    memory.handle("yes")
    history = ConversationManager(
        store=database, session_id=f"desktop_session:{PERSONA_CONTEXT_VERSION}",
    )
    history.add_turn("assistant", "Your favorite game is Sable.")
    database.close()
    router = MagicMock()
    router.send.side_effect = RuntimeError("Provider must not answer a saved-profile overview")

    _, _, _, reopened, turns = run_callbacks(tmp_path, monkeypatch, [
        ("What do you remember about me?", info),
    ], router=router, controller=controller)
    try:
        assert "Your favorite game is Hades." in turns[-1].content
        assert "Sable" not in turns[-1].content
        assert turns[-1].provider == "local"
        router.send.assert_not_called()
    finally:
        reopened.close()
