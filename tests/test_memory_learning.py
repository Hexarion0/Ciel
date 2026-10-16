"""User-selected learning modes, durable recall controls, and personal fact reads."""

import pytest

from ciel.memory import MemoryStore
from ciel.memory.context import recall_context_memories
from ciel.memory.read import answer_memory_read
from ciel.memory.service import MemoryService


@pytest.fixture
def store():
    """Keep synthetic personal details in an isolated database."""
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def test_automatic_learning_saves_new_facts_and_confirms_corrections(store):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle("My current project is Atlas")
    original = store.get_fact("user:current_project")
    assert "Should I remember" in service.handle("Actually, my current project is Borealis")
    assert store.get_fact("user:current_project").metadata["value"] == "Atlas"
    assert "updated" in service.handle("yes")
    current = store.get_fact("user:current_project")
    assert current.id == original.id
    assert current.metadata["value"] == "Borealis"
    assert current.metadata["revisions"][0]["value"] == "Atlas"
    assert original.source == "user_statement"
    assert current.source == "user_confirmed"
    assert not service.has_pending_confirmation


def test_explicit_conflicting_replacement_also_requires_confirmation(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("My current project is Atlas")
    assert "Should I remember" in service.handle("Remember my current project is Borealis")
    service.handle("no")
    assert store.get_fact("user:current_project").metadata["value"] == "Atlas"


def test_correction_confirmation_can_be_configured_off(store):
    service = MemoryService(store, learning_mode="automatic", confirm_corrections=False)
    service.handle("My current project is Atlas")
    assert "updated" in service.handle("My current project is Borealis")


@pytest.mark.parametrize("statement", [
    "Maybe my current project is Atlas", "My current project is probably Atlas",
    "She said my current project is Atlas", "What if my current project is Atlas?",
    '"My current project is Atlas"', "My current project is Atlas?",
    "I feel tired today", "My password is example-password",
    "My favorite game is CS2 and can you open Steam",
])
def test_automatic_learning_does_not_infer_ambiguous_or_unrecognized_facts(store, statement):
    assert MemoryService(store, learning_mode="automatic").handle(statement) is None
    assert store.count_memories() == 0


def test_explicit_only_mode_still_answers_and_requires_command_to_save(store):
    service = MemoryService(store, learning_mode="explicit")
    assert service.handle("My current project is Atlas") is None
    assert not service.has_pending_confirmation
    assert "saved" in service.handle("Remember my current project is Atlas")
    assert service.handle("What's my current project?") == "Your current project is Atlas."
    assert service.handle("Actually, my current project is Borealis") is None
    assert store.get_fact("user:current_project").metadata["value"] == "Atlas"


@pytest.mark.parametrize("statement,key,value,question", [
    ("I live in Ulaanbaatar", "home_city", "Ulaanbaatar", "What's my home city?"),
    ("My timezone is Asia/Hovd", "timezone", "Asia/Hovd", "What's my timezone?"),
    ("My preferred language is Mongolian", "preferred_language", "Mongolian",
     "What's my preferred language?"),
    ("I prefer concise answers", "response_style", "concise answers",
     "What's my response style?"),
])
def test_clear_additional_profile_details_are_learned_and_answered_locally(
    store, statement, key, value, question,
):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle(statement)
    assert store.get_fact("user:" + key).metadata["value"] == value
    assert value in service.handle(question)


def test_recall_pause_keeps_facts_and_survives_restart(tmp_path):
    path = tmp_path / "memory.db"
    memory = MemoryStore(path)
    service = MemoryService(memory, learning_mode="automatic")
    service.handle("My current project is Atlas")
    assert "paused" in service.handle("Pause memory recall")
    assert memory.get_fact("user:current_project") is not None
    assert recall_context_memories(memory, "Atlas") == []
    assert "paused" in service.handle("What's my current project?")
    assert "Atlas" not in service.handle("What's my current project?")
    memory.close()
    reopened = MemoryStore(path)
    try:
        service = MemoryService(reopened)
        assert not service.recall_enabled
        assert "resumed" in service.handle("Resume memory recall")
        assert "Atlas" in service.handle("What's my current project?")
        assert recall_context_memories(reopened, "Atlas")
    finally:
        reopened.close()


def test_paused_recall_does_not_disable_learning_or_authorize_background_commands(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("Pause memory recall")
    assert "saved" in service.handle("My goal is finish Atlas")
    assert service.handle("Resume memory recall", authorized=False) is None
    assert not service.recall_enabled
    assert store.get_fact("user:goal") is not None


def test_direct_profile_read_redacts_values_forgotten_through_another_key(store):
    service = MemoryService(store)
    service.handle("Remember my current project is Sable")
    service.handle("Remember my favorite game is Sable")
    service.handle("Forget my current project")
    assert "Sable" not in service.handle("What's my favorite game?")


def test_paused_timeline_reads_do_not_use_saved_dates(store):
    service = MemoryService(store)
    service.handle("Remember the project started September 28, 2025")
    service.handle("Pause memory recall")
    assert "paused" in answer_memory_read(store, "Development date")


def test_local_timeline_does_not_reconstruct_a_date_forgotten_through_another_fact(store):
    service = MemoryService(store)
    service.handle("Remember the project started September 28, 2025")
    service.handle("Remember my favorite movie is 2025-09-28")
    service.handle("Forget my favorite movie")
    reply = answer_memory_read(store, "Development date")
    assert "usable confirmed project start date" in reply
    assert "September 28" not in reply
    assert "development day " not in reply


def test_invalid_learning_mode_fails_clearly(store):
    with pytest.raises(ValueError, match="learning mode"):
        MemoryService(store, learning_mode="unknown")


@pytest.mark.parametrize("statement", ["Remember", "Remember that", "Remember it", "Note that"])
def test_empty_or_referential_remember_does_not_save_literal_placeholder(store, statement):
    assert "Please state the detail" in MemoryService(store).handle(statement)
    assert store.count_memories() == 0


def test_concurrent_fact_creation_cannot_skip_replacement_confirmation(store, monkeypatch):
    service = MemoryService(store, learning_mode="automatic")
    original = store.upsert_fact
    raced = []

    def competing_update(key, item, **kwargs):
        if not raced:
            raced.append(True)
            original(key, type(item)(content="Your current project is Borealis.",
                                     metadata={"value": "Borealis"}))
        return original(key, item, **kwargs)

    monkeypatch.setattr(store, "upsert_fact", competing_update)
    assert "Should I remember" in service.handle("My current project is Atlas")
    assert store.get_fact("user:current_project").metadata["value"] == "Borealis"
    assert "updated" in service.handle("yes")
    assert store.get_fact("user:current_project").metadata["value"] == "Atlas"
