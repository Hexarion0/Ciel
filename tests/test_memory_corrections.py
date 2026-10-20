"""Natural corrections use the actual memory handler and preserve confirmation policy."""

import pytest

from ciel.memory import MemoryStore
from ciel.memory.context import recall_context_memories
from ciel.memory.service import MemoryService, is_memory_statement


@pytest.fixture
def service():
    store = MemoryStore(":memory:")
    memory = MemoryService(store, learning_mode="automatic")
    memory.handle("Remember my favorite game is Sable")
    yield memory
    store.close()


@pytest.mark.parametrize("statement,value", [
    ("By the way, my favorite game is Hades and CS2.", "Hades and CS2"),
    ("I told you many times my favorite game is Hades.", "Hades"),
    ("I've told you before that my favorite game is Hades.", "Hades"),
    ("I told many times my fav game is Hades.", "Hades"),
    ("I already told you my favorite game is Hades.", "Hades"),
    ("Actually, my fav game is Hades.", "Hades"),
    ("My favorite game is 'Hades'.", "Hades"),
    ("My favorite game is ‘Hades’.", "Hades"),
    ("My favorite game is Hades, not Sable.", "Hades"),
])
def test_complete_natural_correction_reaches_local_handler(service, statement, value):
    assert is_memory_statement(statement)
    reply = service.handle(statement)
    assert reply.startswith("Should I remember")
    assert "replace" in reply
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Sable"
    assert "updated" in service.handle("yes")
    assert service.store.get_fact("user:favorite_game").metadata["value"] == value


@pytest.mark.parametrize("correction", [
    "It's Hades.", "It is Hades.", "Its 'Hades'", 'Its "Hades" not that',
    "No, it's Hades, not Sable.", "Actually, Hades.", "I meant Hades.",
])
def test_contextual_spelling_correction_waits_for_confirmation(service, correction):
    service.handle("What's my favorite game?")
    assert service.is_contextual_correction(correction)
    assert not service.has_pending_confirmation
    reply = service.handle(correction)
    assert reply.startswith("Should I remember")
    assert "Your favorite game is Hades." in reply
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Sable"
    assert "updated" in service.handle("yes")
    current = service.store.get_fact("user:favorite_game")
    assert current.metadata["value"] == "Hades"
    assert current.metadata["revisions"][0]["value"] == "Sable"
    assert service.store.count_memories() == 1
    assert "Sable" not in "\n".join(recall_context_memories(service.store, "favorite game"))


@pytest.mark.parametrize("correction", [
    "It's Hades?", "Is it Hades", "It's not Hades", "It's probably Hades",
    "It's Hades or CS2", "It's Hades and can you open Steam",
    "It's Hades but I mostly play Sable", "No, forget my favorite game",
    "No, open Steam", "No, tell me about Hades", "No, stop listening",
    "No, actually I don't know", "It's Hades, not a favorite",
    '"It is Hades"', "It's Hades. Can you open Steam",
])
def test_contextual_correction_classifier_does_not_swallow_ambiguous_speech(
    service, correction,
):
    assert not service.is_contextual_correction(correction)
    assert service.handle(correction) is None
    assert not service.has_pending_confirmation
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Sable"


@pytest.mark.parametrize("reason", ["unrelated", "cancel", "restart", "unauthorized"])
def test_short_correction_requires_a_current_authorized_topic(service, reason):
    if reason == "unrelated":
        service.handle("What's the weather?")
    elif reason == "cancel":
        service.cancel_proposal()
    elif reason == "restart":
        service = MemoryService(service.store)
    else:
        service.handle("What's my favorite game?", authorized=False)
    assert not service.is_contextual_correction("It's Hades")
    assert service.handle("It's Hades") is None
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Sable"


@pytest.mark.parametrize("confirmation", ["yeah yeah yeah", "Yes, yes.", "yes, save it"])
def test_natural_affirmation_only_confirms_a_live_pending_proposal(service, confirmation):
    service.handle("Actually, my favorite game is Hades")
    assert "updated" in service.handle(confirmation)
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Hades"


@pytest.mark.parametrize("confirmation", ["yeah yeah yeah", "Yes, yes.", "yes, save it"])
def test_affirmation_without_proposal_does_not_create_a_memory(service, confirmation):
    service.cancel_proposal()
    assert service.handle(confirmation) is None
    assert service.store.count_memories() == 1


def test_contextual_correction_respects_explicit_only_mode(service):
    memory = MemoryService(service.store, learning_mode="explicit")
    memory.handle("What's my favorite game?")
    assert not memory.is_contextual_correction("It's Hades")
    assert memory.handle("It's Hades") is None
    assert not memory.has_pending_confirmation


def test_corrected_value_and_local_recall_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    memory = MemoryService(store, learning_mode="automatic")
    memory.handle("Remember my favorite game is Sable")
    memory.handle('Its "Hades" not that')
    memory.handle("yes")
    store.close()
    reopened = MemoryStore(path)
    try:
        assert MemoryService(reopened).handle("What's my favorite game?") == (
            "Your favorite game is Hades."
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("question", [
    "What's my favorite game and favorite food?",
    "What are my goal and current project?",
    "What do you remember about me?",
    "What is your favorite game?",
    "My friend said their favorite game is Hades",
    'She said "my favorite game is Hades"',
    'Explain the phrase "my favorite game"',
    "What do you remember about my favorite game and the project start date?",
    "What do you know about my favorite game and when the project started?",
])
def test_multiple_or_unspecified_topics_do_not_authorize_a_short_correction(service, question):
    service.handle(question)
    assert service.focus_key is None
    assert not service.is_contextual_correction("It's Hades")
    assert service.handle("It's Hades") is None
    assert not service.has_pending_confirmation


def test_fav_shorthand_reads_the_saved_value_locally(service):
    from ciel.memory.read import memory_read_intent

    assert memory_read_intent("What's my fav game?") == "recall"
    assert service.handle("What's my fav game?") == "Your favorite game is Sable."


def test_unknown_rejected_value_does_not_get_saved_as_part_of_favorite(service):
    assert service.handle("My favorite game is Hades, not a favorite") is None
    assert not service.has_pending_confirmation
    assert service.store.get_fact("user:favorite_game").metadata["value"] == "Sable"
