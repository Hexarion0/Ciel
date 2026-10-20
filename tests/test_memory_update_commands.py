"""Explicit memory updates use the same durable write and confirmation flow as remember."""

import pytest

from ciel.memory import MemoryStore
from ciel.memory.service import MemoryService, is_memory_confirmation, is_memory_statement


@pytest.fixture
def memory():
    store = MemoryStore(":memory:")
    service = MemoryService(store)
    service.handle("Remember my favorite game is Sable")
    yield service
    store.close()


@pytest.mark.parametrize("command", [
    "Update my favorite game to Hades",
    "Change my favorite game to Hades.",
    "Replace my favorite game with Hades",
    "Please update my favorite game to Hades",
    "Can you update my favorite game to Hades?",
    "Could you please change my favorite game to Hades?",
    "Would you replace my favorite game with Hades?",
    "Update my memory: my favorite game is Hades",
    "Update your memory with my favorite game is Hades",
    "Please update the saved memory: my favorite game is Hades",
    "Change my fav game to 'Hades'",
])
@pytest.mark.parametrize("mode", ["confirm", "automatic", "explicit"])
def test_update_commands_propose_a_real_replacement_in_every_learning_mode(memory, command, mode):
    service = MemoryService(memory.store, learning_mode=mode)
    assert is_memory_statement(command)
    assert "Should I remember this correction?" in service.handle(command)
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Sable"
    assert "I've updated that." in service.handle("yes, update that")
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Hades"
    assert memory.store.count_memories() == 1


@pytest.mark.parametrize("command,key,value", [
    ("Update my name to Nova", "preferred_name", "Nova"),
    ("Change my preferred name to Nova", "preferred_name", "Nova"),
    ("Update my current project to Borealis", "current_project", "Borealis"),
    ("Replace my active project with Borealis", "current_project", "Borealis"),
    ("Change my goal to finish school", "goal", "finish school"),
    ("Update my occupation to designer", "occupation", "designer"),
    ("Change my job to designer", "occupation", "designer"),
    ("Update my home city to Tokyo", "home_city", "Tokyo"),
    ("Update my time zone to Asia/Tokyo", "timezone", "Asia/Tokyo"),
    ("Replace my preferred language with Japanese", "preferred_language", "Japanese"),
    ("Update my response style to short replies", "response_style", "short replies"),
    ("Update my memory: I live in Tokyo", "home_city", "Tokyo"),
])
def test_known_personal_fields_use_their_existing_fact_keys(memory, command, key, value):
    service = MemoryService(memory.store, learning_mode="explicit")
    assert is_memory_statement(command)
    assert "I've saved that." in service.handle(command)
    assert memory.store.get_fact("user:" + key).metadata["value"] == value
    assert memory.store.count_memories() == 2


@pytest.mark.parametrize("command", [
    "Update my memory", "Please update your memory.",
    "Can you update my memory?", "Update my favorite game", "Change my goal",
    "Replace my favorite game", "Update my favorite game to",
    "Update my memory: my keyboard is white",
    "Update my memory: remember that my keyboard is white",
    "Replace my memory with my keyboard is white",
    "Update my favorite game to maybe Hades",
    "Update my favorite game to Hades or Sable",
    "Change my favorite game to Hades and open Steam",
    "Change my favorite game to Hades and close Steam",
    "Update my favorite game to Hades, then pause memory recall",
    "Update my favorite game to Hades. Can you open Steam",
    "Update my favorite game to not Hades",
    'Update my memory: "my favorite game is Hades"',
    "Update my memory: she said my favorite game is Hades",
])
def test_incomplete_unsupported_or_ambiguous_updates_ask_for_a_detail_without_saving(
    memory, command,
):
    assert is_memory_statement(command)
    reply = memory.handle(command)
    assert "Which saved detail" in reply
    assert not memory.has_pending_confirmation
    assert memory.store.count_memories() == 1
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Sable"


@pytest.mark.parametrize("command", [
    "Update my favorite game to Hades and set an alarm for 8",
    "Change my home city to Tokyo and send a message to my mom",
    "Update my favorite game to Hades and delete my old memories",
    "Change my home city to Tokyo. Set an alarm for 8",
    "Update my favorite game to Hades; run a script",
    "Change my home city to Tokyo and turn the volume down",
])
def test_second_commands_neither_replace_existing_facts_nor_create_new_ones(
    memory, command,
):
    assert is_memory_statement(command)
    assert "Which saved detail" in memory.handle(command)
    assert not memory.has_pending_confirmation
    assert memory.store.count_memories() == 1
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Sable"
    assert memory.store.get_fact("user:home_city") is None


@pytest.mark.parametrize("command,key,value", [
    ("Update my favorite game to Ori and the Blind Forest", "favorite_game",
     "Ori and the Blind Forest"),
    ("Update my home city to Trinidad and Tobago", "home_city", "Trinidad and Tobago"),
])
def test_conjunctions_inside_names_remain_valid_update_values(memory, command, key, value):
    reply = memory.handle(command)
    if memory.has_pending_confirmation:
        reply = memory.handle("yes update it")
    assert "I've " in reply
    assert memory.store.get_fact("user:" + key).metadata["value"] == value


@pytest.mark.parametrize("text", [
    '"Update my favorite game to Hades"',
    "She said update my favorite game to Hades",
    "How do I update my favorite game to Hades?",
    "If I change my favorite game to Hades",
    "Update my browser", "Change my password to secret", "Update her favorite game to Hades",
])
def test_reports_questions_and_unrelated_commands_are_not_memory_updates(memory, text):
    assert not is_memory_statement(text)
    assert memory.handle(text) is None
    assert not memory.has_pending_confirmation
    assert memory.store.count_memories() == 1


@pytest.mark.parametrize("answer", [
    "yes update it", "Yes, update that.", "yes remember it", "yes, remember that",
    "yep", "yup", "save that", "confirm it",
])
def test_bounded_update_confirmations_need_a_live_authorized_proposal(memory, answer):
    assert is_memory_confirmation(answer)
    assert memory.handle(answer) is None
    memory.handle("My favorite game is Hades")
    assert "updated" in memory.handle(answer)
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Hades"


@pytest.mark.parametrize("answer", [
    "okay", "sure", "do it", "go ahead", "yes but not Hades", "yep and open Steam",
    "confirm it if I say so", "yes update it tomorrow",
])
def test_ambiguous_or_conditional_confirmations_do_not_change_saved_facts(memory, answer):
    memory.handle("My favorite game is Hades")
    assert not is_memory_confirmation(answer)
    assert memory.handle(answer) is None
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Sable"


def test_unauthorized_update_and_confirmation_never_save(memory):
    assert memory.handle("Update my favorite game to Hades", authorized=False) is None
    assert not memory.has_pending_confirmation
    memory.handle("Update my favorite game to Hades")
    assert memory.handle("yes update it", authorized=False) is None
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Sable"


def test_explicit_updates_keep_configured_replacement_confirmation_opt_out(memory):
    service = MemoryService(memory.store, learning_mode="explicit", confirm_corrections=False)
    assert "I've updated that." in service.handle("Update my favorite game to Hades")
    assert not service.has_pending_confirmation
    assert memory.store.get_fact("user:favorite_game").metadata["value"] == "Hades"


def test_update_reuses_one_fact_preserves_revisions_and_survives_restart(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    service.handle("Remember my favorite game is Sable")
    service.handle("Change my favorite game to Hades")
    service.handle("yep")
    store.close()
    reopened = MemoryStore(path)
    try:
        current = reopened.get_fact("user:favorite_game")
        assert current.metadata["value"] == "Hades"
        assert current.metadata["revisions"][0]["value"] == "Sable"
        assert reopened.count_memories() == 1
        assert MemoryService(reopened).handle("What's my favorite game?") == (
            "Your favorite game is Hades."
        )
    finally:
        reopened.close()
