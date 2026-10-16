"""Automatic learning accepts clear declarations and rechecks changed confirmations."""

from datetime import datetime

import pytest

from ciel.memory import MemoryItem, MemoryStore
from ciel.memory.context import recall_context_memories
from ciel.memory.service import MemoryService, is_memory_statement


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


@pytest.mark.parametrize("statement", [
    "Did another project begin September 28, 2025",
    "Is the project start date September 28, 2025",
    "Was the project started September 28, 2025",
    "Could the project have started September 28, 2025",
    "Where did the project begin September 28, 2025",
])
def test_unpunctuated_questions_never_create_a_fact_or_proposal(store, statement):
    service = MemoryService(store, learning_mode="automatic")
    assert service.handle(statement, datetime(2026, 10, 16)) is None
    assert not service.has_pending_confirmation
    assert store.count_memories() == 0


@pytest.mark.parametrize("statement", [
    "The Apollo project began September 28, 2025",
    "The project started September 28, 2025",
    "CIEL development began September 28, 2025",
])
def test_broad_project_dates_require_confirmation_even_with_automatic_learning(store, statement):
    service = MemoryService(store, learning_mode="automatic", confirm_corrections=False)
    assert service.handle(statement, datetime(2026, 10, 16)).startswith("Should I remember")
    assert store.get_fact("ciel:project_start_date") is None
    assert "saved" in service.handle("yes")


def test_explicit_project_date_command_still_saves_without_a_proposal(store):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle("Remember the project started September 28, 2025")
    assert store.get_fact("ciel:project_start_date").metadata["value"] == "2025-09-28"


@pytest.mark.parametrize("correction", ["Actually, all good", "No, it's Hades"])
def test_short_inferred_correction_requires_confirmation_when_no_fact_exists(store, correction):
    service = MemoryService(store, learning_mode="automatic", confirm_corrections=False)
    assert service.handle("What's my favorite game?") is None
    assert service.handle(correction).startswith("Should I remember")
    assert store.get_fact("user:favorite_game") is None
    service.handle("no")
    assert store.get_fact("user:favorite_game") is None


@pytest.mark.parametrize("statement", [
    "My home city is London and I work as a developer",
    "I live in London but only until Friday",
    "I live in London, I work as a developer",
    "My goal is finish Atlas and then can you open Steam",
    "My favorite game is Hades but I mostly play CS2",
    "My name is Alice or Bob",
    "I work as a developer if I get the job",
])
def test_compound_or_conditional_profiles_are_not_automatically_saved(store, statement):
    service = MemoryService(store, learning_mode="automatic")
    assert service.handle(statement) is None
    assert not service.has_pending_confirmation
    assert store.count_memories() == 0


@pytest.mark.parametrize("statement,key,value", [
    ("My current project is War and Peace", "current_project", "War and Peace"),
    ("I work as a research and development engineer", "occupation",
     "a research and development engineer"),
    ("My preferred language is English and Mongolian", "preferred_language",
     "English and Mongolian"),
])
def test_conjunctions_in_a_single_supported_value_remain_valid(store, statement, key, value):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle(statement)
    assert store.get_fact("user:" + key).metadata["value"] == value


def test_explicit_remember_preserves_the_exact_compound_note(store):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle("Remember I live in London and I work as a developer")
    assert store.count_memories() == 1


def test_a_newer_conflicting_value_invalidates_the_previous_confirmation(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("My current project is Atlas")
    service.handle("My current project is Borealis")
    other = MemoryService(store, learning_mode="automatic", confirm_corrections=False)
    other.handle("My current project is Cygnus")

    reply = service.handle("yes")

    assert "changed" in reply
    assert "Should I remember" in reply
    assert store.get_fact("user:current_project").metadata["value"] == "Cygnus"
    assert "updated" in service.handle("yes")
    assert store.get_fact("user:current_project").metadata["value"] == "Borealis"


def test_new_fact_created_while_a_proposal_waits_also_requires_reconfirmation(store):
    service = MemoryService(store)
    service.handle("My current project is Atlas")
    MemoryService(store, learning_mode="automatic").handle("My current project is Borealis")
    assert "Should I remember" in service.handle("yes")
    assert store.get_fact("user:current_project").metadata["value"] == "Borealis"


def test_forgotten_fact_is_not_resurrected_by_a_stale_confirmation(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("My current project is Atlas")
    service.handle("My current project is Borealis")
    MemoryService(store).handle("Forget my current project")
    assert "Should I remember" in service.handle("yes")
    assert store.get_fact("user:current_project") is None


def test_language_and_delivery_preferences_survive_newer_favorites(store):
    service = MemoryService(store, learning_mode="automatic")
    for statement in [
        "My preferred language is Mongolian", "I prefer concise answers",
        "My favorite game is Hades", "My favorite food is noodles", "My favorite color is blue",
    ]:
        service.handle(statement)
    context = "\n".join(recall_context_memories(store, "Tell me a joke"))
    assert "Mongolian" in context
    assert "concise answers" in context


def test_forgotten_date_is_not_reconstructed_by_derived_timeline_context(store):
    service = MemoryService(store)
    service.handle("Remember the project started September 28, 2025")
    service.handle("Remember my favorite movie is 2025-09-28")
    service.handle("Forget my favorite movie")
    context = "\n".join(recall_context_memories(store, "development day"))
    assert "2025-09-28" not in context
    assert "elapsed calendar days" not in context
    assert "development day " not in context


@pytest.mark.parametrize("statement", [
    "My favorite game is Hades", "My current project is Atlas", "I live in Ulaanbaatar",
    "My timezone is Asia/Hovd", "I prefer concise answers", "Call me Alex",
    "Remember that my keyboard has silent switches", "Forget my favorite game",
    "Forget that", "The project started September 28, 2025",
])
def test_supported_memory_statement_recognition_is_read_only(store, statement):
    assert is_memory_statement(statement)
    assert store.count_memories() == 0
    service = MemoryService(store, learning_mode="automatic")
    assert service.handle(statement, authorized=False) is None
    assert store.count_memories() == 0


@pytest.mark.parametrize("statement", [
    "Is the project start date September 28, 2025", "My name is Alice or Bob",
    "My current project is Atlas?", "I live in London and I work as a developer",
    "Maybe my goal is finish Atlas", "Actually, all good", "No, it's Hades",
    "Call me tomorrow", "Tell me about memory", "The Apollo project began September 28, 2025",
])
def test_ambiguous_or_unrelated_turns_are_not_classified_as_memory_declarations(statement):
    assert not is_memory_statement(statement)


def test_declaration_recognition_does_not_override_explicit_only_mode(store):
    statement = "My current project is Atlas"
    assert is_memory_statement(statement)
    assert MemoryService(store, learning_mode="explicit").handle(statement) is None
    assert store.get_fact("user:current_project") is None


def test_existing_clear_replacement_can_still_opt_out_of_confirmation(store):
    service = MemoryService(store, learning_mode="automatic", confirm_corrections=False)
    service.handle("My current project is Atlas")
    assert "updated" in service.handle("My current project is Borealis")
    assert store.get_fact("user:current_project").metadata["value"] == "Borealis"


def test_pending_baseline_uses_content_rather_than_unrelated_metadata(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("My current project is Atlas")
    service.handle("My current project is Borealis")
    current = store.get_fact("user:current_project")
    store.upsert_fact("user:current_project", MemoryItem(
        current.content, metadata={"value": "Atlas", "extra_note": "metadata changed"},
    ))
    assert "updated" in service.handle("yes")


def test_labelled_project_entries_keep_their_metadata_and_confirmation_policy(store):
    from ciel.memory.projects import list_project_memories

    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle("Project Atlas decision: use SQLite")
    assert "saved" in service.handle("Project Atlas decision: use local speech recognition")
    assert "saved" in service.handle("Project Atlas next step: build the journal")
    assert "Should I remember" in service.handle("Project Atlas next step: benchmark retrieval")
    entries = list_project_memories(store, "Atlas")
    assert len(entries) == 3
    assert {item.metadata["entry_kind"] for item in entries} == {"decision", "next_step"}
    assert all(item.source == "user_statement" for item in entries)
    assert not any("benchmark" in item.content for item in entries)
    service.handle("yes")
    assert any("benchmark" in item.content for item in list_project_memories(store, "Atlas"))


def test_explicit_project_command_preserves_metadata_in_explicit_only_mode(store):
    from ciel.memory.projects import list_project_memories

    service = MemoryService(store, learning_mode="explicit")
    statement = "Project Atlas constraint: all recordings remain local"
    assert is_memory_statement(statement)
    assert service.handle(statement) is None
    assert "saved" in service.handle("Remember " + statement)
    entries = list_project_memories(store, "Atlas")
    assert len(entries) == 1
    assert entries[0].metadata["entry_kind"] == "constraint"
    assert entries[0].metadata["value"] == "all recordings remain local"
