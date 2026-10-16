"""End-to-end local facts, corrections, recall, and forgetting without provider calls."""

from datetime import datetime

import pytest

from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from ciel.memory.context import recall_context_memories
from ciel.memory.recall import rank_memories
from ciel.memory.service import MemoryService


@pytest.fixture
def store():
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


def handle_confirmed(service, text, now=None):
    """Exercise existing persistence cases with the new explicit confirmation turn."""
    reply = service.handle(text, now)
    if reply is not None and reply.startswith("Should I remember"):
        return service.handle("yes", now)
    return reply


def test_natural_fact_and_correction_replace_one_current_record(store):
    service = MemoryService(store)
    assert "saved" in handle_confirmed(service, "My favorite game is CS2.")
    first = store.get_fact("user:favorite_game")
    assert handle_confirmed(service, "What's my favorite game?") == "Your favorite game is CS2."
    assert "updated" in handle_confirmed(service, "No, it's Valorant.")
    current = store.get_fact("user:favorite_game")
    assert current.id == first.id
    assert current.metadata["value"] == "Valorant"
    assert store.count_memories() == 1
    assert current.metadata["revisions"][0]["value"] == "CS2"
    assert "CS2" not in "\n".join(recall_context_memories(store, "games"))


def test_correction_and_profile_survive_restart(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    handle_confirmed(service, "My favorite food is pizza.")
    handle_confirmed(service, "Actually, my favorite food is pasta.")
    handle_confirmed(service, "Call me hexarion.")
    store.close()
    reopened = MemoryStore(path)
    try:
        assert reopened.get_fact("user:favorite_food").metadata["value"] == "pasta"
        assert reopened.get_fact("user:preferred_name").metadata["value"] == "hexarion"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "text",
    [
        "What if my favorite game is CS2?",
        "She said my favorite game is CS2.",
        "Maybe my favorite game is CS2",
        "Call me tomorrow",
        "I feel tired today",
    ],
)
def test_questions_other_people_and_ambiguous_statements_are_not_facts(store, text):
    assert MemoryService(store).handle(text) is None
    assert store.count_memories() == 0


def test_explicit_notes_are_deduplicated(store):
    service = MemoryService(store)
    handle_confirmed(service, "Remember that I own a red keyboard.")
    handle_confirmed(service, "Remember that I own a red keyboard.")
    assert store.count_memories() == 1


def test_start_date_correction_is_separate_from_first_commit(store):
    service = MemoryService(store)
    now = datetime(2026, 10, 3)
    handle_confirmed(service, "The project started September 28th.", now)
    handle_confirmed(service, "The first commit was September 29th.", now)
    assert store.get_fact("ciel:project_start_date").metadata["value"] == "2026-09-28"
    assert store.get_fact("ciel:first_commit_date").metadata["value"] == "2026-09-29"
    handle_confirmed(service, "What development day is it?", now)
    handle_confirmed(service, "No, September 27th.", now)
    assert store.get_fact("ciel:project_start_date").metadata["value"] == "2026-09-27"


def test_alias_recall_ranks_relevant_fact_above_newer_generic_notes(store):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    handle_confirmed(service, "Remember that we worked on code today.")
    ranked = rank_memories(store, "What do I enjoy playing?")
    assert ranked[0][1].metadata["value"] == "CS2"


def test_forget_redacts_history_and_summary_but_keeps_archive(store):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    handle_confirmed(service, "Actually, my favorite game is Valorant.")
    manager = ConversationManager(store=store, session_id="test")
    manager.add_turn("user", "My favorite game is CS2.")
    manager.add_turn("assistant", "You enjoy Valorant.")
    store.save_memory(
        MemoryItem(
            content="The user plays Valorant.",
            memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": "test"},
        )
    )
    assert "forgotten" in handle_confirmed(service, "Forget my favorite game.")
    assert store.get_fact("user:favorite_game") is None
    model_context = "\n".join(m.content for m in manager.get_active_messages("Persona"))
    assert "CS2" not in model_context
    assert "Valorant" not in model_context
    assert "CS2" in store.get_recent_turns("test")[0].content
    handle_confirmed(service, "My favorite game is Valorant.")
    assert store.redact_forgotten("Valorant") == "Valorant"


def test_unknown_forget_does_not_delete_previous_focus(store):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    reply = handle_confirmed(service, "Forget my passport number.")
    assert "couldn't find" in reply
    assert store.get_fact("user:favorite_game") is not None


def test_forget_that_uses_last_saved_fact_and_zero_match_is_honest(store):
    service = MemoryService(store)
    handle_confirmed(service, "Remember that I have a red keyboard.")
    assert "forgotten" in handle_confirmed(service, "Forget that.")
    assert "couldn't find" in handle_confirmed(service, "Forget my red keyboard.")


def test_existing_structured_anchor_is_updated_in_place(tmp_path):
    path = tmp_path / "legacy.db"
    store = MemoryStore(path)
    memory_id = store.save_memory(
        MemoryItem(
            content="CIEL project began on 2026-09-28.",
            memory_type=MemoryType.PROJECT,
            metadata={
                "project": "ciel",
                "fact_key": "project_start_date",
                "value": "2026-09-28",
            },
        )
    )
    store.close()
    store = MemoryStore(path)
    try:
        handle_confirmed(MemoryService(store),
            "Actually, the project started September 27.", datetime(2026, 10, 3)
        )
        assert store.get_fact("ciel:project_start_date").id == memory_id
        assert store.count_memories() == 1
    finally:
        store.close()


def test_unrelated_question_clears_ambiguous_correction_focus(store):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    handle_confirmed(service, "What's the time right now?")
    assert handle_confirmed(service, "No, it's Valorant.") is None
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_forgetting_persists_across_restart(tmp_path):
    path = tmp_path / "forgotten.db"
    store = MemoryStore(path)
    service = MemoryService(store)
    handle_confirmed(service, "My favorite food is pizza.")
    handle_confirmed(service, "Forget my favorite food.")
    store.close()
    store = MemoryStore(path)
    try:
        assert store.get_fact("user:favorite_food") is None
        assert "pizza" not in store.redact_forgotten("Your favorite food is pizza.")
    finally:
        store.close()


def test_note_object_is_redacted_in_paraphrased_summary(store):
    service = MemoryService(store)
    handle_confirmed(service, "Remember that I own a red keyboard.")
    handle_confirmed(service, "Forget that.")
    assert "red keyboard" not in store.redact_forgotten("The user owns a red keyboard.")


def test_unknown_profile_query_does_not_create_a_fact(store):
    service = MemoryService(store)
    assert handle_confirmed(service, "What's my favorite game?") is None
    assert store.count_memories() == 0


def test_parallel_key_updates_do_not_create_conflicting_current_records(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "parallel.db"
    first = MemoryStore(path)
    second = MemoryStore(path)

    def save(store, value):
        return store.upsert_fact(
            "user:favorite_game",
            MemoryItem(content=f"Your favorite game is {value}.", metadata={"value": value}),
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ids = list(
                executor.map(lambda args: save(*args), [(first, "CS2"), (second, "Valorant")])
            )
        assert ids[0] == ids[1]
        assert first.count_memories() == 1
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "correction", ["No, wait.", "No, it's incorrect.", "Actually, never mind."]
)
def test_vague_objection_does_not_replace_favorite(store, correction):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    assert handle_confirmed(service, correction) is None
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_curly_apostrophe_correction(store):
    service = MemoryService(store)
    handle_confirmed(service, "My favorite game is CS2.")
    assert "updated" in handle_confirmed(service, "No, it’s Valorant.")
    assert store.get_fact("user:favorite_game").metadata["value"] == "Valorant"


def test_proposal_requires_acceptance_and_rejection_does_not_save(store):
    service = MemoryService(store)
    assert "Should I remember" in service.handle("My favorite game is CS2.")
    assert store.get_fact("user:favorite_game") is None
    assert "won't save" in service.handle("no")
    assert store.count_memories() == 0
    service.handle("My favorite game is CS2.")
    assert "saved" in service.handle("yes")
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"


def test_explicit_remember_saves_without_proposal(store):
    assert "saved" in MemoryService(store).handle("Remember my favorite game is CS2")
    assert store.get_fact("user:favorite_game") is not None


def test_proposed_correction_does_not_overwrite_until_confirmed(store):
    service = MemoryService(store)
    service.handle("Remember my favorite game is CS2")
    assert "Should I remember" in service.handle("No, it's Valorant")
    assert store.get_fact("user:favorite_game").metadata["value"] == "CS2"
    assert "updated" in service.handle("yes")
    assert store.get_fact("user:favorite_game").metadata["value"] == "Valorant"


@pytest.mark.parametrize("reason", ["timeout", "unrelated", "unauthorized", "restart", "cancel"])
def test_stale_or_unauthorized_confirmation_never_saves(store, monkeypatch, reason):
    clock = [100.0]
    monkeypatch.setattr("ciel.memory.service.time.monotonic", lambda: clock[0])
    service = MemoryService(store)
    service.handle("My favorite game is CS2")
    if reason == "timeout":
        clock[0] += 61
    elif reason == "unrelated":
        service.handle("What's the weather?")
    elif reason == "unauthorized":
        service.handle("yes", authorized=False)
    elif reason == "restart":
        service = MemoryService(store)
    else:
        service.cancel_proposal()
    assert service.handle("yes") is None
    assert store.count_memories() == 0


def test_revised_proposal_saves_only_latest_candidate(store):
    service = MemoryService(store)
    service.handle("My favorite game is CS2")
    service.handle("Actually, it's Valorant")
    service.handle("yes")
    assert store.get_fact("user:favorite_game").metadata["value"] == "Valorant"
    assert store.count_memories() == 1


def test_spoken_correction_fillers_preserve_exact_favorite_value(store):
    service = MemoryService(store)
    reply = service.handle("No, like my my favorite game is Standoff 2.")
    assert reply is not None and reply.startswith("Should I remember")
    assert store.get_fact("user:favorite_game") is None
    assert "saved" in service.handle("yes")
    assert store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
