"""Project decisions and unfinished work keep exact project identities and source records."""

from ciel.memory import MemoryItem, MemoryStore
from ciel.memory.projects import list_project_memories, parse_project_entry
from ciel.memory.recall import rank_memories
from ciel.memory.service import MemoryService


def test_project_journal_preserves_multiple_decisions_and_separates_projects():
    store = MemoryStore(":memory:")
    try:
        for statement in [
            "Project Atlas decision: use SQLite", "Project Atlas decision: keep backups",
            "Project Atlas constraint: work offline", "Project Atlas next step: test restart",
            "Project Borealis next step: write a proposal",
        ]:
            key, item = parse_project_entry(statement)
            store.upsert_fact(key, item)
        entries = list_project_memories(store, "ATLAS")
        assert len(entries) == 4
        assert {entry.metadata["project_name"] for entry in entries} == {"Atlas"}
        assert len([entry for entry in entries if entry.metadata["entry_kind"] == "decision"]) == 2
        assert "Atlas" in rank_memories(store, "Atlas next step")[0][1].content
    finally:
        store.close()


def test_next_step_updates_one_record_and_keeps_revisions():
    store = MemoryStore(":memory:")
    try:
        first_key, first = parse_project_entry("Project Atlas next step: test restart")
        store.upsert_fact(first_key, first)
        key, item = parse_project_entry("Project atlas next step: ship the build")
        store.upsert_fact(key, item)
        assert key == first_key
        current = list_project_memories(store, "Atlas")[0]
        assert current.id == first.id
        assert current.metadata["revisions"][0]["value"] == "test restart"
    finally:
        store.close()


def test_quoted_questions_and_unlabeled_plans_do_not_become_project_entries():
    for text in [
        "Should Project Atlas decision: use SQLite?", "Project Atlas decision: use SQLite?",
        '"Project Atlas decision: use SQLite"', "We could use SQLite for Atlas",
    ]:
        assert parse_project_entry(text) is None


def test_project_entries_follow_learning_and_replacement_confirmation_modes():
    store = MemoryStore(":memory:")
    try:
        service = MemoryService(store, learning_mode="automatic")
        assert "saved" in service.handle("Project Atlas next step: test restart")
        assert "Should I remember" in service.handle("Project Atlas next step: ship the build")
        assert list_project_memories(store, "Atlas")[0].metadata["value"] == "test restart"
        assert "updated" in service.handle("yes")
        assert list_project_memories(store, "Atlas")[0].metadata["value"] == "ship the build"
        assert "saved" in service.handle("Project Atlas decision: use SQLite")
        assert "saved" in service.handle("Project Atlas decision: keep backups")
        assert len(list_project_memories(store, "Atlas")) == 3
        explicit = MemoryService(store, learning_mode="explicit")
        assert explicit.handle("Project Atlas constraint: stay offline") is None
        assert "saved" in explicit.handle("Remember project Atlas constraint: stay offline")
    finally:
        store.close()


def test_named_project_forgetting_does_not_delete_the_assistant_timeline_or_user_goal():
    store = MemoryStore(":memory:")
    try:
        service = MemoryService(store, learning_mode="automatic")
        service.handle("Remember the project started September 28, 2025")
        service.handle("My goal is finish the prototype")
        service.handle("Project Atlas next step: benchmark retrieval")
        service.handle("Project Atlas goal: ship the app")
        assert "forgotten" in service.handle("Forget Project Atlas next step")
        assert store.get_fact("ciel:project_start_date") is not None
        assert store.get_fact("user:goal") is not None
        assert len(list_project_memories(store, "Atlas")) == 1
        assert "forgotten" in service.handle("Forget Project Atlas goal")
        assert store.get_fact("user:goal") is not None
        assert "couldn't find" in service.handle("Forget Project Unknown next step")
        assert store.get_fact("ciel:project_start_date") is not None
        assert "forgotten" in service.handle("Forget project start date")
    finally:
        store.close()


def test_forgetting_detects_older_ambiguous_matches_across_the_full_collection():
    store = MemoryStore(":memory:")
    try:
        service = MemoryService(store, learning_mode="automatic")
        service.handle("Project Atlas decision: use SQLite")
        for index in range(2001):
            store.save_memory(MemoryItem(content=f"Unrelated archive note {index}"))
        service.handle("Project Atlas decision: keep backups")
        assert "several saved facts" in service.handle("Forget Project Atlas decision")
        assert len(list_project_memories(store, "Atlas")) == 2
    finally:
        store.close()
