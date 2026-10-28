"""Private structured facts remain private through every application-owned answer path."""

from datetime import datetime

import pytest

from ciel.memory.context import recall_context_memories
from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.read import answer_memory_read
from ciel.memory.service import PRIVATE_MEMORY, MemoryService
from ciel.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def _private(store, key, content, value, *, kind=MemoryType.FACT):
    store.upsert_fact(key, MemoryItem(content, memory_type=kind,
                                    metadata={"value": value, "local_only": True}))


def test_private_profile_overview_and_recall_anchors_require_local_context(store):
    _private(store, "user:preferred_name", "Your preferred name is SecretName.", "SecretName",
             kind=MemoryType.PREFERENCE)
    _private(store, "user:goal", "Your goal is SecretGoal.", "SecretGoal", kind=MemoryType.PROJECT)
    store.upsert_fact("user:occupation", MemoryItem("Your occupation is developer."))
    public = " ".join(recall_context_memories(store, "SecretName SecretGoal"))
    assert "SecretName" not in public and "SecretGoal" not in public
    assert "developer" in public
    private = " ".join(recall_context_memories(store, "SecretName SecretGoal", local_only=True))
    assert "SecretName" in private and "SecretGoal" in private
    public_overview = answer_memory_read(store, "What do you know about me?")
    assert "SecretName" not in public_overview and "SecretGoal" not in public_overview
    private_overview = answer_memory_read(store, "What do you know about me?", local_only=True)
    assert "SecretName" in private_overview and "SecretGoal" in private_overview


@pytest.mark.parametrize("query,key,content,value", [
    ("What's my favorite game?", "user:favorite_game", "Your favorite game is SecretGame.",
     "SecretGame"),
    ("What's my goal?", "user:goal", "Your goal is SecretGoal.", "SecretGoal"),
    ("What is my name?", "user:preferred_name", "Your preferred name is SecretName.", "SecretName"),
    ("What do you call me?", "user:preferred_name",
     "Your preferred name is SecretName.", "SecretName"),
])
def test_direct_private_profile_answers_require_local_context(store, query, key, content, value):
    _private(store, key, content, value)
    service = MemoryService(store)
    assert service.handle(query) == PRIVATE_MEMORY
    assert service.handle(query, local_only=True) == content


def test_private_timeline_cannot_be_reconstructed_in_public_context(store):
    _private(store, "ciel:project_start_date", "CIEL project development began on 2026-10-01.",
             "2026-10-01", kind=MemoryType.PROJECT)
    assert answer_memory_read(store, "Development day", datetime(2026, 10, 25)) == PRIVATE_MEMORY
    assert "development day 25" in answer_memory_read(
        store, "Development day", datetime(2026, 10, 25), local_only=True,
    )
    assert "2026-10-01" not in " ".join(recall_context_memories(store, "development date"))
    assert "2026-10-01" in " ".join(recall_context_memories(
        store, "development date", local_only=True,
    ))


def test_local_explicit_writes_and_corrections_remain_private_and_move_to_private_file(store):
    service = MemoryService(store)
    service.handle("Remember my favorite game is PublicGame")
    reply = service.handle("Remember my favorite game is SecretGame", local_only=True)
    assert "correction" in reply
    service.handle("yes", local_only=True)
    fact = store.get_fact("user:favorite_game")
    assert fact.metadata["local_only"] is True
    assert fact.metadata["value"] == "SecretGame"
    assert "SecretGame" not in (store.workspace.root / "MEMORY.md").read_text()
    assert "SecretGame" in (store.workspace.root / "memory/private/MEMORY.md").read_text()
    assert "SecretGame" not in " ".join(excerpt.text for excerpt in store.workspace.bootstrap())
    assert service.handle("What's my favorite game?") == PRIVATE_MEMORY
    public = service.handle("Remember my favorite game is PublicReplacement")
    assert public == PRIVATE_MEMORY
    assert "SecretGame" not in public
    assert store.get_fact("user:favorite_game").metadata["value"] == "SecretGame"


def test_private_proposal_cannot_be_confirmed_from_a_public_turn(store):
    service = MemoryService(store, learning_mode="confirm")
    assert "Should I remember" in service.handle("My favorite game is SecretGame", local_only=True)
    assert service.handle("yes") == PRIVATE_MEMORY
    assert store.get_fact("user:favorite_game") is None
    assert not service.has_pending_confirmation


def test_privacy_added_while_public_proposal_waits_blocks_public_confirmation(store):
    service = MemoryService(store, learning_mode="automatic")
    service.handle("My favorite game is PublicGame")
    service.handle("My favorite game is ReplacementGame")
    original = store.get_fact("user:favorite_game")
    original.metadata["local_only"] = True
    store.upsert_fact("user:favorite_game", original)
    assert service.handle("yes") == PRIVATE_MEMORY
    assert store.get_fact("user:favorite_game").metadata["value"] == "PublicGame"


def test_store_writes_cannot_downgrade_an_existing_private_fact(store):
    _private(store, "user:favorite_game", "Your favorite game is SecretGame.", "SecretGame")
    store.upsert_fact("user:favorite_game", MemoryItem("Your favorite game is NewGame.",
                                                     metadata={"value": "NewGame"}))
    updated = store.get_fact("user:favorite_game")
    assert updated.metadata["local_only"] is True
    updated.metadata.pop("local_only")
    store.save_memory(updated)
    assert store.get_fact("user:favorite_game").metadata["local_only"] is True
    assert all(document.local_only for document in store.workspace.documents()
               if document.text.strip())


def test_public_forget_does_not_select_private_fact_or_echo_its_content(store):
    _private(store, "user:favorite_game", "Your favorite game is SecretGame.", "SecretGame")
    service = MemoryService(store)
    assert service.handle("Forget my favorite game") == PRIVATE_MEMORY
    assert store.get_fact("user:favorite_game") is not None
    assert "forgotten" in service.handle("Forget my favorite game", local_only=True)
    assert store.get_fact("user:favorite_game") is None


def test_semantic_results_cannot_bypass_structured_context_privacy(store):
    _private(store, "user:favorite_game", "Your favorite game is SecretGame.", "SecretGame")
    fact = store.get_fact("user:favorite_game")

    class PrivateSemanticResult:
        def hybrid_rank(self, _query, _lexical, limit=4):
            return [(1.0, fact)]

    assert recall_context_memories(store, "unknown paraphrase",
                                   semantic_recall=PrivateSemanticResult()) == []
    assert "SecretGame" in " ".join(recall_context_memories(
        store, "unknown paraphrase", semantic_recall=PrivateSemanticResult(), local_only=True,
    ))


def test_local_automatic_learning_keeps_source_policy(store):
    service = MemoryService(store, learning_mode="automatic")
    assert "saved" in service.handle("My goal is build a private prototype", local_only=True)
    fact = store.get_fact("user:goal")
    assert fact.metadata["local_only"] is True
    assert fact.source == "user_statement"
    assert recall_context_memories(store, "prototype") == []
