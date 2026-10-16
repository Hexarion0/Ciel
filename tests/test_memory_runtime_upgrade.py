"""Memory controls remain application-owned through the real runtime callbacks."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.config import Settings
from ciel.memory import MemoryItem, MemoryStore
from ciel.memory.read import answer_memory_read
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks


@pytest.mark.parametrize("decision", ["answer", "silent", "invalid"])
def test_explicit_pause_routes_locally_even_if_controller_misclassifies(
    tmp_path, monkeypatch, decision,
):
    router = MagicMock()
    content = "invalid JSON" if decision == "invalid" else json.dumps({
        "action": decision, "confidence": 1,
    })
    router.send.return_value = LLMResponse(content, "fake", "fake")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Pause memory recall", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert store.get_setting("recall_enabled") is False
        assert "paused" in turns[-1].content
        assert turns[-1].provider == "local"
    finally:
        store.close()


def test_runtime_uses_selected_automatic_mode_and_confirms_existing_corrections(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("MEMORY_LEARNING_MODE", "automatic")
    router = MagicMock()
    router.send.return_value = LLMResponse("A normal response.", "fake", "fake")
    typed = {"input_source": "keyboard"}
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("My current project is Atlas", typed),
        ("My current project is Borealis", typed),
        ("no", typed),
        ("What's my current project?", typed),
    ], router=router, controller=False)
    try:
        assert store.get_fact("user:current_project").metadata["value"] == "Atlas"
        assert any(turn.content.startswith("Should I remember") for turn in turns)
        assert turns[-1].content == "Your current project is Atlas."
    finally:
        store.close()


def test_supported_statements_save_locally_before_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_LEARNING_MODE", "automatic")
    router = MagicMock()
    router.send.return_value = LLMResponse(json.dumps({
        "action": "answer", "confidence": 1,
    }), "fake", "fake")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Project Atlas decision: use SQLite", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert store.count_memories() == 1
        assert turns[-1].provider == "local"
        assert "saved" in turns[-1].content
        router.send.assert_not_called()
    finally:
        store.close()


def test_paused_prompt_omits_saved_name_and_includes_authoritative_status(tmp_path, monkeypatch):
    seed = MemoryStore(tmp_path / "memory.db")
    seed.upsert_fact("user:preferred_name", MemoryItem(
        content="Your preferred name is SyntheticName.", metadata={"value": "SyntheticName"},
    ))
    seed.set_setting("recall_enabled", False)
    seed.close()
    router = MagicMock()
    router.send.return_value = LLMResponse("Hello.", "fake", "fake")
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Tell me a joke", {"input_source": "keyboard"})],
        router=router, controller=False,
    )
    try:
        prompt = router.send.call_args.args[0][0].content
        assert "SyntheticName" not in prompt
        assert "saved-fact recall is paused" in prompt
    finally:
        store.close()


def test_memory_status_is_local_application_evidence():
    store = MemoryStore(":memory:")
    try:
        assert "enabled" in answer_memory_read(store, "Memory status")
        store.set_setting("recall_enabled", False)
        assert "paused" in answer_memory_read(store, "Is memory recall paused?")
    finally:
        store.close()


def test_memory_configuration_exposes_bounds_and_modes():
    memory = Settings(
        _env_file=None, memory_learning_mode="automatic", memory_confirm_corrections=True,
        memory_recall_limit=7, memory_recall_max_chars=1200, memory_max_context_chars=3000,
    ).memory
    assert memory.learning_mode == "automatic"
    assert memory.confirm_corrections
    assert memory.recall_limit == 7
    assert memory.recall_max_chars == 1200
    assert memory.max_context_chars == 3000
