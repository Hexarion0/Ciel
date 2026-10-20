"""Explicit updates must persist before the assistant reports success or reads them back."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.memory import MemoryStore
from ciel.memory.service import MemoryService
from tests.test_ambient import run_callbacks
from tests.test_memory_runtime_corrections import _conversational_router


def _seed(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        MemoryService(store).handle("Remember my favorite game is Sable")
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("info", [
    {"input_source": "keyboard"}, {"wake_verified": True, "stt_confidence": .99},
])
@pytest.mark.parametrize("command,confirmation", [
    ("Update my favorite game to Hades", "yes, update that"),
    ("Can you change my favorite game to Hades?", "yep"),
    ("Replace my favorite game with Hades", "save that"),
    ("Update my memory: my favorite game is Hades", "confirm it"),
])
def test_update_confirmation_and_overview_use_real_database_without_provider(
    tmp_path, monkeypatch, controller, info, command, confirmation,
):
    _seed(tmp_path)
    router = MagicMock()
    router.send.side_effect = RuntimeError("Update and saved-fact recall must work offline")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (command, info), (confirmation, info), ("What's my favorite game?", info),
        ("What do you remember about me?", info),
    ], router=router, controller=controller)
    try:
        replies = [turn for turn in turns if turn.role == "assistant"]
        assert replies[0].content.startswith("Should I remember this correction?")
        assert replies[1].content == "I've updated that. Your favorite game is Hades."
        assert replies[2].content == "Your favorite game is Hades."
        assert "Your favorite game is Hades." in replies[3].content
        assert "Sable" not in replies[3].content
        assert all(reply.provider == "local" for reply in replies)
        assert store.get_fact("user:favorite_game").metadata["value"] == "Hades"
        assert store.count_memories() == 1
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("command", [
    "Update my memory", "Update my favorite game", "Update my memory: my keyboard is white",
])
def test_incomplete_updates_ask_locally_instead_of_claiming_model_success(
    tmp_path, monkeypatch, controller, command,
):
    _seed(tmp_path)
    router = MagicMock()
    router.send.side_effect = RuntimeError("No model should promise a save for this request")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (command, {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        assert "Which saved detail" in turns[-1].content
        assert "haven't changed anything" in turns[-1].content
        assert store.get_fact("user:favorite_game").metadata["value"] == "Sable"
        assert store.count_memories() == 1
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_ambient_confirmation_of_an_update_still_needs_direct_address(
    tmp_path, monkeypatch, controller,
):
    _seed(tmp_path)
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Update my favorite game to Hades", {"input_source": "keyboard"}),
        ("yes, update that", {}),
        ("What's my favorite game?", {"input_source": "keyboard"}),
    ], router=_conversational_router(), controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Sable"
        replies = [turn.content for turn in turns if turn.role == "assistant"]
        assert "I haven't saved that" in replies[1]
        assert replies[-1] == "Your favorite game is Sable."
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_canceled_update_confirmation_does_not_replace_a_fact(tmp_path, monkeypatch, controller):
    _seed(tmp_path)
    canceled = Event()
    canceled.set()
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("Update my favorite game to Hades", {"input_source": "keyboard"}),
        ("yes update it", {"input_source": "keyboard", "cancel_event": canceled}),
    ], router=_conversational_router(), controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Sable"
    finally:
        store.close()
