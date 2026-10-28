"""Focused spelling corrections reach real memory writes in both runtime paths."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.memory import MemoryStore
from ciel.memory.service import MemoryService
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks


def _seed_game(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        MemoryService(store).handle("Remember my favorite game is Stand of Two")
    finally:
        store.close()


def _conversational_router(reply="Let's keep talking about that."):
    router = MagicMock()

    def send(*_args, **kwargs):
        assert kwargs.get("purpose") != "controller"
        if kwargs.get("purpose") == "speech_gate":
            return LLMResponse('{"listener":"assistant","confidence":0.99}', "light", "test")
        return LLMResponse(reply, "chat", "test")

    router.send.side_effect = send
    return router


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("info", [
    {"input_source": "keyboard"}, {"wake_verified": True, "stt_confidence": 0.99},
])
@pytest.mark.parametrize("correction", ["Its 'Standoff 2'", "It's Standoff 2 not that"])
def test_focused_correction_and_repeated_confirmation_save_without_provider(
    tmp_path, monkeypatch, controller, info, correction,
):
    _seed_game(tmp_path)
    router = MagicMock()
    router.send.side_effect = RuntimeError("provider is offline")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("What's my favorite game?", info), (correction, info), ("yeah yeah yeah", info),
        ("What's my favorite game?", info),
    ], router=router, controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
        assistant = [turn for turn in turns if turn.role == "assistant"]
        assert "Should I remember this correction?" in assistant[1].content
        assert "I've updated that." in assistant[2].content
        assert assistant[-1].content == "Your favorite game is Standoff 2."
        assert all(turn.provider == "local" for turn in assistant)
        router.send.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_explicit_user_topic_survives_a_model_answer_for_a_short_correction(
    tmp_path, monkeypatch, controller,
):
    _seed_game(tmp_path)
    router = _conversational_router("I remember Stand of Two as your favorite game.")
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Can you explain what you remember about my favorite game?", {"input_source": "keyboard"}),
        ("Its 'Standoff 2'", {"input_source": "keyboard"}),
        ("yeah yeah yeah", {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
        assert turns[-1].provider == "local"
        assert "I've updated that." in turns[-1].content
        assert router.send.call_count == 1
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("question", [
    "What do you remember about me?", "Tell me about a game.",
    "What are my favorite game and favorite food?",
])
def test_assistant_claims_and_broad_recall_cannot_select_a_correction_topic(
    tmp_path, monkeypatch, controller, question,
):
    _seed_game(tmp_path)
    router = _conversational_router(
        "Your favorite game is Stand of Two. Your favorite food is soup.",
    )
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (question, {"input_source": "keyboard"}),
        ("Its 'Standoff 2'", {"input_source": "keyboard"}),
        ("yeah yeah yeah", {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Stand of Two"
        assert not any("I've updated that." in turn.content for turn in turns)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_accepted_ambient_memory_followup_cannot_claim_a_save_or_confirm_later(
    tmp_path, monkeypatch, controller, pending,
):
    _seed_game(tmp_path)
    router = _conversational_router("I would update my memory.")
    initial = "My favorite game is Standoff 2" if pending else "What's my favorite game?"
    followup = "yeah yeah yeah" if pending else "Its 'Standoff 2'"
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (initial, {"input_source": "keyboard"}), (followup, {}),
        ("yeah yeah yeah", {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Stand of Two"
        replies = [turn.content for turn in turns if turn.role == "assistant"]
        assert "I haven't saved that." in replies[1]
        assert "Ciel" in replies[1]
        assert not any("I've updated that." in reply for reply in replies)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_explicit_learning_mode_requires_a_complete_remember_command(
    tmp_path, monkeypatch, controller,
):
    _seed_game(tmp_path)
    monkeypatch.setenv("MEMORY_LEARNING_MODE", "explicit")
    router = _conversational_router()
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("What's my favorite game?", {"input_source": "keyboard"}),
        ("Its 'Standoff 2'", {"input_source": "keyboard"}),
        ("yeah yeah yeah", {"input_source": "keyboard"}),
        ("Remember my favorite game is Standoff 2", {"input_source": "keyboard"}),
        ("yeah yeah yeah", {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        replies = [turn.content for turn in turns if turn.role == "assistant"]
        assert not any("I've updated that." in reply for reply in replies[:-1])
        assert "I've updated that." in replies[-1]
        assert store.get_fact("user:favorite_game").metadata["value"] == "Standoff 2"
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_canceled_confirmation_cannot_mutate_a_focused_correction(
    tmp_path, monkeypatch, controller,
):
    _seed_game(tmp_path)
    canceled = Event()
    canceled.set()
    _, _, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("What's my favorite game?", {"input_source": "keyboard"}),
        ("Its 'Standoff 2'", {"input_source": "keyboard"}),
        ("yeah yeah yeah", {"input_source": "keyboard", "cancel_event": canceled}),
    ], router=_conversational_router(), controller=controller)
    try:
        assert store.get_fact("user:favorite_game").metadata["value"] == "Stand of Two"
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("correction", [False, True])
def test_memory_proposals_explain_the_explicit_confirmation_address_when_needed(
    tmp_path, monkeypatch, capsys, controller, typed, ambient, correction,
):
    if correction:
        _seed_game(tmp_path)
    monkeypatch.setenv("WAKE_WORD", "hello Nova")
    monkeypatch.setenv("MEMORY_LEARNING_MODE", "confirm")
    info = {"input_source": "keyboard"} if typed else {"wake_verified": True}
    router = MagicMock()
    router.send.side_effect = RuntimeError("provider is offline")
    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("My favorite game is Standoff 2", info),
    ], router=router, controller=controller, ambient=ambient)
    try:
        reply = next(turn.content for turn in turns if turn.role == "assistant")
        yes_action = "replace the saved value" if correction else "save it"
        no_action = "keep it" if correction else "skip it"
        if ambient and not typed:
            assert (f'Say "hello Nova, yes" to {yes_action} or '
                    f'"hello Nova, no" to {no_action}.') in reply
        else:
            assert f"Say yes to {yes_action} or no to {no_action}." in reply
            assert "hello Nova, yes" not in reply
        if typed:
            tts.begin_stream.assert_not_called()
            assert reply in capsys.readouterr().out
        tts.speak.assert_called_once()
        assert tts.speak.call_args.args[0] == reply
        router.send.assert_not_called()
        current = store.get_fact("user:favorite_game")
        assert (current.metadata["value"] if current else None) == (
            "Stand of Two" if correction else None
        )
    finally:
        store.close()
