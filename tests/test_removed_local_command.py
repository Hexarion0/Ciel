"""Retired local-mode commands cannot start fresh desktop work."""

from unittest.mock import MagicMock

import pytest

from ciel.internet import InternetService
from ciel.memory import ConversationManager, MemoryItem, MemoryStore, MemoryType
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("text,info", [
    ("/local", {"input_source": "keyboard"}),
    ("/LOCAL remember my favorite game is CS2", {"input_source": "keyboard"}),
    ("Ciel, /local search Mongolia news", {"wake_verified": True}),
    ("Hello\n/local remember my name is Secret", {"input_source": "keyboard"}),
])
def test_retired_command_is_rejected_before_desktop_models_memory_or_search(
    tmp_path, monkeypatch, controller, text, info,
):
    router, prepare, lookup = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(InternetService, "prepare", prepare)
    monkeypatch.setattr(InternetService, "lookup", lookup)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [(text, info)], router=router, controller=controller,
    )
    try:
        router.send.assert_not_called()
        prepare.assert_not_called()
        lookup.assert_not_called()
        assert turns == []
        assert store.count_memories() == 0
        assert "no longer available" in tts.speak.call_args.args[0]
    finally:
        store.close()


def test_legacy_private_summary_remains_protected_without_recent_dialogue(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    try:
        store.save_memory(MemoryItem(
            content="Confidential older conversation.", memory_type=MemoryType.CONVERSATION,
            metadata={"session_id": "legacy", "local_only": True},
        ))
        messages = ConversationManager(store=store, session_id="legacy").get_gate_messages()
        assert ModelRouter.requires_local(messages)
        assert not any("Confidential older conversation." in item.content for item in messages)
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("retired_text", [
    "/local remember my favorite game is CS2.", "Hello\n  /LOCAL remember my name is Secret",
])
def test_interrupted_retired_command_cannot_reappear_in_next_request(
    tmp_path, monkeypatch, controller, retired_text,
):
    import json
    from threading import Event

    canceled = Event()
    canceled.set()
    router = MagicMock()

    def send(_messages, **kwargs):
        content = (json.dumps({"action": "answer", "confidence": 0.99})
                   if kwargs.get("purpose") == "controller" else "Here is an explanation.")
        return LLMResponse(content, "test", "test")

    router.send.side_effect = send
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, explain this setting.", {"wake_verified": True,
                                          "supersedes_cancel_event": canceled})],
        observed=[("Ciel, " + retired_text, {
            "superseded": True, "cancel_event": canceled,
        })], router=router, controller=controller,
    )
    try:
        assert router.send.called
        assert all(not ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list)
        assert not any("remember my name is Secret" in message.content
                       for call in router.send.call_args_list for message in call.args[0])
        assert not any("/local" in turn.content for turn in turns)
        assert store.count_memories() == 0
    finally:
        store.close()
