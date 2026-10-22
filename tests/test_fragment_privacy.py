"""Speech fragments preserve old privacy boundaries without reviving retired commands."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.internet import InternetResult, InternetService
from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.memory.bridge import desktop_conversation_id
from ciel.memory.manager import ConversationManager
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _decision


def _seed_legacy_private_context(tmp_path, source):
    """Write an archive from an older release before constructing the runtime."""
    store = MemoryStore(tmp_path / "memory.db")
    try:
        manager = ConversationManager(store=store, session_id=desktop_conversation_id())
        manager.add_turn("user", "/local Explain this setting." if source == "history" else
                         "Explain this setting.")
        if source == "summary":
            store.save_memory(MemoryItem(
                content="An earlier private discussion about this setting.",
                memory_type=MemoryType.CONVERSATION,
                metadata={"session_id": desktop_conversation_id(), "local_only": True},
            ))
    finally:
        store.close()


def _long_chain(*, address_all=False, retired_start=False):
    """Include an empty interruption and more clips than the five-fragment tail."""
    texts = [
        "Ciel, explain this setting.", "Include an example.", "", "Use simple words.",
        "Keep it brief.", "Mention drawbacks.", "Cover alternatives.",
        "Finish with a recommendation.",
    ]
    if retired_start:
        texts[0] = "Ciel, /LoCaL retired secret marker"
    if address_all:
        texts = [texts[0], *["Ciel, " + text if text else "" for text in texts[1:]]]
    events = [Event() for _text in texts]
    for event in events[:-1]:
        event.set()
    observed = [
        (text, {
            "superseded": True, "cancel_event": events[index],
            **({"supersedes_cancel_event": events[index - 1]} if index else {}),
        })
        for index, text in enumerate(texts[:-1])
    ]
    final = (texts[-1], {
        "cancel_event": events[-1], "supersedes_cancel_event": events[-2],
    })
    return observed, final


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("source", ["history", "summary"])
def test_legacy_privacy_survives_normal_fragment_cap_restart_and_summary(
    tmp_path, monkeypatch, controller, source,
):
    _seed_legacy_private_context(tmp_path, source)
    observed, final = _long_chain(address_all=not controller)
    router = MagicMock()
    router.send.side_effect = ([
        _decision("answer"),
    ] if controller else []) + [LLMResponse("An explanation.", "chat", "ollama")]
    prepare, lookup = MagicMock(), MagicMock()
    monkeypatch.setattr(InternetService, "prepare", prepare)
    monkeypatch.setattr(InternetService, "lookup", lookup)
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [final], observed=observed, router=router, controller=controller,
    )
    try:
        assert router.send.call_count == (2 if controller else 1)
        assert all(ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list)
        assert sum(turn.content.startswith("/local") for turn in turns) == (source == "history")
        assert any(turn.content == "Finish with a recommendation." for turn in turns)
        prepare.assert_not_called()
        lookup.assert_not_called()
    finally:
        store.close()

    # Reopen the archive independently with a smaller window, then summarize
    # older turns. The old privacy decision must survive both kinds of clipping.
    store = MemoryStore(tmp_path / "memory.db")
    try:
        restored = ConversationManager(
            store=store, session_id=desktop_conversation_id(), max_turns=2,
            auto_summarize_threshold=2,
        )
        assert ModelRouter.requires_local(restored.get_gate_messages(limit=1))
        assert ModelRouter.requires_local(restored.get_active_messages("Continue normally."))
        summary_provider = MagicMock()
        summary_provider.name = "ollama"
        summary_provider.send.return_value = LLMResponse("A private explanation.", "chat", "ollama")
        assert restored.summarize_older_turns(summary_provider) == "A private explanation."
        assert ModelRouter.requires_local(summary_provider.send.call_args.args[0])
        assert store.get_session_summary(desktop_conversation_id()).metadata["local_only"]
        assert ModelRouter.requires_local(restored.get_gate_messages(limit=1))
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_superseded_retired_prefix_is_never_retained_merged_or_persisted(
    tmp_path, monkeypatch, controller,
):
    observed, final = _long_chain(address_all=True, retired_start=True)
    router = MagicMock()
    router.send.side_effect = ([
        _decision("answer"),
    ] if controller else []) + [LLMResponse("An explanation.", "chat", "nim")]
    monkeypatch.setattr(InternetService, "prepare", lambda *_args, **_kwargs: InternetResult())
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [final], observed=observed, router=router, controller=controller,
    )
    try:
        assert router.send.call_count == (2 if controller else 1)
        assert all(not ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list)
        assert all("retired secret marker" not in message.content
                   for call in router.send.call_args_list for message in call.args[0])
        assert all("/local" not in turn.content.casefold() for turn in turns)
        assert store.count_memories() == 0
    finally:
        store.close()


def test_retired_transcript_does_not_establish_an_active_ambient_conversation(
    tmp_path, monkeypatch,
):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"other","confidence":0.99}', "gate", "nim",
    )
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Could you explain something else?", {"supersedes_cancel_event": Event()})],
        observed=[("Ciel, /local retired secret marker", {
            "superseded": True, "cancel_event": canceled,
        })],
        router=router, controller=False,
    )
    try:
        router.send.assert_not_called()
        assert turns == []
        assert store.count_memories() == 0
    finally:
        store.close()
