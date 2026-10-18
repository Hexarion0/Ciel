"""Bounded pending speech must not discard an earlier explicit private route."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.internet import InternetService
from ciel.memory.manager import ConversationManager
from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _decision


def _long_private_chain(*, address_all=False, empty_middle=False):
    """Return linked clips whose original route falls outside the five-fragment tail."""
    texts = [
        "Ciel, /local explain this setting.", "Include an example.", "Use simple words.",
        "Keep it brief.", "Mention drawbacks.", "Cover alternatives.",
        "Finish with a recommendation.",
    ]
    if empty_middle:
        texts.insert(2, "")
    if address_all:
        texts = [texts[0], *["Ciel, " + text if text else "" for text in texts[1:-1]], texts[-1]]
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
@pytest.mark.parametrize("empty_middle", [False, True])
def test_private_route_survives_fragment_cap_and_persisted_history(
    tmp_path, monkeypatch, controller, empty_middle,
):
    observed, final = _long_private_chain(address_all=not controller, empty_middle=empty_middle)
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer") if controller else LLMResponse(
            '{"listener":"assistant","confidence":0.99}', "gate", "nim",
        ),
        LLMResponse("An explanation.", "chat", "nim"),
    ]
    prepare, lookup = MagicMock(), MagicMock()
    monkeypatch.setattr(InternetService, "prepare", prepare)
    monkeypatch.setattr(InternetService, "lookup", lookup)
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [final], observed=observed, router=router, controller=controller,
    )
    try:
        assert router.send.call_count == 2
        assert all(ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list)
        assert any(turn.content == "/local explain this setting." for turn in turns)
        restored = ConversationManager(store=store, session_id=turns[0].session_id)
        assert ModelRouter.requires_local(restored.get_gate_messages(limit=1))
        assert ModelRouter.requires_local(restored.get_active_messages("Continue normally."))
        assert store.count_memories() == 0
        prepare.assert_not_called()
        lookup.assert_not_called()
        # A new manager uses persisted route policy even after the original
        # fragment leaves the recent window and moves into a summary.
        restored = ConversationManager(
            store=store, session_id=turns[0].session_id, max_turns=2,
            auto_summarize_threshold=2,
        )
        summary_provider = MagicMock()
        summary_provider.name = "ollama"
        summary_provider.send.return_value = LLMResponse("A private explanation.", "chat", "ollama")
        assert restored.summarize_older_turns(summary_provider) == "A private explanation."
        assert ModelRouter.requires_local(summary_provider.send.call_args.args[0])
        assert ModelRouter.requires_local(restored.get_gate_messages(limit=1))
    finally:
        store.close()


def test_contextual_reply_and_web_policy_inherit_the_uncapped_private_route(tmp_path, monkeypatch):
    observed, final = _long_private_chain()
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer"), LLMResponse("The sample took 18 days.", "chat", "nim"),
        _decision("clarify", speech="Which recommendation do you mean?"),
        LLMResponse("A brief recommendation.", "chat", "nim"),
    ]
    _, _, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Describe the sample.", {"input_source": "keyboard"}), final],
        observed_after={0: observed}, router=router, controller=True,
    )
    try:
        assert router.send.call_count == 4
        assert all(ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list[2:])
    finally:
        store.close()

    lookup = MagicMock()
    monkeypatch.setattr(InternetService, "lookup", lookup)
    router = MagicMock()
    router.send.return_value = _decision("search", query="sample alternatives")
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Search for more alternatives.", {"input_source": "keyboard"})],
        router=router, controller=True,
    )
    try:
        assert ModelRouter.requires_local(router.send.call_args.args[0])
        lookup.assert_not_called()
        assert "private" in turns[-1].content
    finally:
        store.close()


def test_unaccepted_long_private_chain_cannot_trigger_a_web_lookup(tmp_path, monkeypatch):
    observed, final = _long_private_chain()
    router = MagicMock()
    router.send.return_value = _decision("search", query="sample alternatives")
    lookup = MagicMock()
    monkeypatch.setattr(InternetService, "lookup", lookup)
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [final], observed=observed, router=router, controller=True,
    )
    try:
        router.send.assert_called_once()
        assert ModelRouter.requires_local(router.send.call_args.args[0])
        lookup.assert_not_called()
        assert "private" in turns[-1].content
    finally:
        store.close()


@pytest.mark.parametrize("replacement", ["typed", "unlinked"])
def test_replacement_does_not_persist_or_inherit_abandoned_private_fragments(
    tmp_path, monkeypatch, replacement,
):
    observed, _final = _long_private_chain(empty_middle=True)
    info = {"cancel_event": Event(), "supersedes_cancel_event": Event()}
    if replacement == "typed":
        info["input_source"] = "keyboard"
    router = MagicMock()
    router.send.side_effect = [_decision("answer"), LLMResponse("A new answer.", "chat", "nim")]
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [("Ciel, describe another sample.", info)],
        observed=observed, router=router, controller=True,
    )
    try:
        assert all(not ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list)
        assert not any("/local" in turn.content for turn in turns)
        assert store.count_memories() == 0
    finally:
        store.close()


@pytest.mark.parametrize("address_final", [False, True])
def test_rejected_private_chain_does_not_revive_or_persist_on_later_speech(
    tmp_path, monkeypatch, address_final,
):
    observed, final = _long_private_chain()
    if address_final:
        final = ("Ciel, " + final[0], final[1])
    rejected = final[1]["cancel_event"]
    router = MagicMock()
    router.send.side_effect = [
        _decision("silent", intent="quiet"),
        _decision("answer"), LLMResponse("A new answer.", "chat", "nim"),
    ]

    def utterances():
        yield final
        rejected.set()
        yield "Ciel, describe another sample.", {
            "cancel_event": Event(), "supersedes_cancel_event": rejected,
        }

    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances(), observed=observed, router=router, controller=True,
    )
    try:
        assert router.send.call_count == 3
        assert ModelRouter.requires_local(router.send.call_args_list[0].args[0])
        assert all(not ModelRouter.requires_local(call.args[0])
                   for call in router.send.call_args_list[1:])
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "describe another sample.",
        ]
        assert store.count_memories() == 0
    finally:
        store.close()


def test_abandoned_private_ram_transcript_still_keeps_legacy_gate_local(tmp_path, monkeypatch):
    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.send.return_value = LLMResponse(
        '{"listener":"other","confidence":0.99}', "gate", "nim",
    )
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Could you explain something else?", {"supersedes_cancel_event": Event()})],
        observed=[("Ciel, /local explain this setting.", {
            "superseded": True, "cancel_event": canceled,
        })],
        router=router, controller=False,
    )
    try:
        router.send.assert_called_once()
        assert ModelRouter.requires_local(router.send.call_args.args[0])
        assert turns == []
        assert store.count_memories() == 0
    finally:
        store.close()
