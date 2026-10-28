"""The strict listener policy remains an admission gate for the native main flow."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks
from tests.test_controller_runtime import _internet, _seed_legacy_private_summary


def _router(listener="uncertain", confidence=0.7):
    router = MagicMock()

    def send(_messages, **options):
        if options.get("purpose") == "speech_gate":
            return LLMResponse(json.dumps({
                "listener": listener, "confidence": confidence, "interpretation": "",
            }), "gate", "nim")
        return LLMResponse("A spoken response.", "main", "nim")

    router.send.side_effect = send
    return router


@pytest.mark.parametrize("listener,confidence", [("uncertain", 0.7), ("other", 0.99)])
def test_strict_native_flow_rejects_unadmitted_followup_before_history_or_tools(
    tmp_path, monkeypatch, listener, confidence,
):
    monkeypatch.setenv("AMBIENT_FOLLOWUP_POLICY", "strict")
    internet = _internet(monkeypatch, [])
    router = _router(listener, confidence)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            ("Hey Ciel.", {"wake_verified": True}),
            ("She said you should order the blue one.", {}),
        ], router=router, controller=True,
    )
    try:
        assert [call.kwargs.get("purpose", "conversation")
                for call in router.send.call_args_list] == ["conversation", "speech_gate"]
        assert [turn.content for turn in turns if turn.role == "user"] == ["Hey Ciel."]
        tts.speak.assert_called_once()
        internet.lookup.assert_not_called()
    finally:
        store.close()


def test_strict_native_flow_admits_confident_followup_with_private_context(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AMBIENT_FOLLOWUP_POLICY", "strict")
    router = _router("assistant", 0.99)
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            ("Hey Ciel.", {"wake_verified": True}), ("Could you explain that?", {}),
        ], router=router, controller=True,
        initialize_conversation=_seed_legacy_private_summary,
    )
    try:
        assert [call.kwargs.get("purpose", "conversation")
                for call in router.send.call_args_list] == [
                    "conversation", "speech_gate", "conversation",
                ]
        gate_messages = router.send.call_args_list[1].args[0]
        assert any(message.local_only for message in gate_messages)
        assert [turn.content for turn in turns if turn.role == "user"] == [
            "Hey Ciel.", "Could you explain that?",
        ]
        assert tts.speak.call_count == 2
    finally:
        store.close()


@pytest.mark.parametrize("info", [{"wake_verified": True}, {"input_source": "keyboard"}])
def test_strict_explicit_input_skips_inferred_admission(tmp_path, monkeypatch, info):
    monkeypatch.setenv("AMBIENT_FOLLOWUP_POLICY", "strict")
    router = _router()
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [("Hey Ciel.", info)], router=router, controller=True,
    )
    try:
        router.send.assert_called_once()
        assert router.send.call_args.kwargs.get("purpose", "conversation") == "conversation"
    finally:
        store.close()


def test_cancellation_during_strict_admission_prevents_persistence_and_reply(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("AMBIENT_FOLLOWUP_POLICY", "strict")
    canceled = Event()
    router = _router("assistant", 0.99)
    send = router.send.side_effect

    def cancel_gate(messages, **options):
        if options.get("purpose") == "speech_gate":
            canceled.set()
        return send(messages, **options)

    router.send.side_effect = cancel_gate
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            ("Hey Ciel.", {"wake_verified": True}),
            ("Could you explain that?", {"cancel_event": canceled}),
        ], router=router, controller=True,
    )
    try:
        assert canceled.is_set()
        assert [turn.content for turn in turns if turn.role == "user"] == ["Hey Ciel."]
        assert router.send.call_count == 2
        tts.speak.assert_called_once()
    finally:
        store.close()
