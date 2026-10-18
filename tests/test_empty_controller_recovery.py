"""Controller recovery retains observer-only requests without duplicating accepted turns."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.providers.base import LLMResponse
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks


def _answer_decision() -> LLMResponse:
    return LLMResponse(json.dumps({
        "intent": "request", "action": "answer", "confidence": 0.99,
    }), "test-controller", "test")


@pytest.mark.parametrize("ambient", [True, False])
@pytest.mark.parametrize("private", [False, True])
def test_empty_controller_recovery_includes_observer_only_query_once(
    tmp_path, monkeypatch, ambient, private,
):
    canceled, replacement = Event(), Event()
    canceled.set()
    router = MagicMock()
    generation_messages = []
    query = ("/local " if private else "") + "explain this setting."

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            return _answer_decision()
        generation_messages.append(list(messages))
        return LLMResponse("Here is the explanation.", "test-chat", "test")

    router.send.side_effect = send
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("", {"cancel_event": replacement, "supersedes_cancel_event": canceled})],
        ambient=ambient, router=router, controller=True,
        observed=[("Ciel, " + query, {
            "superseded": True, "cancel_event": canceled,
        })],
    )
    try:
        assert len(generation_messages) == 1
        assert [message.content for message in generation_messages[0]
                if message.role == "user"] == [query]
        assert [(turn.role, turn.content) for turn in turns] == [
            ("user", query), ("assistant", "Here is the explanation."),
        ]
        if private:
            assert all(ModelRouter.requires_local(call.args[0])
                       for call in router.send.call_args_list)
        tts.speak.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("private", [False, True])
def test_empty_controller_recovery_does_not_duplicate_an_accepted_canceled_query(
    tmp_path, monkeypatch, private,
):
    canceled, replacement = Event(), Event()
    router = MagicMock()
    generation_messages = []
    query = ("/local " if private else "") + "explain this setting."

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            return _answer_decision()
        generation_messages.append(list(messages))
        if len(generation_messages) == 1:
            canceled.set()
            return LLMResponse("Discard this reply.", "test-chat", "test")
        return LLMResponse("Here is the explanation.", "test-chat", "test")

    router.send.side_effect = send
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, " + query, {"cancel_event": canceled}),
         ("", {"cancel_event": replacement, "supersedes_cancel_event": canceled})],
        router=router, controller=True,
    )
    try:
        assert len(generation_messages) == 2
        for messages in generation_messages:
            assert [message.content for message in messages if message.role == "user"] == [
                query,
            ]
        assert [(turn.role, turn.content) for turn in turns] == [
            ("user", query), ("assistant", "Here is the explanation."),
        ]
        if private:
            assert all(ModelRouter.requires_local(call.args[0])
                       for call in router.send.call_args_list)
        tts.speak.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("final", ["", "Keep the answer short."])
def test_accepted_canceled_request_and_observed_addition_recover_without_duplicate_history(
    tmp_path, monkeypatch, private, final,
):
    initial_event, observer_event, final_event = Event(), Event(), Event()
    observer_event.set()
    query = ("/local " if private else "") + "explain this setting."
    addition = "And include an example."
    generation_messages = []
    router = MagicMock()

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            return _answer_decision()
        generation_messages.append(list(messages))
        if len(generation_messages) == 1:
            initial_event.set()
            return LLMResponse("Discard this reply.", "test-chat", "test")
        return LLMResponse("Here is the complete explanation.", "test-chat", "test")

    router.send.side_effect = send
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, " + query, {"cancel_event": initial_event}),
         (final, {"cancel_event": final_event, "supersedes_cancel_event": observer_event})],
        observed_after={0: [(addition, {
            "superseded": True, "cancel_event": observer_event,
            "supersedes_cancel_event": initial_event,
        })]},
        router=router, controller=True,
    )
    try:
        expected = [query, addition, *([final] if final else [])]
        assert len(generation_messages) == 2
        assert [message.content for message in generation_messages[-1]
                if message.role == "user"] == expected
        assert [(turn.role, turn.content) for turn in turns] == [
            *(("user", fragment) for fragment in expected),
            ("assistant", "Here is the complete explanation."),
        ]
        if private:
            assert all(ModelRouter.requires_local(call.args[0])
                       for call in router.send.call_args_list)
        tts.speak.assert_called_once()
    finally:
        store.close()


def test_empty_observer_does_not_reset_the_already_started_playback_guard(tmp_path, monkeypatch):
    initial_event, observer_event, final_event = Event(), Event(), Event()
    observer_event.set()
    router = MagicMock()
    generation_messages = []

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            return _answer_decision()
        generation_messages.append(list(messages))
        return LLMResponse("The explanation has started.", "test-chat", "test")

    def speak(_text, **controls):
        controls["on_start"]()
        initial_event.set()
        return False

    router.send.side_effect = send
    tts = MagicMock()
    tts.speak.side_effect = speak
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("Ciel, explain this setting.", {"cancel_event": initial_event}),
         ("", {"cancel_event": final_event, "supersedes_cancel_event": observer_event})],
        observed_after={0: [("", {
            "superseded": True, "cancel_event": observer_event,
            "supersedes_cancel_event": initial_event,
        })]},
        router=router, controller=True, tts=tts,
    )
    try:
        assert len(generation_messages) == 1
        assert [(turn.role, turn.content) for turn in turns] == [
            ("user", "explain this setting."),
            ("assistant", "The explanation has started."),
        ]
        tts.speak.assert_called_once()
    finally:
        store.close()
