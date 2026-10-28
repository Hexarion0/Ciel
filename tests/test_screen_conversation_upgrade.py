"""Screen tasks survive speech repairs and execute another look instead of stale prose."""

import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from ciel.providers.base import LLMResponse, ToolCall
from ciel.providers.router import ModelRouter
from ciel.screen.service import ScreenResult
from tests.test_controller_runtime import _seed_legacy_private_summary
from tests.test_screen_integration import chat as chat
from tests.test_screen_integration import (
    install_screen,
    run_callbacks,
    stub_runtime_internet,
)

ORIGINAL = "What anime am I watching?"
CLARIFICATION = "Did you mean take another look at the anime on your screen?"
SEQUENCE = [ORIGINAL, "Kitten chick again.", "Can you check again? I meant.",
            "Check again.", "Again."]


@pytest.fixture
def screen_intent(monkeypatch):
    """Retain the optional legacy speech classifier for controller-disabled callbacks."""
    def classify(_router, query, *_args, **_kwargs):
        if "Kitten chick again" in query:
            return SimpleNamespace(action="clarify", confidence=0.99, question=CLARIFICATION)
        if "Tell me a joke" in query:
            return SimpleNamespace(action="new_topic", confidence=0.99, question="")
        return SimpleNamespace(action="refresh", confidence=0.99, question="")

    classifier = Mock(side_effect=classify)
    monkeypatch.setattr("ciel.screen.service.interpret_screen_intent", classifier, raising=False)
    return classifier


def visual_result(number):
    """Distinct observations make stale text replay visible without desktop capture."""
    return ScreenResult(f"Fresh screen observation {number}.", "selected-image-provider",
                        "actual-vision-model", "synthetic-image")


def ordinary_router():
    """Keep any accidental general reply deterministic and completely offline."""
    router = MagicMock()

    def send(messages, **kwargs):
        purpose = kwargs.get("purpose")
        if purpose == "speech_gate":
            text = json.dumps({"listener": "assistant", "confidence": 0.99,
                               "interpretation": ""})
        else:
            assert purpose != "controller"
            query = next(message.content for message in reversed(messages)
                         if message.role == "user")
            if "Kitten chick again" in query:
                return LLMResponse(CLARIFICATION, "text-model", "text-provider")
            if "reassess that scene" in query:
                assert "check_screen" in {tool["function"]["name"] for tool in kwargs["tools"]}
                return LLMResponse("", "text-model", "text-provider", tool_calls=(
                    ToolCall("fresh_scene", "check_screen", "{}"),
                ))
            text = "I already told you about that screen."
        return LLMResponse(text, "text-model", "text-provider")

    router.send.side_effect = send
    return router


def assert_original_visual_focus(screen, count):
    requests = [call.args[0] for call in screen.answer.call_args_list]
    assert len(requests) == count
    assert all(ORIGINAL in request.query and not request.previous for request in requests)


def test_chat_reported_speech_repair_sequence_keeps_task_and_obtains_new_observations(
    chat, screen_intent,
):
    session, screen = chat
    session.router.send.side_effect = ordinary_router().send.side_effect
    screen.answer.side_effect = [visual_result(index) for index in range(1, 5)]

    replies = [session.reply(utterance) for utterance in SEQUENCE]

    assert replies == [visual_result(1).text, CLARIFICATION, *[
        visual_result(index).text for index in range(2, 5)
    ]]
    assert_original_visual_focus(screen, 4)
    session.router.send.assert_called_once()
    assert session.router.send.call_args.kwargs["purpose"] == "conversation"
    screen_intent.assert_not_called()
    session.internet.prepare.assert_not_called()


@pytest.mark.parametrize("controller", [False, True])
def test_desktop_reported_speech_repair_sequence_uses_fresh_visual_results(
    tmp_path, monkeypatch, screen_intent, controller,
):
    results = iter(visual_result(index) for index in range(1, 5))
    created = install_screen(monkeypatch, on_answer=lambda *_args, **_kwargs: next(results))
    internet = stub_runtime_internet(monkeypatch)
    router = ordinary_router()
    utterances = [(utterance, {"wake_verified": True} if index == 0 else {})
                  for index, utterance in enumerate(SEQUENCE)]
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances, router=router, controller=controller,
    )
    try:
        expected = [visual_result(1).text, CLARIFICATION, *[
            visual_result(index).text for index in range(2, 5)
        ]]
        assert [call.args[0] for call in tts.speak.call_args_list] == expected
        assert [turn.content for turn in turns if turn.role == "assistant"] == expected
        assert_original_visual_focus(created[0], 4)
        if controller:
            router.send.assert_called_once()
            assert router.send.call_args.kwargs["purpose"] == "conversation"
            screen_intent.assert_not_called()
        else:
            assert all(call.kwargs.get("purpose") == "speech_gate"
                       for call in router.send.call_args_list)
        internet.lookup.assert_not_called()
    finally:
        store.close()


def test_chat_pause_retains_screen_task_but_cancel_ends_it(chat, screen_intent):
    session, screen = chat
    session.reply(ORIGINAL)
    session.reply("Wait.")
    session.reply("Again.")
    assert_original_visual_focus(screen, 2)

    session.reply("Cancel.")
    session.reply("Again.")
    assert screen.answer.call_count == 2


def test_uncertain_chat_screen_followup_uses_one_native_main_request(chat, screen_intent):
    session, screen = chat
    session.router.send.side_effect = ordinary_router().send.side_effect
    session.reply(ORIGINAL)
    assert session.reply("Could you reassess that scene for me?") == screen.answer.return_value.text
    assert screen.answer.call_count == 2
    request = screen.answer.call_args.args[0]
    assert request.original_query == ORIGINAL
    assert "reassess that scene" in request.query and not request.previous
    session.router.send.assert_called_once()
    assert session.router.send.call_args.kwargs["purpose"] == "conversation"
    screen_intent.assert_not_called()


def test_uncertain_desktop_screen_followup_uses_one_native_main_request(
    tmp_path, monkeypatch, screen_intent,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    router = ordinary_router()
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"input_source": "keyboard"}),
            ("Could you reassess that scene for me?", {"input_source": "keyboard"}),
        ], router=router, controller=True,
    )
    try:
        screen = created[0]
        assert screen.answer.call_count == 2
        request = screen.answer.call_args.args[0]
        assert request.original_query == ORIGINAL
        assert "reassess that scene" in request.query and not request.previous
        router.send.assert_called_once()
        assert router.send.call_args.kwargs["purpose"] == "conversation"
        assert [turn.content for turn in turns if turn.role == "assistant"] == [
            screen.answer.return_value.text,
        ] * 2
        screen_intent.assert_not_called()
    finally:
        store.close()


@pytest.mark.parametrize("query", [
    "Can you actually see my screen?", "What did you mean by that?", "I liked that.",
    "Recheck your reasoning about this scene.", "Check your answer about this screen.",
])
def test_chat_native_tool_cannot_capture_for_discussion_or_acknowledgment(chat, query):
    session, screen = chat
    session.reply(ORIGINAL)
    session.router.send.side_effect = [
        LLMResponse("", "text-model", "text-provider", tool_calls=(
            ToolCall("unrequested_capture", "check_screen", "{}"),
        )),
        LLMResponse("We can discuss the earlier observation.", "text-model", "text-provider"),
    ]

    session.reply(query)

    screen.answer.assert_called_once()


@pytest.mark.parametrize("query", [
    "Can you actually see my screen?", "What did you mean by that?", "I liked that.",
    "Recheck your reasoning about this scene.", "Check your answer about this screen.",
])
def test_desktop_native_tool_cannot_capture_for_discussion_or_acknowledgment(
    tmp_path, monkeypatch, query,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    router = MagicMock()
    router.send.side_effect = [
        LLMResponse("", "text-model", "text-provider", tool_calls=(
            ToolCall("unrequested_capture", "check_screen", "{}"),
        )),
        LLMResponse("We can discuss the earlier observation.", "text-model", "text-provider"),
    ]
    _, _, _, store, _ = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"input_source": "keyboard"}),
            (query, {"input_source": "keyboard"}),
        ], router=router, controller=True,
    )
    try:
        created[0].answer.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_desktop_pause_retains_screen_task_but_cancel_ends_it(
    tmp_path, monkeypatch, screen_intent, controller,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    _, _, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"input_source": "keyboard"}),
            ("Wait.", {"input_source": "keyboard"}),
            ("Again.", {"input_source": "keyboard"}),
            ("Cancel.", {"input_source": "keyboard"}),
            ("Again.", {"input_source": "keyboard"}),
        ], router=ordinary_router(), controller=controller,
    )
    try:
        assert_original_visual_focus(created[0], 2)
    finally:
        store.close()


def test_chat_clear_new_topic_ends_implicit_screen_task(chat, screen_intent):
    session, screen = chat
    session.reply(ORIGINAL)
    session.reply("Tell me a joke about a penguin.")
    session.reply("Again.")
    screen.answer.assert_called_once()


@pytest.mark.parametrize("controller", [False, True])
def test_desktop_clear_new_topic_ends_implicit_screen_task(
    tmp_path, monkeypatch, screen_intent, controller,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    _, _, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"input_source": "keyboard"}),
            ("Tell me a joke about a penguin.", {"input_source": "keyboard"}),
            ("Again.", {"input_source": "keyboard"}),
        ], router=ordinary_router(), controller=controller,
    )
    try:
        created[0].answer.assert_called_once()
    finally:
        store.close()


@pytest.mark.parametrize("invalidation", ["cancel", "reset", "task_cancel"])
def test_invalidation_during_uncertain_chat_interpretation_cannot_start_another_capture(
    chat, screen_intent, invalidation,
):
    session, screen = chat
    session.reply(ORIGINAL)
    canceled = Event()

    def cancel_then_refresh(_messages, **kwargs):
        assert kwargs["purpose"] == "conversation"
        assert "check_screen" in {tool["function"]["name"] for tool in kwargs["tools"]}
        if invalidation == "cancel":
            canceled.set()
        elif invalidation == "reset":
            session.conversation.clear_session()
        else:
            screen.clear()
        return LLMResponse("", "text-model", "text-provider", tool_calls=(
            ToolCall("stale_scene", "check_screen", "{}"),
        ))

    session.router.send.side_effect = cancel_then_refresh
    assert session.reply("Could you reassess that scene for me?", canceled) == ""
    screen.answer.assert_called_once()
    screen_intent.assert_not_called()
    if invalidation == "reset":
        assert session.conversation.get_recent_turns() == []


@pytest.mark.parametrize("controller", [False, True])
def test_merged_screen_correction_preserves_private_routing(
    tmp_path, monkeypatch, screen_intent, controller,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    canceled = Event()
    canceled.set()
    _, _, _, store, _turns = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"wake_verified": True}),
            ("again? I meant.", {"cancel_event": Event(),
                                   "supersedes_cancel_event": canceled}),
        ], observed_after={0: [("Ciel, can you check", {
            "superseded": True, "cancel_event": canceled,
        })]}, router=ordinary_router(), controller=controller,
        initialize_conversation=_seed_legacy_private_summary,
    )
    try:
        assert_original_visual_focus(created[0], 2)
        assert all(ModelRouter.requires_local(call.args[1])
                   for call in created[0].answer.call_args_list)
        assert all(call.kwargs.get("local_only") or any(
            isinstance(arg, list) and ModelRouter.requires_local(arg) for arg in call.args
        ) for call in screen_intent.call_args_list)
    finally:
        store.close()
