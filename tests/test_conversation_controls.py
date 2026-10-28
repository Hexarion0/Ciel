"""Spoken cancellation is application-owned before either listener/model path."""

from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import InternetConfig
from ciel.controller import conversation_control, is_explicit_web_request
from ciel.internet import InternetResult
from ciel.providers.base import LLMResponse, ToolCall
from tests.test_ambient import run_callbacks


def _services(monkeypatch):
    service = MagicMock()
    service.config = InternetConfig()
    service.prepare.return_value = InternetResult()
    service.lookup.return_value = InternetResult(attempted=True, reply="A fresh lookup ran.")
    monkeypatch.setattr("ciel.internet.InternetService", lambda *_args, **_kwargs: service)
    router = MagicMock()

    def respond(messages, **kwargs):
        assert kwargs.get("purpose") != "controller"
        query = next(message.content for message in reversed(messages) if message.role == "user")
        if "tools" in kwargs and query == "Search anti-cheat program.":
            assert "web_search" in {tool["function"]["name"] for tool in kwargs["tools"]}
            return LLMResponse("", "test", "test", tool_calls=(
                ToolCall("anti_cheat_search", "web_search", '{"query":"anti-cheat program"}'),
            ))
        return LLMResponse("Let's think it through together.", "test", "test")

    router.send.side_effect = respond
    return service, router


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("command,reply", [
    ("Wait.", "Take your time."),
    ("Don't.", "Got it, quiet now."),
    ("Do not search.", "Okay, I won't search."),
])
def test_active_control_bypasses_models_and_all_internet_work(
    tmp_path, monkeypatch, controller, command, reply,
):
    internet, router = _services(monkeypatch)
    _, tts, _, store, _ = run_callbacks(tmp_path, monkeypatch, [
        ("Ciel, what time is it?", {}), (command, {}),
    ], router=router, controller=controller)
    try:
        router.send.assert_not_called()
        internet.prepare.assert_not_called()
        internet.lookup.assert_not_called()
        assert tts.speak.call_args.args[0] == reply
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_no_search_blocks_model_chosen_lookup_until_fresh_user_lookup(
    tmp_path, monkeypatch, controller,
):
    internet, router = _services(monkeypatch)
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Do not search.", {"input_source": "keyboard"}),
        ("Should I try it?", {"input_source": "keyboard"}),
        ("Search anti-cheat program.", {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        retrieval = internet.lookup if controller else internet.prepare
        retrieval.assert_called_once()
        assert "anti-cheat" in retrieval.call_args.args[0]
        assert "Let's think it through together." in [turn.content for turn in turns]
        assert any("paused online lookups" in call.args[0][0].content
                   for call in router.send.call_args_list)
        if controller:
            assert "web_search" not in {
                tool["function"]["name"] for tool in router.send.call_args_list[0].kwargs["tools"]
            }
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_superseded_cancellation_is_not_merged_back_into_an_old_request(
    tmp_path, monkeypatch, controller,
):
    internet, router = _services(monkeypatch)
    canceled = Event()
    canceled.set()
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Ciel, what time is it?", {}),
        ("What do you think?", {"input_source": "keyboard"}),
    ], router=router, controller=controller, observed_after={0: [
        ("Do not search.", {"superseded": True, "cancel_event": canceled}),
    ]})
    try:
        internet.lookup.assert_not_called()
        internet.prepare.assert_not_called()
        assert not any("Do not search.\nWhat" in turn.content for turn in turns)
    finally:
        store.close()


@pytest.mark.parametrize("text", [
    '"Do not search."', "She said don't search.", "Don't forget my favorite game.",
    "Don't search engines keep old pages?", "Dad, do not search.",
])
def test_reported_negation_and_other_commands_are_not_conversation_controls(text):
    assert conversation_control(text) is None
    assert not is_explicit_web_request(text)


@pytest.mark.parametrize("text", [
    "Can u search the news in Mongolia?", "Could you look that up?", "Look it up, please.",
])
def test_affirmative_lookups_include_casual_wording(text):
    assert is_explicit_web_request(text)


@pytest.mark.parametrize("controller", [False, True])
@pytest.mark.parametrize("text", [
    "Don't search for anti-cheat programs, instead explain the tradeoffs.",
    "Don't search, just explain it from what you know.",
    "Do not search; tell me what you think.",
])
def test_do_not_search_instead_explain_keeps_new_question_and_blocks_network(
    tmp_path, monkeypatch, controller, text,
):
    internet, router = _services(monkeypatch)
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (text, {"input_source": "keyboard"}),
    ], router=router, controller=controller)
    try:
        internet.lookup.assert_not_called()
        internet.prepare.assert_not_called()
        assert router.send.call_count == 1
        messages = router.send.call_args.args[0]
        current_user = next(message for message in reversed(messages) if message.role == "user")
        assert current_user.content == text
        assert turns[-1].content == "Let's think it through together."
    finally:
        store.close()


@pytest.mark.parametrize("controller", [False, True])
def test_unaddressed_cancellation_outside_conversation_is_not_accepted(
    tmp_path, monkeypatch, controller,
):
    internet, router = _services(monkeypatch)
    _, tts, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Do not search.", {}),
    ], router=router, controller=controller)
    try:
        internet.lookup.assert_not_called()
        internet.prepare.assert_not_called()
        tts.speak.assert_not_called()
        assert not turns
    finally:
        store.close()
