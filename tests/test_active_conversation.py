"""Active ambient dialogue follows turn-taking without repeating a wake phrase."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.audio.ambient import AmbientConversation
from ciel.providers.base import ChatMessage, LLMResponse


def judgment_router(result):
    """Return a provider stub for a listener classification, including invalid data."""
    router = MagicMock()
    content = result if isinstance(result, str) else json.dumps(result)
    router.send.return_value = LLMResponse(content, "test", "test")
    return router


@pytest.fixture
def active_conversation(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.ambient.time.monotonic", lambda: clock[0])
    ambient = AmbientConversation()
    ambient.record_addressed("user", "Ciel, what are you good at?")
    ambient.record_addressed("assistant", "What's something you'd like to work on today?")
    return ambient, clock


@pytest.mark.parametrize("listener, confidence", [
    ("assistant", 0), ("assistant", 0.6), ("assistant", 1),
    ("uncertain", 0), ("uncertain", 0.6), ("uncertain", 1),
    ("other", 0), ("other", 0.6), ("other", 0.849),
])
def test_active_dialogue_accepts_next_turn_without_another_address(
    active_conversation, listener, confidence,
):
    ambient, clock = active_conversation
    clock[0] += 6
    router = judgment_router({"listener": listener, "confidence": confidence})
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert decision.addressed and not decision.explicit
    assert decision.reason == "active_conversation"
    assert decision.confidence == confidence


@pytest.mark.parametrize("transcript", [
    "I've got a better idea.", "Actually, tell me something interesting.",
    "What would you add next?", "I don't know. You choose.",
])
def test_active_policy_handles_new_topics_and_unseen_wording(active_conversation, transcript):
    ambient, _clock = active_conversation
    router = judgment_router({"listener": "uncertain", "confidence": 0.6})
    decision = ambient.decide(transcript, router, [])
    assert decision.addressed and not decision.explicit
    assert decision.reason == "active_conversation"


@pytest.mark.parametrize("confidence", [0.85, 1])
def test_clear_change_of_listener_ends_active_dialogue(active_conversation, confidence):
    ambient, _clock = active_conversation
    router = judgment_router({"listener": "other", "confidence": confidence})
    decision = ambient.decide("Can you hand me the remote?", router, [])
    assert not decision.addressed and decision.reason == "other_listener"
    assert ambient.deadline == 0
    assert not ambient.interaction
    assert "remote" in ambient.context_note()
    router.reset_mock()
    assert not ambient.decide("And what about tomorrow?", router, []).addressed
    router.send.assert_not_called()


def test_family_address_remains_silent_after_assistant_question(active_conversation):
    ambient, _clock = active_conversation
    router = MagicMock()
    decision = ambient.decide("Mom, what do you want to talk about?", router, [])
    assert not decision.addressed and decision.reason == "addressed_to_someone_else"
    assert ambient.deadline == 0
    assert not ambient.interaction
    router.send.assert_not_called()


@pytest.mark.parametrize("policy, addressed", [("conversation", True), ("strict", False)])
def test_legacy_uncertain_listener_result_respects_selected_policy(policy, addressed):
    ambient = AmbientConversation(followup_policy=policy)
    ambient.record_addressed("user", "Ciel, let's chat.")
    ambient.record_addressed("assistant", "What would you like to talk about?")
    router = judgment_router({"addressed": False, "confidence": 0.6})
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert decision.addressed is addressed
    assert not decision.explicit
    assert decision.reason == ("active_conversation" if addressed else "uncertain_intent")


@pytest.mark.parametrize("listener, confidence", [
    ("uncertain", 0.99), ("assistant", 0.84), ("other", 0.6),
])
def test_strict_policy_keeps_uncertainty_silent(listener, confidence):
    ambient = AmbientConversation(followup_policy="strict")
    ambient.record_addressed("user", "Ciel, let's chat.")
    ambient.record_addressed("assistant", "What would you like to talk about?")
    router = judgment_router({"listener": listener, "confidence": confidence})
    assert not ambient.decide("What do you want to talk about?", router, []).addressed


def test_strict_policy_accepts_clear_followup_without_memory_permission():
    ambient = AmbientConversation(followup_policy="strict")
    ambient.record_addressed("user", "Ciel, let's chat.")
    ambient.record_addressed("assistant", "What would you like to talk about?")
    router = judgment_router({"listener": "assistant", "confidence": 0.95})
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert decision.addressed and not decision.explicit
    assert decision.reason == "clear_followup"


@pytest.mark.parametrize("state", ["fresh", "user_only", "expired", "reset"])
def test_active_default_requires_recent_assistant_turn(active_conversation, state):
    ambient, clock = active_conversation
    if state == "fresh":
        ambient = AmbientConversation()
    elif state == "user_only":
        ambient = AmbientConversation()
        ambient.record_addressed("user", "Ciel, let's chat.")
    elif state == "expired":
        clock[0] = ambient.deadline + 1
    else:
        ambient.reset()
    router = judgment_router({"listener": "uncertain", "confidence": 0.6})
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert not decision.addressed
    if state == "user_only":
        payload = json.loads(router.send.call_args.args[0][1].content)
        assert payload["active_conversation"] is False
    else:
        assert decision.reason == "outside_followup_window"
        router.send.assert_not_called()


def test_speech_onset_keeps_long_followup_active(active_conversation):
    ambient, clock = active_conversation
    began = clock[0] + 6
    clock[0] = ambient.deadline + 15
    router = judgment_router({"listener": "uncertain", "confidence": 0.6})
    decision = ambient.decide(
        "What do you want to talk about? I have a few ideas.", router, [], started_at=began,
    )
    assert decision.addressed and decision.reason == "active_conversation"
    assert not decision.explicit


def test_new_address_after_idle_cannot_reuse_stale_assistant_turn(active_conversation):
    ambient, clock = active_conversation
    clock[0] = 130.0
    router = judgment_router({"listener": "uncertain", "confidence": 0.6})
    direct_text = "Ciel, explain a new feature."
    assert ambient.decide(direct_text, router, []).explicit
    ambient.record_addressed("user", direct_text)
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert not decision.addressed
    payload = json.loads(router.send.call_args.args[0][1].content)
    assert payload["active_conversation"] is False
    assert payload["recent_dialogue"] == [{"role": "user", "content": direct_text}]


@pytest.mark.parametrize("result", [
    "not JSON", "[]", "null", {},
    {"listener": "assistant", "confidence": None},
    {"listener": "assistant", "confidence": True},
    {"listener": "assistant", "confidence": "0.99"},
    {"listener": "assistant", "confidence": -0.1},
    {"listener": "assistant", "confidence": 1.1},
    {"listener": "assistant", "confidence": float("nan")},
    {"listener": "assistant", "confidence": float("inf")},
    {"listener": "unknown", "confidence": 0.99},
    {"listener": [], "confidence": 0.99},
    {"addressed": "false", "confidence": 0.6},
])
def test_invalid_judgment_cannot_open_active_reply(active_conversation, result):
    ambient, _clock = active_conversation
    deadline = ambient.deadline
    interaction = list(ambient.interaction)
    router = judgment_router(result)
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert not decision.addressed and decision.reason == "invalid_judgment"
    assert ambient.deadline == deadline
    assert list(ambient.interaction) == interaction


def test_provider_failure_stays_silent_without_ending_active_dialogue(active_conversation):
    ambient, _clock = active_conversation
    deadline = ambient.deadline
    router = MagicMock()
    router.send.side_effect = RuntimeError("offline")
    decision = ambient.decide("What do you want to talk about?", router, [])
    assert not decision.addressed and decision.reason == "judgment_failed"
    assert ambient.deadline == deadline


def test_gate_receives_actual_recent_reply_and_active_dialogue_state(active_conversation):
    ambient, _clock = active_conversation
    transcript = "What do you want to talk about?"
    router = judgment_router({"listener": "uncertain", "confidence": 0.6})
    stale_dialogue = [ChatMessage("assistant", "A stale unrelated answer.")]
    assert ambient.decide(transcript, router, stale_dialogue).addressed
    request = router.send.call_args
    payload = json.loads(request.args[0][1].content)
    assert payload["active_conversation"] is True
    assert payload["transcript"] == transcript
    assert payload["recent_dialogue"] == [
        {"role": "user", "content": "Ciel, what are you good at?"},
        {"role": "assistant", "content": "What's something you'd like to work on today?"},
    ]
    assert request.kwargs["purpose"] == "speech_gate"


def test_inferred_active_turn_never_becomes_explicit_memory_permission(active_conversation):
    ambient, _clock = active_conversation
    router = judgment_router({"listener": "assistant", "confidence": 1})
    decision = ambient.decide("Remember my name is Alex.", router, [])
    assert decision.addressed and not decision.explicit
