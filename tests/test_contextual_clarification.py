"""Only valid model clarifications can use the existing conversational response stage."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import ControllerConfig
from ciel.controller import AIController, ControllerDecision
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk
from ciel.providers.router import ModelRouter
from tests.test_ambient import run_callbacks


def _decision(action="clarify", **fields) -> LLMResponse:
    """Make a model response independently of application-only decision metadata."""
    return LLMResponse(json.dumps({
        "intent": "unclear" if action == "clarify" else "request",
        "action": action, "confidence": 0.9, "speech": "Which number do you mean?",
        **fields,
    }), "test-controller", "test")


def test_context_resolution_flag_is_private_readonly_and_cannot_be_provider_supplied():
    decision = ControllerDecision.model_validate_json(json.dumps({
        "action": "clarify", "confidence": 0.9,
        "resolve_with_context": True, "_resolve_with_context": True,
    }))

    assert decision.resolve_with_context is False
    assert "resolve_with_context" not in decision.model_dump()
    assert "_resolve_with_context" not in decision.model_dump()
    assert "resolve_with_context" not in decision.model_json_schema()["properties"]
    with pytest.raises((AttributeError, ValueError)):
        decision.resolve_with_context = True


@pytest.mark.parametrize("active,assistant,confidence,pending,tool_result,expected", [
    (True, True, 0.9, False, None, True),
    (True, True, 0.5, False, None, True),
    (False, True, 0.9, False, None, False),
    (True, False, 0.9, False, None, False),
    (True, True, 0.49, False, None, False),
    (True, True, 0.9, True, None, False),
    (True, True, 0.9, False, {}, False),
])
def test_only_confident_active_model_clarifications_with_dialogue_are_delegated(
    active, assistant, confidence, pending, tool_result, expected,
):
    router = MagicMock()
    router.send.return_value = _decision(confidence=confidence)
    messages = [ChatMessage("user", "How long did the sample prototype take?")]
    if assistant:
        messages.append(ChatMessage("assistant", "It took 18 days."))

    choice = AIController(router, ControllerConfig()).decide(
        "That's a big number.", messages, addressed=True, active=active,
        internet_enabled=True, pending_local=pending, tool_result=tool_result,
    )

    assert choice.action == "clarify"
    assert choice.resolve_with_context is expected


@pytest.mark.parametrize("text,response,assistant,options", [
    ("Yes, go ahead.", _decision(), "I can search again or explain it.", {}),
    ("What's the weather tomorrow?", _decision("search", query="weather tomorrow"),
     "What would you like to check?", {}),
    ("Search the sample release.", _decision("search", query="sample release", confidence=0.8),
     "The sample release is unverified.", {}),
    ("Search the sample release.", _decision("search", query="sample release"),
     "The sample release is unverified.", {"internet_enabled": False}),
    ("What number was that?", TimeoutError("synthetic timeout"), "It took 18 days.", {}),
    ("What number was that?", LLMResponse("invalid JSON", "test", "test"),
     "It took 18 days.", {}),
])
def test_application_or_failure_clarifications_never_gain_context_resolution(
    text, response, assistant, options,
):
    router = MagicMock()
    if isinstance(response, Exception):
        router.send.side_effect = response
    else:
        router.send.return_value = response
    choice = AIController(router, ControllerConfig()).decide(
        text, [ChatMessage("assistant", assistant)], addressed=True, active=True,
        **{"internet_enabled": True, **options},
    )

    assert choice.action == "clarify"
    assert choice.resolve_with_context is False


@pytest.mark.parametrize("rechecked,expected", [
    (_decision(), True),
    (LLMResponse("invalid JSON", "test", "test"), False),
    (TimeoutError("synthetic recheck timeout"), False),
    (_decision("silent", intent="other_listener"), False),
])
def test_only_valid_model_recheck_clarification_can_use_context(rechecked, expected):
    """A second model decision has the same context path and failure guards as the first."""
    router = MagicMock()
    router.send.side_effect = [
        _decision("silent", intent="other_listener"), rechecked,
    ]

    choice = AIController(router, ControllerConfig()).decide(
        "That's a big number.", [ChatMessage("assistant", "It took 18 days.")],
        addressed=False, active=True, internet_enabled=True,
    )

    assert router.send.call_count == 2
    assert choice.action == "clarify"
    assert choice.resolve_with_context is expected


def test_strict_unaddressed_followup_cannot_use_contextual_resolution():
    router = MagicMock()
    router.send.return_value = _decision()

    choice = AIController(router, ControllerConfig()).decide(
        "That's a big number.", [ChatMessage("assistant", "It took 18 days.")],
        addressed=False, active=True, internet_enabled=True, strict_followups=True,
    )

    assert choice.action == "clarify"
    assert choice.resolve_with_context is False


def test_private_web_policy_clarification_cannot_be_delegated_to_chat():
    router = MagicMock()
    router.send.return_value = _decision("search", query="sample release")
    choice = AIController(router, ControllerConfig()).decide(
        "Search the sample release.", [ChatMessage("user", "/local Earlier discussion."),
                                        ChatMessage("assistant", "It remains unverified.")],
        addressed=True, active=True, internet_enabled=True,
    )

    assert choice.action == "clarify"
    assert "private" in choice.speech
    assert choice.resolve_with_context is False


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("private", [False, True])
def test_model_clarification_uses_full_history_without_any_tool_authority(
    tmp_path, monkeypatch, streaming, private,
):
    """A normal active clarification reaches chat in both playback and privacy modes."""
    from ciel.actions import ActionRegistry
    from ciel.internet import InternetService
    from ciel.memory.service import MemoryService

    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    memory_handle, action_handle, lookup = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(MemoryService, "handle", memory_handle)
    monkeypatch.setattr(ActionRegistry, "handle", action_handle)
    monkeypatch.setattr(InternetService, "lookup", lookup)
    question = ("/local " if private else "") + "How long did the sample prototype take?"
    original = "The prototype took 18 days."
    resolved = "Yes, just over two weeks of work."
    router, tts = MagicMock(), MagicMock()
    tts.supports_sentence_pipeline = False
    generations = iter([original, resolved])
    gate_requests, reply_requests = [], []

    def generate(messages):
        reply_requests.append(messages)
        return next(generations)

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            gate_requests.append(messages)
            return _decision("answer") if len(gate_requests) == 1 else _decision()
        return LLMResponse(generate(messages), "test-chat", "test")

    router.send.side_effect = send
    router.stream.side_effect = lambda messages, **_kwargs: iter([
        LLMStreamChunk(generate(messages), "test-chat", "test"),
    ])
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(question, {"input_source": "keyboard"}), ("That's a big number.", {})],
        router=router, tts=tts, controller=True, streaming=streaming,
        persona_file=persona_file,
    )
    try:
        assert len(gate_requests) == len(reply_requests) == 2
        assert [turn.content for turn in turns if turn.role == "assistant"] == [original, resolved]
        assert [(message.role, message.content) for message in reply_requests[-1][1:]] == [
            ("user", question), ("assistant", original), ("user", "That's a big number."),
        ]
        assert "Which number do you mean?" in reply_requests[-1][0].content
        assert all(ModelRouter.requires_local(messages) is private
                   for messages in [*gate_requests, *reply_requests])
        assert turns[-1].provider == "test"
        memory_handle.assert_not_called()
        action_handle.assert_not_called()
        lookup.assert_not_called()
        assert store.count_memories() == 0
    finally:
        store.close()


def test_canceling_contextual_answer_does_not_speak_or_save_stale_model_output(
    tmp_path, monkeypatch,
):
    """A clarification's delegated answer retains the original turn's cancellation token."""
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    canceled = Event()
    router = MagicMock()
    calls = []

    def send(messages, **kwargs):
        calls.append((messages, kwargs))
        if len(calls) == 1:
            return _decision("answer")
        if len(calls) == 2:
            return LLMResponse("The prototype took 18 days.", "test-chat", "test")
        if len(calls) == 3:
            return _decision()
        assert kwargs["cancel_event"] is canceled
        canceled.set()
        return LLMResponse("This canceled answer must not be saved.", "test-chat", "test")

    router.send.side_effect = send
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("How long did the sample prototype take?", {"input_source": "keyboard"}),
         ("That's a big number.", {"cancel_event": canceled})],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert len(calls) == 4
        assert [turn.content for turn in turns if turn.role == "assistant"] == [
            "The prototype took 18 days.",
        ]
        tts.speak.assert_called_once()
        assert store.count_memories() == 0
    finally:
        store.close()


def test_failed_contextual_answer_falls_back_without_claiming_a_result(tmp_path, monkeypatch):
    """A response-stage outage still leaves an honest, audible reply."""
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    router = MagicMock()
    router.send.side_effect = [
        _decision("answer"), LLMResponse("The prototype took 18 days.", "test-chat", "test"),
        _decision(), ConnectionError("synthetic response-stage outage"),
    ]
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("How long did the sample prototype take?", {"input_source": "keyboard"}),
         ("That's a big number.", {})],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert router.send.call_count == 4
        assert turns[-1].provider == "local"
        assert turns[-1].content == "Which number do you mean?"
        assert tts.speak.call_count == 2
        assert store.count_memories() == 0
    finally:
        store.close()
