"""Recent assistant replies survive real persistence and both conversational model stages.

Providers are mocked: these regressions verify context delivery and authorization, while live
synthetic evaluations separately check whether a hosted model uses that context well.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ciel.controller import _CONTEXT_MARKER
from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks


def _decision() -> LLMResponse:
    """Supply a conversational choice without mocking the runtime or memory manager."""
    return LLMResponse(json.dumps({
        "intent": "follow_up", "action": "answer", "confidence": 0.99,
    }), "test-controller", "test")


def _payload(messages) -> dict:
    """Read either supported compact controller request representation."""
    if _CONTEXT_MARKER in messages[0].content:
        return json.loads(messages[0].content.split(_CONTEXT_MARKER, 1)[1])
    return json.loads(messages[-1].content)


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("question,replies,followups", [
    (
        "How long did the sample prototype take?",
        ["The prototype took 18 days.", "Yes, just over two weeks.", "Right, 18 days of work."],
        ["That's a big number.", "The days."],
    ),
    (
        "Describe the sample badge.",
        ["The badge has a silver border.", "The silver border suits it.", "Yes, that border."],
        ["That looks nice.", "The border."],
    ),
    (
        "How long is the sample walk?",
        ["The walk takes 40 minutes.", "It's a fairly long walk.", "Yes, about 40 minutes."],
        ["That's longer than I expected.", "The duration."],
    ),
])
def test_recent_assistant_reply_reaches_followup_gate_and_generation(
    tmp_path, monkeypatch, typed, question, replies, followups,
):
    """Speech and typed reactions retain the exact words and surrounding dialogue."""
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    router = MagicMock()
    requests = [question, *followups]
    answers = iter(replies)
    gate_requests = []
    answer_requests = []

    def send(messages, **kwargs):
        if kwargs.get("purpose") == "controller":
            index = len(gate_requests)
            payload = _payload(messages)
            assert messages[-1].content == requests[index]
            assert payload["transcribed"] is not typed
            if index:
                previous = payload["recent_dialogue"]
                assert previous[-2]["role"] == "user"
                assert previous[-2]["text"] == requests[index - 1]
                assert previous[-1]["role"] == "assistant"
                assert previous[-1]["text"] == replies[index - 1]
                assert payload["active_conversation"] is True
                assert payload["direct_address"] is typed
            gate_requests.append(payload)
            return _decision()

        index = len(answer_requests)
        dialogue = [(message.role, message.content) for message in messages[1:]]
        expected = []
        for previous in range(index):
            expected.extend([
                ("user", requests[previous]), ("assistant", replies[previous]),
            ])
        assert dialogue == [*expected, ("user", requests[index])]
        answer_requests.append(dialogue)
        return LLMResponse(next(answers), "test-chat", "test")

    router.send.side_effect = send
    info = {"input_source": "keyboard"} if typed else {}
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(question, {**info, "wake_verified": not typed}),
         *((followup, dict(info)) for followup in followups)],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert len(gate_requests) == len(answer_requests) == 3
        assert [turn.content for turn in turns if turn.role == "user"] == requests
        assert [turn.content for turn in turns if turn.role == "assistant"] == replies
        assert tts.speak.call_count == 3
        assert store.count_memories() == 0
    finally:
        store.close()


@pytest.mark.parametrize("typed", [False, True])
def test_local_timeline_reply_is_replayed_to_a_short_conversational_reaction(
    tmp_path, monkeypatch, typed,
):
    """Application answers enter the same recent dialogue as generated replies."""
    from ciel.memory import read

    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    seed = MemoryStore(tmp_path / "memory.db")
    seed.upsert_fact("ciel:project_start_date", MemoryItem(
        content="The sample project began on 2026-03-01.",
        memory_type=MemoryType.PROJECT, source="user_explicit",
        metadata={"value": "2026-03-01"},
    ))
    seed.close()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 12, 4, 1, tzinfo=tz)

    monkeypatch.setattr(read, "datetime", Clock)
    router = MagicMock()
    router.send.side_effect = [_decision(), LLMResponse(
        "Yes, almost two weeks of work.", "test-chat", "test",
    )]
    info = {"input_source": "keyboard"} if typed else {}
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [("What development day is it?", {**info, "wake_verified": not typed}),
         ("That's quite a number.", info)],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert router.send.call_count == 2
        assert turns[1].provider == "local"
        assert "development day 12" in turns[1].content
        gate = _payload(router.send.call_args_list[0].args[0])
        assert gate["recent_dialogue"][-1]["text"] == turns[1].content
        reply_messages = router.send.call_args_list[1].args[0]
        assert [(message.role, message.content) for message in reply_messages[1:]] == [
            (turn.role, turn.content) for turn in turns[:-1]
        ]
        assert store.count_memories() == 1
        assert store.get_fact("ciel:project_start_date").metadata["value"] == "2026-03-01"
    finally:
        store.close()


@pytest.mark.parametrize("typed", [False, True])
def test_noun_only_reply_keeps_the_topic_from_before_a_controller_clarification(
    tmp_path, monkeypatch, typed,
):
    """An unnecessary clarification cannot erase the assistant answer it referred to."""
    persona_file = tmp_path / "persona.txt"
    persona_file.write_text("", encoding="utf-8")
    question = "How long did the sample prototype take?"
    original_reply = "The prototype took 18 days."
    clarification = "What number are you referring to?"
    router = MagicMock()
    router.send.side_effect = [
        _decision(), LLMResponse(original_reply, "test-chat", "test"),
        LLMResponse(json.dumps({
            "intent": "unclear", "action": "clarify", "confidence": 0.9,
            "speech": clarification,
        }), "test-controller", "test"),
        LLMResponse(clarification, "test-chat", "test"),
        _decision(), LLMResponse("Right, 18 days of work.", "test-chat", "test"),
    ]
    info = {"input_source": "keyboard"} if typed else {}
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(question, {**info, "wake_verified": not typed}),
         ("That's a big number.", dict(info)), ("The days.", dict(info))],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert router.send.call_count == 6
        assert turns[3].provider == "test" and turns[3].model == "test-chat"
        gate_request = router.send.call_args_list[4]
        assert gate_request.kwargs["purpose"] == "controller"
        assert gate_request.args[0][-1].content == "The days."
        gate = _payload(gate_request.args[0])
        assert [(item["role"], item["text"]) for item in gate["recent_dialogue"]] == [
            (turn.role, turn.content) for turn in turns[:4]
        ]
        assert gate["recent_dialogue"][1]["text"] == original_reply
        assert gate["recent_dialogue"][-1]["text"] == clarification
        response_messages = router.send.call_args_list[5].args[0]
        assert [(message.role, message.content) for message in response_messages[1:]] == [
            (turn.role, turn.content) for turn in turns[:-1]
        ]
        assert store.count_memories() == 0
    finally:
        store.close()
