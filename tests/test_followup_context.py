"""Recent assistant replies survive persistence and reach the main responding model.

Providers are mocked: these regressions verify context delivery and authorization, while live
synthetic evaluations separately check whether a hosted model uses that context well.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks


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
    answer_requests = []

    def send(messages, **kwargs):
        assert kwargs["purpose"] == "conversation"
        assert kwargs["max_tokens"] == (4096 if typed else 400)
        index = len(answer_requests)
        dialogue = [(message.role, message.content) for message in messages
                    if message.role != "system"]
        assert ("latest message was typed" in messages[0].content) is typed
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
        assert router.send.call_count == len(answer_requests) == 3
        assert [turn.content for turn in turns if turn.role == "user"] == requests
        assert [turn.content for turn in turns if turn.role == "assistant"] == replies
        assert [call.args[0] for call in tts.speak.call_args_list] == replies
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
    router.send.side_effect = [LLMResponse(
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
        assert router.send.call_count == 1
        assert turns[1].provider == "local"
        assert "development day 12" in turns[1].content
        assert router.send.call_args.kwargs["purpose"] == "conversation"
        reply_messages = router.send.call_args.args[0]
        assert [(message.role, message.content) for message in reply_messages
                if message.role != "system"] == [
            (turn.role, turn.content) for turn in turns[:-1]
        ]
        assert store.count_memories() == 1
        assert store.get_fact("ciel:project_start_date").metadata["value"] == "2026-03-01"
    finally:
        store.close()


@pytest.mark.parametrize("typed", [False, True])
def test_noun_only_reply_keeps_the_topic_from_before_a_model_clarification(
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
        LLMResponse(original_reply, "test-chat", "test"),
        LLMResponse(clarification, "test-chat", "test"),
        LLMResponse("Right, 18 days of work.", "test-chat", "test"),
    ]
    info = {"input_source": "keyboard"} if typed else {}
    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch,
        [(question, {**info, "wake_verified": not typed}),
         ("That's a big number.", dict(info)), ("The days.", dict(info))],
        router=router, controller=True, persona_file=persona_file,
    )
    try:
        assert router.send.call_count == 3
        assert turns[3].provider == "test" and turns[3].model == "test-chat"
        assert all(call.kwargs["purpose"] == "conversation"
                   for call in router.send.call_args_list)
        response_messages = router.send.call_args_list[2].args[0]
        assert [(message.role, message.content) for message in response_messages
                if message.role != "system"] == [
            (turn.role, turn.content) for turn in turns[:-1]
        ]
        assert store.count_memories() == 0
    finally:
        store.close()
