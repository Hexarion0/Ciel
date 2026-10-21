"""Unmatched local decisions still answer with saved memory and recent user corrections."""

import json
from datetime import datetime
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.memory.context import recall_context_memories
from ciel.providers.base import LLMResponse
from tests.test_ambient import run_callbacks


def decision(action: str) -> LLMResponse:
    """Supply a controller choice independently of the final conversational reply."""
    return LLMResponse(json.dumps({"action": action, "confidence": 0.99}), "light", "nim")


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("query,reply", [
    ("Which development day would that make today?", "It's development day 17."),
    ("How does your memory work?", "Yes. I can recall saved facts and our recent conversation."),
])
def test_unmatched_memory_tool_uses_recalled_context(
    tmp_path, monkeypatch, typed, query, reply,
):
    """The controller cannot turn a readable saved fact into a tool-match failure."""
    seed = MemoryStore(tmp_path / "memory.db")
    seed.save_memory(MemoryItem(
        content="RAPHAEL project development began on 2026-09-28.",
        memory_type=MemoryType.PROJECT,
        source="user_explicit",
        metadata={"project": "ciel", "fact_key": "project_start_date", "value": "2026-09-28"},
    ))
    seed.close()
    monkeypatch.setattr(
        "ciel.memory.context.recall_context_memories",
        lambda store, query="", **kwargs: recall_context_memories(
            store, query, now=datetime(2026, 10, 14), **kwargs,
        ),
    )
    router = MagicMock()
    choices = [decision("local"), LLMResponse(reply, "chat", "nim")]
    utterances = [(query, {"input_source": "keyboard"} if typed else {})]
    if not typed:
        choices = [decision("answer"), LLMResponse("Hello.", "chat", "nim"), *choices]
        utterances.insert(0, ("Hey Ciel.", {"wake_verified": True}))
    router.send.side_effect = choices

    _, _, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, utterances, router=router, controller=True,
    )
    try:
        assert turns[-1].content == reply
        call = router.send.call_args
        assert call.kwargs.get("purpose") != "controller"
        prompt = call.args[0][0].content
        assert "development day 17" in prompt
        assert "2026-09-28" in prompt
        assert "The application stores history" in prompt
        assert "No local action or memory write was performed" in prompt
        assert store.count_memories() == 1
        assert call.args[0][-1].content == query
    finally:
        store.close()


def test_unauthorized_local_memory_request_uses_application_acknowledgment(tmp_path, monkeypatch):
    """An inferred follow-up cannot save a fact or ask a model to claim it did."""
    router = MagicMock()
    router.send.side_effect = [
        decision("answer"), LLMResponse("Hello.", "chat", "nim"),
        decision("local"),
    ]
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel.", {"wake_verified": True}),
        ("Remember my favorite game is CS2.", {}),
    ], router=router, controller=True)
    try:
        assert store.get_fact("user:favorite_game") is None
        assert turns[-1].content.startswith("I haven't saved that.")
        assert turns[-1].provider == "local"
        assert router.send.call_count == 3
        assert router.send.call_args.kwargs["purpose"] == "controller"
    finally:
        store.close()


def test_controller_and_reply_preserve_name_history_corrections(tmp_path, monkeypatch):
    """A previous assistant name remains context without becoming its current name."""
    router = MagicMock()
    router.send.side_effect = [
        decision("answer"), LLMResponse("Understood, you renamed me.", "chat", "nim"),
        decision("answer"), LLMResponse("Yes, Raphael was my previous name.", "chat", "nim"),
    ]
    # These conversational phrasings exercise the reply model. Exact declarative
    # renames already have a deterministic, read-only handler.
    correction = "I used to call you Rafael, then I changed your name to Ciel."
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        (correction, {"input_source": "keyboard"}),
        ("Before that, I called you Raphael.", {"input_source": "keyboard"}),
    ], router=router, controller=True)
    try:
        controller_call = router.send.call_args_list[2]
        controller_prompt = controller_call.args[0][0].content
        assert correction in controller_prompt
        assert "current or former name" in controller_prompt
        reply_messages = router.send.call_args.args[0]
        assert any(message.role == "user" and message.content == correction
                   for message in reply_messages)
        assert "Acknowledge supplied former names" in reply_messages[0].content
        assert turns[-1].content == "Yes, Raphael was my previous name."
        assert store.count_memories() == 0  # No unconfirmed identity fact is silently saved.
    finally:
        store.close()


@pytest.mark.parametrize("invalid_decision", [False, True])
def test_spoken_development_date_and_memory_correction_never_search_online(
    tmp_path, monkeypatch, invalid_decision,
):
    """Replay the supplied failure through the real callback with a misrouting model."""
    from ciel.memory import read
    from tests.test_controller_runtime import _internet

    service = _internet(monkeypatch, [])
    seed = MemoryStore(tmp_path / "memory.db")
    seed.upsert_fact("ciel:project_start_date", MemoryItem(
        content="CIEL development began on 2026-09-28.",
        memory_type=MemoryType.PROJECT, source="user_explicit",
        metadata={"value": "2026-09-28"},
    ))
    seed.close()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 16, 3, 54, tzinfo=tz)

    monkeypatch.setattr(read, "datetime", Clock)
    wrong = LLMResponse(
        "not JSON" if invalid_decision else json.dumps({
            "action": "search", "confidence": .95, "query": "development date",
        }), "light", "nim",
    )
    router = MagicMock()
    router.send.side_effect = [
        decision("answer"), LLMResponse("Hello.", "chat", "nim"),
        *([wrong] * (4 if invalid_decision else 2)),
    ]
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel.", {"wake_verified": True}),
        ("Development date.", {}),
        ("It's in your memory.", {}),
    ], router=router, controller=True)
    try:
        service.lookup.assert_not_called()
        service.read_page.assert_not_called()
        replies = [turn for turn in turns if turn.role == "assistant"]
        assert all("development day 19" in turn.content for turn in replies[-2:])
        assert all("September 28, 2026" in turn.content for turn in replies[-2:])
        assert all(turn.provider == "local" for turn in replies[-2:])
        assert store.count_memories() == 1
        assert router.send.call_count == (6 if invalid_decision else 4)
    finally:
        store.close()


def test_memory_correction_supersedes_garbled_interrupted_request(tmp_path, monkeypatch):
    """The latest clear fragment remains the memory request after an interrupted clarification."""
    from tests.test_controller_runtime import _internet

    service = _internet(monkeypatch, [])
    first, second = Event(), Event()
    router = MagicMock()
    router.send.side_effect = [
        decision("answer"), LLMResponse("Hello.", "chat", "nim"),
        LLMResponse("not JSON", "light", "nim"),
        LLMResponse("not JSON", "light", "nim"),
        LLMResponse('{"action":"search","confidence":0.95,"query":"development date"}',
                    "light", "nim"),
    ]
    tts = MagicMock()

    def speak(text, **kwargs):
        if kwargs.get("on_start"):
            kwargs["on_start"]()
        if "try again" in text:
            first.set()
            return False
        return True

    tts.speak.side_effect = speak
    _, _, _, store, turns = run_callbacks(tmp_path, monkeypatch, [
        ("Hey Ciel.", {"wake_verified": True}),
        ("No, I meant to development it.", {"cancel_event": first}),
        ("Development date.", {"cancel_event": second, "supersedes_cancel_event": first}),
    ], router=router, tts=tts, controller=True)
    try:
        service.lookup.assert_not_called()
        assert turns[-1].content == "I don't have a confirmed project start date saved."
        assert turns[-1].provider == "local"
        assert store.count_memories() == 0
        latest_request = router.send.call_args.args[0][-1].content
        assert "No, I meant to development it." in latest_request
        assert "Development date." in latest_request
    finally:
        store.close()
