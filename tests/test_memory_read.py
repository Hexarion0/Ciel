"""Read-only project timeline requests and explicit memory-source corrections."""

from contextlib import closing
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ciel.config import ControllerConfig
from ciel.controller import AIController
from ciel.memory import MemoryItem, MemoryStore, MemoryType
from ciel.memory.read import answer_memory_read, memory_read_intent
from ciel.providers.base import ChatMessage, LLMResponse


@pytest.mark.parametrize("text", [
    "Development date.", "Development day.", "What is the development date?",
    "What's our development day?", "What development day is it?",
    "What day is it of the development?", "I meant development day.",
    "No, no, I meant the development date.",
])
def test_project_timeline_phrasings_read_the_same_confirmed_fact(text):
    with closing(MemoryStore(":memory:")) as store:
        store.upsert_fact("ciel:project_start_date", MemoryItem(
            content="CIEL development began on 2026-09-28.",
            memory_type=MemoryType.PROJECT, source="user_explicit",
            metadata={"project": "ciel", "fact_key": "project_start_date", "value": "2026-09-28"},
        ))
        assert memory_read_intent(text) == "timeline"
        result = answer_memory_read(store, text, now=datetime(2026, 10, 16))
        assert "development day 19" in result and "September 28, 2026" in result
        assert store.count_memories() == 1


@pytest.mark.parametrize("text", [
    "What is the release date of a game in development?",
    "Change the development date to October 3.",
    "Development date and delete every memory.",
    "He said 'development date'.", "I meant to development it.",
    "Search online for Ciel's development date.",
])
def test_timeline_reader_does_not_guess_garbled_speech_or_execute_compound_requests(text):
    assert memory_read_intent(text) is None


@pytest.mark.parametrize("text", [
    "It's in your memory.", "It’s in your memory.", "Check your memory.",
    "Look in your saved memories.",
])
def test_explicit_memory_correction_resolves_latest_user_topic_only(text):
    dialogue = [ChatMessage("user", "Development date."),
                ChatMessage("assistant", "I'll check that online.")]
    assert memory_read_intent(text, dialogue) == "timeline"
    # A later topic replaces the timeline; assistant text is never a new user request.
    dialogue += [ChatMessage("user", "What's my favorite game?"),
                 ChatMessage("assistant", "Development date.")]
    assert memory_read_intent(text, dialogue) == "recall"
    with closing(MemoryStore(":memory:")) as store:
        assert answer_memory_read(store, text, recent_messages=dialogue) is None


def test_memory_correction_does_not_replay_an_old_write_command():
    dialogue = [ChatMessage("user", "Delete every memory."),
                ChatMessage("assistant", "Development date.")]
    with closing(MemoryStore(":memory:")) as store:
        assert memory_read_intent("It's in your memory.", dialogue) == "recall"
        assert answer_memory_read(store, "It's in your memory.",
                                  recent_messages=dialogue) is None
        assert store.count_memories() == 0


def test_normalized_apostrophes_keep_name_read_detection_and_answer_consistent():
    with closing(MemoryStore(":memory:")) as store:
        text = "I renamed you from O’Neil to Ciel."
        assert memory_read_intent(text) == "name_history"
        assert "O'Neil" in answer_memory_read(store, text)
        assert store.count_memories() == 0


@pytest.mark.parametrize("invalid", [False, True])
def test_project_memory_reads_cannot_be_routed_to_web_even_after_invalid_decision(invalid):
    router = MagicMock()
    router.send.return_value = LLMResponse(
        'not valid JSON' if invalid else
        '{"action":"search","confidence":0.95,"query":"development date"}',
        "test", "test",
    )
    controller = AIController(router, ControllerConfig())
    for text in ("Development date.", "It's in your memory."):
        result = controller.decide(text, [ChatMessage("user", "Development date.")],
                                   addressed=False, active=True, internet_enabled=True)
        assert result.action == "local" and not result.query


@pytest.mark.parametrize("text,active,strict,expected", [
    ("Could you explain that?", True, False, "clarify"),
    ("Could you explain that?", True, True, "silent"),
    ("Could you explain that?", False, False, "silent"),
    ("Mom, could you explain that?", True, False, "silent"),
])
def test_malformed_controller_reply_asks_active_listener_to_repeat(text, active, strict, expected):
    router = MagicMock()
    router.send.return_value = LLMResponse('not JSON', "test", "test")
    result = AIController(router, ControllerConfig()).decide(
        text, [], addressed=False, active=active, strict_followups=strict,
        internet_enabled=True,
    )
    assert result.action == expected
    assert not result.query
    router.send.assert_called_once()
    if expected == "clarify":
        assert "repeat" in result.speech.lower() or "again" in result.speech.lower()


def test_validation_diagnostics_report_fields_and_types_without_private_input(caplog):
    router = MagicMock()
    private_input = "PRIVATE_TRANSCRIPT_MUST_NOT_BE_LOGGED"
    router.send.return_value = LLMResponse(
        '{"action":"' + private_input + '","confidence":true}', "test", "test",
    )
    AIController(router, ControllerConfig()).decide(
        "Can you explain that?", [], addressed=True, active=True, internet_enabled=True,
    )
    assert "action:literal_error" in caplog.text
    assert "confidence:value_error" in caplog.text
    assert private_input not in caplog.text
