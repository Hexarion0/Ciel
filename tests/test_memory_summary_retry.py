"""Background summary failures retain work without retrying on every reply."""

import json
from unittest.mock import MagicMock

import pytest

from ciel.memory import ConversationManager, MemoryStore
from ciel.memory import manager as manager_module
from ciel.memory.summary import SUMMARY_MAX_CHARS, SummaryValidationError, normalize_summary
from ciel.providers.base import LLMResponse
from ciel.providers.priority import BackgroundDeferred


@pytest.fixture
def conversation():
    store = MemoryStore(":memory:")
    manager = ConversationManager(
        store, max_turns=2, auto_summarize_threshold=2, summary_interval=2,
    )
    for index in range(8):
        manager.add_turn("user", f"Earlier task {index}")
    yield manager
    if manager._summary_thread is not None:
        manager._summary_thread.join(timeout=2)
    store.close()


@pytest.fixture
def retry_clock(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(manager_module, "monotonic", lambda: clock[0])
    return clock


def provider_reply(output="Recovered summary"):
    """Return a fake provider without sending requests or persisting private output."""
    provider = MagicMock()
    provider.send.return_value = LLMResponse(output, "test", "test")
    return provider


def structured_summary(**overrides):
    """Build a valid compact summary, with optional deliberately invalid fields."""
    return json.dumps({
        "overview": "Earlier tasks.", "user_context": [], "decisions": [],
        "open_threads": [], "superseded": [], **overrides,
    })


def finish_summary(manager):
    """Wait only for the fake-provider worker, failing if it cannot finish promptly."""
    assert manager._summary_thread is not None
    manager._summary_thread.join(timeout=2)
    assert not manager._summary_thread.is_alive()


def test_failed_summary_does_not_retry_on_each_new_reply(conversation, retry_clock):
    provider = provider_reply("{broken JSON")

    assert conversation.schedule_summary(provider)
    finish_summary(conversation)
    assert conversation.get_summary() is None
    conversation.add_turn("assistant", "The next reply.")

    assert not conversation.schedule_summary(provider)
    assert conversation.summarize_older_turns(provider) is None
    assert provider.send.call_count == 1
    assert conversation.get_total_turn_count() == 9


def test_failed_summary_retries_after_cooldown_without_losing_pending_turns(
    conversation, retry_clock,
):
    provider = provider_reply()
    provider.send.side_effect = [
        LLMResponse("{broken JSON", "test", "test"), provider.send.return_value,
    ]
    pending_ids = [turn.id for turn in conversation.get_recent_turns(limit=20)]
    assert conversation.schedule_summary(provider)
    finish_summary(conversation)

    retry_clock[0] += manager_module._SUMMARY_RETRY_SECONDS - 0.1
    assert not conversation.schedule_summary(provider)
    retry_clock[0] += 0.1
    assert conversation.schedule_summary(provider)
    finish_summary(conversation)

    assert provider.send.call_count == 2
    assert conversation.get_summary() == "Recovered summary"
    summary = conversation.store.get_session_summary("default")
    assert summary.metadata["last_turn_id"] == pending_ids[-3]
    assert [turn.id for turn in conversation.get_recent_turns(limit=20)] == pending_ids


def test_foreground_deferral_does_not_start_a_failure_cooldown(conversation, retry_clock):
    provider = provider_reply()
    provider.send.side_effect = [
        BackgroundDeferred("Foreground conversation is active"), provider.send.return_value,
    ]
    assert conversation.schedule_summary(provider)
    finish_summary(conversation)
    assert conversation.get_summary() is None

    assert conversation.schedule_summary(provider)
    finish_summary(conversation)

    assert provider.send.call_count == 2
    assert conversation.get_summary() == "Recovered summary"


@pytest.mark.parametrize("other_manager", [False, True])
def test_reset_generation_bypasses_an_existing_failure_cooldown(
    conversation, retry_clock, other_manager,
):
    provider = provider_reply("{broken JSON")
    assert conversation.schedule_summary(provider)
    finish_summary(conversation)
    resetter = ConversationManager(conversation.store) if other_manager else conversation
    assert resetter.clear_session() == 8
    for index in range(8):
        conversation.add_turn("user", f"New generation task {index}")
    provider.send.return_value = LLMResponse("New generation summary", "test", "test")

    assert conversation.schedule_summary(provider)
    finish_summary(conversation)

    assert provider.send.call_count == 2
    assert conversation.get_summary() == "New generation summary"


def test_in_flight_failure_cannot_defer_work_created_after_reset(conversation, retry_clock):
    provider = provider_reply()

    def reset_then_fail(*_args, **_kwargs):
        conversation.clear_session()
        for index in range(8):
            conversation.add_turn("user", f"New generation task {index}")
        raise ValueError("Private provider detail")

    provider.send.side_effect = reset_then_fail
    assert conversation.schedule_summary(provider)
    finish_summary(conversation)
    provider.send.side_effect = None

    assert conversation.schedule_summary(provider)
    finish_summary(conversation)

    assert provider.send.call_count == 2
    assert conversation.get_summary() == "Recovered summary"


@pytest.mark.parametrize("output,category", [
    ("{private malformed response", "malformed_json"),
    (structured_summary(decisions="private field value"), "schema"),
    (structured_summary(open_threads=["private " + "x" * 4000]), "length"),
    ("", "empty"),
])
def test_summary_validation_logs_only_a_safe_category(
    conversation, retry_clock, caplog, output, category,
):
    provider = provider_reply(output)

    assert conversation.summarize_older_turns(provider) is None

    assert f"category={category}" in caplog.text
    assert "private" not in caplog.text
    assert "ValidationError" not in caplog.text
    assert conversation.get_summary() is None
    assert not conversation.schedule_summary(provider)
    assert provider.send.call_count == 1


def test_provider_value_error_has_a_distinct_safe_failure_category(
    conversation, retry_clock, caplog,
):
    provider = provider_reply()
    provider.send.side_effect = ValueError("private provider response")

    assert conversation.summarize_older_turns(provider) is None

    assert "provider_ValueError" in caplog.text
    assert "category=" not in caplog.text
    assert "private" not in caplog.text
    assert not conversation.schedule_summary(provider)
    assert provider.send.call_count == 1


@pytest.mark.parametrize("fields", [
    {"overview": "o" * 401},
    {"decisions": ["d" * 161]},
    {"decisions": [
        "The user chose to keep the existing local voice and postpone cloud changes until "
        "microphone stability is verified; this decision does not authorize uploads or "
        "installation.",
    ]},
    {"decisions": [f"User chose supported option {index}." for index in range(5)]},
    {"decisions": ["  Keep this condition: no uploads until the user agrees.\n"]},
])
def test_complete_typed_summary_preserves_all_notes_within_context_budget(conversation, fields):
    output = structured_summary(**fields)
    assert len(output) < SUMMARY_MAX_CHARS

    summary = conversation.summarize_older_turns(provider_reply(output))

    assert summary is not None
    assert json.loads(summary) == json.loads(output)
    assert conversation.store.get_session_summary("default").metadata["last_turn_id"] == 6


def test_oversized_summary_keeps_previous_cursor_and_all_pending_turns(
    conversation, retry_clock,
):
    previous = conversation.summarize_older_turns(provider_reply(structured_summary()))
    previous_cursor = conversation.store.get_session_summary("default").metadata["last_turn_id"]
    for index in range(4):
        conversation.add_turn("user", f"New pending task {index}")
    pending_ids = [turn.id for turn in conversation.get_recent_turns(limit=20)]
    provider = provider_reply(structured_summary(decisions=["x" * 4000]))

    assert conversation.summarize_older_turns(provider) is None

    assert conversation.get_summary() == previous
    assert (conversation.store.get_session_summary("default").metadata["last_turn_id"]
            == previous_cursor)
    assert [turn.id for turn in conversation.get_recent_turns(limit=20)] == pending_ids
    assert not conversation.schedule_summary(provider)
    assert provider.send.call_count == 1


@pytest.mark.parametrize("fields", [
    {"overview": " \n\t"}, {"decisions": [" \n\t"]},
    {"decisions": [123]}, {"overview": 123}, {"unexpected": "extra field"},
])
def test_summary_length_flexibility_preserves_strict_content_schema(conversation, fields):
    assert conversation.summarize_older_turns(provider_reply(structured_summary(**fields))) is None
    assert conversation.get_summary() is None


def test_structured_summary_keeps_the_existing_hard_serialized_budget():
    fields = json.loads(structured_summary(overview=""))
    overhead = len(json.dumps(fields, ensure_ascii=False, separators=(",", ":")))
    fields["overview"] = "x" * (SUMMARY_MAX_CHARS - overhead)

    normalized, summary_format = normalize_summary(json.dumps(fields))

    assert len(normalized) == SUMMARY_MAX_CHARS
    assert summary_format == "structured-v1"
    assert json.loads(normalized) == fields
    fields["overview"] += "x"
    with pytest.raises(SummaryValidationError) as error:
        normalize_summary(json.dumps(fields))
    assert error.value.category == "length"
