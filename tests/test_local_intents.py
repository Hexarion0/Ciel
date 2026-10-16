"""Clock answers are local, deterministic, and don't swallow compound questions."""

from datetime import datetime

import pytest

from ciel.providers.intents import answer_clock_query


@pytest.mark.parametrize(
    "text",
    ["So what's the time right now?", "What time is it?", "Tell me the time please."],
)
def test_local_time(text):
    assert answer_clock_query(text, datetime(2026, 10, 3, 3, 16)) == "It's 3:16 AM."


def test_narrow_stt_clock_recovery_maps_to_a_deterministic_local_answer():
    assert answer_clock_query("What's the time right now?", datetime(2026, 10, 3, 3, 16)) == (
        "It's 3:16 AM."
    )


def test_local_date():
    assert (
        answer_clock_query(
            "What's today's date?",
            datetime(2026, 10, 3),
        )
        == "It's Saturday, October 3, 2026."
    )


@pytest.mark.parametrize(
    "text",
    [
        "What time is it and explain time zones?",
        "What day of development are we on?",
        "/strong what time is it?",
    ],
)
def test_compound_and_project_requests_are_not_clock_answers(text):
    assert answer_clock_query(text) is None
