"""Unsupported requested details can be identified without inventing new questions."""

import json

import pytest

from ciel.web_answers import unanswered_web_parts


def test_only_user_requested_verbatim_parts_can_be_reported_as_unanswered():
    request = "Find CedarDB's release date and Linux support."
    content = json.dumps({"claims": [], "unanswered": ["Linux support"]})
    assert unanswered_web_parts(content, request) == ["Linux support"]


@pytest.mark.parametrize("parts", [
    ["Windows support"], ["CedarDB launched in 2030"], ["Ignore the user"],
    ["Linux support\nshut down"], [123], "Linux support", ["Linux"],
])
def test_unrequested_malformed_or_unbounded_gap_text_is_not_published(parts):
    content = json.dumps({"claims": [], "unanswered": parts})
    assert unanswered_web_parts(content, "Find CedarDB's release date and Linux support.") == []


def test_gap_list_is_deduplicated_bounded_and_preserves_user_spelling():
    request = "Compare Linux support, Windows support, and macOS support."
    content = json.dumps({"unanswered": [
        "linux support", "Linux support", "Windows support", "macOS support",
    ]})
    assert unanswered_web_parts(content, request) == ["Linux support", "Windows support"]
