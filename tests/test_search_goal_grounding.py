"""Rewritten search keywords must not replace the user's actual research goal."""

import json
from unittest.mock import Mock

import pytest

from ciel.config import ControllerConfig, InternetConfig
from ciel.controller import AIController, ControllerDecision
from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import ChatMessage, LLMResponse


def _report(*, original, clarification, subject, query, quote, claims):
    """Run real report validation on supplied evidence, without a network or private data."""
    router, internet = Mock(), Mock()
    internet.config = InternetConfig()
    def reply(request, **_options):
        if "WEB GOAL CHECK" in request[0].content:
            candidates = json.loads(request[-1].content)["claims"]
            return LLMResponse(json.dumps({"relevant_claims": [
                claim["index"] for claim in candidates
                if not any(phrase in claim["text"] for phrase in (
                    "aired on", "released on",
                ))
            ]}), "test", "test")
        if "WEB ANSWER FORMAT" in request[0].content:
            return LLMResponse(json.dumps({"claims": claims}), "test", "test")
        return LLMResponse('{"action":"answer","confidence":1}', "test", "test")

    router.send.side_effect = reply
    messages = [
        ChatMessage("system", "Answer the current request using verified sources."),
        ChatMessage("user", original),
        ChatMessage("assistant", clarification),
        ChatMessage("user", subject),
    ]
    lookup = InternetResult(attempted=True, query=query, sources=[
        InternetSource("Synthetic reference", "https://example.org/reference", content=quote),
    ])
    result = AIController(router, ControllerConfig()).run_web(
        ControllerDecision(action="search", confidence=1, query=query),
        subject, messages, internet, lambda _message: None, lambda: True,
        initial_lookup=lookup, channel="text",
    )
    internet.lookup.assert_not_called()
    return result, router


@pytest.mark.parametrize("claims_present", [True, False])
def test_related_episode_fact_cannot_replace_reading_goal_after_series_clarification(
    claims_present,
):
    """Both synthesized claims and exhaustion fallbacks must respect the retained goal."""
    quote = "Amber Wings episode 92 aired on Saturday."
    claims = [{"text": quote, "quote": quote, "source": 1}] if claims_present else []

    result, _router = _report(
        original="Which light novel volume should I read after the last anime episode?",
        clarification="Which series do you mean?", subject="Amber Wings.",
        query="Amber Wings anime last episode", quote=quote, claims=claims,
    )

    assert "episode 92" not in result.text.casefold()
    assert "aired on Saturday" not in result.text
    assert "couldn't" in result.text.casefold() or "which" in result.text.casefold()


@pytest.mark.parametrize("claims_present", [True, False])
def test_related_release_fact_cannot_replace_platform_support_goal(claims_present):
    """The answer contract applies to ordinary software questions, too."""
    quote = "AuroraClient version 7 was released on Friday."
    claims = [{"text": quote, "quote": quote, "source": 1}] if claims_present else []

    result, _router = _report(
        original="Search whether that client supports Linux on ARM64.",
        clarification="What is the client called?", subject="AuroraClient.",
        query="AuroraClient latest release", quote=quote, claims=claims,
    )

    assert "version 7" not in result.text.casefold()
    assert "released on Friday" not in result.text


def test_supported_continuation_survives_clarification_and_an_unverified_extra():
    quote = (
        "Amber Wings anime episode 12 ends at light novel volume 3. "
        "Continue reading from volume 4."
    )
    answer = "Continue reading from volume 4."

    result, router = _report(
        original="Which light novel volume should I read after anime episode 12?",
        clarification="Which series do you mean?", subject="Amber Wings.",
        query="Amber Wings anime episode 12", quote=quote, claims=[
            {"text": answer, "quote": quote, "source": 1},
            {"text": "Volume 99 is also required.", "quote": quote, "source": 1},
        ],
    )

    assert answer in result.text
    assert "99" not in result.text
    assert "couldn't verify" in result.text
    report_call = next(call for call in router.send.call_args_list
                       if "WEB ANSWER FORMAT" in call.args[0][0].content)
    prompt = "\n".join(message.content for message in report_call.args[0])
    assert "light novel volume" in prompt
    assert "Amber Wings" in prompt


def test_supported_portion_of_platform_question_survives_unverified_extra():
    quote = "AuroraClient supports Linux desktop systems."

    result, _router = _report(
        original="Search whether the client supports Linux and ARM64.",
        clarification="Which client are you asking about?", subject="AuroraClient.",
        query="AuroraClient Linux ARM64 support", quote=quote, claims=[
            {"text": quote, "quote": quote, "source": 1},
            {"text": "ARM64 support arrived in version 19.", "quote": quote, "source": 1},
        ],
    )

    assert quote in result.text
    assert "version 19" not in result.text
    assert "couldn't verify" in result.text
