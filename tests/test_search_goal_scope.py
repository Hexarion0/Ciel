"""The user's scope and supported citations survive search planning and presentation."""

import json
from unittest.mock import Mock

from ciel.config import ControllerConfig, InternetConfig
from ciel.controller import AIController, ControllerDecision
from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import ChatMessage, LLMResponse
from tests.test_controller import _web_model_responses


def test_restored_user_season_overrides_stale_model_search_scope():
    quote = "Amber Wings anime season 5 ends at volume 12. Continue reading from volume 13."
    router, internet = Mock(), Mock(config=InternetConfig())
    router.send.side_effect = _web_model_responses([LLMResponse(json.dumps({"claims": [{
        "text": "Continue reading from volume 13.", "quote": quote, "source": 1,
    }]}), "test", "test")])
    internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
        "Adaptation guide", "https://example.org/guide", content=quote,
    )])
    messages = [
        ChatMessage("system", "Answer the user's question."),
        ChatMessage("user", "Which light novel volume follows anime season 5?"),
        ChatMessage("assistant", "Which series do you mean?"),
        ChatMessage("user", "Amber Wings."),
    ]

    result = AIController(router, ControllerConfig()).run_web(
        ControllerDecision(action="search", confidence=1, query="Amber Wings season 4 anime"),
        "Amber Wings.", messages, internet, lambda _text: None, lambda: True, channel="text",
    )

    assert "volume 13" in result.text
    assert "season 4" not in internet.lookup.call_args.args[0]
    assert "season 5" in internet.lookup.call_args.args[0]
    assert "season 4" not in result.lookup.query
    assert "season 5" in internet.lookup.call_args.kwargs["objective"]


def test_only_cited_sources_are_displayed_and_numbers_remain_consistent():
    quote = "AuroraClient supports Linux desktop systems."
    sources = [
        InternetSource("Character profile", "https://example.org/characters/one",
                       content="A fictional character runs a shop."),
        InternetSource("Unrelated release", "https://example.org/release",
                       content="An unrelated release was announced."),
        InternetSource("AuroraClient platform support", "https://example.org/platforms",
                       content=quote),
    ]
    router, internet = Mock(), Mock(config=InternetConfig())
    router.send.side_effect = _web_model_responses([LLMResponse(json.dumps({"claims": [{
        "text": quote, "quote": quote, "source": 3,
    }]}), "test", "test")])
    internet.lookup.return_value = InternetResult(attempted=True, sources=sources)
    question = "Does AuroraClient support Linux?"

    result = AIController(router, ControllerConfig()).run_web(
        ControllerDecision(action="search", confidence=1, query="AuroraClient Linux support"),
        question, [ChatMessage("system", "Answer accurately."), ChatMessage("user", question)],
        internet, lambda _text: None, lambda: True, channel="text",
    )

    assert result.text == quote + " [1]"
    assert result.sources == [sources[2]]
    assert result.lookup.sources == result.sources
    assert "characters" not in result.lookup.context
