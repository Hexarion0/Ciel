"""Representative search follow-up, evidence, and citation regressions with no network."""

import json
from unittest.mock import Mock

import pytest

from ciel.config import ControllerConfig, InternetConfig
from ciel.controller import AIController, ControllerDecision
from ciel.internet import (
    InternetResult,
    InternetService,
    InternetSource,
    _current_scope,
    source_identity,
)
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.web_answers import verified_web_answer


def service_with_results(monkeypatch, results, *, pages=0):
    """Keep query resolution and evidence selection real while mocking network boundaries."""
    service = InternetService(InternetConfig(max_results=2, max_pages=pages))
    monkeypatch.setattr(service, "_search", Mock(return_value=results))
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=AssertionError("Unexpected read")))
    return service


def test_scope_correction_replaces_previous_season_without_losing_named_topic(monkeypatch):
    service = service_with_results(monkeypatch, [])
    result = service.prepare("Actually season 5", recent_messages=[
        ChatMessage("user", "Search Amber Works season 4 anime ending light novel"),
        ChatMessage("assistant", "Season 3 ends at volume 8."),
    ])
    assert result.attempted
    assert "Amber Works" in result.query
    assert result.query.casefold().count("season 5") == 1
    assert "season 4" not in result.query.casefold()
    assert "season 3" not in result.query.casefold() and "volume 8" not in result.query
    assert _current_scope("Amber Works season 4", "Compare season 4 with season 5") == (
        "Amber Works season 4"
    )


@pytest.mark.parametrize("text", ["Wait", "Don't search the web for Python releases"])
def test_direct_retrieval_entry_point_respects_conversation_controls(monkeypatch, text):
    service = service_with_results(monkeypatch, [])
    assert not service.prepare(text).attempted
    service._search.assert_not_called()


def test_tracking_duplicates_do_not_displace_distinct_relevant_sources(monkeypatch):
    service = service_with_results(monkeypatch, [
        {"title": "Krita release overview", "href": "https://example.org/krita?utm_source=a",
         "body": "Krita release notes and painting changes."},
        {"title": "Krita release notes", "href": "https://example.org/krita?utm_source=b#notes",
         "body": "Krita release notes and painting changes."},
        {"title": "Krita release notes", "href": "https://docs.example.org/krita",
         "body": "Krita release notes for the latest painting release."},
    ])
    result = service.prepare("Search Krita release notes")
    assert len(result.sources) == 2
    assert len({source_identity(source.url) for source in result.sources}) == 2
    assert any(source.url.startswith("https://docs.example.org/") for source in result.sources)
    assert source_identity("https://example.org/read?id=1") != source_identity(
        "https://example.org/read?id=2",
    )


def test_unrelated_landing_page_does_not_become_read_evidence(monkeypatch):
    service = service_with_results(monkeypatch, [{
        "title": "Krita release notes", "href": "https://example.org/krita",
        "body": "Krita release notes document new painting tools.",
    }], pages=1)
    service._read_page.side_effect = None
    service._read_page.return_value = InternetSource(
        "Unrelated mortgage offers", "https://example.org/offers",
        content="Compare the latest mortgage offers and interest rates.",
    )
    result = service.prepare("Search Krita release notes")
    assert len(result.sources) == 1
    assert not result.sources[0].content
    assert "mortgage" not in result.context
    assert "search_snippet_only" in result.context


def test_matching_version_number_cannot_verify_an_unrelated_named_subject():
    quote = "Python version 3.14 is the stable interpreter release."
    lookup = InternetResult(query="Python stable version", sources=[InternetSource(
        "Python release notes", "https://example.org/python", content=quote,
    )])

    def claim(subject):
        return json.dumps({"claims": [{
            "text": f"{subject} version 3.14 is stable.", "source": 1, "quote": quote,
        }]})

    assert verified_web_answer(claim("Python"), lookup) == "Python version 3.14 is stable."
    assert "couldn't verify" in verified_web_answer(claim("Blender"), lookup)


def test_controller_search_uses_explicit_corrected_scope_before_fetching():
    quote = "Amber Works season 5 has ten episodes."
    router = Mock()
    router.send.return_value = LLMResponse(json.dumps({"claims": [{
        "text": quote, "source": 1, "quote": quote,
    }]}), "test", "test")
    controller = AIController(router, ControllerConfig())
    internet = Mock(config=InternetConfig())
    internet.lookup.return_value = InternetResult(sources=[InternetSource(
        "Amber Works season 5", "https://example.org/amber", content=quote,
    )])
    initial = ControllerDecision(intent="request", action="search", confidence=1,
                                 query="Amber Works season 4 episodes")
    text = "Actually season 5"
    reply = controller.run_web(initial, text, [ChatMessage("user", text)], internet,
                               lambda _text: None, lambda: True)
    assert "season 5" in internet.lookup.call_args.args[0]
    assert "season 4" not in internet.lookup.call_args.args[0]
    assert reply.text == quote
