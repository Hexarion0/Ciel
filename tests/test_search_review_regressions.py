"""Answer ranking and redirected reads preserve local, source-owned evidence."""

import json
import time
from unittest.mock import Mock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetResult, InternetService, InternetSource, _goal_score
from ciel.search_providers import SearchProviderError
from ciel.web_answers import verified_web_answer


def test_latest_sidebar_cannot_promote_another_subject_over_the_requested_release(monkeypatch):
    service = InternetService(InternetConfig(provider="parallel", max_pages=0, max_results=1))
    correct = "Astral episode 18 aired on October 28, 2026."
    sidebar = ("Astral episode release information appears here.\n\n"
               "Boreal latest episode is episode 19, which aired yesterday.")
    monkeypatch.setattr(service, "_search", Mock(return_value=[
        {"href": "https://example.org/astral", "title": "Astral episode release",
         "body": correct, "content": correct},
        {"href": "https://example.org/weekly", "title": "Television weekly round-up",
         "body": sidebar, "content": sidebar},
    ]))

    result = service.lookup("Astral latest episode", objective=(
        "What is the latest released episode of Astral?"
    ))

    assert [source.url for source in result.sources] == ["https://example.org/astral"]


@pytest.mark.parametrize("child_heading", ["Episode releases", "Release notes"])
def test_current_release_can_use_its_own_subject_heading(child_heading):
    source = InternetSource("Release guide", "https://example.org/astral", content=(
        f"# Astral\n\n## {child_heading}\n\n"
        "The latest released episode is episode 18, which aired October 28, 2026."
    ))

    assert _goal_score("What is the latest released episode of Astral?", source)[0] > 0


def test_another_named_section_cannot_borrow_a_parent_subject_for_current_ranking():
    source = InternetSource("Astral release guide", "https://example.org/astral", content=(
        "# Astral\n\n## Boreal episode guide\n\n"
        "The latest released episode is episode 19, which aired October 28, 2026."
    ))

    assert _goal_score("What is the latest released episode of Astral?", source)[0] == 0


@pytest.mark.parametrize("provider", ["parallel", "ddgs"])
def test_redirected_read_cannot_inherit_another_urls_evidence_or_metadata(monkeypatch, provider):
    old_url, new_url = "https://old.example.org/cedar", "https://new.example.org/cedar"
    old_quote = "Cedar supports Linux and runs offline."
    new_quote = "Cedar privacy and local storage documentation describes other operating systems."
    service = InternetService(InternetConfig(provider=provider, max_pages=1))
    row = {"href": old_url, "title": "Cedar platform guide", "body": old_quote,
           "date": "2026-10-29", "date_kind": "published", "publisher": "Old publisher"}
    if provider == "parallel":
        row["content"] = old_quote
        monkeypatch.setattr("ciel.parallel_search.fetch_parallel", Mock(
            side_effect=SearchProviderError("Unavailable"),
        ))
    monkeypatch.setattr(service, "_search", Mock(return_value=[row]))
    monkeypatch.setattr(service, "_read_page", Mock(return_value=InternetSource(
        "Cedar replacement overview", new_url, content=new_quote,
    )))

    result = service.lookup("Cedar Linux offline privacy and local storage", objective=(
        "Find Cedar Linux offline support, privacy and local storage requirements"
    ))

    sources = {source.url: source for source in result.sources}
    assert sources[new_url].content == new_quote
    assert sources[new_url].snippet == ""
    assert sources[new_url].published_at == sources[new_url].date_kind == ""
    assert sources[new_url].publisher == ""
    assert old_quote in (sources[old_url].content or sources[old_url].snippet)
    for index, source in enumerate(result.sources, 1):
        if source.url != new_url:
            continue
        report = json.dumps({"claims": [{
            "text": old_quote, "quote": old_quote, "source": index,
        }]})
        assert "couldn't verify" in verified_web_answer(report, result)


def test_a_redirected_source_cannot_validate_an_earlier_provider_passage_id(monkeypatch):
    from ciel.web_answers import web_evidence_context

    quote = "Cedar supports Linux and runs offline."
    original = InternetSource("Cedar guide", "https://old.example.org/cedar", content=quote,
                              content_kind="provider_excerpt")
    old_lookup = InternetResult(query="Cedar Linux offline support", sources=[original])
    record = json.loads(web_evidence_context(old_lookup).split("\n", 1)[1])["sources"][0]
    passage_id = next(passage["id"] for passage in record["passages"]
                      if passage["text"] == quote)
    service = InternetService(InternetConfig(provider="parallel", max_pages=1))
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", Mock(
        side_effect=SearchProviderError("Unavailable"),
    ))
    monkeypatch.setattr(service, "_read_page", Mock(return_value=InternetSource(
        "Cedar privacy", "https://new.example.org/cedar",
        content="Cedar privacy and local storage documentation describes other operating systems.",
    )))
    sources = [original]

    service._expand_excerpts(
        "Find Cedar Linux offline support, privacy and local storage requirements",
        [original], sources, time.monotonic() + 5, None, lambda _sources: None,
    )

    redirected = next(source for source in sources if source.url != original.url)
    lookup = InternetResult(query=old_lookup.query, sources=[redirected])
    report = json.dumps({"claims": [{"text": quote, "passage": passage_id, "source": 1}]})
    assert "couldn't verify" in verified_web_answer(report, lookup)
