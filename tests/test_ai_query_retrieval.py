"""AI retrieval keywords remain separate from the owner's evidence requirements."""

from unittest.mock import Mock

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetService, InternetSource


def _hit(slug, title, content, *, date=""):
    """Return a public synthetic provider excerpt."""
    return {"href": f"https://example.org/{slug}", "title": title, "body": content,
            "content": content, "date": date}


def _service(monkeypatch, rows, **settings):
    """Keep evidence selection active without contacting a search provider or a page."""
    service = InternetService(InternetConfig(provider="parallel", **settings))
    monkeypatch.setattr(service, "_search", Mock(return_value=rows))
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=OSError("Unavailable")))
    return service


def test_paraphrased_query_keeps_goal_evidence_and_rejects_keyword_only_matches(monkeypatch):
    query = "CedarDB platform compatibility"
    goal = "Does CedarDB support Linux offline deployment?"
    service = _service(monkeypatch, [
        _hit("android", "CedarDB platform compatibility",
             "CedarDB Android cloud platform compatibility notes."),
        _hit("install", "Installation guide", "CedarDB supports Linux and can operate offline."),
    ], max_pages=0)

    result = service.lookup(query, objective=goal)

    assert [source.url for source in result.sources] == ["https://example.org/install"]
    assert service._search.call_args.args[0] == query
    assert service._search.call_args.kwargs["objective"] == goal
    assert result.query == query


def test_duplicate_selection_prefers_evidence_for_goal_over_query_repetition(monkeypatch):
    service = _service(monkeypatch, [
        _hit("install", "CedarDB platform compatibility", "CedarDB Linux offline support notes."),
        _hit("install", "CedarDB Linux offline deployment",
             "CedarDB supports Linux offline deployment."),
    ], max_pages=0)

    result = service.lookup("CedarDB platform compatibility", objective=(
        "Does CedarDB support Linux offline deployment?"
    ))

    assert len(result.sources) == 1
    assert result.sources[0].title == "CedarDB Linux offline deployment"


def test_short_query_cannot_drop_the_requested_installment_during_selection(monkeypatch):
    service = _service(monkeypatch, [
        _hit("wrong", "Astral adaptation guide",
             "Astral anime season 3 ends at volume 9 of the light novel."),
        _hit("guide", "Astral adaptation guide",
             "Astral anime season 2 ends at volume 6 of the light novel."),
    ], max_pages=0)

    result = service.lookup("Astral adaptation guide", objective=(
        "Where does the Astral anime season 2 end in the light novel?"
    ))

    assert [source.url for source in result.sources] == ["https://example.org/guide"]


def test_locally_read_page_is_checked_against_goal_instead_of_search_keywords(monkeypatch):
    service = InternetService(InternetConfig(provider="ddgs", max_pages=1))
    monkeypatch.setattr(service, "_search", Mock(return_value=[{
        "href": "https://example.org/install", "title": "CedarDB platform compatibility",
        "body": "CedarDB Linux offline deployment platform compatibility notes.",
    }]))
    page = InternetSource("Installation", "https://example.org/install", content=(
        "CedarDB supports Linux and can operate offline."
    ))
    monkeypatch.setattr(service, "_read_page", Mock(return_value=page))

    result = service.lookup("CedarDB platform compatibility", objective=(
        "Does CedarDB support Linux offline deployment?"
    ))

    assert result.sources[0].content == page.content
    assert result.sources[0].content_kind == ""


@pytest.mark.parametrize("provider_fetch", [True, False])
def test_expanded_mapping_uses_goal_for_provider_and_local_reads(monkeypatch, provider_fetch):
    partial = "Astral LN endpoint guide explains light novel continuation after anime season 2."
    complete = "Astral season 2 ends at volume 6 of the light novel."
    service = _service(monkeypatch, [_hit("guide", "Astral guide", partial)], max_pages=1)
    fetch = Mock(return_value=[_hit("guide", "Astral guide", complete)])
    if not provider_fetch:
        fetch.side_effect = OSError("Provider fetch unavailable")
        service._read_page.side_effect = None
        service._read_page.return_value = InternetSource(
            "Astral guide", "https://example.org/guide", content=complete,
        )
    monkeypatch.setattr("ciel.parallel_search.fetch_parallel", fetch)

    result = service.lookup("Astral LN endpoint", objective=(
        "Where does the Astral anime season 2 end in the light novel?"
    ))

    assert complete in result.sources[0].content
    fetch.assert_called_once()
    if provider_fetch:
        service._read_page.assert_not_called()
    else:
        service._read_page.assert_called_once()


def test_news_objective_keeps_news_selection_when_ai_query_omits_news_word(monkeypatch):
    query = "Mongolia railway"
    service = _service(monkeypatch, [
        _hit("old-story", "Mongolia opens a rail link", "Mongolia opens a new rail link.",
             date="2026-10-20"),
        _hit("new-story", "Mongolia opens a rail link", "Mongolia opens a new rail link.",
             date="2026-10-28"),
    ], max_results=1, max_pages=0)

    result = service.lookup(query, objective="Search the latest news in Mongolia.")

    assert result.kind == "news"
    assert result.sources[0].url == "https://example.org/new-story"
    assert service._search.call_args.args[0] == query


def test_failed_ai_lookup_retains_news_classification_from_original_goal(monkeypatch):
    service = _service(monkeypatch, [], max_pages=0)
    service._search.side_effect = OSError("Provider unavailable")

    result = service.lookup("Mongolia railway", objective="Search the latest news in Mongolia.")

    assert result.attempted and not result.sources
    assert result.kind == "news"
    assert result.query == "Mongolia railway"
