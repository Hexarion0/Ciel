"""Free search chains: DuckDuckGo (default) or Yahoo first, Parallel when blocked."""

from unittest.mock import Mock

import pytest

from ciel import duckduckgo_search, yahoo_search
from ciel.config import InternetConfig, Settings
from ciel.internet import InternetService, InternetSource
from ciel.scraped_search import EngineUnavailable
from ciel.web_answers import extract_citations, renumber_citations


@pytest.fixture(autouse=True)
def _fresh_engine_state():
    yahoo_search.reset()
    duckduckgo_search.reset()
    yield
    yahoo_search.reset()
    duckduckgo_search.reset()


def _service(**kwargs):
    return InternetService(InternetConfig(provider="yahoo", **kwargs))


def test_duckduckgo_is_the_default_provider():
    assert InternetConfig().provider == "duckduckgo"
    assert Settings(_env_file=None).internet.provider == "duckduckgo"


def test_duckduckgo_rows_are_used_with_configured_region_and_safe_search(monkeypatch):
    service = InternetService(InternetConfig(
        provider="duckduckgo", max_pages=0,
        duckduckgo_region="uk-en", duckduckgo_safe_search="strict",
    ))
    search = Mock(return_value=[{
        "title": "Python 3.14 release", "href": "https://www.python.org/downloads/",
        "body": "Python 3.14 is out.", "content": "Injected excerpt",
    }])
    monkeypatch.setattr(duckduckgo_search, "search_duckduckgo", search)
    monkeypatch.setattr(yahoo_search, "search_yahoo",
                        Mock(side_effect=AssertionError("Yahoo should not run")))
    monkeypatch.setattr("ciel.parallel_search.search_parallel",
                        Mock(side_effect=AssertionError("Parallel should not run")))

    result = service.lookup("latest Python release",
                            search_queries=["latest Python release", "Python 3.14"])

    assert search.call_args.args[0][0] == "latest Python release"
    assert search.call_args.kwargs["region"] == "uk-en"
    assert search.call_args.kwargs["safe_search"] == "strict"
    assert [source.url for source in result.sources] == ["https://www.python.org/downloads/"]
    # DuckDuckGo rows are search previews; only Parallel carries source excerpts.
    assert result.sources[0].content == ""


def test_parallel_answers_when_duckduckgo_is_blocked(monkeypatch):
    service = InternetService(InternetConfig(provider="duckduckgo", max_pages=0))
    monkeypatch.setattr(duckduckgo_search, "search_duckduckgo", Mock(
        side_effect=EngineUnavailable("DuckDuckGo asked this network to solve a CAPTCHA"),
    ))
    parallel = Mock(return_value=[{
        "title": "Python release notes", "href": "https://www.python.org/",
        "body": "Python.", "content": "Python 3.14 was released in October 2025.",
    }])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", parallel)

    result = service.lookup("Python release notes")

    parallel.assert_called_once()
    assert result.sources[0].content_kind == "provider_excerpt"


def test_direct_mode_returns_duckduckgo_rows(monkeypatch):
    import time

    service = InternetService(InternetConfig(provider="duckduckgo"))
    url = "https://example.org/python?id=1&region=eu"
    monkeypatch.setattr(duckduckgo_search, "search_duckduckgo", Mock(return_value=[
        {"title": "Python notes", "href": url, "body": "Notes."},
    ]))

    sources = service.search_direct(["python notes"], deadline=time.monotonic() + 10)

    assert [(source.url, source.snippet) for source in sources] == [(url, "Notes.")]


def test_yahoo_rows_are_used_and_pages_read_locally(monkeypatch):
    service = _service(max_pages=1)
    search = Mock(return_value=[{
        "title": "Python 3.15.2 release", "href": "https://www.python.org/downloads/",
        "body": "Python 3.15.2 is the latest Python release.",
        # Yahoo rows never carry provider excerpts; ignore one if present.
        "content": "Injected excerpt",
    }])
    monkeypatch.setattr(yahoo_search, "search_yahoo", search)
    monkeypatch.setattr("ciel.parallel_search.search_parallel",
                        Mock(side_effect=AssertionError("Parallel should not run")))
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=OSError("offline")))

    result = service.lookup("latest Python release", search_queries=[
        "latest Python release", "Python 3.15 release",
    ])

    # The engine removes the repeated first phrasing itself.
    assert search.call_args.args[0] == ["latest Python release", "latest Python release",
                                        "Python 3.15 release"]
    assert [source.url for source in result.sources] == ["https://www.python.org/downloads/"]
    assert result.sources[0].content == ""
    assert result.sources[0].snippet.startswith("Python 3.15.2")


def test_parallel_answers_when_yahoo_is_blocked(monkeypatch):
    service = _service(max_pages=1)
    monkeypatch.setattr(yahoo_search, "search_yahoo", Mock(
        side_effect=EngineUnavailable("Yahoo asked this network to solve a CAPTCHA"),
    ))
    excerpt = "Spain won the 2026 FIFA World Cup final against Argentina."
    parallel = Mock(return_value=[{
        "title": "Spain win the 2026 World Cup", "href": "https://www.fifa.com/final",
        "body": "Spain beat Argentina.", "content": excerpt,
    }])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", parallel)
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=OSError("offline")))

    result = service.lookup("2026 World Cup final winner Spain Argentina",
                            search_queries=["2026 World Cup winner"])

    assert parallel.call_args.kwargs["search_queries"] == ["2026 World Cup winner"]
    assert [source.url for source in result.sources] == ["https://www.fifa.com/final"]
    # Parallel rows keep their attributable excerpts after a fallback.
    assert result.sources[0].content == excerpt
    assert result.sources[0].content_kind == "provider_excerpt"


def test_nearly_spent_budget_skips_yahoo_for_parallel(monkeypatch):
    service = _service(max_pages=0, timeout_seconds=2.0)
    yahoo = Mock(side_effect=AssertionError("Too little time for Yahoo"))
    monkeypatch.setattr(yahoo_search, "search_yahoo", yahoo)
    parallel = Mock(return_value=[{"title": "Python release notes",
                                   "href": "https://www.python.org/", "body": "Python."}])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", parallel)

    service.lookup("Python release notes")

    yahoo.assert_not_called()
    parallel.assert_called_once()


def test_result_url_ampersands_are_kept(monkeypatch):
    service = _service(max_pages=0)
    url = "https://example.org/python?id=1&region=eu&timestamp=5"
    monkeypatch.setattr(yahoo_search, "search_yahoo", Mock(return_value=[
        {"title": "Python release notes", "href": url, "body": "Python release notes."},
    ]))

    result = service.lookup("Python release notes")

    assert [source.url for source in result.sources] == [url]


def test_same_titled_article_on_two_paths_is_listed_once(monkeypatch):
    service = _service(max_pages=0)
    title = "Spain crowned FIFA World Cup 2026 Champions in New York New Jersey"
    monkeypatch.setattr(yahoo_search, "search_yahoo", Mock(return_value=[
        {"title": title, "href": "https://inside.fifa.com/organisation/news/spain-crowned",
         "body": "Spain crowned World Cup 2026 champions."},
        {"title": title, "href": "https://inside.fifa.com/news/spain-crowned",
         "body": "Spain crowned World Cup 2026 champions."},
    ]))

    result = service.lookup("Spain World Cup 2026 champions")

    assert len(result.sources) == 1


def test_citations_are_renumbered_to_the_displayed_sources():
    sources = [InternetSource(f"Source {n}", f"https://s{n}.example/") for n in range(1, 9)]
    reply = ("Set the quantum [2]. Disable power saving [1][8] in 2024 [2024].\n"
             "```python\nrates = xs[0] + ys[8]\nprint([8])\n```\nSee also [8].")

    cited = extract_citations(reply, sources)
    renumbered = renumber_citations(reply, sources, cited)

    assert [source.title for source in cited] == ["Source 1", "Source 2", "Source 8"]
    assert renumbered == (
        "Set the quantum [2]. Disable power saving [1][3] in 2024 [2024].\n"
        "```python\nrates = xs[0] + ys[8]\nprint([8])\n```\nSee also [3]."
    )
