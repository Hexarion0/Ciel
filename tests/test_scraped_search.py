"""Shared keyless-engine machinery: phrasings, cache, cooldowns, and HTTP failures."""

from unittest.mock import Mock

import httpx
import pytest

from ciel import scraped_search
from ciel.scraped_search import EngineBlocked, EngineUnavailable, ScrapedEngine

_REAL_CLIENT = httpx.Client


def _row(name):
    return {"title": name, "href": f"https://{name}.example/", "body": ""}


def test_phrasings_are_interleaved_by_rank_without_duplicates():
    pages = {"first": [_row("a"), _row("shared")], "second": [_row("shared"), _row("b")]}
    engine = ScrapedEngine("Test", lambda query, _date, _timeout: pages[query])

    rows = engine.search(["first", "second", "first"], timeout=2, maximum=10)

    assert [row["href"] for row in rows] == [
        "https://a.example/", "https://shared.example/", "https://b.example/",
    ]


def test_results_are_cached_per_query():
    fetch = Mock(return_value=[_row("a")])
    engine = ScrapedEngine("Test", fetch)

    engine.search(["Same Query"], timeout=2, maximum=5)
    engine.search(["same query"], timeout=2, maximum=5)

    assert fetch.call_count == 1


def test_empty_results_are_unavailable_but_not_cached_or_cooled_down():
    fetch = Mock(side_effect=[[], [_row("a")]])
    engine = ScrapedEngine("Test", fetch)

    with pytest.raises(EngineUnavailable, match="no results"):
        engine.search(["flaky"], timeout=2, maximum=5)
    assert not engine.blocked()  # An empty page is a real answer, not an outage.

    assert [row["href"] for row in engine.search(["flaky"], timeout=2, maximum=5)] == [
        "https://a.example/",
    ]


def test_one_blocked_phrasing_keeps_the_others():
    def fetch(query, _date, _timeout):
        if query == "blocked":
            raise EngineBlocked("captcha")
        return [_row("a")]

    engine = ScrapedEngine("Test", fetch)

    rows = engine.search(["blocked", "fine"], timeout=2, maximum=5)

    assert [row["href"] for row in rows] == ["https://a.example/"]
    assert not engine.blocked()


def test_fetch_page_reports_status_and_bounds_size(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503, text="busy")

    monkeypatch.setattr("ciel.scraped_search.httpx.Client", lambda **kwargs: _REAL_CLIENT(
        **{**kwargs, "transport": httpx.MockTransport(handler)},
    ))
    status, html = scraped_search.fetch_page("Test", "https://search.example/", {"q": "x"}, 2)

    assert (status, html) == (503, "busy")
    assert "Mozilla/5.0" in seen[0].headers["user-agent"]
    assert seen[0].url.params["q"] == "x"


def test_network_errors_start_the_short_cooldown():
    engine = ScrapedEngine("Test", Mock(side_effect=httpx.ConnectError("down")))

    with pytest.raises(EngineUnavailable, match=r"failed \(ConnectError\)"):
        engine.search(["q"], timeout=2, maximum=5)
    with pytest.raises(EngineUnavailable, match="cooling down after a failed request"):
        engine.search(["q"], timeout=2, maximum=5)


def test_news_searches_get_a_date_filter():
    assert scraped_search.freshness("openai news today", news=True) == "d"
    assert scraped_search.freshness("ai news this month", news=True) == "m"
    assert scraped_search.freshness("ai news", news=True) == "w"
    assert scraped_search.freshness("ai news today", news=False) == ""


def _stalled_engine(release):
    def stalled(*_args, **_kwargs):
        release.wait(5)
        return [_row("late")]
    return ScrapedEngine("Slow", stalled)


def test_a_timeout_with_a_full_budget_starts_the_cooldown(monkeypatch):
    import threading

    monkeypatch.setattr(scraped_search, "SHORT_BUDGET_SECONDS", 0.2)
    release = threading.Event()
    engine = _stalled_engine(release)
    try:
        with pytest.raises(EngineUnavailable, match="timed out"):
            engine.search(["q"], timeout=0.5, maximum=5)
        assert engine.blocked()
    finally:
        release.set()


def test_a_cancel_ends_an_in_flight_search_promptly():
    import threading
    import time

    release, cancel = threading.Event(), threading.Event()
    engine = _stalled_engine(release)
    threading.Timer(0.2, cancel.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="canceled"):
            engine.search(["q"], timeout=3, maximum=5, cancel_event=cancel)
        assert time.monotonic() - started < 1.0
        assert not engine.blocked()
    finally:
        release.set()


def test_cached_results_are_served_during_a_cooldown():
    fetch = Mock(side_effect=[[_row("a")], EngineBlocked("CAPTCHA")])
    engine = ScrapedEngine("Test", fetch)

    first = engine.search(["cached query"], timeout=2, maximum=5)
    with pytest.raises(EngineUnavailable, match="CAPTCHA"):
        engine.search(["new query"], timeout=2, maximum=5)

    assert engine.search(["Cached Query"], timeout=2, maximum=5) == first
    with pytest.raises(EngineUnavailable, match="cooling down"):
        engine.search(["new query"], timeout=2, maximum=5)
    assert fetch.call_count == 2
