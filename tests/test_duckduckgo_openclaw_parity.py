"""OpenClaw parity: each case from OpenClaw's DuckDuckGo provider tests, run against Ciel.

Source: openclaw/openclaw ``extensions/duckduckgo/src/ddg-search-provider.test.ts``.
The fixture HTML below is OpenClaw's own. Cases that test OpenClaw's tool plumbing
(``count`` argument validation, TTL configuration) map to Ciel's equivalents; the
intentional differences are listed in ``ciel.duckduckgo_search``'s docstring.
"""

import threading

import httpx
import pytest

from ciel import duckduckgo_search as ddg
from ciel.scraped_search import EngineUnavailable

_REAL_CLIENT = httpx.Client


def result_html(url, title, snippet=""):
    """OpenClaw's ``resultHtml`` fixture helper, verbatim."""
    return f'<a class="result__a" href="{url}">{title}</a><a class="result__snippet">{snippet}</a>'


@pytest.fixture(autouse=True)
def _fresh_state():
    ddg.reset()
    yield
    ddg.reset()


@pytest.fixture
def served(monkeypatch):
    """Serve one fixed page for every request and record the requests."""
    state = {"requests": [], "response": httpx.Response(200, text="")}

    def handler(request):
        state["requests"].append(request)
        return state["response"]

    monkeypatch.setattr("ciel.scraped_search.httpx.Client", lambda **kwargs: _REAL_CLIENT(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    return state


def _search(served, html, query="query", **options):
    served["response"] = httpx.Response(200, text=html)
    return ddg.search_duckduckgo([query], timeout=2, maximum=10, **options)


def test_rejects_an_already_canceled_search_without_fetching(served):
    canceled = threading.Event()
    canceled.set()

    with pytest.raises(TimeoutError, match="canceled"):
        ddg.search_duckduckgo(["pre-canceled"], timeout=2, maximum=5, cancel_event=canceled)
    assert served["requests"] == []


def test_preserves_http_status_for_search_failure_guidance(served):
    served["response"] = httpx.Response(429, text="rate limited")

    with pytest.raises(EngineUnavailable, match=r"HTTP 429"):
        ddg.search_duckduckgo(["rate limited"], timeout=2, maximum=5)


def test_bounds_successful_html_bodies(served):
    served["response"] = httpx.Response(200, content=b"x" * (2 * 1024 * 1024))

    with pytest.raises(EngineUnavailable, match="oversized"):
        ddg.search_duckduckgo(["bounded response"], timeout=2, maximum=5)


@pytest.mark.parametrize(("options", "region", "safe_search"), [
    ({"region": "de-de"}, "de-de", "-1"),
    ({"region": "   "}, None, "-1"),
    ({}, None, "-1"),
    ({"safe_search": "strict"}, None, "1"),
    ({"safe_search": "off"}, None, "-2"),
])
def test_sends_region_and_safe_search_in_the_request(served, options, region, safe_search):
    from urllib.parse import parse_qs

    served["response"] = httpx.Response(200, text=result_html("https://example.com", "Example"))
    ddg.search_duckduckgo([f"DuckDuckGo {sorted(options.items())}"], timeout=2, maximum=5,
                          **options)

    (request,) = served["requests"]
    form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
    assert [form.get("kl"), form.get("kp")] == [region, safe_search]


def test_parses_href_first_links_and_highlights_without_mistaking_snippets_for_a_challenge(
    served,
):
    html = ('<a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com" '
            'class="result__a">&#38; Caf<b>é</b></a>\n'
            '    <a class="result__snippet">Fast&nbsp;&hellip; caf<b>é</b></a>\n'
            "    " + result_html("https://example.org/direct", "Coding Challenge",
                                 "A coding challenge."))

    rows = _search(served, html, query="href ordering")

    assert [row["href"] for row in rows] == ["https://example.com", "https://example.org/direct"]
    assert "& Café" in rows[0]["title"]
    assert "Fast ... café" in rows[0]["body"]
    assert "Coding Challenge" in rows[1]["title"]
    assert "A coding challenge." in rows[1]["body"]
    assert "Caf é" not in str(rows)


def test_rejects_bot_challenge_pages(served):
    challenge = '<form>Are you a human?<div class="g-recaptcha">captcha</div></form>'

    with pytest.raises(EngineUnavailable, match="CAPTCHA"):
        _search(served, challenge, query="bot challenge")


def test_repeated_searches_are_served_from_the_cache(served):
    page = result_html("https://example.com/1", "Result 1")

    first = _search(served, page, query="DuckDuckGo current TTL")
    second = _search(served, page, query="DuckDuckGo current TTL")

    assert first == second
    assert len(served["requests"]) == 1


def test_honest_user_agent_and_only_valid_results_with_their_snippets(served):
    html = "".join([
        result_html("https://example.com/empty-title", "", "Discarded empty-title snippet"),
        result_html("https://example.com/first", "First", "First snippet"),
        result_html("", "Missing URL", "Discarded missing-URL snippet"),
        '<a class="result__a" href="https://example.com/second">Second</a>',
        result_html("https://example.com/third", "Third", "Third snippet"),
    ])
    served["response"] = httpx.Response(200, text=html)

    rows = ddg.search_duckduckgo(["DuckDuckGo valid result count"], timeout=2, maximum=2)

    (request,) = served["requests"]
    agent = request.headers["user-agent"]
    assert agent.startswith("ciel-duckduckgo/")
    assert not any(marker in agent for marker in ("Mozilla", "Chrome", "AppleWebKit"))
    assert [(row["href"], row["body"]) for row in rows] == [
        ("https://example.com/first", "First snippet"),
        ("https://example.com/second", ""),
    ]
    assert "Discarded" not in str(rows)


def test_honors_the_result_limit(served):
    html = "".join(result_html(f"https://example.com/{i}", f"Result {i}") for i in (1, 2, 3))
    served["response"] = httpx.Response(200, text=html)

    rows = ddg.search_duckduckgo(["openclaw docs"], timeout=2, maximum=2)

    assert len(rows) == 2


def test_cancels_a_slow_search_at_the_deadline(monkeypatch):
    import time

    release = threading.Event()

    def stalled(*_args, **_kwargs):
        release.wait(5)
        return []

    monkeypatch.setattr(ddg, "_fetch", stalled)
    started = time.monotonic()
    try:
        with pytest.raises(EngineUnavailable, match="timed out"):
            ddg.search_duckduckgo(["1000 deadline"], timeout=0.5, maximum=5)
        assert time.monotonic() - started < 1.5
    finally:
        release.set()


@pytest.mark.parametrize(("entity", "expected"), [
    ("&nbsp;", " "), ("&ndash;", "-"), ("&mdash;", "--"), ("&hellip;", "..."),
    ("&#8230;", "..."), ("&#x27;", "'"), ("&amp;lt;b&amp;gt;", "&lt;b&gt;"),
])
def test_entities_decode_once_with_openclaws_text_forms(served, entity, expected):
    (row,) = _search(served, result_html("https://example.com/", f"a{entity}b"))

    assert row["title"] == f"a{expected}b"
