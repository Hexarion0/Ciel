"""Keyless DuckDuckGo search, modeled on OpenClaw's DuckDuckGo web_search provider."""

import httpx
import pytest

from ciel import duckduckgo_search as ddg
from ciel.scraped_search import EngineUnavailable

_REAL_CLIENT = httpx.Client


def _result(href, title, snippet=None, *, ad=False):
    """One result block in the shape html.duckduckgo.com serves."""
    classes = "result results_links results_links_deep web-result" + (" result--ad" if ad else "")
    snippet_html = (f'<a class="result__snippet" href="{href}">{snippet}</a>'
                    if snippet is not None else "")
    return (f'<div class="{classes}"><div class="links_main links_deep result__body">'
            f'<h2 class="result__title"><a rel="nofollow" class="result__a" href="{href}">'
            f"{title}</a></h2>{snippet_html}</div></div>")


def _wrapped(target):
    """DuckDuckGo's protocol-relative redirect link, HTML-escaped as in the page."""
    from urllib.parse import quote

    return f"//duckduckgo.com/l/?uddg={quote(target, safe='')}&amp;rut=2f32e697"


RESULTS_PAGE = "".join([
    _result(_wrapped("https://www.python.org/downloads/release/python-3140/"),
            "Python Release Python 3.14.0 | Python.org",
            "For more details on the changes to <b>Python</b> <b>3.14</b>, see What&#x27;s "
            "new in Caf<b>é</b>&nbsp;&hellip;"),
    # Ads point at DuckDuckGo's own click tracker, never a uddg target.
    _result("https://duckduckgo.com/y.js?ad_domain=shop.example&amp;u3=https%3A%2F%2Fx",
            "Sponsored Python course", "Buy now.", ad=True),
    # A result with no snippet must not borrow the next result's snippet.
    _result("https://docs.python.org/3.14/whatsnew/3.14.html",
            "What&#x27;s new in Python 3.14"),
    _result(_wrapped("https://example.org/coding-challenge?id=1&region=eu"),
            "Coding Challenge &amp; Puzzles", "A coding challenge."),
    _result(_wrapped("https://example.com/empty-title"), "", "Discarded empty-title snippet"),
    _result("", "Missing URL", "Discarded missing-URL snippet"),
])

CHALLENGE_PAGE = """<!DOCTYPE html><html><head><title>DuckDuckGo</title></head><body>
<form id="img-form" action="//duckduckgo.com/anomaly.js?sv=html&cc=botnet" method="POST"></form>
<form id="challenge-form" action="//duckduckgo.com/anomaly.js?sv=html&cc=botnet"></form>
</body></html>"""


@pytest.fixture(autouse=True)
def _fresh_duckduckgo_state():
    ddg.reset()
    yield
    ddg.reset()


def _serve(monkeypatch, handler):
    monkeypatch.setattr("ciel.scraped_search.httpx.Client", lambda **kwargs: _REAL_CLIENT(
        **{**kwargs, "transport": httpx.MockTransport(handler)},
    ))


def _form(request):
    from urllib.parse import parse_qs

    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


def test_parse_results_unwraps_redirects_and_cleans_text():
    rows = ddg.parse_results(RESULTS_PAGE)

    assert rows == [
        {"title": "Python Release Python 3.14.0 | Python.org",
         "href": "https://www.python.org/downloads/release/python-3140/",
         "body": "For more details on the changes to Python 3.14, see What's new in Café ..."},
        {"title": "What's new in Python 3.14",
         "href": "https://docs.python.org/3.14/whatsnew/3.14.html", "body": ""},
        {"title": "Coding Challenge & Puzzles",
         "href": "https://example.org/coding-challenge?id=1&region=eu",
         "body": "A coding challenge."},
    ]


def test_match_highlights_inside_words_do_not_split_them():
    page = _result("https://example.com/", "Caf<b>é</b> guide", "caf<b>é</b> au lait")

    (row,) = ddg.parse_results(page)

    assert (row["title"], row["body"]) == ("Café guide", "café au lait")


def test_href_before_class_and_non_web_targets_are_handled():
    page = (
        '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffirst.example%2F" class="result__a">'
        "First</a>"
        '<a class="result__a" href="javascript:alert(1)">Script</a>'
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=ftp%3A%2F%2Ffiles.example">Ftp</a>'
        '<a class="result__a" href="https://duckduckgo.com/about">About DuckDuckGo</a>'
    )

    assert [row["href"] for row in ddg.parse_results(page)] == ["https://first.example/"]


def test_challenge_detection_needs_real_bot_check_markers():
    assert ddg.is_challenge(CHALLENGE_PAGE, status=202)
    assert ddg.is_challenge(CHALLENGE_PAGE)
    assert ddg.is_challenge('<form>Are you a human?<div class="g-recaptcha"></div></form>')
    # DuckDuckGo answers a challenged request with 202 and no results.
    assert ddg.is_challenge("<html><body></body></html>", status=202)
    # Result text that mentions a challenge or a CAPTCHA is not a bot check.
    assert not ddg.is_challenge(RESULTS_PAGE + '<div class="g-recaptcha">')
    assert not ddg.is_challenge("<p>No results for: are you a human captcha</p>")


def test_search_posts_the_query_with_an_honest_user_agent(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    rows = ddg.search_duckduckgo(["python 3.14 release"], timeout=2, maximum=5)

    assert rows[0]["href"] == "https://www.python.org/downloads/release/python-3140/"
    (request,) = seen
    assert request.method == "POST"
    assert (request.url.host, request.url.path) == ("html.duckduckgo.com", "/html/")
    assert _form(request) == {"q": "python 3.14 release", "kp": "-1"}
    agent = request.headers["user-agent"]
    assert agent.startswith("ciel-duckduckgo/")
    assert not any(marker in agent for marker in ("Mozilla", "Chrome", "AppleWebKit"))


@pytest.mark.parametrize(("region", "safe_search", "expected"), [
    ("de-de", "moderate", {"kl": "de-de", "kp": "-1"}),
    ("   ", "strict", {"kp": "1"}),
    ("", "off", {"kp": "-2"}),
    ("tw-tzh", "moderate", {"kl": "tw-tzh", "kp": "-1"}),
])
def test_region_and_safe_search_are_sent(monkeypatch, region, safe_search, expected):
    seen = []

    def handler(request):
        seen.append(_form(request))
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    ddg.search_duckduckgo(["python"], timeout=2, maximum=5,
                          region=region, safe_search=safe_search)

    assert seen == [{"q": "python", **expected}]


def test_news_searches_send_a_date_filter(monkeypatch):
    seen = []

    def handler(request):
        seen.append(_form(request))
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    ddg.search_duckduckgo(["world cup news today"], timeout=2, maximum=5, news=True)

    assert seen[0]["df"] == "d"


def test_one_request_per_search_like_openclaw(monkeypatch):
    seen = []

    def handler(request):
        seen.append(_form(request)["q"])
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    ddg.search_duckduckgo(["first phrasing", "second phrasing", "third"],
                          timeout=2, maximum=5)

    assert seen == ["first phrasing"]


def test_results_are_cached_per_query_and_options(monkeypatch):
    seen = []

    def handler(request):
        seen.append(_form(request))
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    ddg.search_duckduckgo(["Python"], timeout=2, maximum=5)
    ddg.search_duckduckgo(["python"], timeout=2, maximum=5)
    ddg.search_duckduckgo(["python"], timeout=2, maximum=5, region="uk-en")

    assert [form.get("kl") for form in seen] == [None, "uk-en"]


def test_maximum_limits_returned_rows(monkeypatch):
    _serve(monkeypatch, lambda _request: httpx.Response(200, text=RESULTS_PAGE))

    rows = ddg.search_duckduckgo(["python"], timeout=2, maximum=2)

    assert len(rows) == 2


def test_bot_check_starts_cooldown_without_further_requests(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(202, text=CHALLENGE_PAGE)

    _serve(monkeypatch, handler)
    with pytest.raises(EngineUnavailable, match="CAPTCHA"):
        ddg.search_duckduckgo(["rtx 5090 price"], timeout=2, maximum=5)
    with pytest.raises(EngineUnavailable, match="cooling down"):
        ddg.search_duckduckgo(["another"], timeout=2, maximum=5)

    assert len(calls) == 1
    assert ddg.blocked()


def test_http_errors_are_failed_requests(monkeypatch):
    _serve(monkeypatch, lambda _request: httpx.Response(429, text="rate limited"))

    with pytest.raises(EngineUnavailable, match="HTTP 429"):
        ddg.search_duckduckgo(["rtx 5090 price"], timeout=2, maximum=5)
    assert ddg.blocked()


def test_no_results_page_is_unavailable_without_a_cooldown(monkeypatch):
    _serve(monkeypatch, lambda _request: httpx.Response(
        200, text='<div class="no-results">No results.</div>',
    ))

    with pytest.raises(EngineUnavailable, match="no results"):
        ddg.search_duckduckgo(["zxqv nonsense"], timeout=2, maximum=5)
    assert not ddg.blocked()
