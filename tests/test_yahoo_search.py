"""Keyless Yahoo search: result parsing, redirects, ads, and blocked pages."""

import httpx
import pytest

from ciel import yahoo_search as yahoo
from ciel.scraped_search import EngineUnavailable

_REDIRECT = ("https://r.search.yahoo.com/_ylt=A2R;_ylu=Y29s/RV=2/RE=1795344030/RO=10/"
             "RU={target}/RK=2/RS=2gmVdGKBx-")
_WEATHER = _REDIRECT.format(
    target="https%3a%2f%2fwww.weather.example%2fkhovd%3fdays%3d7%26units%3dc",
)
_AD = _REDIRECT.format(target="https%3a%2f%2fwww.bing.com%2faclick%3fld%3dx")
RESULTS_PAGE = f"""
<ol><li><div class="dd fst algo algo-sr relsrch Sr" data-918="65fc">
  <div class="compTitle options-toggle"><a class="d-ib va-top" target="_blank"
     href="{_WEATHER}">
    <div class="d-ib p-abs"><span class="fc-141414 d-b">weather.example</span>
      https://www.weather.example › khovd</div>
    <h3 style="display:block" class="title"><span class="d-b fz-20">Weather Khovd - 7 day
      &amp; hourly</span></h3></a></div>
  <div class="compText aAbs"><p class="fc-dustygray"> Today's <b>weather</b> forecast
    for Khovd. </p></div>
</div></li>
<li><div class="dd algo algo-sr relsrch Sr">
  <div class="compTitle"><a href="{_AD}">
    <h3 class="title">Sponsored weather app</h3></a></div>
</div></li>
<li><div class="dd algo algo-sr relsrch Sr">
  <div class="compTitle"><a href="https://news.yahoo.com/weather"><h3>Yahoo Weather</h3></a></div>
</div></li>
<li><div class="dd algo algo-sr relsrch Sr">
  <div class="compTitle"><a href="https://example.net/no-heading">No heading here</a></div>
</div></li>
<li><div class="dd algo algo-sr relsrch Sr">
  <div class="compTitle"><a href="https://docs.example.org/guide">
    <h3 class="title">Plain link guide</h3></a></div>
</div></li>
<li><div class="dd AlsoTry_M"><div class="compTitle"><h2 class="title">Searches related
  to khovd weather</h2></div></div></li></ol>
"""


@pytest.fixture(autouse=True)
def _fresh_yahoo_state():
    yahoo.reset()
    yield
    yahoo.reset()


_REAL_CLIENT = httpx.Client


def _serve(monkeypatch, handler):
    monkeypatch.setattr("ciel.scraped_search.httpx.Client", lambda **kwargs: _REAL_CLIENT(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


def test_parse_results_unwraps_redirects_and_skips_ads_and_internal_links():
    rows = yahoo.parse_results(RESULTS_PAGE)

    assert rows == [
        {"title": "Weather Khovd - 7 day & hourly",
         "href": "https://www.weather.example/khovd?days=7&units=c",
         "body": "Today's weather forecast for Khovd."},
        {"title": "Plain link guide", "href": "https://docs.example.org/guide", "body": ""},
    ]


def test_challenge_detection_needs_real_bot_check_markers():
    assert yahoo.is_challenge(
        "<p>Our systems have detected unusual traffic from your computer network.</p>")
    assert yahoo.is_challenge('<form action="/captcha/verify"><div class="g-recaptcha">')
    assert yahoo.is_challenge("<p>Unable to process request</p>", status=999)
    assert not yahoo.is_challenge(RESULTS_PAGE + '<div class="g-recaptcha">')
    # A no-results page that echoes the query, or a script mentioning captcha.
    assert not yahoo.is_challenge("<p>No results for: how to solve captcha not a robot</p>")
    assert not yahoo.is_challenge('<script>var s="g-recaptcha";</script><p>No results</p>')


def test_redirected_yahoo_pages_stay_but_ads_and_assets_go():
    def result(href):
        return (f'<div class="dd algo algo-sr"><div class="compTitle"><a href="{href}">'
                f"<h3>Title for {href[-20:]}</h3></a></div></div>")

    page = "".join(result(href) for href in (
        _REDIRECT.format(target="https%3a%2f%2ffinance.yahoo.com%2fquote%2fNVDA%C2%A0"),
        _REDIRECT.format(target="https%3a%2f%2fwww.bing.com.%2faclick%3fld%3dx"),
        _REDIRECT.format(target="https%3a%2f%2fs.yimg.com%2fpixel"),
        _REDIRECT.format(target="https%3a%2f%2fads.bing.net%2fclick"),
        _REDIRECT.format(target="https%3a%2f%2fshop.example%2fc%2b%2b%3fq%3da+b"),
        "https://example.org/RU=https%3a%2f%2fevil.example/RK=2",
    ))

    assert [row["href"] for row in yahoo.parse_results(page)] == [
        "https://finance.yahoo.com/quote/NVDA",
        "https://shop.example/c++?q=a+b",
        "https://example.org/RU=https%3a%2f%2fevil.example/RK=2",
    ]


def test_search_sends_query_and_news_date_filter(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text=RESULTS_PAGE)

    _serve(monkeypatch, handler)
    rows = yahoo.search_yahoo(["khovd weather this month"], timeout=2, maximum=5, news=True)

    assert rows[0]["href"] == "https://www.weather.example/khovd?days=7&units=c"
    (request,) = seen
    assert request.url.host == "search.yahoo.com"
    assert request.url.params["p"] == "khovd weather this month"
    assert request.url.params["btf"] == "m"
    assert "Mozilla/5.0" in request.headers["user-agent"]


def test_bot_check_starts_cooldown_without_further_requests(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text='<form action="/captcha"><div class="g-recaptcha">')

    _serve(monkeypatch, handler)
    with pytest.raises(EngineUnavailable, match="CAPTCHA"):
        yahoo.search_yahoo(["rtx 5090 price"], timeout=2, maximum=5)
    with pytest.raises(EngineUnavailable, match="cooling down"):
        yahoo.search_yahoo(["another"], timeout=2, maximum=5)

    assert len(calls) == 1


def test_consent_redirect_is_a_failed_request(monkeypatch):
    _serve(monkeypatch, lambda _request: httpx.Response(
        302, headers={"Location": "https://consent.yahoo.com/v2/collectConsent"},
    ))

    with pytest.raises(EngineUnavailable, match="HTTP 302"):
        yahoo.search_yahoo(["rtx 5090 price"], timeout=2, maximum=5)
    assert yahoo.blocked()


def test_a_late_failure_does_not_shorten_a_captcha_cooldown(monkeypatch):
    from ciel import scraped_search

    engine = scraped_search.ScrapedEngine("Test", lambda *_args: [])
    engine._cool_down(scraped_search.BLOCK_COOLDOWN_SECONDS, "a CAPTCHA")
    engine._cool_down(scraped_search.FAILURE_COOLDOWN_SECONDS, "a failed request")

    with pytest.raises(EngineUnavailable, match="after a CAPTCHA"):
        engine.search(["q"], timeout=1, maximum=5)


def test_a_stalled_engine_cannot_outlast_its_budget():
    import threading
    import time

    from ciel import scraped_search

    release = threading.Event()

    def stalled(_query, _date, _timeout):
        release.wait(5)
        return [{"title": "late", "href": "https://late.example/", "body": ""}]

    engine = scraped_search.ScrapedEngine("Slow", stalled)
    started = time.monotonic()
    try:
        with pytest.raises(EngineUnavailable, match="timed out"):
            engine.search(["q"], timeout=0.5, maximum=5)
        assert time.monotonic() - started < 1.5
        # Half a second is the caller's leftover budget, not a sign the engine is down.
        assert not engine.blocked()
    finally:
        release.set()


def test_case_variants_are_sent_once():
    from unittest.mock import Mock

    from ciel import scraped_search

    fetch = Mock(return_value=[{"title": "A", "href": "https://a.example/", "body": ""}])
    engine = scraped_search.ScrapedEngine("Test", fetch)

    engine.search(["RTX 5090 price", "rtx 5090 price", "rtx  5090 PRICE"], timeout=1, maximum=5)

    assert fetch.call_count == 1
