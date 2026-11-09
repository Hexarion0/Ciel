"""Keyless Yahoo web search through its public results page.

Yahoo's web results come from Bing. It has no free API, so the results page is
parsed directly. Select it with ``INTERNET_PROVIDER=yahoo``; when it blocks the
network, fails, or returns nothing, the search goes to Parallel. Caching and cooldowns live
in ``ciel.scraped_search``.
"""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import unquote, urlsplit

from ciel.scraped_search import EngineBlocked, EngineFailed, ScrapedEngine, fetch_page, text

ENDPOINT = "https://search.yahoo.com/search"
# Each organic result is a div whose class list includes "algo-sr".
_RESULT_START = re.compile(r'<div\b[^>]*\bclass="[^"]*\balgo-sr\b[^"]*"', re.I)
_LINK = re.compile(r'<a\b[^>]*\bhref="([^"]+)"', re.I)
_TITLE = re.compile(r"<h3\b[^>]*>(.*?)</h3>", re.I | re.S)
_BODY = re.compile(r'<div\b[^>]*\bclass="[^"]*\bcompText\b[^"]*"[^>]*>(.*?)</div>', re.I | re.S)
# Bot-check page markers, not bare words a results page could echo back.
_CHALLENGE = re.compile(
    r"unusual traffic from your (?:computer )?network|g-recaptcha|h-captcha|hcaptcha\.com"
    r"|<form\b[^>]*\baction=\"[^\"]*captcha|id=\"captcha", re.I,
)
_SCRIPT = re.compile(r"<script\b.*?</script>", re.I | re.S)
_REDIRECT_HOST = re.compile(r"(?:^|\.)search\.yahoo\.com$")
# Links straight to these hosts are Yahoo or Bing navigation, not results.
_INTERNAL_HOSTS = ("yahoo.com", "yahoo.net", "yimg.com", "bing.com", "bing.net")
# Ad and asset hosts are dropped even when a redirect points at them.
_ALWAYS_SKIPPED = ("yimg.com", "bing.net")


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").removeprefix("www.")


def _matches(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _target(href: str) -> tuple[str, bool]:
    """Unwrap Yahoo's ``r.search.yahoo.com/.../RU=<url>/RK=...`` redirect.

    Returns (url, unwrapped). ``unquote``, not ``unquote_plus``: a literal ``+``
    in the target is percent-encoded, so a bare ``+`` must stay a plus.
    """
    href = unescape(href)
    if "/RU=" in href and _REDIRECT_HOST.search(_host(href)):
        target = href.split("/RU=", 1)[1].split("/RK=", 1)[0].split("/RS=", 1)[0]
        # Yahoo sometimes leaves a trailing non-breaking space (%C2%A0) in targets.
        return unquote(target).strip(), True
    return href, False


def is_challenge(html: str, status: int = 200) -> bool:
    """Report a bot-check page (or Yahoo's 999 rate-limit status)."""
    if status == 999:
        return True
    return not _RESULT_START.search(html) and bool(_CHALLENGE.search(_SCRIPT.sub("", html)))


def parse_results(html: str) -> list[dict[str, str]]:
    """Extract organic results; ads and Yahoo/Bing-internal links are skipped."""
    starts = [match.start() for match in _RESULT_START.finditer(html)]
    results: list[dict[str, str]] = []
    for index, start in enumerate(starts):
        block = html[start:starts[index + 1] if index + 1 < len(starts) else len(html)]
        link = _LINK.search(block)
        title = _TITLE.search(block)
        if not link or not title:
            continue
        try:
            url, unwrapped = _target(link.group(1))
            host = _host(url)
        except ValueError:
            continue  # A malformed address in one row must not discard the page.
        if not url.startswith(("http://", "https://")):
            continue
        # Redirected results may be real Yahoo pages (finance, news); Bing ad
        # clicks and asset hosts never are.
        if _matches(host, _ALWAYS_SKIPPED if unwrapped else _INTERNAL_HOSTS):
            continue
        if _matches(host, ("bing.com",)) and urlsplit(url).path.startswith("/aclick"):
            continue
        body = _BODY.search(block)
        if heading := text(title.group(1)):
            results.append({"title": heading, "href": url,
                            "body": text(body.group(1)) if body else ""})
    return results


def _fetch(query: str, freshness: str, timeout: float) -> list[dict[str, str]]:
    params = {"p": query}
    if freshness:
        params["btf"] = freshness
    status, html = fetch_page("Yahoo", ENDPOINT, params, timeout)
    if is_challenge(html, status):
        raise EngineBlocked("Yahoo asked this network to solve a CAPTCHA")
    if status != 200:
        # Includes redirects to a consent page, which needs a browser.
        raise EngineFailed(f"Yahoo search returned HTTP {status}")
    return parse_results(html)


# The lambda resolves _fetch at call time so tests can replace it.
_ENGINE = ScrapedEngine("Yahoo", lambda query, date, timeout: _fetch(query, date, timeout))
blocked = _ENGINE.blocked
reset = _ENGINE.reset
search_yahoo = _ENGINE.search
