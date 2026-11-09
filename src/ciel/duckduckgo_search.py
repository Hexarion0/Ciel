"""Keyless DuckDuckGo web search through its non-JavaScript HTML results page.

A port of OpenClaw's DuckDuckGo ``web_search`` provider
(``extensions/duckduckgo/src/ddg-client.ts``); ``tests/test_duckduckgo_openclaw_parity.py``
runs OpenClaw's own test cases against it. Shared with OpenClaw: one request per
search, an honest User-Agent rather than a browser imitation, ``result__a`` links
with the ``result__snippet`` before the next result, ``uddg`` redirects unwrapped,
match highlights joined inside words, entities decoded once with OpenClaw's text
forms, region (``kl``) and SafeSearch (``kp``, moderate by default), bot-check pages
reported instead of parsed, a size cap, a deadline, and a 15-minute result cache.

Intentional differences:

- The query is POSTed as DuckDuckGo's own form does. OpenClaw sends a GET, which
  DuckDuckGo challenged for new queries from networks where POST still worked.
- News queries add a ``df`` date filter (day, week, or month).
- Links to DuckDuckGo itself, which include its ``/y.js`` ad clicks, are skipped.
- A block or failure starts a cooldown, and ``ciel.internet`` then sends the search
  to Parallel; OpenClaw only reports the error.

Caching and cooldowns live in ``ciel.scraped_search``.
"""

from __future__ import annotations

import re
import threading
from html import unescape
from urllib.parse import parse_qs, urlsplit

from ciel import __version__
from ciel.scraped_search import EngineBlocked, EngineFailed, ScrapedEngine, fetch_page

ENDPOINT = "https://html.duckduckgo.com/html/"
# Identify Ciel rather than impersonating a browser, as OpenClaw does.
USER_AGENT = f"ciel-duckduckgo/{__version__} (+https://github.com/Hexarion0/Ciel)"
SAFE_SEARCH_LEVELS = ("strict", "moderate", "off")
_SAFE_SEARCH_PARAM = {"strict": "1", "moderate": "-1", "off": "-2"}
# Region codes look like us-en, de-de, tw-tzh, or wt-wt (no region).
REGION_PATTERN = re.compile(r"[a-z]{2}-[a-z]{2,3}")

_RESULT_LINK = re.compile(
    r'<a\b(?=[^>]*\bclass="[^"]*\bresult__a\b[^"]*")([^>]*)>(.*?)</a>', re.I | re.S,
)
_NEXT_RESULT = re.compile(r'<a\b(?=[^>]*\bclass="[^"]*\bresult__a\b[^"]*")[^>]*>', re.I)
_SNIPPET = re.compile(
    r'<a\b(?=[^>]*\bclass="[^"]*\bresult__snippet\b[^"]*")[^>]*>(.*?)</a>', re.I | re.S,
)
_HREF = re.compile(r'\bhref="([^"]*)"', re.I)
# Match highlights can sit inside a word ("Caf<b>é</b>"); drop them without a space.
_HIGHLIGHT = re.compile(r"</?b\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:[a-z][a-z0-9]*|#\d+|#x[0-9a-f]+);", re.I)
# OpenClaw's plain-text forms for these decoded characters.
_ENTITY_TEXT = {"\u00a0": " ", "\u2013": "-", "\u2014": "--", "\u2026": "..."}
# Bot-check markup, not bare words a results page could echo back from the query.
_CHALLENGE = re.compile(
    r'\bclass="[^"]*\bg-recaptcha\b|\bid="challenge-form"|\bname="challenge"'
    r'|\baction="[^"]*anomaly\.js', re.I,
)
_SCRIPT = re.compile(r"<script\b.*?</script>", re.I | re.S)
_INTERNAL_HOST = "duckduckgo.com"


def _decode(match: re.Match[str]) -> str:
    decoded = unescape(match.group(0))
    return _ENTITY_TEXT.get(decoded, decoded)


def _clean(fragment: str) -> str:
    """Visible text: highlights joined, tags spaced, entities decoded exactly once."""
    stripped = _TAG.sub(" ", _HIGHLIGHT.sub("", fragment))
    return " ".join(_ENTITY.sub(_decode, stripped).split())


def _target(href: str) -> str:
    """Resolve a result link to its destination, unwrapping ``/l/?uddg=`` redirects.

    Returns an empty string for non-web links and DuckDuckGo's own pages, which
    include its ad click tracker (``/y.js``).
    """
    url = unescape(href).strip()
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".")
        if host == _INTERNAL_HOST or host.endswith("." + _INTERNAL_HOST):
            uddg = parse_qs(parts.query).get("uddg")
            if not uddg:
                return ""
            url = uddg[0].strip()
            parts = urlsplit(url)
            host = (parts.hostname or "").rstrip(".")
        if parts.scheme not in {"http", "https"} or not host:
            return ""
        if host == _INTERNAL_HOST or host.endswith("." + _INTERNAL_HOST):
            return ""
    except ValueError:
        return ""  # A malformed address in one row must not discard the page.
    return url


def is_challenge(html: str, status: int = 200) -> bool:
    """Report a bot-check page; DuckDuckGo serves them with HTTP 202 and no results."""
    if _RESULT_LINK.search(html):
        return False
    return status == 202 or bool(_CHALLENGE.search(_SCRIPT.sub("", html)))


def parse_results(html: str) -> list[dict[str, str]]:
    """Extract organic results; each snippet is read only up to the next result."""
    results: list[dict[str, str]] = []
    for match in _RESULT_LINK.finditer(html):
        href = _HREF.search(match.group(1))
        url = _target(href.group(1)) if href else ""
        title = _clean(match.group(2))
        if not url or not title:
            continue
        trailing = html[match.end():]
        following = _NEXT_RESULT.search(trailing)
        scope = trailing[:following.start()] if following else trailing
        snippet = _SNIPPET.search(scope)
        results.append({"title": title, "href": url,
                        "body": _clean(snippet.group(1)) if snippet else ""})
    return results


def _fetch(
    query: str, freshness: str, timeout: float, *,
    region: str = "", safe_search: str = "moderate",
) -> list[dict[str, str]]:
    params = {"q": query}
    if region:
        params["kl"] = region
    params["kp"] = _SAFE_SEARCH_PARAM.get(safe_search, _SAFE_SEARCH_PARAM["moderate"])
    if freshness:
        params["df"] = freshness
    status, html = fetch_page("DuckDuckGo", ENDPOINT, params, timeout,
                              method="POST", user_agent=USER_AGENT)
    if is_challenge(html, status):
        raise EngineBlocked("DuckDuckGo asked this network to solve a CAPTCHA")
    if status != 200:
        raise EngineFailed(f"DuckDuckGo search returned HTTP {status}")
    return parse_results(html)


# One request per search, as in OpenClaw; parallel phrasings invite bot checks.
# The lambda resolves _fetch at call time so tests can replace it.
_ENGINE = ScrapedEngine(
    "DuckDuckGo", lambda query, date, timeout, **options: _fetch(query, date, timeout, **options),
    max_queries=1,
)
blocked = _ENGINE.blocked
reset = _ENGINE.reset


def search_duckduckgo(
    queries: list[str], *, timeout: float, maximum: int, news: bool = False,
    cancel_event: threading.Event | None = None, region: str = "", safe_search: str = "moderate",
) -> list[dict[str, str]]:
    """Search the first phrasing; raises ``EngineUnavailable`` so the caller can fall back."""
    region = region.strip().lower()
    options = {"safe_search": safe_search if safe_search in _SAFE_SEARCH_PARAM else "moderate"}
    if REGION_PATTERN.fullmatch(region):
        options["region"] = region
    return _ENGINE.search(queries, timeout=timeout, maximum=maximum, news=news,
                          cancel_event=cancel_event, options=options)
