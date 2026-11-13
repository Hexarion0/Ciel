"""Shared machinery for keyless search engines read through their HTML results pages.

Each engine supplies a ``fetch(query, freshness, timeout, **options)`` that returns
result rows (``title``, ``href``, ``body``) or raises one of the errors below. ``ScrapedEngine``
adds a per-query cache, a cooldown after a CAPTCHA or failure (so later turns skip
a blocked engine instantly), and interleaving of several phrasings by rank.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for
from html import unescape

import httpx

from ciel.providers.streaming import pooled_client
from ciel.search_providers import SearchProviderError

CACHE_TTL_SECONDS = 15 * 60
BLOCK_COOLDOWN_SECONDS = 5 * 60
# Timeouts, connection errors, and HTTP failures skip an engine briefly so a down or
# blackholed endpoint does not delay every search before the fallback.
FAILURE_COOLDOWN_SECONDS = 60
# A timeout under this budget says more about the caller's leftover time than the
# engine, so it does not start the shared cooldown for every session.
SHORT_BUDGET_SECONDS = 2.5
_CANCEL_POLL_SECONDS = 0.1
MAX_QUERIES = 3
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_CACHE_ENTRIES = 128
# Results pages serve a stripped page or a challenge to non-browser agents.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
_TAG = re.compile(r"<[^>]+>")

Rows = list[dict[str, str]]


class EngineUnavailable(SearchProviderError):
    """The engine blocked, failed, or returned nothing; the caller may fall back."""


class EngineBlocked(EngineUnavailable):
    """A CAPTCHA or bot-check page; starts the long cooldown."""


class EngineFailed(EngineUnavailable):
    """A timeout or HTTP failure; starts the short cooldown."""


class EngineTimedOut(EngineFailed):
    """The request outlasted its budget."""


def text(fragment: str) -> str:
    """Visible text of an HTML fragment, whitespace collapsed."""
    return " ".join(unescape(_TAG.sub(" ", fragment)).split())


def freshness(query: str, news: bool) -> str:
    """Day, week, or month date filter for news searches; empty otherwise."""
    if not news:
        return ""
    if re.search(r"\b(?:today|tonight)\b", query, re.I):
        return "d"
    if re.search(r"\bmonth\b", query, re.I):
        return "m"
    return "w"


def fetch_page(
    name: str, url: str, params: dict[str, str], timeout: float, *,
    method: str = "GET", user_agent: str = USER_AGENT,
) -> tuple[int, str]:
    """Fetch a results page with a size cap and a wall-clock deadline; no redirects.

    ``params`` go in the query string for GET and in a form body for POST.
    """
    # httpx timeouts limit each phase; the deadline also bounds a slow trickle. A slow
    # first byte (over 2s from some networks) needs most of the budget on its own.
    deadline = time.monotonic() + timeout
    with pooled_client(timeout * 0.75, follow_redirects=False,
                      trust_env=False) as client:
        with client.stream(
            method, url,
            **({"data": params} if method == "POST" else {"params": params}),
            headers={"User-Agent": user_agent, "Accept": "text/html",
                     "Accept-Language": "en-US,en;q=0.8"},
        ) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise EngineUnavailable(f"{name} returned an oversized page")
                if time.monotonic() > deadline:
                    raise EngineTimedOut(f"{name} search timed out")
            return response.status_code, body.decode(response.encoding or "utf-8",
                                                      errors="replace")


class ScrapedEngine:
    """Cache, cooldown, and multi-phrasing search around one engine's fetch."""

    def __init__(
        self, name: str, fetch: Callable[..., Rows], *, max_queries: int = MAX_QUERIES,
    ) -> None:
        self.name = name
        self._fetch = fetch
        self._max_queries = max(1, min(max_queries, MAX_QUERIES))
        self._lock = threading.Lock()
        self._cache: dict[tuple, tuple[float, Rows]] = {}
        self._blocked_until = 0.0
        self._block_reason = ""

    def blocked(self) -> bool:
        """Report whether a recent CAPTCHA or failure means requests should be skipped."""
        with self._lock:
            return time.monotonic() < self._blocked_until

    def reset(self) -> None:
        """Forget cached results and any cooldown (tests and manual retries)."""
        with self._lock:
            self._cache.clear()
            self._blocked_until = 0.0
            self._block_reason = ""

    def _cool_down(self, seconds: float, reason: str) -> None:
        with self._lock:
            until = time.monotonic() + seconds
            # A late failure from another session must not shorten a CAPTCHA cooldown.
            if until > self._blocked_until:
                self._blocked_until, self._block_reason = until, reason

    def _cooling_reason(self) -> str:
        with self._lock:
            return self._block_reason if time.monotonic() < self._blocked_until else ""

    @staticmethod
    def _key(query: str, date_filter: str, options: dict[str, str]) -> tuple:
        return query.casefold(), date_filter, tuple(sorted(options.items()))

    def _lookup(self, key: tuple) -> Rows | None:
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < CACHE_TTL_SECONDS:
                return list(hit[1])
        return None

    def _cached(
        self, query: str, date_filter: str, timeout: float, options: dict[str, str],
    ) -> Rows:
        key = self._key(query, date_filter, options)
        if (hit := self._lookup(key)) is not None:
            return hit
        now = time.monotonic()
        rows = self._fetch(query, date_filter, timeout, **options)
        if not rows:
            return rows  # A transient empty page should not pin the query to the fallback.
        with self._lock:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                self._cache.pop(min(self._cache, key=lambda item: self._cache[item][0]))
            self._cache[key] = (now, rows)
        return list(rows)

    def search(
        self, queries: list[str], *, timeout: float, maximum: int, news: bool = False,
        cancel_event: threading.Event | None = None, options: dict[str, str] | None = None,
    ) -> Rows:
        """Search up to ``max_queries`` phrasings at once and interleave results by rank.

        ``options`` reach the engine's fetch and its cache key (region, safe search).
        Raises ``EngineUnavailable`` when the engine is cooling down, every phrasing
        failed, or no organic results came back.
        """
        name = self.name
        cleaned = [" ".join(q.split()) for q in queries if isinstance(q, str)]
        # Phrasings differing only in case share a cache entry; send each once.
        unique: dict[str, str] = {}
        for phrasing in cleaned:
            if phrasing:
                unique.setdefault(phrasing.casefold(), phrasing)
        phrasings = list(unique.values())[:self._max_queries]
        if not phrasings:
            raise EngineUnavailable(f"No {name} query was supplied")
        options = dict(options or {})
        if reason := self._cooling_reason():
            # Results fetched before the block are still good while it lasts.
            cached = [rows for rows in (self._lookup(self._key(q, freshness(q, news), options))
                                        for q in phrasings) if rows]
            if cached:
                return _merge(cached, maximum)
            raise EngineUnavailable(f"{name} is cooling down after {reason}")
        if cancel_event is not None and cancel_event.is_set():
            raise TimeoutError("Web request canceled")
        timeout = max(0.5, timeout)
        pool = ThreadPoolExecutor(max_workers=len(phrasings))
        try:
            futures = [pool.submit(self._cached, q, freshness(q, news), timeout, options)
                       for q in phrasings]
            # Hard wall-clock bound: a stalled connect or read cannot outlast the
            # budget, and a cancel ends the wait. Unfinished requests end on their
            # own httpx timeouts.
            deadline = time.monotonic() + timeout
            pending = set(futures)
            while pending and (left := deadline - time.monotonic()) > 0:
                if cancel_event is not None and cancel_event.is_set():
                    raise TimeoutError("Web request canceled")
                _done, pending = wait_for(pending, timeout=min(left, _CANCEL_POLL_SECONDS))
            outcomes: list[Rows | Exception] = []
            for future in futures:
                if not future.done():
                    outcomes.append(EngineTimedOut(f"{name} search timed out"))
                    continue
                try:
                    outcomes.append(future.result())
                except (httpx.HTTPError, OSError, ValueError, SearchProviderError) as error:
                    outcomes.append(error)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        batches = [item for item in outcomes if isinstance(item, list)]
        if not batches:
            if any(isinstance(item, EngineBlocked) for item in outcomes):
                self._cool_down(BLOCK_COOLDOWN_SECONDS, "a CAPTCHA")
                raise EngineUnavailable(f"{name} asked this network to solve a CAPTCHA")
            reason = next((str(item) for item in outcomes
                           if isinstance(item, SearchProviderError)), "")
            budget_ran_out = timeout < SHORT_BUDGET_SECONDS and all(
                isinstance(item, (EngineTimedOut, httpx.TimeoutException)) for item in outcomes)
            if (not reason or any(isinstance(item, EngineFailed) for item in outcomes)) \
                    and not budget_ran_out:
                # Network errors (httpx, OSError) and HTTP failures, not a malformed page.
                self._cool_down(FAILURE_COOLDOWN_SECONDS, "a failed request")
            if not reason:
                kinds = sorted({type(item).__name__ for item in outcomes})
                reason = f"{name} search failed ({', '.join(kinds)})"
            raise EngineUnavailable(reason)
        merged = _merge(batches, maximum)
        if not merged:
            raise EngineUnavailable(f"{name} returned no results")
        return merged


def _merge(batches: list[Rows], maximum: int) -> Rows:
    """Interleave result batches by rank, keeping each address once."""
    merged: dict[str, dict[str, str]] = {}
    for rank in range(max(len(batch) for batch in batches)):
        for batch in batches:
            if rank < len(batch):
                merged.setdefault(batch[rank]["href"], batch[rank])
    return list(merged.values())[:maximum]
