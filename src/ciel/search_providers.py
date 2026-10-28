"""Explicit optional search APIs; saved credentials never enable a fallback."""

import math
from typing import Any
from urllib.parse import urlsplit

import httpx

from ciel.config import InternetConfig
from ciel.logging import SecretMaskingFilter

_BRAVE_WEB_URL = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
_TAVILY_URL = "https://api.tavily.com/search"


class SearchProviderError(RuntimeError):
    """A safe diagnostic which contains no provider response, query, or secret."""


def _text(value: Any, limit: int = 8000) -> str:
    """Retain textual result data without stringifying arbitrary response objects."""
    return SecretMaskingFilter._redact(value).strip()[:limit] if isinstance(value, str) else ""


def _normalise(row: dict, *, provider: str) -> dict[str, str] | None:
    """Project each provider's result onto the free-search result contract."""
    url = _text(row.get("url"), 2048)
    if "[REDACTED_SECRET]" in url:
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    profile = row.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    return {
        "href": url,
        "title": _text(row.get("title"), 500),
        "body": _text(row.get("description") if provider == "brave" else row.get("content")),
        "date": _text(row.get("page_age") or row.get("published_date"), 100),
        "date_kind": "published_or_modified",
        "publisher": _text(row.get("publisher") or profile.get("long_name"), 200),
    }


def search_api(
    config: InternetConfig, query: str, *, news: bool, timeout: float, maximum: int,
) -> list[dict[str, str]]:
    """Search only the selected Brave/Tavily API, with one bounded request and no retry.

    Free DDGS retrieval belongs to the caller. Fixed endpoints, disabled redirects, and
    disabled environment proxies keep credentials at the selected provider. Exceptions
    expose only a static category or HTTP status, never request/response text.
    """
    if config.provider not in {"brave", "tavily"}:
        raise SearchProviderError("Free search must use the selected free search path.")
    name = "Brave Search" if config.provider == "brave" else "Tavily"
    secret = config.brave_api_key if config.provider == "brave" else config.tavily_api_key
    key = secret.get_secret_value().strip() if secret else ""
    if not key:
        variable = "BRAVE_SEARCH_API_KEY" if config.provider == "brave" else "TAVILY_API_KEY"
        raise SearchProviderError(f"{name} is selected but {variable} is missing. Run ciel setup.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise SearchProviderError("Search request timeout must be a positive finite number.")
    query = SecretMaskingFilter._redact(query).strip()
    if not query or maximum <= 0:
        return []
    maximum = min(maximum, 20)
    try:
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            if config.provider == "brave":
                response = client.get(
                    _BRAVE_NEWS_URL if news else _BRAVE_WEB_URL,
                    headers={"X-Subscription-Token": key, "Accept": "application/json"},
                    params={"q": query, "count": maximum, "country": "ALL"},
                )
            else:
                response = client.post(
                    _TAVILY_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "query": query, "topic": "news" if news else "general",
                        "search_depth": "basic", "auto_parameters": False,
                        "max_results": maximum, "include_answer": False,
                        "include_raw_content": False, "include_published_date": True,
                    },
                )
            if response.status_code != 200:
                raise SearchProviderError(f"{name} search failed (HTTP {response.status_code}).")
            payload = response.json()
    except httpx.TimeoutException:
        raise SearchProviderError(f"{name} search request timed out.") from None
    except httpx.RequestError:
        raise SearchProviderError(f"{name} search connection failed.") from None
    except (TypeError, ValueError):
        raise SearchProviderError(f"{name} returned an invalid search response.") from None
    if not isinstance(payload, dict):
        raise SearchProviderError(f"{name} returned an invalid search response.")
    group = payload.get("web") if config.provider == "brave" and not news else payload
    rows = group.get("results") if isinstance(group, dict) else None
    if not isinstance(rows, list):
        raise SearchProviderError(f"{name} returned an invalid search response.")
    normalised = []
    for row in rows:
        result = _normalise(row, provider=config.provider) if isinstance(row, dict) else None
        if result is not None:
            normalised.append(result)
        if len(normalised) >= maximum:
            break
    return normalised
