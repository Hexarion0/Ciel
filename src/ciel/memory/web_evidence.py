"""Bounded, untrusted web evidence retained separately from conversation text."""

import ipaddress
import json
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from ciel.logging import SecretMaskingFilter

MAX_WEB_EVIDENCE_SOURCES = 10
MAX_WEB_EVIDENCE_JSON_CHARS = 256_000
_SOURCE_TEXT_LIMITS = {
    "title": 300,
    "snippet": 1600,
    "content": 9000,
    "published_at": 100,
    "date_kind": 32,
    "fetched_at": 100,
    "publisher": 300,
}
_CREDENTIAL_PARAMETERS = {
    "access_token", "api_key", "apikey", "auth", "authorization", "key", "password",
    "secret", "token",
}


def _text(value: Any, limit: int) -> str | None:
    """Copy text with known secrets removed and nonprinting controls discarded."""
    if not isinstance(value, str):
        return None
    redacted = SecretMaskingFilter._redact(value)
    return "".join(
        char for char in redacted
        if char in "\n\t" or (
            ord(char) >= 32 and not 127 <= ord(char) <= 159 and not 0xD800 <= ord(char) <= 0xDFFF
        )
    ).strip()[:limit]


def _source_url(value: Any) -> str | None:
    """Accept public-looking HTTP(S) addresses without credentials; never resolve DNS."""
    if (
        not isinstance(value, str) or not value or len(value) > 2048 or "\\" in value
        or any(
            char.isspace() or ord(char) < 32 or 127 <= ord(char) <= 159
            or 0xD800 <= ord(char) <= 0xDFFF for char in value
        )
    ):
        return None
    decoded = unquote(unquote(value))
    if SecretMaskingFilter._redact(decoded) != decoded:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if (
            parsed.scheme.lower() not in {"http", "https"} or not hostname
            or parsed.username is not None or parsed.password is not None
            or any(key.casefold() in _CREDENTIAL_PARAMETERS for key, _ in parse_qsl(parsed.query))
        ):
            return None
        parsed.port
        hostname = hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
            return None
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            if "." not in hostname:
                return None
        else:
            if not address.is_global:
                return None
    except ValueError:
        return None
    return value


def normalize_web_evidence(value: Any) -> dict[str, Any]:
    """Allowlist bounded evidence data without interpreting remote text as instructions.

    These fields are only historical source material. Callers must explicitly choose
    when to use them; ordinary conversation history does not replay their contents.
    """
    if not isinstance(value, dict):
        return {}
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in {"web", "news"}:
        return {}
    query = _text(value.get("query"), 1600)
    retrieved_at = _text(value.get("retrieved_at"), 100)
    sources = value.get("sources")
    if not query or not retrieved_at or not isinstance(sources, list):
        return {}
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources[:MAX_WEB_EVIDENCE_SOURCES]:
        if not isinstance(source, dict):
            continue
        url = _source_url(source.get("url"))
        if not url or url in seen:
            continue
        fields = {
            key: _text(source.get(key, ""), limit)
            for key, limit in _SOURCE_TEXT_LIMITS.items()
        }
        if any(text is None for text in fields.values()):
            continue
        if fields["date_kind"] not in {
            "", "published", "modified", "reported", "published_or_modified",
        }:
            fields["date_kind"] = ""
        seen.add(url)
        result.append({"url": url, **fields})
        if len(result) == MAX_WEB_EVIDENCE_SOURCES:
            break
    if not result and sources:
        return {}
    return {
        "query": query,
        "kind": value["kind"],
        "retrieved_at": retrieved_at,
        "sources": result,
    }


def decode_web_evidence(value: Any) -> dict[str, Any]:
    """Fail closed on corrupt or oversized stored JSON, then validate every field."""
    if not isinstance(value, str) or len(value) > MAX_WEB_EVIDENCE_JSON_CHARS:
        return {}
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        return {}
    return normalize_web_evidence(decoded)
