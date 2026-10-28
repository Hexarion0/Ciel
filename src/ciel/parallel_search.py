"""Bounded, anonymous access to Parallel's free Search MCP endpoint."""

import codecs
import json
import math
import time
from threading import Event
from typing import Any
from urllib.parse import urlsplit

import httpx

from ciel.logging import SecretMaskingFilter
from ciel.search_providers import SearchProviderError

_ENDPOINT = "https://search.parallel.ai/mcp"
_PROTOCOL = "2025-03-26"
_MAX_RESPONSE_BYTES = 1_048_576


def _remaining(deadline: float, cancel_event: Event | None = None) -> float:
    """Spend one overall budget across the MCP handshake and streamed result."""
    if cancel_event is not None and cancel_event.is_set():
        raise SearchProviderError("Parallel search was canceled.")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("Parallel deadline elapsed")
    return remaining


def _invalid() -> SearchProviderError:
    return SearchProviderError("Parallel returned an invalid search response.")


def _rpc_result(payload: Any, request_id: int, *, event: bool = False) -> dict | None:
    """Accept only the matching JSON-RPC result; server instructions are never used."""
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        raise _invalid()
    if "id" not in payload and event and isinstance(payload.get("method"), str):
        return None
    identity = payload.get("id")
    if type(identity) is not int or identity != request_id:
        if event:
            return None
        raise _invalid()
    if "error" in payload or not isinstance(payload.get("result"), dict):
        raise _invalid()
    return payload["result"]


def _read_result(
    response: httpx.Response, request_id: int, deadline: float, cancel_event: Event | None,
) -> dict:
    """Bound JSON/SSE bytes, decode split Unicode, and close after our SSE response."""
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in {"application/json", "text/event-stream"}:
        raise _invalid()
    # Request identity transfer to keep decompression from expanding an untrusted chunk.
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise _invalid()
    length = response.headers.get("content-length", "")
    if length.isdigit() and int(length) > _MAX_RESPONSE_BYTES:
        raise SearchProviderError("Parallel search response exceeded the size limit.")
    decoder = codecs.getincrementaldecoder("utf-8")()
    body, pending, data = [], "", []
    size = 0

    def event_result() -> dict | None:
        if not data:
            return None
        payload = json.loads("\n".join(data))
        data.clear()
        return _rpc_result(payload, request_id, event=True)

    for chunk in response.iter_bytes():
        _remaining(deadline, cancel_event)
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise SearchProviderError("Parallel search response exceeded the size limit.")
        text = decoder.decode(chunk)
        if media_type == "application/json":
            body.append(text)
            continue
        pending += text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                result = event_result()
                if result is not None:
                    return result
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
    _remaining(deadline, cancel_event)
    final = decoder.decode(b"", final=True)
    if media_type == "application/json":
        result = _rpc_result(json.loads("".join(body) + final), request_id)
        if result is not None:
            return result
    else:
        pending += final
        if pending.startswith("data:"):
            data.append(pending[5:].removeprefix(" ").removesuffix("\r"))
        result = event_result()
        if result is not None:
            return result
    raise _invalid()


def _request(
    client: httpx.Client, headers: dict[str, str], payload: dict, deadline: float,
    *, notification: bool = False, cancel_event: Event | None = None,
) -> tuple[dict | None, str]:
    """Send to the fixed anonymous endpoint without redirects, retries, or credentials."""
    with client.stream(
        "POST", _ENDPOINT, headers=headers, json=payload,
        timeout=_remaining(deadline, cancel_event),
    ) as response:
        _remaining(deadline, cancel_event)
        expected_status = 202 if notification else 200
        if response.status_code != expected_status:
            raise SearchProviderError(f"Parallel search failed (HTTP {response.status_code}).")
        session = response.headers.get("mcp-session-id", "")
        if len(session) > 256 or any(ord(char) < 33 or ord(char) > 126 for char in session):
            raise _invalid()
        result = None if notification else _read_result(
            response, payload["id"], deadline, cancel_event,
        )
        return result, session


def _text(value: Any, limit: int) -> str:
    return SecretMaskingFilter._redact(value).strip()[:limit] if isinstance(value, str) else ""


def _rows(result: dict, maximum: int) -> list[dict[str, str]]:
    """Parse Parallel's observed structuredContent or equivalent JSON text fallback."""
    if result.get("isError", False) is not False:
        raise SearchProviderError("Parallel search tool failed.")
    if "structuredContent" in result:
        payload = result["structuredContent"]
    else:
        content = result.get("content")
        if not isinstance(content, list) or len(content) != 1:
            raise _invalid()
        item = content[0]
        if not isinstance(item, dict) or item.get("type") != "text":
            raise _invalid()
        payload = json.loads(item.get("text", ""))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise _invalid()
    found = []
    for row in payload["results"]:
        if not isinstance(row, dict):
            continue
        excerpts = row.get("excerpts")
        if not isinstance(excerpts, list) or any(not isinstance(part, str) for part in excerpts):
            continue
        url = _text(row.get("url"), 2048)
        if "[REDACTED_SECRET]" in url:
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            if parsed.username or parsed.password:
                continue
        except ValueError:
            continue
        excerpt = _text("\n\n".join(excerpts), 25_000)
        date = _text(row.get("publish_date"), 100)
        found.append({
            "href": url, "title": _text(row.get("title"), 500),
            "body": excerpt, "content": excerpt,
            "date": date, "date_kind": "published" if date else "", "publisher": "",
        })
        if len(found) >= maximum:
            break
    return found


def search_parallel(
    query: str, *, timeout: float, maximum: int, session_id: str,
    cancel_event: Event | None = None, objective: str = "",
) -> list[dict[str, str]]:
    """Search Parallel's anonymous free tier with a stable per-conversation identifier.

    The MCP protocol session is local to this call. The supplied conversation identifier
    is reused across calls for the service's anonymous rate accounting. Dense excerpts
    are returned as evidence; callers still validate relevance, URLs, and citations.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise SearchProviderError("Search request timeout must be a positive finite number.")
    query = _text(query, 2000)
    objective = _text(objective, 5000) or query
    if not query or maximum <= 0:
        return []
    if not isinstance(session_id, str) or not session_id or len(session_id) > 100:
        raise SearchProviderError("Parallel search requires a valid conversation identifier.")
    if any(ord(char) < 33 or ord(char) > 126 for char in session_id):
        raise SearchProviderError("Parallel search requires a valid conversation identifier.")
    deadline = time.monotonic() + timeout
    keywords = " ".join(query.split())
    if len(keywords) > 200:
        prefix = keywords[:201]
        keywords = prefix.rsplit(" ", 1)[0] if " " in prefix else prefix[:200]
    headers = {"Accept": "application/json, text/event-stream", "Accept-Encoding": "identity"}
    try:
        _remaining(deadline, cancel_event)
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            initialized, session = _request(client, headers, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": _PROTOCOL, "capabilities": {},
                    "clientInfo": {"name": "ciel", "version": "0.3.6"},
                },
            }, deadline, cancel_event=cancel_event)
            if initialized is None or initialized.get("protocolVersion") != _PROTOCOL:
                raise _invalid()
            headers["MCP-Protocol-Version"] = _PROTOCOL
            if session:
                headers["Mcp-Session-Id"] = session
            _request(client, headers, {
                "jsonrpc": "2.0", "method": "notifications/initialized",
            }, deadline, notification=True, cancel_event=cancel_event)
            result, _session = _request(client, headers, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "web_search", "arguments": {
                        "objective": objective, "search_queries": [keywords],
                        "session_id": session_id,
                    },
                },
            }, deadline, cancel_event=cancel_event)
            if result is None:
                raise _invalid()
            found = _rows(result, min(maximum, 20))
            _remaining(deadline, cancel_event)
            return found
    except httpx.TimeoutException:
        raise SearchProviderError("Parallel search request timed out.") from None
    except httpx.RequestError:
        raise SearchProviderError("Parallel search connection failed.") from None
    except (TypeError, ValueError):
        raise _invalid() from None
