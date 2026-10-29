"""Bounded, anonymous search and page extraction through Parallel's free MCP endpoint."""

import codecs
import ipaddress
import json
import math
import time
from dataclasses import dataclass
from threading import Event
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx

from ciel.logging import SecretMaskingFilter
from ciel.search_providers import SearchProviderError

_ENDPOINT = "https://mcp.parallel.ai/mcp"
_PROTOCOL = "2025-03-26"
_MAX_RESPONSE_BYTES = 1_048_576
_EXCERPT_GAP = "\n\n[... omitted source text ...]\n\n"


@dataclass
class _ResponseBudget:
    """Share one response-size ceiling across protocol setup and extracted evidence."""

    remaining: int

    def consume(self, amount: int) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            raise SearchProviderError("Parallel search response exceeded the size limit.")


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
    budget: _ResponseBudget,
) -> dict:
    """Bound JSON/SSE bytes, decode split Unicode, and close after our SSE response."""
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in {"application/json", "text/event-stream"}:
        raise _invalid()
    # Request identity transfer to keep decompression from expanding an untrusted chunk.
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise _invalid()
    length = response.headers.get("content-length", "")
    if length.isdigit() and int(length) > budget.remaining:
        raise SearchProviderError("Parallel search response exceeded the size limit.")
    decoder = codecs.getincrementaldecoder("utf-8")()
    body, pending, data = [], "", []

    def event_result() -> dict | None:
        if not data:
            return None
        payload = json.loads("\n".join(data))
        data.clear()
        return _rpc_result(payload, request_id, event=True)

    for chunk in response.iter_bytes():
        _remaining(deadline, cancel_event)
        budget.consume(len(chunk))
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
    budget: _ResponseBudget,
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
            response, payload["id"], deadline, cancel_event, budget,
        )
        return result, session


def _text(value: Any, limit: int) -> str:
    return SecretMaskingFilter._redact(value).strip()[:limit] if isinstance(value, str) else ""


def _public_url(value: Any) -> str:
    """Reject private URL literals and credentials before sending pages to a service."""
    message = "Parallel fetch requires public HTTP(S) pages without credentials."
    if (
        not isinstance(value, str) or not value or len(value) > 2048
        or any(ord(char) <= 32 or ord(char) == 127 or char == "\\" for char in value)
        or SecretMaskingFilter._redact(value) != value or "[REDACTED_SECRET" in value
    ):
        raise SearchProviderError(message)
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").rstrip(".").casefold()
        if (
            parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in {None, 80 if parsed.scheme == "http" else 443}
            or host == "localhost"
            or host.endswith((".localhost", ".local", ".internal", ".home"))
            or any(key.casefold() in {"token", "access_token", "api_key", "password", "secret"}
                   for key, _value in parse_qsl(parsed.query))
        ):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or "%" in host or all(
                part.isdigit() or part.startswith("0x") for part in host.split(".")
            ):
                raise ValueError from None
        else:
            addresses = [address]
            if isinstance(address, ipaddress.IPv6Address):
                addresses.extend(item for item in (address.ipv4_mapped, address.sixtofour) if item)
                addresses.extend(address.teredo or ())
            if any(not item.is_global or item.is_multicast or item.is_reserved
                   for item in addresses):
                raise ValueError
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    except ValueError:
        raise SearchProviderError(message) from None


def _queries(values: list[str] | None, fallback: str = "") -> list[str]:
    """Keep at most three distinct, redacted keyword queries in their supplied order."""
    found, seen = [], set()
    for value in values if isinstance(values, list) else []:
        query = " ".join(_text(value, 2000).split())
        if len(query) > 200:
            prefix = query[:201]
            query = prefix.rsplit(" ", 1)[0] if " " in prefix else prefix[:200]
        if not query or query.casefold() in seen:
            continue
        found.append(query)
        seen.add(query.casefold())
        if len(found) == 3:
            break
    if not found and fallback:
        return _queries([fallback])
    return found


def _rows(result: dict, maximum: int, *, full_content: bool = False) -> list[dict[str, str]]:
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
        if not isinstance(row, dict) or "error" in row or row.get("error_type"):
            continue
        excerpts = row.get("excerpts")
        if not isinstance(excerpts, list) or any(not isinstance(part, str) for part in excerpts):
            continue
        try:
            url = _public_url(row.get("url"))
        except SearchProviderError:
            continue
        excerpt = _text(_EXCERPT_GAP.join(excerpts), 25_000)
        content = _text(row.get("full_content"), 25_000) if full_content else ""
        date = _text(row.get("publish_date"), 100)
        found.append({
            "href": url, "title": _text(row.get("title"), 500),
            "body": excerpt, "content": content or excerpt,
            "date": date, "date_kind": "published" if date else "", "publisher": "",
        })
        if len(found) >= maximum:
            break
    return found


def _validate_request(timeout: float, session_id: str) -> None:
    """Keep invalid budgets and identifiers out of the network client."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise SearchProviderError("Search request timeout must be a positive finite number.")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 100:
        raise SearchProviderError("Parallel search requires a valid conversation identifier.")
    if any(ord(char) < 33 or ord(char) > 126 for char in session_id):
        raise SearchProviderError("Parallel search requires a valid conversation identifier.")


def _call_tool(
    name: str, arguments: dict, *, timeout: float, maximum: int,
    cancel_event: Event | None = None, full_content: bool = False,
) -> list[dict[str, str]]:
    """Share a bounded anonymous MCP handshake and result parser for both read tools."""
    deadline = time.monotonic() + timeout
    budget = _ResponseBudget(_MAX_RESPONSE_BYTES)
    headers = {"Accept": "application/json, text/event-stream", "Accept-Encoding": "identity"}
    try:
        _remaining(deadline, cancel_event)
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            initialized, session = _request(client, headers, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": _PROTOCOL, "capabilities": {},
                    "clientInfo": {"name": "ciel", "version": "0.3.6"},
                },
            }, deadline, budget, cancel_event=cancel_event)
            if initialized is None or initialized.get("protocolVersion") != _PROTOCOL:
                raise _invalid()
            headers["MCP-Protocol-Version"] = _PROTOCOL
            if session:
                headers["Mcp-Session-Id"] = session
            _request(client, headers, {
                "jsonrpc": "2.0", "method": "notifications/initialized",
            }, deadline, budget, notification=True, cancel_event=cancel_event)
            result, _session = _request(client, headers, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": name, "arguments": arguments,
                },
            }, deadline, budget, cancel_event=cancel_event)
            if result is None:
                raise _invalid()
            found = _rows(result, min(maximum, 20), full_content=full_content)
            _remaining(deadline, cancel_event)
            return found
    except httpx.TimeoutException:
        raise SearchProviderError("Parallel search request timed out.") from None
    except httpx.RequestError:
        raise SearchProviderError("Parallel search connection failed.") from None
    except (TypeError, ValueError):
        raise _invalid() from None


def search_parallel(
    query: str, *, timeout: float, maximum: int, session_id: str,
    cancel_event: Event | None = None, objective: str = "", search_queries: list[str] | None = None,
) -> list[dict[str, str]]:
    """Search related keyword queries in one anonymous call with one overall budget.

    The MCP protocol session is local to this call. The supplied conversation identifier
    is reused across calls for the service's anonymous rate accounting. Dense excerpts
    are returned as evidence; callers still validate relevance, URLs, and citations.
    """
    _validate_request(timeout, session_id)
    query = _text(query, 2000)
    if not query or maximum <= 0:
        return []
    return _call_tool("web_search", {
        "objective": _text(objective, 5000) or query,
        "search_queries": _queries(search_queries, query), "session_id": session_id,
    }, timeout=timeout, maximum=maximum, cancel_event=cancel_event)


def fetch_parallel(
    urls: list[str], *, timeout: float, maximum: int, session_id: str,
    cancel_event: Event | None = None, objective: str = "", search_queries: list[str] | None = None,
    full_content: bool = False,
) -> list[dict[str, str]]:
    """Extract evidence from up to twenty public pages, retaining successful partial results.

    Callers validate requested and returned URLs against their own public-page policy.
    Provider diagnostics and failed-page content are never projected as source evidence.
    """
    _validate_request(timeout, session_id)
    if not urls or maximum <= 0:
        return []
    selected = []
    for value in urls:
        url = _public_url(value)
        if url not in selected:
            selected.append(url)
        if len(selected) == 20:
            break
    arguments: dict[str, Any] = {
        "urls": selected, "session_id": session_id, "full_content": bool(full_content),
    }
    objective = _text(objective, 200)
    queries = _queries(search_queries)
    if objective:
        arguments["objective"] = objective
    if queries:
        arguments["search_queries"] = queries
    return _call_tool("web_fetch", arguments, timeout=timeout, maximum=maximum,
                      cancel_event=cancel_event, full_content=bool(full_content))
