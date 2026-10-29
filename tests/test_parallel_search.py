"""Anonymous Parallel MCP requests retain evidence and fail safely."""

import json
from threading import Event

import httpx
import pytest

from ciel import parallel_search
from ciel.parallel_search import search_parallel
from ciel.search_providers import SearchProviderError

SESSION = "1234567890abcdef1234567890abcdef"


@pytest.fixture
def transport(monkeypatch):
    """Observe real HTTP request construction with no network access."""
    requests, options, responses = [], [], []
    original_client = httpx.Client

    def respond(request):
        requests.append(request)
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return httpx.Response(request=request, **item)

    def client(**kwargs):
        options.append(kwargs)
        return original_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr("ciel.parallel_search.httpx.Client", client)
    return requests, options, responses


def _handshake(responses, *, session="opaque-mcp-session"):
    headers = {"Mcp-Session-Id": session} if session else {}
    responses.extend([
        {"status_code": 200, "headers": headers, "json": {
            "jsonrpc": "2.0", "id": 1,
            "result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}},
        }},
        {"status_code": 202},
    ])


def _results(responses, rows, *, structured=True):
    data = {"search_id": "search_example", "results": rows, "session_id": SESSION}
    result = {"isError": False, "content": [{"type": "text", "text": json.dumps(data)}]}
    if structured:
        result["structuredContent"] = data
    responses.append({"status_code": 200, "json": {
        "jsonrpc": "2.0", "id": 2, "result": result,
    }})


@pytest.mark.parametrize("structured", [True, False])
def test_anonymous_mcp_preserves_order_and_dense_excerpts(transport, structured):
    requests, options, responses = transport
    _handshake(responses)
    _results(responses, [{
        "url": "https://docs.python.org/3/tutorial/datastructures.html",
        "title": "Python list comprehensions", "publish_date": "2026-10-24",
        "excerpts": ["First supporting passage.", "Second supporting passage."],
    }, {"url": "https://example.org/second", "title": None,
        "publish_date": None, "excerpts": ["Another result."]}], structured=structured)

    found = search_parallel("Python list comprehensions", timeout=8, maximum=1, session_id=SESSION)

    assert found == [{
        "href": "https://docs.python.org/3/tutorial/datastructures.html",
        "title": "Python list comprehensions", "date": "2026-10-24",
        "date_kind": "published", "publisher": "",
        "body": "First supporting passage.\n\n[... omitted source text ...]\n\n"
                "Second supporting passage.",
        "content": "First supporting passage.\n\n[... omitted source text ...]\n\n"
                   "Second supporting passage.",
    }]
    assert len(requests) == 3
    assert all(str(request.url) == "https://mcp.parallel.ai/mcp" for request in requests)
    assert all("authorization" not in request.headers for request in requests)
    assert all(request.headers["accept"] == "application/json, text/event-stream"
               for request in requests)
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False
    assert requests[1].headers["mcp-session-id"] == "opaque-mcp-session"
    assert requests[2].headers["mcp-protocol-version"] == "2025-03-26"
    assert json.loads(requests[1].content) == {
        "jsonrpc": "2.0", "method": "notifications/initialized",
    }
    assert json.loads(requests[2].content)["params"] == {
        "name": "web_search", "arguments": {
            "objective": "Python list comprehensions",
            "search_queries": ["Python list comprehensions"], "session_id": SESSION,
        },
    }


class _Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def test_sse_ignores_notifications_and_stops_after_correlated_result(transport):
    _requests, _options, responses = transport
    _handshake(responses, session="")
    payload = {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": {
        "results": [{"url": "https://example.org", "excerpts": ["Café documentation."]}],
    }}}
    event = "event: message\r\ndata: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n"
    encoded = event.encode()
    boundary = encoded.index("é".encode()) + 1
    chunks = _Chunks([
        b': keepalive\n\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n',
        b'data: {"jsonrpc":"2.0","id":99,"result":{"private":"ignore"}}\n\n',
        encoded[:boundary], encoded[boundary:], b"data: invalid trailing event\n\n",
    ])
    responses.append({"status_code": 200, "headers": {"Content-Type": "text/event-stream"},
                      "stream": chunks})

    found = search_parallel("documentation", timeout=8, maximum=3, session_id=SESSION)

    assert found[0]["content"] == "Café documentation."
    assert chunks.closed


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_error_is_safe_and_does_not_retry(transport, status):
    requests, _options, responses = transport
    _handshake(responses)
    responses.append({"status_code": status, "text": "private-query secret-response"})
    with pytest.raises(SearchProviderError) as error:
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION)
    assert str(error.value) == f"Parallel search failed (HTTP {status})."
    assert len(requests) == 3


@pytest.mark.parametrize("result", [
    {"isError": True, "content": [{"type": "text", "text": "private-query"}]},
    {"structuredContent": {"results": "private-query"}},
    {"structuredContent": []},
    {"content": [{"type": "text", "text": "private-query"}]},
])
def test_invalid_tool_output_never_leaks_response(transport, result):
    _requests, _options, responses = transport
    _handshake(responses)
    responses.append({"status_code": 200, "json": {
        "jsonrpc": "2.0", "id": 2, "result": result,
    }})
    with pytest.raises(SearchProviderError) as error:
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION)
    assert "private-query" not in str(error.value)


@pytest.mark.parametrize("payload", [
    {"jsonrpc": "2.0", "id": 99, "result": {}},
    {"jsonrpc": "2.0", "id": True, "result": {}},
    {"jsonrpc": "2.0", "id": 1, "error": {"message": "private-query"}},
    {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "unknown"}},
])
def test_protocol_failures_stop_before_search(transport, payload):
    requests, _options, responses = transport
    responses.append({"status_code": 200, "json": payload})
    with pytest.raises(SearchProviderError, match="invalid search response"):
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION)
    assert len(requests) == 1


def test_malformed_rows_and_credential_urls_are_not_projected(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [None, {"url": "file:///private", "excerpts": ["text"]},
                         {"url": "https://name:pass@example.com", "excerpts": ["text"]},
                         {"url": "https://example.com", "excerpts": "wrong shape"},
                         {"url": "https://example.org", "excerpts": ["kept"], "title": []}])
    assert search_parallel("query", timeout=8, maximum=5, session_id=SESSION) == [{
        "href": "https://example.org", "title": "", "body": "kept", "content": "kept",
        "date": "", "date_kind": "", "publisher": "",
    }]


@pytest.mark.parametrize("failure, category", [
    (httpx.ConnectError("private-query"), "connection failed"),
    (httpx.ReadTimeout("private-query"), "timed out"),
])
def test_network_errors_hide_internal_exception(transport, failure, category):
    requests, _options, responses = transport
    responses.append(failure)
    with pytest.raises(SearchProviderError, match=category) as error:
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION)
    assert "private-query" not in str(error.value)
    assert error.value.__suppress_context__
    assert len(requests) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_never_opens_client(transport, timeout):
    requests, options, _responses = transport
    with pytest.raises(SearchProviderError, match="positive finite"):
        search_parallel("query", timeout=timeout, maximum=5, session_id=SESSION)
    assert not requests and not options


def test_timeout_budget_is_shared_by_handshake_and_search(transport, monkeypatch):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [])
    now = [0.0]
    monkeypatch.setattr("ciel.parallel_search.time.monotonic", lambda: now[0])
    original_send = httpx._client.Client.send

    def send(self, request, **kwargs):
        response = original_send(self, request, **kwargs)
        now[0] += 3.0
        return response

    monkeypatch.setattr(httpx._client.Client, "send", send)
    with pytest.raises(SearchProviderError, match="timed out"):
        search_parallel("query", timeout=5, maximum=5, session_id=SESSION)
    assert len(requests) == 2
    assert requests[1].extensions["timeout"]["read"] == 2.0


def test_oversized_stream_stops_at_byte_limit(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    chunks = _Chunks([b" " * 600_000, b" " * 600_000, b"private-query"])
    responses.append({"status_code": 200, "headers": {"Content-Type": "application/json"},
                      "stream": chunks})
    with pytest.raises(SearchProviderError, match="response exceeded"):
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION)
    assert chunks.closed


def test_stream_budget_does_not_reset_for_each_chunk(transport, monkeypatch):
    _requests, _options, responses = transport
    _handshake(responses)
    now = [0.0]
    monkeypatch.setattr("ciel.parallel_search.time.monotonic", lambda: now[0])

    class SlowStream(_Chunks):
        def __iter__(self):
            yield b': keepalive\n\n'
            now[0] = 9.0
            yield b': keepalive\n\n'
            pytest.fail("read continued after the deadline")

    chunks = SlowStream([])
    responses.append({"status_code": 200, "headers": {"Content-Type": "text/event-stream"},
                      "stream": chunks})
    with pytest.raises(SearchProviderError, match="timed out"):
        search_parallel("query", timeout=8, maximum=5, session_id=SESSION)
    assert chunks.closed


def test_dense_excerpts_remain_evidence_instead_of_becoming_short_snippets(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    excerpt = "A detailed supported passage. " * 400
    _results(responses, [{"url": "https://example.org", "excerpts": [excerpt]}])
    found = search_parallel("query", timeout=8, maximum=5, session_id=SESSION)
    assert len(found[0]["content"]) > 8000
    assert found[0]["content"] == excerpt.strip()


@pytest.mark.parametrize("session", [None, "", "a" * 101, "session\nunsafe"])
def test_invalid_conversation_identifier_stops_before_network(transport, session):
    requests, options, _responses = transport
    with pytest.raises(SearchProviderError, match="conversation identifier"):
        search_parallel("query", timeout=8, maximum=5, session_id=session)
    assert not requests and not options


@pytest.mark.parametrize("stage, expected_calls", [
    ("initialize", 1), ("notifications/initialized", 2),
])
def test_cancellation_during_handshake_never_sends_query(
    transport, monkeypatch, stage, expected_calls,
):
    requests, _options, responses = transport
    _handshake(responses)
    canceled = Event()
    original_send = httpx._client.Client.send

    def send(self, request, **kwargs):
        response = original_send(self, request, **kwargs)
        if json.loads(request.content)["method"] == stage:
            canceled.set()
        return response

    monkeypatch.setattr(httpx._client.Client, "send", send)
    with pytest.raises(SearchProviderError, match="canceled"):
        search_parallel("private-query", timeout=8, maximum=5, session_id=SESSION,
                        cancel_event=canceled)
    assert len(requests) == expected_calls
    assert all("private-query" not in request.content.decode() for request in requests)


def test_cancellation_while_reading_closes_stream_without_waiting_for_more(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    canceled = Event()

    class CanceledStream(_Chunks):
        def __iter__(self):
            yield b": keepalive\n\n"
            canceled.set()
            yield b": interrupted\n\n"
            pytest.fail("read continued after cancellation")

    chunks = CanceledStream([])
    responses.append({"status_code": 200, "headers": {"Content-Type": "text/event-stream"},
                      "stream": chunks})
    with pytest.raises(SearchProviderError, match="canceled"):
        search_parallel("query", timeout=8, maximum=5, session_id=SESSION, cancel_event=canceled)
    assert chunks.closed


def test_long_objective_keeps_full_question_and_uses_bounded_keyword_query(transport):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [])
    query = "Explain the behavior of Python list comprehensions with nested loops and filters. " * 5
    search_parallel(query, timeout=8, maximum=5, session_id=SESSION)
    arguments = json.loads(requests[-1].content)["params"]["arguments"]
    assert arguments["objective"] == query.strip()
    keywords = arguments["search_queries"][0]
    assert len(keywords) <= 200
    assert query.startswith(keywords + " ")


def test_related_queries_are_sanitized_deduplicated_and_batched(transport):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [])
    secret = "sk-syntheticprivatekey12345"
    search_parallel(
        "original question", timeout=8, maximum=5, session_id=SESSION,
        objective="Find the requested explanation",
        search_queries=["  Python\nlist comprehension ", "python list comprehension", "",
                        f"Python filters {secret}", "Python nested loops " * 30, "extra query"],
    )
    arguments = json.loads(requests[-1].content)["params"]["arguments"]
    queries = arguments["search_queries"]
    assert arguments["objective"] == "Find the requested explanation"
    assert len(queries) == 3
    assert queries[0] == "Python list comprehension"
    assert "[REDACTED_SECRET]" in queries[1]
    assert all(0 < len(query) <= 200 and "\n" not in query for query in queries)
    assert all(secret not in request.content.decode() for request in requests)
    assert len(requests) == 3


def test_empty_related_queries_fall_back_to_original_query(transport):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [])
    search_parallel("original question", timeout=8, maximum=5, session_id=SESSION,
                    search_queries=["", "  ", None])
    assert json.loads(requests[-1].content)["params"]["arguments"]["search_queries"] == [
        "original question",
    ]


@pytest.mark.parametrize("structured", [True, False])
def test_fetch_batches_urls_and_preserves_full_content_with_partial_errors(transport, structured):
    requests, options, responses = transport
    _handshake(responses)
    data = {
        "extract_id": "extract_example", "session_id": SESSION,
        "results": [{"url": "https://example.org/read", "title": "Useful page",
                     "excerpts": ["Relevant excerpt."], "full_content": "Full source passage."},
                    {"url": "https://example.com/other", "excerpts": ["Other source."]}],
        "errors": [{"url": "https://example.org/failed", "error_type": "private-query",
                    "content": "private-error-secret"}],
    }
    result = {"content": [{"type": "text", "text": json.dumps(data)}]}
    if structured:
        result["structuredContent"] = data
    responses.append({"status_code": 200, "json": {
        "jsonrpc": "2.0", "id": 2, "result": result,
    }})
    found = parallel_search.fetch_parallel(
        ["https://example.org/read", "https://example.org/read", "https://example.org/failed",
         "https://example.com/other"], timeout=8, maximum=2, session_id=SESSION,
        objective="Focus on Python behavior " * 20, search_queries=["Python nested loops"],
        full_content=True,
    )
    assert [row["href"] for row in found] == [
        "https://example.org/read", "https://example.com/other",
    ]
    assert found[0]["body"] == "Relevant excerpt."
    assert found[0]["content"] == "Full source passage."
    assert found[1]["body"] == found[1]["content"] == "Other source."
    assert options[0]["trust_env"] is False
    assert options[0]["follow_redirects"] is False
    assert all(str(request.url) == "https://mcp.parallel.ai/mcp" for request in requests)
    assert all("authorization" not in request.headers for request in requests)
    params = json.loads(requests[-1].content)["params"]
    assert params["name"] == "web_fetch"
    arguments = params["arguments"]
    assert arguments["urls"] == [
        "https://example.org/read", "https://example.org/failed", "https://example.com/other",
    ]
    assert len(arguments["objective"]) <= 200
    assert arguments["session_id"] == SESSION
    assert arguments["search_queries"] == ["Python nested loops"]
    assert arguments["full_content"] is True


@pytest.mark.parametrize("url", [
    "file:///private", "https://username:password@example.com/path", "http://127.0.0.1/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "https://localhost/",
    "https://computer.local/", "https://10.0.0.1/path", "http://169.254.169.254/",
    "http://2130706433/", "https://example.org/?api_key=private", "https://example.org/\nsecret",
])
def test_fetch_rejects_private_or_credential_urls_before_network(transport, url):
    requests, options, _responses = transport
    with pytest.raises(SearchProviderError, match="public HTTP") as error:
        parallel_search.fetch_parallel([url], timeout=8, maximum=5, session_id=SESSION)
    assert url not in str(error.value)
    assert not requests and not options


def test_fetch_bounds_url_batch_and_result_count(transport):
    requests, _options, responses = transport
    _handshake(responses)
    rows = [{"url": f"https://example.org/{index}", "excerpts": [f"Passage {index}."]}
            for index in range(25)]
    _results(responses, rows)
    found = parallel_search.fetch_parallel([row["url"] for row in rows], timeout=8, maximum=30,
                                          session_id=SESSION)
    assert len(json.loads(requests[-1].content)["params"]["arguments"]["urls"]) == 20
    assert len(found) == 20


def test_fetch_skips_error_rows_and_does_not_expose_provider_diagnostics(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [
        {"url": "https://example.org/failed", "error": "private-error", "excerpts": ["wrong"]},
        {"url": "https://example.org/ok", "excerpts": ["Supported."],
         "title": "sk-syntheticprivatekey12345"},
    ])
    found = parallel_search.fetch_parallel(["https://example.org/failed", "https://example.org/ok"],
                                          timeout=8, maximum=5, session_id=SESSION)
    assert [row["href"] for row in found] == ["https://example.org/ok"]
    assert found[0]["title"] == "[REDACTED_SECRET]"


def test_fetch_all_page_errors_returns_empty_evidence(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    responses.append({"status_code": 200, "json": {
        "jsonrpc": "2.0", "id": 2, "result": {"structuredContent": {
            "results": [], "errors": [{"url": "https://example.org", "content": "private"}],
        }},
    }})
    assert parallel_search.fetch_parallel(["https://example.org"], timeout=8, maximum=5,
                                         session_id=SESSION) == []


def test_fetch_cancellation_stops_before_disclosing_urls(transport):
    requests, options, _responses = transport
    canceled = Event()
    canceled.set()
    with pytest.raises(SearchProviderError, match="canceled"):
        parallel_search.fetch_parallel(["https://example.org/private-query"], timeout=8,
                                       maximum=5, session_id=SESSION, cancel_event=canceled)
    assert not requests and not options


def test_fetch_shares_stream_size_limit_and_closes_on_failure(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    chunks = _Chunks([b" " * 600_000, b" " * 600_000, b"private-query"])
    responses.append({"status_code": 200, "headers": {"Content-Type": "application/json"},
                      "stream": chunks})
    with pytest.raises(SearchProviderError, match="response exceeded"):
        parallel_search.fetch_parallel(["https://example.org"], timeout=8, maximum=5,
                                       session_id=SESSION)
    assert chunks.closed


def test_byte_budget_covers_handshake_and_tool_response_together(transport, monkeypatch):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [{"url": "https://example.org", "excerpts": ["passage" * 40]}])
    result_size = len(json.dumps(responses[-1]["json"], separators=(",", ":")).encode())
    monkeypatch.setattr("ciel.parallel_search._MAX_RESPONSE_BYTES", result_size + 10)
    with pytest.raises(SearchProviderError, match="response exceeded"):
        parallel_search.fetch_parallel(["https://example.org"], timeout=8, maximum=5,
                                       session_id=SESSION)
    assert len(requests) == 3


@pytest.mark.parametrize("url", ["http://127.1/", "http://0x7f.0.0.1/", "http://0177.0.0.1/"])
def test_fetch_rejects_obscured_loopback_literals_before_network(transport, url):
    requests, options, _responses = transport
    with pytest.raises(SearchProviderError, match="public HTTP"):
        parallel_search.fetch_parallel([url], timeout=8, maximum=5, session_id=SESSION)
    assert not requests and not options


def test_fetch_cancellation_during_result_stream_closes_it(transport):
    _requests, _options, responses = transport
    _handshake(responses)
    canceled = Event()

    class CanceledStream(_Chunks):
        def __iter__(self):
            yield b": keepalive\n\n"
            canceled.set()
            yield b": interrupted\n\n"
            pytest.fail("read continued after cancellation")

    chunks = CanceledStream([])
    responses.append({"status_code": 200, "headers": {"Content-Type": "text/event-stream"},
                      "stream": chunks})
    with pytest.raises(SearchProviderError, match="canceled"):
        parallel_search.fetch_parallel(["https://example.org"], timeout=8, maximum=5,
                                       session_id=SESSION, cancel_event=canceled)
    assert chunks.closed


def test_fetch_timeout_budget_includes_handshake(transport, monkeypatch):
    requests, _options, responses = transport
    _handshake(responses)
    now = [0.0]
    monkeypatch.setattr("ciel.parallel_search.time.monotonic", lambda: now[0])
    original_send = httpx._client.Client.send

    def send(self, request, **kwargs):
        response = original_send(self, request, **kwargs)
        now[0] += 3.0
        return response

    monkeypatch.setattr(httpx._client.Client, "send", send)
    with pytest.raises(SearchProviderError, match="timed out"):
        parallel_search.fetch_parallel(["https://example.org"], timeout=5, maximum=5,
                                       session_id=SESSION)
    assert len(requests) == 2


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_fetch_http_failures_are_safe_and_do_not_retry(transport, status):
    requests, _options, responses = transport
    _handshake(responses)
    responses.append({"status_code": status, "text": "private-response sk-syntheticsecret123"})
    with pytest.raises(SearchProviderError) as error:
        parallel_search.fetch_parallel(["https://example.org"], timeout=8, maximum=5,
                                       session_id=SESSION)
    assert str(error.value) == f"Parallel search failed (HTTP {status})."
    assert len(requests) == 3
