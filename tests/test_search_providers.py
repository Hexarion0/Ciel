"""Selected search APIs use fixed endpoints and never spend another provider's quota."""

import json
from unittest.mock import Mock

import httpx
import pytest

from ciel.config import InternetConfig
from ciel.search_providers import SearchProviderError, search_api


@pytest.fixture
def api_transport(monkeypatch):
    """Intercept real HTTP request construction without contacting either provider."""
    requests = []
    options = []
    responses = []
    original_client = httpx.Client

    def respond(request):
        requests.append(request)
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        status, kwargs = item
        return httpx.Response(status, request=request, **kwargs)

    def client(**kwargs):
        options.append(kwargs)
        return original_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr("ciel.search_providers.httpx.Client", client)
    return requests, options, responses


@pytest.mark.parametrize("news", [False, True])
def test_brave_uses_only_selected_endpoint_and_normalises_article_metadata(api_transport, news):
    requests, options, responses = api_transport
    result = {
        "title": "Mongolia opens library", "url": "https://publisher.example/news/library",
        "description": "The new library opened on Tuesday.", "page_age": "2026-10-20T09:00:00",
        "profile": {"long_name": "Mongolian Publisher"},
    }
    payload = {"results": [result]} if news else {"web": {"results": [result]}}
    responses.append((200, {"json": payload}))
    config = InternetConfig(provider="brave", brave_api_key="brave-test", tavily_api_key="unused")

    found = search_api(config, "Mongolia news", news=news, timeout=12, maximum=4)

    assert found == [{
        "href": result["url"], "title": result["title"], "body": result["description"],
        "date": result["page_age"], "date_kind": "published_or_modified",
        "publisher": "Mongolian Publisher",
    }]
    assert len(requests) == 1
    assert requests[0].url.host == "api.search.brave.com"
    assert requests[0].url.path == f"/res/v1/{'news' if news else 'web'}/search"
    assert requests[0].url.params["q"] == "Mongolia news"
    assert requests[0].url.params["count"] == "4"
    assert requests[0].headers["x-subscription-token"] == "brave-test"
    assert "unused" not in str(requests[0].headers)
    assert options == [{"timeout": 12, "trust_env": False, "follow_redirects": False}]


@pytest.mark.parametrize("news", [False, True])
def test_tavily_uses_basic_depth_without_ai_answer_or_automatic_cost_upgrade(api_transport, news):
    requests, _options, responses = api_transport
    responses.append((200, {"json": {"results": [{
        "url": "https://publisher.example/article", "title": "Latest story",
        "content": "A supported search excerpt.", "published_date": "Tue, 20 Oct 2026 09:00 GMT",
    }]}}))
    config = InternetConfig(provider="tavily", tavily_api_key="tvly-test", brave_api_key="unused")

    found = search_api(config, "Монгол мэдээ", news=news, timeout=10, maximum=100)

    assert found[0]["body"] == "A supported search excerpt."
    assert found[0]["date"] == "Tue, 20 Oct 2026 09:00 GMT"
    assert found[0]["date_kind"] == "published_or_modified"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.tavily.com/search"
    assert requests[0].headers["authorization"] == "Bearer tvly-test"
    assert "unused" not in str(requests[0].headers)
    assert json.loads(requests[0].content) == {
        "query": "Монгол мэдээ", "topic": "news" if news else "general",
        "search_depth": "basic", "auto_parameters": False, "max_results": 20,
        "include_answer": False, "include_raw_content": False, "include_published_date": True,
    }


@pytest.mark.parametrize("provider, field", [
    ("brave", "brave_api_key"), ("tavily", "tavily_api_key"),
])
@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_selected_key_never_uses_another_saved_key(monkeypatch, provider, field, key):
    client = Mock()
    monkeypatch.setattr("ciel.search_providers.httpx.Client", client)
    values = {"brave_api_key": "other-saved-key", "tavily_api_key": "other-saved-key", field: key}
    config = InternetConfig(provider=provider, **values)

    with pytest.raises(SearchProviderError, match="is selected.*is missing.*ciel setup"):
        search_api(config, "private query", news=False, timeout=8, maximum=5)

    client.assert_not_called()


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_provider_error_never_retries_follows_redirect_or_exposes_response(api_transport, status):
    requests, _options, responses = api_transport
    responses.append((status, {"text": "secret-test-key private query"}))

    with pytest.raises(SearchProviderError) as error:
        search_api(InternetConfig(provider="brave", brave_api_key="secret-test-key"),
                   "private query", news=False, timeout=8, maximum=5)

    assert str(error.value) == f"Brave Search search failed (HTTP {status})."
    assert len(requests) == 1


@pytest.mark.parametrize("failure, expected", [
    (httpx.ConnectError("secret-test-key"), "connection failed"),
    (httpx.ReadTimeout("secret-test-key"), "timed out"),
])
def test_network_failure_has_safe_diagnostic(api_transport, failure, expected):
    requests, _options, responses = api_transport
    responses.append(failure)
    with pytest.raises(SearchProviderError, match=expected) as error:
        search_api(InternetConfig(provider="tavily", tavily_api_key="secret-test-key"),
                   "private query", news=True, timeout=8, maximum=5)
    assert "secret-test-key" not in str(error.value)
    assert error.value.__suppress_context__
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [None, [], {"results": "invalid"}, {"web": []}])
def test_malformed_results_are_reported_without_response_details(api_transport, payload):
    _requests, _options, responses = api_transport
    responses.append((200, {"json": payload}))
    with pytest.raises(SearchProviderError, match="invalid search response"):
        search_api(InternetConfig(provider="brave", brave_api_key="secret"),
                   "query", news=False, timeout=8, maximum=5)


def test_result_projection_ignores_malformed_rows_and_non_http_urls(api_transport):
    _requests, _options, responses = api_transport
    responses.append((200, {"json": {"results": [
        None, {"url": "file:///private"}, {"url": "https://user:secret@example.com/"},
        {"url": "https://example.com/valid", "title": [], "content": {"secret": "text"}},
    ]}}))
    found = search_api(InternetConfig(provider="tavily", tavily_api_key="secret"),
                       "query", news=False, timeout=8, maximum=5)
    assert found == [{
        "href": "https://example.com/valid", "title": "", "body": "", "date": "",
        "date_kind": "published_or_modified", "publisher": "",
    }]


@pytest.mark.parametrize("provider, field", [
    ("brave", "brave_api_key"), ("tavily", "tavily_api_key"),
])
def test_arbitrary_configured_key_is_removed_from_outbound_query_and_returned_text(
    api_transport, provider, field,
):
    requests, _options, responses = api_transport
    key = "synthetic-prefixless-api-credential-12345"
    row = {
        "url": "https://example.com/article", "title": key,
        "description": f"Received {key}", "content": f"Received {key}",
    }
    payload = {"web": {"results": [row]}} if provider == "brave" else {"results": [row]}
    responses.append((200, {"json": payload}))

    found = search_api(InternetConfig(provider=provider, **{field: key}),
                       f"search docs {key}", news=False, timeout=8, maximum=5)

    assert key not in str(requests[0].url)
    assert key not in requests[0].content.decode()
    assert found[0]["title"] == "[REDACTED_SECRET]"
    assert found[0]["body"] == "Received [REDACTED_SECRET]"
    # The one permitted credential transmission is authentication to the chosen API.
    authentication = "x-subscription-token" if provider == "brave" else "authorization"
    assert key in requests[0].headers[authentication]
