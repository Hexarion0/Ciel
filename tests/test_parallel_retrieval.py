"""Parallel excerpts remain attributable evidence without redundant page downloads."""

import json
import threading
import time
from unittest.mock import Mock
from uuid import UUID

import pytest

from ciel.config import InternetConfig
from ciel.internet import InternetService
from ciel.providers.base import ChatMessage
from ciel.search_providers import SearchProviderError
from ciel.web_answers import verified_web_answer, web_answer_messages


def test_parallel_excerpts_support_answers_without_another_page_download(monkeypatch):
    service = InternetService(InternetConfig(provider="parallel", max_pages=2))
    quote = "Python supports asynchronous generators."
    monkeypatch.setattr(service, "_search", Mock(return_value=[{
        "href": "https://docs.python.org/3/reference/expressions.html",
        "title": "Python expressions", "body": "Python language reference.",
        "content": quote, "date": "2026-10-20", "date_kind": "published_or_modified",
    }]))
    reader = Mock(side_effect=OSError("A second download should be unnecessary"))
    monkeypatch.setattr(service, "_read_page", reader)

    result = service.lookup("Python asynchronous generators")

    assert len(result.sources) == 1
    assert result.sources[0].content == quote
    assert result.sources[0].content_kind == "provider_excerpt"
    reader.assert_not_called()
    context = json.loads(result.context.split("\n", 1)[1])
    assert context["outcome"] == "searched"
    assert context["sources"][0]["kind"] == "provider_excerpt"
    assert context["sources"][0]["page_excerpt"] == quote
    answer = json.dumps({"claims": [{"text": quote, "source": 1, "quote": quote}]})
    assert verified_web_answer(answer, result) == quote
    messages = web_answer_messages([
        ChatMessage("system", "Answer accurately."),
        ChatMessage("user", "What does Python support?"),
    ], result)
    assert any(quote in message.content for message in messages)


def test_parallel_relevance_uses_the_excerpt_beyond_the_search_preview(monkeypatch):
    service = InternetService(InternetConfig(provider="parallel", max_pages=0))
    monkeypatch.setattr(service, "_search", Mock(return_value=[{
        "href": "https://docs.python.org/3/library/asyncio.html",
        "title": "Language documentation", "body": "Read the language reference.",
        "content": "Python asyncio provides asynchronous networking and concurrency.",
    }]))

    result = service.lookup("Python asyncio networking")

    assert len(result.sources) == 1
    assert "Python asyncio" in result.sources[0].content


@pytest.mark.parametrize("provider", ["ddgs", "brave", "tavily"])
def test_other_providers_cannot_promote_an_arbitrary_content_field(monkeypatch, provider):
    service = InternetService(InternetConfig(provider=provider, max_pages=0))
    monkeypatch.setattr(service, "_search", Mock(return_value=[{
        "href": "https://docs.python.org/3/", "title": "Python reference",
        "body": "Python reference documentation.", "content": "Unverified extra provider field.",
    }]))

    result = service.lookup("Python reference")

    assert len(result.sources) == 1
    assert not result.sources[0].content
    assert "Unverified extra" not in result.context


def test_parallel_excerpt_still_obeys_public_url_validation(monkeypatch):
    service = InternetService(InternetConfig(provider="parallel", max_pages=0))
    monkeypatch.setattr(service, "_search", Mock(return_value=[{
        "href": "http://127.0.0.1/private", "title": "Python reference",
        "body": "Python reference documentation.", "content": "Python private network data.",
    }]))

    assert not service.lookup("Python reference").sources


def test_parallel_selection_sends_only_query_and_stable_random_session(monkeypatch):
    search = Mock(return_value=[])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", search)
    ddgs = Mock(side_effect=AssertionError("The selected provider must remain Parallel"))
    monkeypatch.setattr("ddgs.DDGS", ddgs)
    service = InternetService(InternetConfig(
        provider="parallel", brave_api_key="unused-brave", tavily_api_key="unused-tavily",
    ))

    service._search("Python documentation", 5, time.monotonic() + 10, None)
    service._search("Python generators", 5, time.monotonic() + 10, None)
    first, second = search.call_args_list
    assert first.args == ("Python documentation",)
    assert first.kwargs["session_id"] == second.kwargs["session_id"]
    assert UUID(first.kwargs["session_id"]).version == 4
    assert 0 < first.kwargs["timeout"] <= 10
    assert first.kwargs["maximum"] == 10
    assert set(first.kwargs) == {"session_id", "timeout", "maximum", "cancel_event"}
    assert first.kwargs["cancel_event"] is None
    service.reset_session()
    service._search("Python generators", 5, time.monotonic() + 10, None)
    assert search.call_args.kwargs["session_id"] != first.kwargs["session_id"]
    ddgs.assert_not_called()


def test_parallel_error_is_reported_without_switching_providers(monkeypatch):
    search = Mock(side_effect=SearchProviderError("Parallel Search is temporarily rate limited."))
    monkeypatch.setattr("ciel.parallel_search.search_parallel", search)
    service = InternetService(InternetConfig(provider="parallel"))

    result = service.lookup("Python documentation")

    assert result.attempted and not result.sources
    assert result.reply == "Parallel Search is temporarily rate limited."
    search.assert_called_once()


def test_private_or_canceled_requests_never_contact_parallel(monkeypatch):
    search = Mock()
    monkeypatch.setattr("ciel.parallel_search.search_parallel", search)
    service = InternetService(InternetConfig(provider="parallel"))
    canceled = threading.Event()
    canceled.set()

    assert not service.lookup("private query", local_only=True).attempted
    assert not service.lookup("canceled query", cancel_event=canceled).attempted
    search.assert_not_called()
