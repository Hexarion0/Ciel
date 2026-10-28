"""Parallel receives the desired answer separately from search keywords."""

import json
import time
from unittest.mock import Mock

from ciel.config import InternetConfig
from ciel.internet import InternetService
from ciel.parallel_search import search_parallel
from tests.test_parallel_search import SESSION, _handshake, _results
from tests.test_parallel_search import transport as parallel_transport

transport = parallel_transport


def test_parallel_keeps_original_objective_when_keywords_only_name_an_episode(transport):
    requests, _options, responses = transport
    _handshake(responses)
    _results(responses, [])
    objective = "Find which light novel volume follows Amber Wings anime episode 12."

    search_parallel("Amber Wings anime episode 12", objective=objective,
                    timeout=8, maximum=5, session_id=SESSION)

    arguments = json.loads(requests[-1].content)["params"]["arguments"]
    assert arguments["objective"] == objective
    assert arguments["search_queries"] == ["Amber Wings anime episode 12"]


def test_internet_passes_resolved_question_to_parallel_without_unrelated_history(monkeypatch):
    search = Mock(return_value=[])
    monkeypatch.setattr("ciel.parallel_search.search_parallel", search)
    service = InternetService(InternetConfig(provider="parallel"))
    objective = "Find which novel volume follows the requested anime episode."

    service._search("Amber Wings episode 12", 5, time.monotonic() + 10, None,
                    objective=objective)

    assert search.call_args.args == ("Amber Wings episode 12",)
    assert search.call_args.kwargs["objective"] == objective


def test_lookup_keeps_objective_through_retrieval_boundary(monkeypatch):
    service = InternetService(InternetConfig(provider="parallel", max_pages=0))
    search = Mock(return_value=[])
    monkeypatch.setattr(service, "_search", search)

    result = service.lookup("Amber Wings episode 12", objective="Find the novel continuation.")

    assert result.attempted
    assert search.call_args.kwargs["objective"] == "Find the novel continuation."
