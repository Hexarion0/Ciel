"""Synthetic evaluation must prove that every search query came from the model."""

import json
import os
import time

import pytest

from ciel.providers.base import LLMResponse, ToolCall
from scripts import evaluate_conversation_search as evaluator


class EvaluationRouter:
    """Emit one tool request and a supported answer without contacting a provider."""

    def __init__(
        self, tool: str = "web_search", *, search_queries: tuple[str, ...] = (),
        later_queries: tuple[str, ...] = (),
    ) -> None:
        self.tool = tool
        self.search_queries = search_queries
        self.later_queries = later_queries
        self.sent_tool = False

    def send(self, messages, **kwargs) -> LLMResponse:
        if kwargs.get("purpose") == "verification":
            return LLMResponse('{"relevant_claims":[1]}', "synthetic", "fixture")
        if not self.sent_tool:
            self.sent_tool = True
            arguments = ({"url": "https://example.org/synthetic/reference"}
                         if self.tool == "read_page" else
                         {"query": " Tensura   Minecraft mod wiki "})
            if self.search_queries:
                arguments["search_queries"] = self.search_queries
            return LLMResponse("", "synthetic", "fixture", tool_calls=(ToolCall(
                "fixture-tool", self.tool, json.dumps(arguments),
            ),))
        fact = evaluator.CASES[0].fact
        return LLMResponse(json.dumps({
            "claims": [{"text": fact, "quote": fact, "source": 1}],
            "search_queries": self.later_queries,
        }), "synthetic", "fixture")


@pytest.fixture(autouse=True)
def isolated_evaluation(monkeypatch):
    """Keep the evaluator independent of production configuration and progress models."""
    for name in tuple(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.setattr("ciel.chat.session.ToolProgress.warm", lambda *_args, **_kwargs: None)


def run_evaluation(router: EvaluationRouter) -> dict:
    return evaluator.evaluate(evaluator.CASES[0], router,
                              deadline=time.monotonic() + 10, maximum=6)


def test_page_read_without_a_search_cannot_pass_the_query_evaluation():
    result = run_evaluation(EvaluationRouter("read_page"))

    assert result["lookups"] and all(call["read_page"] for call in result["lookups"])
    assert result["checks"]["useful_supported_answer"]
    assert not result["checks"]["searched"]
    assert not result["checks"]["queries_written_by_model"]
    assert result["status"] == "failed"


def test_an_unwritten_secondary_query_fails_even_when_the_primary_was_emitted(monkeypatch):
    original = evaluator.FixtureInternet.lookup

    def extra_application_query(self, query, **kwargs):
        kwargs["search_queries"] = [*kwargs.get("search_queries", []), "Unwritten query"]
        return original(self, query, **kwargs)

    monkeypatch.setattr(evaluator.FixtureInternet, "lookup", extra_application_query)
    result = run_evaluation(EvaluationRouter())

    assert result["checks"]["searched"]
    assert result["checks"]["useful_supported_answer"]
    assert not result["checks"]["queries_written_by_model"]
    assert result["status"] == "failed"


def test_a_query_emitted_after_its_lookup_cannot_retroactively_pass(monkeypatch):
    original = evaluator.FixtureInternet.lookup
    future_query = "Tensura Minecraft mod future lead"

    def premature_application_query(self, query, **kwargs):
        kwargs["search_queries"] = [future_query]
        return original(self, future_query, **kwargs)

    monkeypatch.setattr(evaluator.FixtureInternet, "lookup", premature_application_query)
    result = run_evaluation(EvaluationRouter(later_queries=(future_query,)))

    assert any(future_query in call.get("search_queries", []) for call in result["model_calls"])
    assert result["checks"]["searched"]
    assert result["checks"]["useful_supported_answer"]
    assert not result["checks"]["queries_written_by_model"]
    assert result["status"] == "failed"


def test_a_model_query_and_supported_answer_pass_the_evaluation():
    result = run_evaluation(EvaluationRouter())

    assert result["status"] == "passed"
    assert result["checks"]["queries_written_by_model"]


def test_every_model_query_in_a_normalized_batch_can_pass_the_evaluation():
    result = run_evaluation(EvaluationRouter(search_queries=(
        "Tensura Minecraft mod CurseForge", "Tensura Minecraft mod installation",
    )))

    assert result["status"] == "passed"
    assert result["checks"]["queries_written_by_model"]
    assert len(result["lookups"][0]["search_queries"]) == 3
