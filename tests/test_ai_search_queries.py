"""The main responding model writes retrieval keywords; the task stays separate."""

from ciel.providers.base import ChatMessage
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools


def test_contextual_search_sends_the_model_query_without_the_conversation(make_tools):
    goal = "Hey Ciel, do you know about Tensura anime mod in Minecraft?"
    query = "Tensura Minecraft mod CurseForge"
    tools = make_tools("Yes, sure.", objective=goal, contextual_web=True)
    tools.execute(call("web_search", query=query))
    request = tools.internet.lookup.call_args
    assert request.args[0] == query
    assert request.kwargs["search_queries"] == [query]
    assert request.kwargs["objective"] == goal
    tools.router.send.assert_not_called()


def test_batch_only_uses_the_first_ai_query_for_the_selected_search_provider(make_tools):
    goal = "Find the light novel continuation after Tensura season 4."
    queries = ["Tensura season 4 anime light novel adaptation endpoint",
               "Tensura season 4 where to start reading novel"]
    tools = make_tools(goal, objective=goal)
    tools.execute(call("web_search", search_queries=queries))
    request = tools.internet.lookup.call_args
    assert request.args[0] == queries[0]
    assert request.kwargs["search_queries"] == queries
    assert request.kwargs["objective"] == goal


def test_main_model_can_use_a_series_alias_after_an_explicit_name_correction(make_tools):
    messages = [ChatMessage("user", "Search TensorFlow mod for Minecraft"),
                ChatMessage("assistant", "What do you mean?"),
                ChatMessage("user", "Tensura*")]
    query = "That Time I Got Reincarnated as a Slime Minecraft mod"
    tools = make_tools("Tensura*", messages=messages)
    tools.execute(call("web_search", query=query))
    request = tools.internet.lookup.call_args
    assert request.args[0] == query
    assert "Tensura" in request.kwargs["objective"]
    assert "TensorFlow" not in request.kwargs["objective"]


def test_a_missing_or_acknowledgement_query_asks_the_main_model_to_formulate_it(make_tools):
    goal = "Tensura anime mod in Minecraft"
    tools = make_tools("Yes, sure.", objective=goal, contextual_web=True)
    for arguments in ({}, {"query": "Yes, sure."}):
        result = tools.execute(call("web_search", **arguments))
        assert result.is_error
        assert result.content["task"] == goal
        tools.internet.lookup.assert_not_called()
    tools.execute(call("web_search", query="Tensura Minecraft mod wiki"))
    assert tools.internet.lookup.call_args.args[0] == "Tensura Minecraft mod wiki"


def test_a_model_research_lead_cannot_replace_the_requested_outcome(make_tools):
    goal = "Tensura season 4 light novel continuation"
    query = "Tensura season 4 latest episode"
    tools = make_tools("Where should I continue reading?", objective=goal, contextual_web=True)
    tools.execute(call("web_search", query=query))
    request = tools.internet.lookup.call_args
    assert request.args[0] == query
    assert request.kwargs["objective"] == tools.lookup.query == goal
