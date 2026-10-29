"""Accepting a single lookup offer retains the owner's corrected question."""

import pytest

from ciel.controller import resolve_web_goal
from ciel.providers.base import ChatMessage
from ciel.search_context import replace_search_subject
from ciel.task_context import resolve_task_context
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools

QUESTION = "Hey Ciel, do you know about Tinsura anime mod in Minecraft?"
CORRECTED = "Hey Ciel, do you know about Tensura anime mod in Minecraft?"
OFFER = "I haven't heard of a Tensura mod for Minecraft either. Want me to look it up?"


def _history(offer=OFFER):
    return [
        ChatMessage("user", QUESTION),
        ChatMessage("assistant", "I'm not familiar with that mod."),
        ChatMessage("user", "I meant Tensura."),
        ChatMessage("assistant", offer),
    ]


@pytest.mark.parametrize("question", [
    QUESTION, "Do you know about Tinsura anime mod in Minecraft?",
    "Have you heard of Tinsura anime mod in Minecraft?",
    "Ciel, can you tell me about Tinsura anime mod in Minecraft?",
])
def test_conversational_name_repair_preserves_the_lead_in_and_topic(question):
    assert replace_search_subject(question, "Tensura") == question.replace("Tinsura", "Tensura")


@pytest.mark.parametrize("question,subject,expected", [
    ("Search Ciel offline support", "Ciel Plus", "Search Ciel Plus offline support"),
    ("Do you know about Ciel offline support?", "Ciel Plus",
     "Do you know about Ciel Plus offline support?"),
    ("Search Know offline support", "Knowledge", "Search Knowledge offline support"),
    ("Hey Arnold anime continuation?", "Tensura", "Tensura anime continuation?"),
    ("Hello Neighbor game release date?", "Tensura", "Tensura game release date?"),
])
def test_conversational_framing_words_can_still_be_the_actual_named_subject(
    question, subject, expected,
):
    assert replace_search_subject(question, subject) == expected


@pytest.mark.parametrize("acceptance", ["Yes, sure.", "Yes", "Sure!", "Go ahead", "Yes please"])
@pytest.mark.parametrize("offer", [
    OFFER, "Would you like me to search the web for it?",
    "Shall I look up Tensura anime mod in Minecraft?",
    "Do you want me to search for Tensura anime mod in Minecraft?",
    "Want me to look it up for you?", "Want me to search for information about it?",
])
def test_accepting_one_lookup_offer_retains_only_the_corrected_owner_task(acceptance, offer):
    history = _history(offer)
    task = resolve_task_context(acceptance, history)
    assert task.kind == "acceptance" and task.task_type == "web"
    assert task.request == CORRECTED and task.corrected_subject == "Tensura"
    assert task.anchor_index == 0
    goal = resolve_web_goal(acceptance, history, acceptance)
    assert "Tensura anime mod in Minecraft" in goal and "Tinsura" not in goal
    assert goal == CORRECTED
    assert history[0].content == QUESTION and history[2].content == "I meant Tensura."


@pytest.mark.parametrize("offer", [
    "", "That's a Minecraft mod.", "Want me to check your screen?",
    "Want me to save that in memory?", "Want me to install it?",
    'She asked: "Want me to look it up?"', "She asked: Want me to look it up?",
    "Want me to look it up or explain what I remember?",
    "Want me to search it and install it?",
    "Want me to search Violet Castle season 9?",
    "Want me to look it up? Or should I explain it?",
])
def test_an_acceptance_without_one_grounded_public_lookup_offer_is_not_a_web_task(offer):
    task = resolve_task_context("Yes, sure.", _history(offer))
    assert task.task_type == "conversation" and task.kind == "new"
    assert task.request == "Yes, sure."


@pytest.mark.parametrize("boundary", [
    "Stop", "Don't search", "Tell me a joke", "Thanks", "Wait",
])
def test_acceptance_cannot_revive_an_offer_after_an_intervening_owner_turn(boundary):
    history = [*_history(), ChatMessage("user", boundary), ChatMessage("assistant", "Okay.")]
    task = resolve_task_context("Yes, sure.", history)
    assert task.task_type == "conversation" and task.request == "Yes, sure."


@pytest.mark.parametrize("question", [
    "Don't search Tensura", 'She said "Search Tensura"', '"Search Tensura"',
    "Thanks", "Yes, sure.",
])
def test_an_offer_does_not_turn_quoted_negated_or_empty_task_words_into_a_search(question):
    task = resolve_task_context("Yes", [
        ChatMessage("user", question), ChatMessage("assistant", "Want me to look it up?"),
    ])
    assert task.kind == "new" and task.task_type == "conversation"


def test_lookup_offer_can_include_politeness_absent_from_the_owners_question():
    task = resolve_task_context("Yes", [
        ChatMessage("user", "Tensura mod for Minecraft?"),
        ChatMessage("assistant", "Want me to look it up for you?"),
    ])
    assert task.kind == "acceptance" and task.request == "Tensura mod for Minecraft?"


@pytest.mark.parametrize("arguments", [{}, {"query": "Yes, sure."}, {"query": "Tinsura"}])
def test_native_lookup_returns_the_accepted_task_for_bad_or_missing_model_keywords(
    make_tools, arguments,
):
    task = resolve_task_context("Yes, sure.", _history())
    tools = make_tools("Yes, sure.", messages=_history(), objective=task.request,
                       contextual_web=task.task_type == "web")
    rejected = tools.execute(call("web_search", **arguments))
    assert rejected.is_error and rejected.content["task"] == CORRECTED
    tools.internet.lookup.assert_not_called()

    query = "Tensura Minecraft mod wiki"
    tools.execute(call("web_search", query=query))
    request = tools.internet.lookup.call_args
    assert request.args[0] == query
    assert request.kwargs["objective"] == CORRECTED
    assert request.kwargs["search_queries"] == [query]


def test_accepted_corrected_lookup_can_use_an_ai_written_alias(make_tools):
    task = resolve_task_context("Yes, sure.", _history())
    tools = make_tools("Yes, sure.", messages=_history(), objective=task.request,
                       contextual_web=True)
    query = "That Time I Got Reincarnated as a Slime Minecraft mod"

    tools.execute(call("web_search", query=query))

    request = tools.internet.lookup.call_args
    assert request.args[0] == query
    assert request.kwargs["search_queries"] == [query]
    assert request.kwargs["objective"] == CORRECTED
