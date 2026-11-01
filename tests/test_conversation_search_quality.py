"""Reported spoken corrections and failed research recover without repeating filler."""

import json

import pytest

from ciel.agent import MainAgent, ToolResult
from ciel.controller import resolve_web_goal
from ciel.internet import InternetResult, InternetSource, _relevant
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.search_context import replace_search_subject
from ciel.task_context import resolve_task_context
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools
from tests.test_unified_agent import TOOLS, Transport, chunk, request

QUESTION = "Hey Ciel, do you know about like Tenzura mod in Minecraft?"
CORRECTED = QUESTION.replace("Tenzura", "Tensura")


def history():
    return [
        ChatMessage("user", QUESTION),
        ChatMessage("assistant", "I'm not familiar with that mod. Tell me more?"),
        ChatMessage("user", "Tensura."),
        ChatMessage("assistant", "Tensura is a Minecraft mod inspired by an anime."),
    ]


def test_spoken_filler_is_not_replaced_instead_of_the_subject():
    assert replace_search_subject(QUESTION, "Tensura") == CORRECTED


def test_bare_close_spelling_correction_keeps_owner_supplied_mod_context():
    messages = history()[:2]
    task = resolve_task_context("Tensura.", messages)
    assert task.kind == "correction" and task.corrected_subject == "Tensura"
    assert task.request == CORRECTED and task.anchor_index == 0
    assert messages[0].content == QUESTION


def test_gameplay_followup_retains_minecraft_without_duplicating_the_new_request():
    query = "I'm currently playing the mod and can you research it so I can understand it?"
    goal = resolve_web_goal(query, history(), "Tensura Minecraft mod beginner guide")
    assert "Minecraft" in goal and "Tensura" in goal and "Tenzura" not in goal
    assert goal.count(query) == 1


@pytest.mark.parametrize("text", ["Naruto.", "Thanks.", "Clear.", "What is Tensura?"])
def test_unrelated_bare_turn_is_not_a_name_repair(text):
    task = resolve_task_context(text, history()[:2])
    assert task.kind != "correction"


@pytest.mark.parametrize("title,url,accepted", [
    ("Oshi no Ko anime synopsis", "https://example.org/oshi", True),
    ("Anime synopsis", "https://example.org/oshi-no-ko", True),
    ("Oshi news about Ko anime", "https://example.org/unrelated", False),
    ("Naruto anime synopsis", "https://example.org/naruto", False),
])
def test_phonetic_compound_title_can_reach_verification(title, url, accepted):
    source = InternetSource(title, url, content="An anime about a doctor and an idol.")
    assert _relevant("What is Osinoko anime about?", source) is accepted


def test_compound_name_matching_does_not_override_explicit_season():
    source = InternetSource("Oshi no Ko season 2", "https://example.org/oshi-no-ko",
                            content="Oshi no Ko season 2 follows the characters.")
    assert not _relevant("Osinoko anime season 4 plot", source)


@pytest.mark.parametrize("initial_sources", [False, True])
def test_empty_general_report_can_try_a_model_proposed_new_lead(make_tools, initial_sources):
    tools = make_tools("What is Osinoko anime about?")
    fact = "Oshi no Ko follows a doctor reborn as the child of an idol."
    first = [InternetSource(
        "Anime index", "https://example.org/index",
        content="An index of many different anime series.",
    )] if initial_sources else []
    tools.internet.lookup.side_effect = [
        InternetResult(attempted=True, sources=first),
        InternetResult(attempted=True, sources=[InternetSource(
            "Oshi no Ko synopsis", "https://example.org/synopsis", content=fact,
        )]),
    ]
    tools.execute(call("web_search", query="Osinoko anime about plot"))
    tools.router.send.side_effect = [
        LLMResponse(json.dumps({"claims": [{"text": fact, "quote": fact, "source": 2
                                             if initial_sources else 1}]}), "main", "fixture"),
        LLMResponse('{"relevant_claims":[1]}', "main", "fixture"),
    ]
    answer = tools.answer_filter(json.dumps({
        "claims": [], "search_queries": ["Oshi no Ko anime synopsis"],
    }))
    assert fact in answer.terminal_text
    assert tools.internet.lookup.call_count == 2
    assert tools.internet.lookup.call_args.args[0] == "Oshi no Ko anime synopsis"
    assert [source.url for source in answer.sources] == ["https://example.org/synopsis"]


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_failure_after_lookup_still_uses_the_evidence_filter(streaming):
    available = [False]
    filtered = []

    def execute(_call):
        available[0] = True
        return ToolResult({"evidence": "No sources returned."}, source="web")

    def fail():
        yield chunk("Unchecked draft")
        raise RuntimeError("Provider connection failed")

    def verify(text):
        filtered.append(text)
        return ToolResult("No supported answer was found.", source="web")

    router = Transport([request()], fail)
    agent = MainAgent(router, TOOLS, execute, buffer_answer=lambda: available[0],
                      answer_filter=verify)
    messages = [ChatMessage("user", "What is that anime about?")]
    answer = ("".join(item.delta for item in agent.stream(messages)) if streaming
              else agent.send(messages).content)
    assert answer == "No supported answer was found."
    assert filtered == [""] and agent.last_result.error == "provider_error"


def test_a_new_query_returning_unchanged_evidence_requests_final_synthesis(make_tools):
    tools = make_tools("Search Cedar release date")
    tools.execute(call("web_search", query="Cedar release date"))
    second = tools.execute(call("web_search", query="Cedar official release notes"))
    assert tools.internet.lookup.call_count == 2
    assert second.metadata["finalize"] is True


def test_rejected_extra_claim_does_not_add_a_generic_repeated_disclaimer(make_tools):
    tools = make_tools("Search Cedar release date")
    tools.execute(call("web_search", query="Cedar release date"))
    answer = tools.answer_filter(json.dumps({"claims": [
        {"text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025."},
        {"text": "Cedar supports lunar travel.", "source": 1,
         "quote": "Cedar launched in 2025."},
    ]}))
    assert answer.terminal_text == "Cedar launched in 2025."


def test_empty_search_does_not_claim_that_related_pages_were_found(make_tools):
    tools = make_tools("Search Cedar release date")
    tools.internet.lookup.return_value = InternetResult(attempted=True)
    tools.execute(call("web_search", query="Cedar release date"))
    answer = tools.answer_filter('{"claims":[]}')
    assert "no usable sources" in answer.terminal_text
    assert "found related pages" not in answer.terminal_text


@pytest.mark.parametrize("query,repeat_expected", [
    ("I'm playing the mod. Research it so I can understand how to play.", False),
    ("Explain it again, including the quest journal step.", True),
    ("Tell me more.", False),
])
def test_followup_omits_an_exact_previously_spoken_definition(make_tools, query, repeat_expected):
    definition = "Ambercraft is a Minecraft mod with races and skills."
    step = "Open the quest journal to see the first progression tasks."
    messages = [
        ChatMessage("user", "Do you know about the Ambercraft Minecraft mod?"),
        ChatMessage("assistant", definition),
        ChatMessage("user", query),
    ]
    tools = make_tools(query, messages=messages)
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
        "Ambercraft guide", "https://example.org/guide", content=definition + " " + step,
    )])
    tools.execute(call("web_search", query="Ambercraft Minecraft mod gameplay"))

    def approve(sent, **_kwargs):
        payload = json.loads(sent[-1].content)
        return LLMResponse(json.dumps({"relevant_claims": [
            claim["index"] for claim in payload["claims"]
        ]}), "main", "fixture")

    tools.router.send.side_effect = approve
    answer = tools.answer_filter(json.dumps({"claims": [
        {"text": definition, "quote": definition, "source": 1},
        {"text": step, "quote": step, "source": 1},
    ]}))
    assert (definition in answer.terminal_text) is repeat_expected
    assert step in answer.terminal_text and answer.sources
