"""Direct web mode hands raw numbered results to the model and trusts its answer."""

import json
from unittest.mock import Mock

import pytest

from ciel.agent_tools import MainTools
from ciel.config import InternetConfig, Settings
from ciel.dialogue import DialogueContext
from ciel.direct_web import DirectWeb, answer_instructions
from ciel.internet import InternetService, InternetSource
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.service import MemoryService
from ciel.providers.base import ChatMessage, ToolCall

ROWS = [
    {"title": "Spain win the 2026 World Cup", "href": "https://www.fifa.com/final",
     "body": "Spain beat Argentina 1-0."},
    # Shares no query words; verified mode's keyword filter would drop it.
    {"title": "Ferran Torres seals it in extra time", "href": "https://news.example/torres",
     "body": "A 106th-minute winner in New Jersey."},
    {"title": "Same story again", "href": "https://www.fifa.com/final#top", "body": "Dup."},
    {"title": "Private", "href": "http://127.0.0.1/admin", "body": "Not public."},
]


def call(name, **arguments):
    return ToolCall("call_1", name, json.dumps(arguments))


@pytest.fixture
def make_tools(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        memory_workspace_path=str(tmp_path / "notes"),
                        memory_semantic_enabled=False, internet_enabled=True,
                        memory_consolidation_enabled=False, internet_mode="direct")
    store = MemoryStore(db_path=settings.memory.db_path,
                        workspace_path=settings.memory.workspace_path)
    conversation = ConversationManager(store=store, session_id="test-direct")
    context = DialogueContext(settings, store, conversation, MemoryService(store))

    def make(query="who won the 2026 world cup", **kwargs):
        internet = Mock()
        internet.search_direct.return_value = [
            InternetSource(f"Result {n}", f"https://r{n}.example/", f"Snippet {n}")
            for n in range(1, 4)
        ]
        return MainTools(settings, Mock(), [ChatMessage("user", query)], query,
                         context=context, internet=internet, current=lambda: True, **kwargs)

    yield make
    context.tools.close()
    conversation.close()
    store.close()


def test_mode_setting_defaults_to_verified():
    assert InternetConfig().mode == "verified"
    assert Settings(_env_file=None, internet_mode="direct").internet.mode == "direct"


def test_search_direct_keeps_unmatched_results_and_drops_unsafe_ones(monkeypatch):
    service = InternetService(InternetConfig(provider="yahoo"))
    monkeypatch.setattr(service, "_search", Mock(return_value=ROWS))

    sources = service.search_direct(["2026 world cup winner"], deadline=1e12)

    assert [source.url for source in sources] == [
        "https://www.fifa.com/final", "https://news.example/torres",
    ]
    assert sources[1].snippet == "A 106th-minute winner in New Jersey."


@pytest.mark.parametrize(("provider", "answered_by", "expected"), [
    ("duckduckgo", "duckduckgo", ["first"]),
    ("duckduckgo", "parallel", ["first", "second"]),
    ("yahoo", "yahoo", ["first", "second"]),
])
def test_search_direct_reports_the_phrasings_actually_searched(
    monkeypatch, provider, answered_by, expected,
):
    service = InternetService(InternetConfig(provider=provider))

    def search(*_args, **_kwargs):
        service._backend.name = answered_by
        return ROWS

    monkeypatch.setattr(service, "_search", search)
    searched = []

    service.search_direct(["first", "second"], deadline=1e12, searched=searched)

    assert searched == expected


def test_direct_calls_refuse_when_web_retrieval_is_disabled(monkeypatch):
    from ciel.search_providers import SearchProviderError

    service = InternetService(InternetConfig(enabled=False))
    monkeypatch.setattr(service, "_search", Mock(side_effect=AssertionError("searched")))
    monkeypatch.setattr(service, "_read_page", Mock(side_effect=AssertionError("read")))

    with pytest.raises(SearchProviderError, match="disabled"):
        service.search_direct(["anything"], deadline=1e12)
    with pytest.raises(ValueError, match="disabled"):
        service.read_direct("https://example.com/", deadline=1e12)


def test_unsearched_phrasings_are_named_for_the_model():
    internet = Mock()

    def search_direct(queries, *, searched, **_kwargs):
        searched.append(queries[0])
        return [InternetSource("A", "https://a.example/", "a")]

    internet.search_direct.side_effect = search_direct
    web = DirectWeb(internet, query="q")

    payload, failed = web.search(["paris weather", "rome weather"], deadline=1e12,
                                 cancel_event=None, run=lambda work: work())

    assert not failed
    assert "search again for: rome weather" in payload["note"]


def test_source_numbers_stay_stable_and_reads_fill_in_text():
    internet = Mock()
    internet.search_direct.side_effect = [
        [InternetSource("A", "https://a.example/", "a"), InternetSource("B", "https://b.example/")],
        [InternetSource("B", "https://b.example/"), InternetSource("C", "https://c.example/")],
    ]
    internet.read_direct.return_value = InternetSource("B page", "https://b.example/",
                                                       content="Full text of B.")
    web = DirectWeb(internet, query="q")

    def run(work):
        return work()

    first, _ = web.search(["one"], deadline=1e12, cancel_event=None, run=run)
    second, _ = web.search(["two"], deadline=1e12, cancel_event=None, run=run)
    page, failed = web.read("https://b.example/", deadline=1e12, cancel_event=None, run=run)

    assert [row["source"] for row in first["results"]] == [1, 2]
    assert [row["source"] for row in second["results"]] == [2, 3]
    assert not failed and page["source"] == 2 and "[2]" in page["note"]
    assert web.sources[1].content == "Full text of B." and web.sources[1].snippet == ""
    repeat, failed = web.search(["one"], deadline=1e12, cancel_event=None, run=run)
    assert failed and "already ran" in repeat["error"]


def test_direct_schemas_drop_the_claims_format(make_tools):
    tools = make_tools()
    descriptions = {tool["function"]["name"]: tool["function"]["description"]
                    for tool in tools.schemas}

    assert "claims JSON" not in descriptions["web_search"]
    assert "numbered results" in descriptions["web_search"]
    assert "already have happened" in descriptions["web_search"]


def test_direct_search_returns_numbered_results_to_the_model(make_tools):
    tools = make_tools(channel="text")

    result = tools.execute(call("web_search", query="2026 World Cup winner",
                                search_queries=["World Cup final 2026"]))

    assert not result.is_error
    assert [row["source"] for row in result.content["results"]] == [1, 2, 3]
    assert "untrusted" in result.content["note"]
    tools.internet.search_direct.assert_called_once()
    assert tools.internet.search_direct.call_args.args[0] == [
        "2026 World Cup winner", "World Cup final 2026",
    ]
    tools.internet.lookup.assert_not_called()
    assert tools.buffer_answer()
    assert tools.answer_instructions() == answer_instructions("text")


def test_failed_page_read_is_not_a_tool_failure(make_tools):
    tools = make_tools()
    tools.internet.read_direct.side_effect = OSError("blocked")

    result = tools.execute(call("read_page", url="https://weather.example/khovd"))

    assert not result.is_error
    assert result.content["error"] == "The page could not be read."


def test_direct_answer_is_trusted_and_citations_match_the_list(make_tools, monkeypatch):
    tools = make_tools()
    tools.execute(call("web_search", query="2026 World Cup winner"))
    monkeypatch.setattr("ciel.agent_tools.relevant_web_claims",
                        Mock(side_effect=AssertionError("direct mode runs no claim checks")))

    result = tools.answer_filter("Spain won [3], beating Argentina [3][1].")

    assert result.content == "Spain won [2], beating Argentina [2][1]."
    assert [source.url for source in result.sources] == [
        "https://r1.example/", "https://r3.example/",
    ]


def test_uncited_direct_answer_shows_no_sources(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="2026 World Cup winner"))

    result = tools.answer_filter("Spain won the final.")

    assert result.content == "Spain won the final."
    assert result.sources == ()


def test_earlier_evidence_keeps_its_numbers_and_new_results_continue(make_tools):
    from ciel.direct_web import prior_evidence
    from ciel.internet import InternetResult

    previous = InternetResult(attempted=True, sources=[
        InternetSource("Old final report", "https://old.example/", "Spain won."),
    ])
    tools = make_tools(previous=previous)

    shown = json.loads(prior_evidence(previous).split("\n", 1)[1])
    assert shown["results"][0]["source"] == 1
    assert tools.buffer_answer()
    result = tools.execute(call("web_search", query="2026 World Cup final scorer"))
    assert [row["source"] for row in result.content["results"]] == [2, 3, 4]
    answer = tools.answer_filter("Spain won [1]; Torres scored [3].")
    assert answer.content == "Spain won [1]; Torres scored [2]."
    assert [s.url for s in answer.sources] == ["https://old.example/", "https://r2.example/"]


def test_empty_draft_gets_a_truthful_fallback_with_sources(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="2026 World Cup winner"))

    result = tools.answer_filter("")

    assert "couldn't finish" in result.content
    assert len(result.sources) == 3


def test_list_and_range_citations_stay_correct(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="2026 World Cup winner"))

    listed = tools.answer_filter("Facts [3] and also [1, 3]; (Reuters)[3].")
    assert listed.content == "Facts [2] and also [1, 2]; (Reuters)[2]."
    ranged = tools.answer_filter("All agree [1-3], see [3].")
    assert ranged.content == "All agree [1-3], see [3]."
    assert len(ranged.sources) == 3


def test_indexing_and_code_are_not_citations():
    from ciel.direct_web import cited_sources

    sources = [InternetSource(f"S{n}", f"https://s{n}.example/") for n in range(1, 4)]

    assert cited_sources("Use xs[2] and `ys[1]` then\n```\nzs[3]\n```", sources) == []
    assert cited_sources("It says so [2].", sources) == [sources[1]]


def test_redirected_read_keeps_the_result_number():
    internet = Mock()
    internet.search_direct.return_value = [
        InternetSource("A", "http://a.example/story", "snippet"),
    ]
    internet.read_direct.return_value = InternetSource(
        "A", "https://www.a.example/story", content="Full story.",
    )
    web = DirectWeb(internet, query="q")

    def run(work):
        return work()

    web.search(["a"], deadline=1e12, cancel_event=None, run=run)
    page, _ = web.read("http://a.example/story", deadline=1e12, cancel_event=None, run=run)

    assert page["source"] == 1 and len(web.sources) == 1
    assert web.sources[0].content == "Full story."


def test_direct_queries_are_secret_masked(monkeypatch):
    service = InternetService(InternetConfig(provider="yahoo"))
    search = Mock(return_value=[])
    monkeypatch.setattr(service, "_search", search)

    service.search_direct(["deploy failing api_key=sk-live-abc123def456"], deadline=1e12)

    assert "sk-live-abc123def456" not in json.dumps(search.call_args.args[0])
    assert "sk-live-abc123def456" not in json.dumps(search.call_args.kwargs)


def _with_earlier_cited_reply(tools):
    tools.messages = [ChatMessage("user", "who won?"),
                      ChatMessage("assistant", "Spain won [1], beating Argentina [2]."),
                      *tools.messages]
    return tools


def test_answer_without_this_turns_sources_drops_copied_markers(make_tools):
    tools = _with_earlier_cited_reply(make_tools(channel="text"))

    result = tools.answer_filter("Torres scored in the 106th minute [1][2]. Code `xs[1]`.")

    assert result.content == "Torres scored in the 106th minute. Code `xs[1]`."


def test_answer_without_sources_keeps_bracketed_numbers_that_were_not_copied(make_tools):
    fresh = make_tools(channel="text")
    earlier = _with_earlier_cited_reply(make_tools(channel="text"))
    text = "x is in [0, 1], step [3] repeats, and the list [1, 2] is sorted."

    assert fresh.answer_filter(text).content == text
    # [1, 2] matches markers from the earlier reply; [0, 1] and [3] never appeared.
    assert earlier.answer_filter(text).content == (
        "x is in [0, 1], step [3] repeats, and the list is sorted.")
