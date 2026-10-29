"""Missing evidence gets one scoped retry while existing support remains attributable."""

import json
import time
from threading import Event

import pytest

from ciel.internet import InternetResult, InternetSource
from ciel.providers.base import LLMResponse
from tests.test_main_tools import call
from tests.test_main_tools import make_tools as make_tools


def report(text, *, source=1, unanswered=(), queries=()):
    return json.dumps({"claims": [{"text": text, "quote": text, "source": source}],
                       "unanswered": list(unanswered), "search_queries": list(queries)})


def synopsis():
    return InternetResult(attempted=True, sources=[InternetSource(
        "Amber Wings synopsis", "https://example.org/synopsis",
        content="Amber Wings season 4 follows the heroes on a journey.",
    )])


def test_missing_reading_mapping_gets_one_targeted_retry_before_verification(make_tools):
    goal = "Search where Amber Wings season 4 continues in the light novel"
    tools = make_tools(goal)
    mapping = "Amber Wings season 4 ends at light novel volume 11."
    tools.internet.lookup.side_effect = [synopsis(), InternetResult(
        attempted=True, sources=[InternetSource(
            "Amber Wings adaptation guide", "https://example.org/guide", content=mapping,
        )],
    )]
    tools.execute(call("web_search", query="Amber Wings season 4 novel"))
    tools.router.send.side_effect = [
        LLMResponse(report(mapping, source=2), "main", "fixture"),
        LLMResponse('{"relevant_claims":[1]}', "main", "fixture"),
    ]
    query = "Amber Wings season 4 light novel where to start reading"
    result = tools.answer_filter(json.dumps({"claims": [], "search_queries": [query]}))

    assert mapping in result.terminal_text
    assert [source.url for source in result.sources] == ["https://example.org/guide"]
    assert tools.internet.lookup.call_count == 2
    retry = tools.internet.lookup.call_args
    assert retry.kwargs["objective"] == goal
    assert retry.args[0] == query
    assert "start reading" in retry.args[0]
    assert "season 4" in retry.args[0] and "volume 11" not in retry.args[0]
    assert [item.kwargs["purpose"] for item in tools.router.send.call_args_list] == [
        "conversation", "verification",
    ]


@pytest.mark.parametrize("draft", [
    '{"claims":[]}', '{"claims":[],"search_queries":["Yes, sure."]}',
    '{"claims":[],"search_queries":["Amber Wings season 9 light novel"]}',
    '{"claims":[],"search_queries":"Amber Wings adaptation guide"}',
])
def test_recovery_needs_a_usable_query_written_by_the_responding_model(make_tools, draft):
    tools = make_tools("Search Amber Wings season 4 light novel continuation")
    tools.internet.lookup.return_value = synopsis()
    tools.execute(call("web_search", query="Amber Wings season 4 novel"))
    result = tools.answer_filter(draft)
    tools.internet.lookup.assert_called_once()
    assert "couldn't verify" in result.terminal_text


@pytest.mark.parametrize("failure", ["lookup", "duplicate", "synthesis"])
def test_failed_recovery_keeps_original_claim_and_source_numbers(make_tools, failure):
    tools = make_tools("Search Cedar Linux support and offline mode")
    quote = "Cedar supports Linux on desktop computers."
    original = InternetResult(attempted=True, sources=[InternetSource(
        "Cedar platforms", "https://example.org/platforms", content=quote,
    )])
    if failure == "lookup":
        retry = InternetResult(attempted=True, reply="The second lookup timed out.")
    elif failure == "duplicate":
        retry = original
    else:
        retry = InternetResult(attempted=True, sources=[InternetSource(
            "Cedar offline guide", "https://example.org/offline",
            content="Cedar supports offline mode on desktop computers.",
        )])
    tools.internet.lookup.side_effect = [original, retry]
    tools.execute(call("web_search", query="Cedar Linux offline mode"))
    if failure == "synthesis":
        tools.settings.internet_research_max_results = 1
        tools.router.send.side_effect = [RuntimeError("unavailable"),
                                        LLMResponse('{"relevant_claims":[1]}', "main", "fixture")]
    result = tools.answer_filter(report(quote, unanswered=["offline mode"],
                                        queries=["Cedar offline mode documentation"]))

    assert quote in result.terminal_text and "offline mode" in result.terminal_text
    assert [source.url for source in result.sources] == ["https://example.org/platforms"]
    assert tools.internet.lookup.call_count == 2
    expected = ["conversation", "verification"] if failure == "synthesis" else ["verification"]
    assert [item.kwargs["purpose"] for item in tools.router.send.call_args_list] == expected


@pytest.mark.parametrize("guard", ["reserve", "cancel", "paused", "retained", "already_read"])
def test_recovery_respects_ownership_budget_and_prior_research(make_tools, guard):
    cancel, paused = Event(), [False]
    tools = make_tools("Search Amber Wings season 4 light novel continuation",
                       cancel_event=cancel, web_paused=lambda: paused[0])
    tools.internet.lookup.return_value = synopsis()
    tools.execute(call("web_search", query="Amber Wings season 4 novel"))
    if guard == "reserve":
        tools._deadline = time.monotonic() + 8
    elif guard == "cancel":
        cancel.set()
    elif guard == "paused":
        paused[0] = True
    elif guard == "retained":
        tools._fresh_lookup = False
    else:
        tools.execute(call("web_search", query="Amber Wings season 4 adaptation guide"))
    before = tools.internet.lookup.call_count
    tools.answer_filter(json.dumps({"claims": [], "search_queries": [
        "Amber Wings season 4 light novel reading guide",
    ]}))
    assert tools.internet.lookup.call_count == before
    tools.router.send.assert_not_called()


def test_cancel_during_recovery_synthesis_discards_answer_without_verification(make_tools):
    cancel = Event()
    tools = make_tools("Search Amber Wings season 4 light novel continuation", cancel_event=cancel)
    quote = "Amber Wings season 4 ends at light novel volume 11."
    tools.internet.lookup.side_effect = [synopsis(), InternetResult(
        attempted=True, sources=[InternetSource(
            "Amber Wings guide", "https://example.org/guide", content=quote,
        )],
    )]
    tools.execute(call("web_search", query="Amber Wings season 4 novel"))

    def stop(*_args, **_kwargs):
        cancel.set()
        return LLMResponse(report(quote, source=2), "main", "fixture")

    tools.router.send.side_effect = stop
    result = tools.answer_filter(json.dumps({"claims": [], "search_queries": [
        "Amber Wings season 4 light novel reading guide",
    ]}))
    assert result.terminal_text == "" and not result.sources
    tools.router.send.assert_called_once()


def test_partial_recovery_preserves_the_original_supported_requested_part(make_tools):
    tools = make_tools("Search Quorum Linux support and offline mode")
    linux = "Quorum supports Linux on desktop computers."
    offline = "Quorum supports offline mode on desktop computers."
    tools.internet.lookup.side_effect = [
        InternetResult(attempted=True, sources=[InternetSource(
            "Quorum platforms", "https://example.org/platforms", content=linux,
        )]),
        InternetResult(attempted=True, sources=[InternetSource(
            "Quorum offline", "https://example.org/offline", content=offline,
        )]),
    ]
    tools.execute(call("web_search", query="Quorum Linux offline mode"))
    tools.router.send.side_effect = [
        LLMResponse(report(offline, source=2), "main", "fixture"),
        LLMResponse('{"relevant_claims":[1,2]}', "main", "fixture"),
    ]
    result = tools.answer_filter(report(linux, unanswered=["offline mode"],
                                        queries=["Quorum offline mode documentation"]))
    assert linux in result.terminal_text and offline in result.terminal_text
    assert {s.url for s in result.sources} == {
        "https://example.org/platforms", "https://example.org/offline",
    }


def test_page_refresh_keeps_an_earlier_source_passage_reference_valid(make_tools):
    from ciel.web_answers import web_evidence_context

    tools = make_tools('Search what happens to other Cedar tasks when one fails')
    quote = 'When one Cedar task fails, the other tasks are cancelled and then awaited.'
    url = 'https://example.org/cedar/tasks'
    tools.internet.lookup.side_effect = [
        InternetResult(attempted=True, sources=[InternetSource(
            'Cedar tasks', url, content=quote,
        )]),
        InternetResult(attempted=True, sources=[InternetSource(
            'Cedar tasks', url, content='Cedar task guide.\n\n' + quote
            + '\n\nA task group handles structured concurrency.',
        )]),
    ]
    tools.execute(call('web_search', query='Cedar task failure cancellation'))
    source = json.loads(web_evidence_context(tools.lookup).split('\n', 1)[1])['sources'][0]
    passage = next(p for p in source['passages'] if p['text'] == quote)
    tools.execute(call('read_page', url=url))
    result = tools.answer_filter(json.dumps({'claims': [{
        'text': quote, 'source': 1, 'passage': passage['id'],
    }]}))
    assert quote in result.terminal_text
    assert [s.url for s in result.sources] == [url]
    assert tools.lookup.sources[0].content.startswith(quote)


def test_repair_targets_an_unsupported_main_answer_despite_supported_background(make_tools):
    tools = make_tools('Search the latest Cedar release')
    version = 'Download the latest Cedar version 4.7.2 stable build.'
    background = 'Cedar supports Linux on desktop computers.'
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
        'Cedar releases', 'https://example.org/cedar', content=version + '\n\n' + background,
    )])
    tools.execute(call('web_search', query='Cedar latest release'))
    draft = {'claims': [
        {'text': 'Cedar 4.7.2 was released on August 18, 2026.', 'quote': version, 'source': 1},
        {'text': background, 'quote': background, 'source': 1},
    ]}
    # The repaired claim uses the supplied version passage rather than the
    # unsupported date the first draft invented.
    repaired = json.loads(report('The latest Cedar version is 4.7.2.'))
    repaired['claims'][0]['quote'] = version
    tools.router.send.side_effect = [
        LLMResponse(json.dumps(repaired), 'main', 'fixture'),
        LLMResponse('{"relevant_claims":[1]}', 'main', 'fixture'),
    ]
    result = tools.answer_filter(json.dumps(draft))
    assert 'latest Cedar version is 4.7.2' in result.terminal_text
    assert 'August' not in result.terminal_text


def test_repaired_main_answer_and_known_parts_fit_the_claim_limit(make_tools):
    tools = make_tools('Search Cedar latest release, Linux support, and offline mode',
                       channel='text')
    version = 'The latest Cedar release is version 4.7.2.'
    linux = 'Cedar supports Linux on desktop computers.'
    offline = 'Cedar works in offline mode without internet access.'
    extra = ['Cedar has a graphical desktop interface.', 'Cedar includes a built in help page.',
             'Cedar can import several document formats.', 'Cedar supports custom desktop themes.']
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
        'Cedar documentation', 'https://example.org/cedar',
        content='\n\n'.join([version, linux, offline, *extra]),
    )])
    tools.execute(call('web_search', query='Cedar latest release Linux offline'))
    repaired = {'claims': [{'text': text, 'quote': text, 'source': 1}
                          for text in [version, *extra]]}
    tools.router.send.side_effect = [LLMResponse(json.dumps(repaired), 'main', 'fixture'),
                                   LLMResponse('{"relevant_claims":[1,2,3]}', 'main', 'fixture')]
    draft = {'claims': [
        {'text': 'Cedar 4.7.2 was released on August 18, 2026.', 'quote': version, 'source': 1},
        {'text': linux, 'quote': linux, 'source': 1},
        {'text': offline, 'quote': offline, 'source': 1},
    ]}
    result = tools.answer_filter(json.dumps(draft))
    assert all(text in result.terminal_text for text in [version, linux, offline])
    assert 'August' not in result.terminal_text
