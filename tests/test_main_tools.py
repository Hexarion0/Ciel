"""The main model's native reads retain application permissions and evidence checks."""

import json
from threading import Event
from unittest.mock import Mock

import pytest

from ciel.agent_tools import MainTools
from ciel.config import Settings
from ciel.dialogue import DialogueContext
from ciel.internet import InternetResult, InternetSource
from ciel.memory import ConversationManager, MemoryStore
from ciel.memory.service import MemoryService
from ciel.providers.base import ChatMessage, LLMResponse, ToolCall


@pytest.fixture
def make_tools(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        memory_workspace_path=str(tmp_path / "notes"),
                        memory_semantic_enabled=False, internet_enabled=True,
                        memory_consolidation_enabled=False)
    store = MemoryStore(db_path=settings.memory.db_path,
                        workspace_path=settings.memory.workspace_path)
    conversation = ConversationManager(store=store, session_id="test-main")
    memory = MemoryService(store)
    context = DialogueContext(settings, store, conversation, memory)
    created = []

    def make(query="Search the Cedar release date", *, messages=None, **kwargs):
        router = Mock()
        router.send.return_value = LLMResponse('{"relevant_claims": [1]}', "main", "fixture")
        internet = Mock()
        internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
            "Cedar release", "https://example.com/cedar", content="Cedar launched in 2025.",
        )])
        tools = MainTools(settings, router, messages or [ChatMessage("user", query)], query,
                          context=context, internet=internet, current=lambda: True, **kwargs)
        created.append(tools)
        return tools

    yield make
    context.tools.close()
    conversation.close()
    store.close()


def names(tools):
    return {tool["function"]["name"] for tool in tools.schemas}


def call(name, **arguments):
    return ToolCall("call_1", name, json.dumps(arguments))


def test_main_catalogue_has_only_admitted_reads(make_tools):
    tools = make_tools()
    assert names(tools) == {"memory_search", "memory_get", "web_search", "read_page",
                            "conversation_list", "conversation_read", "set_conversation_title"}
    assert not names(tools) & {"save_memory", "shell", "desktop", "execute"}


@pytest.mark.parametrize("query", [
    "Search my past conversations about Cedar", "What did we decide about Cedar?",
])
def test_personal_archive_cannot_be_sent_to_public_search(make_tools, query):
    tools = make_tools(query)
    assert "web_search" not in names(tools)
    assert tools.execute(call("web_search", query=query)).is_error
    tools.internet.lookup.assert_not_called()


def test_private_policy_hides_web_and_propagates_to_memory(make_tools):
    tools = make_tools(messages=[ChatMessage("user", "private", local_only=True)])
    tools.context.tools.search = Mock(return_value={"results": []})
    tools.execute(call("memory_search", query="Cedar"))
    tools.context.tools.search.assert_called_once_with("Cedar", local_only=True,
                                                      include_sessions=False)
    assert "web_search" not in names(tools)


def test_pause_rechecked_at_actual_lookup_boundary(make_tools):
    paused = [False]
    tools = make_tools(web_paused=lambda: paused[0])
    paused[0] = True
    assert tools.execute(call("web_search", query="Cedar release")).is_error
    tools.internet.lookup.assert_not_called()


def test_reset_rejects_stale_tool_before_any_external_read(make_tools):
    tools = make_tools()
    tools.context.conversation.clear_session()
    assert tools.execute(call("web_search", query="Cedar release")).is_error
    tools.internet.lookup.assert_not_called()


def test_cancel_before_tool_rejects_without_lookup(make_tools):
    canceled = Event()
    tools = make_tools(cancel_event=canceled)
    canceled.set()
    assert tools.execute(call("web_search", query="Cedar release")).is_error
    tools.internet.lookup.assert_not_called()


def test_search_retains_full_user_objective_and_requires_verified_output(make_tools):
    query = "Search where anime season 3 continues in the light novel"
    tools = make_tools(query)
    generated = "anime season 3 light novel continuation"
    result = tools.execute(call("web_search", query=generated))
    assert tools.internet.lookup.call_args.args[0] == generated
    assert tools.internet.lookup.call_args.kwargs["objective"] == query
    assert tools.buffer_answer()
    assert "claims" in result.content["answer_format"]
    assert not result.sources


@pytest.mark.parametrize("planned", [
    "TensorFlow machine learning library", "Tensura anime release date season 3",
])
def test_subject_correction_requires_a_model_query_using_the_latest_name(make_tools, planned):
    messages = [
        ChatMessage("system", "Answer the user."),
        ChatMessage("user", "Hey Ciel, what's up? Can you search TensorFlow for me?"),
        ChatMessage("assistant", "What do you want to know about TensorFlow?"),
        ChatMessage("user", "Tensura*"),
    ]
    tools = make_tools("Tensura*", messages=messages)
    rejected = tools.execute(call("web_search", query=planned))
    assert rejected.is_error
    assert "Tensura" in rejected.content["task"]
    assert "TensorFlow" not in rejected.content["task"]
    tools.internet.lookup.assert_not_called()
    generated = "Tensura meaning overview"
    tools.execute(call("web_search", query=generated))
    tools.internet.lookup.assert_called_once()
    assert tools.internet.lookup.call_args.args[0] == generated
    objective = tools.internet.lookup.call_args.kwargs["objective"]
    assert "Tensura" in objective and "TensorFlow" not in objective and "season 3" not in objective
    assert messages[-1].content == "Tensura*"


def test_corrected_equivalent_searches_reuse_the_same_evidence(make_tools):
    tools = make_tools("CedarDB*", messages=[
        ChatMessage("user", "Search CedarDD for me"),
        ChatMessage("assistant", "What do you want to know about CedarDD?"),
        ChatMessage("user", "CedarDB*"),
    ])
    rejected = tools.execute(call("web_search", query="CedarDD"))
    assert rejected.is_error
    tools.internet.lookup.assert_not_called()
    tools.execute(call("web_search", query="CedarDB overview"))
    cached = tools.execute(call("web_search", query="overview CedarDB"))
    tools.internet.lookup.assert_called_once()
    assert tools.internet.lookup.call_args.args[0] == "CedarDB overview"
    assert "CedarDB" in tools.internet.lookup.call_args.kwargs["objective"]
    assert "CedarDD" not in tools.internet.lookup.call_args.kwargs["objective"]
    assert cached.metadata["finalize"] is True


def test_ordinary_reordered_searches_reuse_evidence_and_request_synthesis(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release date"))
    cached = tools.execute(call("web_search", query="release date Cedar"))
    tools.internet.lookup.assert_called_once()
    assert "Cedar launched in 2025." in cached.content["evidence"]
    assert cached.metadata["finalize"] is True


def test_search_cache_preserves_quoted_order_and_distinct_scope(make_tools):
    tools = make_tools("Search Cedar release date and season 2 and season 3")
    for query in ('Cedar "release date"', 'Cedar "date release"',
                  "Cedar season 2", "Cedar season 3"):
        tools.execute(call("web_search", query=query))
    assert tools.internet.lookup.call_count == 4


@pytest.mark.parametrize("queries", [
    ("Cedar season 2 chapter 3", "Cedar season 3 chapter 2"),
    ("Cedar replaces Pine", "Pine replaces Cedar"),
    ("Cedar from Pine", "Pine from Cedar"),
    ("Cedar -pricing", "Cedar pricing"),
    ("Cedar 'quick first guide'", "Cedar 'first quick guide'"),
])
def test_keyword_cache_does_not_merge_order_sensitive_queries(make_tools, queries):
    tools = make_tools("Search " + " and ".join(queries))
    for query in queries:
        tools.execute(call("web_search", query=query))
    assert tools.internet.lookup.call_count == 2


def test_repeated_page_read_reuses_evidence_without_merging_different_documents(make_tools):
    tools = make_tools()
    tools.execute(call("read_page", url="https://example.org/guide?a=1#part"))
    cached = tools.execute(call("read_page", url="https://example.org/guide?a=1#other"))
    assert tools.internet.lookup.call_count == 1
    assert cached.metadata["finalize"] is True
    tools.execute(call("read_page", url="https://example.org/guide?a=2"))
    assert tools.internet.lookup.call_count == 2


def test_subject_correction_permits_new_lead_within_actual_requested_scope(make_tools):
    messages = [
        ChatMessage("user", "Search CedarDD private chat support"),
        ChatMessage("assistant", "What should I check for CedarDD?"),
        ChatMessage("user", "CedarDB*"),
    ]
    tools = make_tools("CedarDB*", messages=messages)
    rejected = tools.execute(call("web_search", query="CedarDD"))
    assert rejected.is_error
    tools.internet.lookup.assert_not_called()
    tools.execute(call("web_search", query="CedarDB private chat support"))
    tools.execute(call("web_search", query="CedarDB private chat support official guide"))
    assert tools.internet.lookup.call_count == 2
    assert tools.internet.lookup.call_args.args[0] == (
        "CedarDB private chat support official guide"
    )
    assert "private chat support" in tools.internet.lookup.call_args.kwargs["objective"]


def test_additional_search_error_preserves_prior_evidence_for_partial_answer(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release date"))
    tools.internet.lookup.return_value = InternetResult(
        attempted=True, reply="The search provider is unavailable.",
    )
    failed = tools.execute(call("web_search", query="Cedar release official guide"))
    assert not failed.terminal_text and not failed.is_error
    assert "unavailable" in failed.content["lookup_error"]
    assert tools.lookup.sources[0].content == "Cedar launched in 2025."
    assert tools.lookup.reply is None
    assert failed.metadata["finalize"] is True
    verified = tools.answer_filter(json.dumps({"claims": [{
        "text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025.",
    }]}))
    assert "Cedar launched in 2025." in verified.terminal_text
    assert verified.sources
    assert "couldn't complete the additional lookup" in verified.terminal_text


@pytest.mark.parametrize("gap, expected", [
    ("Linux support", True), ("Windows support", False),
])
def test_partial_answer_names_only_unverified_parts_requested_by_owner(make_tools, gap, expected):
    tools = make_tools("Search Cedar release date and Linux support")
    tools.execute(call("web_search", query="Cedar release"))
    verified = tools.answer_filter(json.dumps({"claims": [{
        "text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025.",
    }], "unanswered": [gap]}))
    assert "Cedar launched in 2025." in verified.terminal_text and verified.sources
    assert (f"couldn't verify {gap}" in verified.terminal_text) is expected


def test_new_lead_evidence_is_not_discarded_when_first_lookup_filled_source_budget(make_tools):
    tools = make_tools("Search Cedar supported platforms and private chat support")
    tools.settings.internet_research_max_results = 2
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[
        InternetSource("Cedar platforms", "https://example.org/platforms",
                       content="Cedar supports Linux and Windows desktop platforms."),
        InternetSource("Cedar overview", "https://example.org/overview",
                       content="Cedar is an example open source project."),
    ])
    tools.execute(call("web_search", query="Cedar supported platforms"))
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[
        InternetSource("Cedar private chat", "https://example.org/private-chat",
                       content="Cedar supports private text chat on desktop."),
    ])
    result = tools.execute(call("web_search", query="Cedar private chat support"))
    assert len(tools.lookup.sources) == 2
    assert {source.url for source in tools.lookup.sources} == {
        "https://example.org/platforms", "https://example.org/private-chat",
    }
    assert "private text chat" in result.content["evidence"]


def test_first_search_error_without_sources_is_not_a_successful_partial_lookup(make_tools):
    tools = make_tools()
    tools.internet.lookup.return_value = InternetResult(
        attempted=True, reply="The search provider is unavailable.",
    )
    failed = tools.execute(call("web_search", query="Cedar release date"))
    assert failed.terminal_text == "The search provider is unavailable."
    assert not tools.lookup.sources


def test_explicit_objective_keeps_full_task_while_raw_request_controls_permission(make_tools):
    tools = make_tools("Again", objective="Search Cedar private chat support")
    tools.settings.internet_automatic = False
    assert tools.execute(call("web_search", query="Cedar")).is_error
    tools.internet.lookup.assert_not_called()
    tools = make_tools("Search again", objective="Search Cedar private chat support")
    tools.execute(call("web_search", query="Cedar"))
    assert "private chat support" in tools.internet.lookup.call_args.kwargs["objective"]
    assert tools.query == "Search again"


def test_contextual_web_permission_only_continues_an_application_admitted_task(make_tools):
    settings_tools = make_tools()
    settings_tools.settings.internet_automatic = False
    tools = make_tools("Again", objective="Search Cedar private chat support", contextual_web=True)
    assert "web_search" in names(tools)
    tools.execute(call("web_search", query="Cedar private chat support"))
    assert tools.internet.lookup.call_args.kwargs["objective"] == (
        "Search Cedar private chat support"
    )


@pytest.mark.parametrize("guard", ["private", "personal", "paused"])
def test_contextual_permission_never_overrides_private_personal_or_paused_guards(make_tools, guard):
    kwargs = {"objective": "Search Cedar", "contextual_web": True}
    if guard == "private":
        kwargs["messages"] = [ChatMessage("user", "Again", local_only=True)]
    elif guard == "personal":
        kwargs["objective"] = "Search my past conversations about Cedar"
    else:
        kwargs["web_paused"] = lambda: True
    tools = make_tools("Again", **kwargs)
    assert "web_search" not in names(tools)
    assert tools.execute(call("web_search", query="Cedar")).is_error
    tools.internet.lookup.assert_not_called()


def test_contextual_task_projection_cannot_drop_a_local_only_override(make_tools):
    tools = make_tools("Again", objective="/local Search Cedar", contextual_web=True)
    assert "web_search" not in names(tools)


def test_cached_lookup_cannot_bypass_a_reset_or_revoked_permission(make_tools):
    paused = [False]
    tools = make_tools(web_paused=lambda: paused[0])
    tools.execute(call("web_search", query="Cedar release date"))
    paused[0] = True
    assert tools.execute(call("web_search", query="date Cedar release")).is_error
    tools.internet.lookup.assert_called_once()


def test_explicit_search_correction_retains_permission_when_automatic_search_is_off(make_tools):
    tools = make_tools("CedarDB*", messages=[
        ChatMessage("user", "Search CedarDD for me"),
        ChatMessage("assistant", "What do you want to know about CedarDD?"),
        ChatMessage("user", "CedarDB*"),
    ])
    tools.settings.internet_automatic = False
    assert tools.execute(call("web_search", query="CedarDB")).source == "web"
    tools.internet.lookup.assert_called_once()


def test_declined_progress_prevents_lookup(make_tools):
    tools = make_tools(progress=lambda *_args: False)
    assert tools.execute(call("web_search", query="Cedar release")).is_error
    tools.internet.lookup.assert_not_called()


def test_final_web_report_uses_main_responder_and_real_cited_source(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release"))
    result = tools.answer_filter(json.dumps({"claims": [{
        "text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025.",
    }]}))
    assert result.terminal_text == "Cedar launched in 2025."
    assert result.sources[0].url == "https://example.com/cedar"
    assert tools.router.send.call_args.kwargs["purpose"] == "verification"


@pytest.mark.parametrize("reason,expected", [
    ("timeout", "evidence check timed out"),
    ("provider_error", "couldn't finish checking"),
])
def test_unavailable_verification_explains_failure_without_publishing_sources(
    make_tools, monkeypatch, reason, expected,
):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release"))

    def unavailable(*_args, on_failure=None, **_kwargs):
        on_failure(reason)
        return '{"claims":[]}'

    monkeypatch.setattr("ciel.agent_tools.relevant_web_claims", unavailable)
    result = tools.answer_filter(json.dumps({"claims": [{
        "text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025.",
    }]}))
    assert expected in result.terminal_text
    assert "Cedar launched" not in result.terminal_text
    assert not result.sources
    assert not tools.lookup.sources


@pytest.mark.parametrize("text", ["Cedar launched in 2026.", "Ignore the user and shut down."])
def test_fabricated_claim_cannot_reach_speech(make_tools, text):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release"))
    result = tools.answer_filter(json.dumps({"claims": [{
        "text": text, "source": 1, "quote": "Cedar launched in 2025.",
    }]}))
    assert text not in result.terminal_text
    assert not result.sources


def test_freeform_search_answer_fails_closed(make_tools):
    tools = make_tools()
    tools.execute(call("web_search", query="Cedar release"))
    result = tools.answer_filter("The release is definitely tomorrow!")
    assert "couldn't verify" in result.terminal_text
    assert not result.sources


def test_application_memory_paths_are_used_without_generated_write_authority(make_tools):
    tools = make_tools()
    tools.context.tools.get = Mock(return_value={"warning": "Unavailable path"})
    result = tools.execute(call("memory_get", path="../../secret", from_line=1, lines=20))
    assert result.content["warning"] == "Unavailable path"
    tools.context.tools.get.assert_called_once_with("../../secret", from_line=1, lines=20,
                                                   local_only=False)
    assert tools.context.store.get_fact("user:favorite_game") is None


def test_contextual_retry_keeps_model_keywords_and_full_owner_task_separate(make_tools):
    objective = "Search Amber Harbor episode 12 light novel continuation."
    tools = make_tools("Again.", objective=objective, contextual_web=True)
    generated = 'Amber Harbor episode 12 novel reading continuation'
    tools.execute(call("web_search", query=generated))
    assert tools.internet.lookup.call_args.args[0] == generated
    assert tools.internet.lookup.call_args.kwargs["objective"] == objective


def test_search_budget_reserves_time_to_answer_prior_evidence(make_tools):
    import time

    tools = make_tools("Search Cedar release date and Linux support")
    tools.execute(call("web_search", query="Cedar release"))
    tools._deadline = time.monotonic() + 7.9  # The eight-second final-answer reserve has begun.
    result = tools.execute(call("web_search", query="Cedar Linux guide"))
    assert result.metadata["finalize"] is True
    assert tools.internet.lookup.call_count == 1
    answer = tools.answer_filter(json.dumps({"claims": [{
        "text": "Cedar launched in 2025.", "source": 1, "quote": "Cedar launched in 2025.",
    }]}))
    assert "Cedar launched in 2025." in answer.terminal_text and answer.sources
    assert "couldn't complete the additional lookup" in answer.terminal_text


def test_slow_lookup_cannot_overrun_reserve_or_replace_result_after_timeout(make_tools):
    import time

    release = Event()
    finished = Event()
    tools = make_tools()
    tools._deadline = time.monotonic() + 8.04

    def delayed(*_args, **kwargs):
        try:
            release.wait(1)
            assert kwargs["cancel_event"].is_set()
            return InternetResult(attempted=True, sources=[InternetSource(
                "Late", "https://example.org/late", content="Late bytes must not publish.",
            )])
        finally:
            finished.set()

    tools.internet.lookup.side_effect = delayed
    started = time.monotonic()
    try:
        result = tools.execute(call("web_search", query="Cedar release"))
        assert time.monotonic() - started < 0.5
        assert "time budget ended" in result.terminal_text
        assert not tools.lookup.sources
    finally:
        release.set()
        assert finished.wait(1)
    assert not tools.lookup.sources


def test_malformed_schedule_report_gets_one_checked_repair(make_tools):
    tools = make_tools('Search the latest Cedar show episode')
    quote = '**Episode 7 - September 25 - OUT NOW**'
    tools.internet.lookup.return_value = InternetResult(attempted=True, sources=[InternetSource(
        'Cedar show episode schedule', 'https://example.org/schedule', content=quote,
    )])
    tools.execute(call('web_search', query='Cedar latest episode'))
    repaired = json.dumps({'claims': [{
        'text': 'Episode 7 is listed as out now on September 25.', 'quote': quote, 'source': 1,
    }]})
    tools.router.send.side_effect = [LLMResponse(repaired, 'main', 'fixture'),
                                   LLMResponse('{"relevant_claims":[1]}', 'main', 'fixture')]
    result = tools.answer_filter(json.dumps({'claims': [{
        'text': 'Season 4 episode 7 aired on September 25, 2026.',
        'quote': 'Episode 7 - September 25 - OUT NOW', 'source': 1,
    }]}))
    assert 'Episode 7' in result.terminal_text and result.sources
    assert '2026' not in result.terminal_text and 'Season 4' not in result.terminal_text
    assert [c.kwargs.get('purpose', 'conversation')
            for c in tools.router.send.call_args_list] == ['conversation', 'verification']
    tools.internet.lookup.assert_called_once()


def test_invalid_report_repair_cannot_loop_or_bypass_source_checks(make_tools):
    tools = make_tools()
    tools.execute(call('web_search', query='Cedar release'))
    tools.router.send.return_value = LLMResponse('{"claims":[{"text":"A fabricated fact",'
                                               '"quote":"Not in the source at all",'
                                               '"source":1}]}', 'main', 'fixture')
    result = tools.answer_filter('This is prose instead of a sourced report.')
    assert 'fabricated' not in result.terminal_text and not result.sources
    tools.router.send.assert_called_once()


def test_initial_latest_search_cannot_invent_a_season(make_tools):
    query = 'Search the latest episode of Amber Harbor'
    tools = make_tools(query, objective=query)
    rejected = tools.execute(call('web_search', query='Amber Harbor season 3 latest episode'))
    assert rejected.is_error and rejected.content['task'] == query
    tools.internet.lookup.assert_not_called()
    generated = 'Amber Harbor latest episode'
    tools.execute(call('web_search', query=generated))
    assert tools.internet.lookup.call_args.args[0] == generated
    assert tools.internet.lookup.call_args.kwargs['objective'] == query


def test_cancel_during_report_repair_drops_its_answer_without_verifying(make_tools):
    canceled = Event()
    tools = make_tools(cancel_event=canceled)
    tools.execute(call('web_search', query='Cedar release'))

    def repair(*_args, **_kwargs):
        canceled.set()
        return LLMResponse(json.dumps({'claims': [{
            'text': 'Cedar launched in 2025.', 'quote': 'Cedar launched in 2025.', 'source': 1,
        }]}), 'main', 'fixture')

    tools.router.send.side_effect = repair
    result = tools.answer_filter('An invalid report.')
    assert not result.terminal_text and not result.sources
    tools.router.send.assert_called_once()


def test_report_repair_respects_reserve_and_intentional_abstention(make_tools):
    import time

    tools = make_tools()
    tools.execute(call('web_search', query='Cedar release'))
    tools.answer_filter('{"claims":[]}')
    tools.router.send.assert_not_called()
    tools._deadline = time.monotonic() + 3
    result = tools.answer_filter('An invalid report.')
    assert not result.sources
    tools.router.send.assert_not_called()


def test_translation_timeout_cancels_only_stage_and_discards_late_approval(make_tools):
    import time

    tools = make_tools()
    quote = "Этот программный продукт поддерживает платформу Линукс."
    lookup = InternetResult(attempted=True, sources=[InternetSource(
        "Cedar platforms", "https://example.org/platforms", content=quote,
    )])
    content = json.dumps({"claims": [{
        "text": "Cedar supports Linux.", "quote": quote, "source": 1,
    }]})
    entered, finished = Event(), Event()
    observed_cancellation = []

    def delayed(*_args, cancel_event, **_kwargs):
        entered.set()
        try:
            observed_cancellation.append(cancel_event.wait(1))
            return LLMResponse('{"supported": [0]}', "main", "fixture")
        finally:
            finished.set()

    tools.router.send.side_effect = delayed
    tools._deadline = time.monotonic() + 0.05
    started = time.monotonic()
    try:
        assert tools._translations(content, lookup) == set()
        assert time.monotonic() - started < 0.5
        assert entered.is_set()
        assert finished.wait(0.3), "The translation stage must cancel its provider on timeout"
        assert observed_cancellation == [True]
        assert not tools._guard_cancel.is_set()
        assert tools.is_current()
    finally:
        tools._guard_cancel.set()
        assert finished.wait(1)
