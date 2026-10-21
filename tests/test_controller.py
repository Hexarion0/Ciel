"""AI decisions precede tools and remain bounded, private, and cancellable."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.providers.base import ChatMessage, LLMResponse


def _response(action, **fields):
    return LLMResponse(json.dumps({"action": action, "confidence": 0.98, **fields}), "test", "nim")


def test_controller_keeps_retry_context_without_keyword_query_stripping():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response(
        "search", query="Tensura season 4 part 2 anime light novel continuation",
        speech="I'll check where that part continues in the novel.",
    )
    controller = AIController(router, ControllerConfig())
    decision = controller.decide("Okay, try again", [
        ChatMessage("user", "Where does Tensura season 4 part 2 continue in the novel?"),
        ChatMessage("assistant", "I couldn't verify that online."),
    ], addressed=True, active=True, internet_enabled=True)
    assert decision.action == "search" and "Tensura" in decision.query
    messages = router.send.call_args.args[0]
    payload = json.loads(messages[-1].content)
    assert payload["current_utterance"] == "Okay, try again"
    assert "Tensura" in payload["previous_dialogue"][0]["text"]
    assert router.send.call_args.kwargs["purpose"] == "controller"
    assert router.send.call_args.kwargs["max_tokens"] <= 256


@pytest.mark.parametrize("text,action", [
    ("Why can't you search it on the internet?", "answer"),
    ("Hey Ciel, what is the time?", "local"),
    ("Which one?", "clarify"),
])
def test_controller_uses_ai_selected_action(text, action):
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response(
        action, speech="Which season did you finish?",
        intent="request",
    )
    decision = AIController(router, ControllerConfig()).decide(
        text, [], addressed=True, active=True, internet_enabled=True,
    )
    assert decision.action == action


@pytest.mark.parametrize("result", [
    "not json", "[]", '{"action":"shell","confidence":1}',
    '{"action":"search","confidence":true,"query":"Tensura"}',
    '{"action":"search","confidence":0.98}',
])
def test_invalid_decisions_never_request_tools(result):
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = LLMResponse(result, "test", "nim")
    controller = AIController(router, ControllerConfig())
    assert controller.decide(
        "What do you mean?", [], addressed=False, active=False, internet_enabled=True,
    ).action == "silent"
    assert controller.decide(
        "Search Tensura", [], addressed=True, active=True, internet_enabled=True,
    ).action == "clarify"


def test_controller_preserves_private_routing_and_blocks_web_actions():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="Tensura")
    result = AIController(router, ControllerConfig()).decide(
        "Search Tensura", [ChatMessage("system", "private", local_only=True)],
        addressed=True, active=True, internet_enabled=True,
    )
    assert result.action == "clarify" and "private" in result.speech
    assert any(message.local_only for message in router.send.call_args.args[0])


def test_canceled_controller_never_calls_model():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    canceled = Event()
    canceled.set()
    router = MagicMock()
    decision = AIController(router, ControllerConfig()).decide(
        "Search Tensura", [], addressed=True, active=True, internet_enabled=True,
        cancel_event=canceled,
    )
    assert decision.action == "silent"
    router.send.assert_not_called()


def _web_fixture(router):
    from ciel.config import ControllerConfig, InternetConfig
    from ciel.controller import AIController, ControllerDecision

    controller = AIController(router, ControllerConfig())
    service = MagicMock()
    service.config = InternetConfig()
    initial = ControllerDecision(action="search", confidence=1, query="Amber S4 LN")
    messages = [ChatMessage("system", "Be concise."), ChatMessage(
        "user", "Where does Amber season 4 end in the light novel?",
    )]
    return controller, service, initial, messages


def test_web_loop_reads_page_after_snippet_and_validates_before_reporting():
    from ciel.internet import InternetResult, InternetSource

    router = MagicMock()
    controller, service, initial, messages = _web_fixture(router)
    quote = "Amber season 4 ends at light novel volume 11."
    url = "https://example.com/amber"
    service.lookup.side_effect = [
        InternetResult(attempted=True, sources=[InternetSource("Guide", url, snippet=quote)]),
        InternetResult(attempted=True, sources=[InternetSource("Guide", url, content=quote)]),
    ]
    report = LLMResponse(json.dumps({"claims": [{
        "text": quote, "source": 1, "quote": quote,
    }]}), "light", "nim")
    # The exact same supported-looking number in a search preview MUST NOT pass.
    router.send.side_effect = [report, _response("read_page", query=url), report]
    acknowledge = MagicMock()
    reply = controller.run_web(initial, messages[-1].content, messages, service,
                               acknowledge, lambda: True)
    assert reply.text == quote and not reply.canceled
    assert len(reply.sources) == 1 and reply.sources[0].content == quote
    assert service.lookup.call_args.kwargs["read_page"] is True
    assert service.lookup.call_count == 2
    acknowledge.assert_called_once()
    assert controller.last_lookup["outcome"] == "supported"


@pytest.mark.parametrize("next_action,queries", [
    ("search", ["Amber S4 LN"]), ("local", [""]),
    ("search", ["Amber novel continuation", "Amber adaptation guide", "unused"]),
])
def test_web_loop_caps_retries_and_never_executes_page_suggested_local_actions(
    next_action, queries,
):
    from ciel.internet import InternetResult

    router = MagicMock()
    controller, service, initial, messages = _web_fixture(router)
    service.lookup.return_value = InternetResult(attempted=True)
    router.send.side_effect = [_response(next_action, query=query) for query in queries]
    reply = controller.run_web(initial, messages[-1].content, messages, service,
                               lambda _text: None, lambda: True)
    assert "couldn't verify" in reply.text
    assert service.lookup.call_count == (3 if len(queries) > 1 else 1)
    assert len({call.args[0] for call in service.lookup.call_args_list}) == (
        service.lookup.call_count
    )


def test_decision_timeout_cancels_provider_and_returns_without_waiting():
    import time

    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    closed = Event()
    router = MagicMock()

    def stalled(_messages, **kwargs):
        assert kwargs["cancel_event"].wait(1)
        closed.set()
        return _response("search", query="Too late")

    router.send.side_effect = stalled
    controller = AIController(router, ControllerConfig(timeout_seconds=0.1))
    start = time.monotonic()
    result = controller.decide("Search it", [], addressed=True, active=True, internet_enabled=True)
    assert result.action == "clarify" and time.monotonic() - start < 0.8
    assert closed.wait(0.5)


def test_cancel_during_web_report_discards_answer_and_cancels_model():
    from ciel.internet import InternetResult, InternetSource

    router = MagicMock()
    controller, service, initial, messages = _web_fixture(router)
    service.lookup.return_value = InternetResult(attempted=True, sources=[
        InternetSource("Guide", "https://example.com/guide", content="Unverified"),
    ])
    canceled = Event()
    closed = Event()

    def cancel_during_send(_messages, **kwargs):
        canceled.set()
        assert kwargs["cancel_event"].wait(1)
        closed.set()
        return LLMResponse('{"claims":[]}', "light", "nim")

    router.send.side_effect = cancel_during_send
    reply = controller.run_web(initial, messages[-1].content, messages, service,
                               lambda _text: None, lambda: not canceled.is_set(),
                               cancel_event=canceled)
    assert reply.canceled and not reply.text
    assert closed.wait(0.5)
    assert service.lookup.call_count == 1


def test_fresh_retry_sources_survive_a_full_previous_source_budget():
    from ciel.internet import InternetResult, InternetSource

    router = MagicMock()
    controller, service, initial, messages = _web_fixture(router)
    quote = "Amber season 4 ends at light novel volume 11."
    service.config.research_max_results = 1
    service.lookup.side_effect = [
        InternetResult(attempted=True, sources=[InternetSource(
            "Unhelpful", "https://example.com/old", content="An unrelated description.",
        )]),
        InternetResult(attempted=True, sources=[InternetSource(
            "Helpful", "https://example.com/new", content=quote,
        )]),
    ]
    router.send.side_effect = [
        LLMResponse('{"claims":[]}', "light", "nim"),
        _response("search", query="Amber season 4 novel adaptation"),
        LLMResponse(json.dumps({"claims": [{"text": quote, "source": 1, "quote": quote}]}),
                    "light", "nim"),
    ]
    reply = controller.run_web(initial, messages[-1].content, messages, service,
                               lambda _text: None, lambda: True)
    assert reply.text == quote
    assert [source.url for source in reply.sources] == ["https://example.com/new"]


def test_latest_turn_is_separate_from_unanswered_history():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response('answer', intent='greeting')
    AIController(router, ControllerConfig()).decide(
        'Good evening, Ciel!', [ChatMessage('user', 'Find the Lantern Festival date.')],
        addressed=True, active=False, internet_enabled=True,
    )
    sent = router.send.call_args.args[0]
    assert sent[-1].role == 'user' and sent[-1].content == 'Good evening, Ciel!'
    assert 'Lantern Festival' not in sent[-1].content
    assert 'Lantern Festival' in sent[-2].content


def test_empty_model_clarification_gets_a_speakable_question():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("clarify", speech="", query="weather tomorrow")
    decision = AIController(router, ControllerConfig()).decide(
        "Weather tomorrow?", [], addressed=True, active=True,
        internet_enabled=True,
    )
    assert decision.action == "clarify" and decision.speech.endswith("?")


def test_page_title_without_url_uses_search_instead_of_a_page_read():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response(
        "read_page", query="official Python tutorial documentation",
    )
    decision = AIController(router, ControllerConfig()).decide(
        "Search the official Python tutorial documentation.", [],
        addressed=True, active=True, internet_enabled=True,
    )
    assert decision.action == "search"
    assert decision.query == "official Python tutorial documentation"


def test_extra_model_fields_and_invalid_budget_hint_cannot_change_action():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = LLMResponse(json.dumps({
        "intent": "request", "action": "search", "confidence": 0.9,
        "query": "official Python tutorial", "research": "all sites",
        "automatic_search": True, "shell": "launch calculator",
    }), "test", "nim")
    decision = AIController(router, ControllerConfig()).decide(
        "Search the official Python tutorial.", [], addressed=True, active=True,
        internet_enabled=True,
    )
    assert decision.action == "search"
    assert decision.query == "official Python tutorial" and not decision.research


def test_generic_weather_without_a_city_cannot_start_a_search():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="weather tomorrow")
    decision = AIController(router, ControllerConfig()).decide(
        "What's the weather tomorrow?", [], addressed=True, active=True,
        internet_enabled=True, memories=["The user likes chess."],
    )
    assert decision.action == "clarify" and "city" in decision.speech


def test_unaddressed_room_speech_cannot_trigger_tools_from_model_choice():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("local")
    controller = AIController(router, ControllerConfig())
    decision = controller.decide(
        "Dad, please turn on the TV.", [], addressed=False, active=True,
        internet_enabled=True,
    )
    assert decision.action == "silent"
    decision = controller.decide(
        "Mom, what's for dinner?", [], addressed=False, active=False,
        internet_enabled=True,
    )
    assert decision.action == "silent"


def test_active_followup_ignores_legacy_model_silence_without_another_listener_check():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.side_effect = [
        _response("silent", intent="other_listener", confidence=0.9),
        _response("search", intent="follow_up", confidence=0.9,
                  query="Amber Works season 4 part 2 light novel continuation"),
    ]
    decision = AIController(router, ControllerConfig()).decide(
        "Can you check that again?", [
            ChatMessage("user", "Where does Amber Works season 4 part 2 end?"),
            ChatMessage("assistant", "I couldn't verify it."),
        ], addressed=False, active=True, internet_enabled=True,
    )
    assert decision.action == "answer" and not decision.query and not decision.speech
    router.send.assert_called_once()


@pytest.mark.parametrize("intent", ["other_listener", "quiet", "request"])
def test_active_followup_is_answered_despite_legacy_model_silence(intent):
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("silent", intent=intent)
    decision = AIController(router, ControllerConfig()).decide(
        "I don't know.", [ChatMessage("assistant", "What would you like to do?")],
        addressed=False, active=True, internet_enabled=True,
    )
    assert decision.action == "answer" and not decision.speech and not decision.query
    router.send.assert_called_once()


def test_explicit_other_person_stays_silent_during_active_chat():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("silent", intent="other_listener")
    decision = AIController(router, ControllerConfig()).decide(
        "Mom what's for dinner?", [ChatMessage("assistant", "How can I help?")],
        addressed=False, active=True, internet_enabled=True,
    )
    assert decision.action == "silent"
    router.send.assert_not_called()


def test_automatic_search_off_requires_an_explicit_lookup_request():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="latest Python release")
    controller = AIController(router, ControllerConfig())
    decision = controller.decide(
        "What is the latest Python release?", [], addressed=True, active=True,
        internet_enabled=True, automatic_search=False,
    )
    assert decision.action == "answer"
    decision = controller.decide(
        "Search for the latest Python release.", [], addressed=True, active=True,
        internet_enabled=True, automatic_search=False,
    )
    assert decision.action == "search"


@pytest.mark.parametrize('intent,action,expected', [
    ('greeting', 'search', 'answer'),
    ('request', 'silent', 'answer'),
    ('unclear', 'search', 'clarify'),
    ('quiet', 'silent', 'answer'),
    ('other_listener', 'silent', 'answer'),
])
def test_intent_and_action_must_agree_before_tools(intent, action, expected):
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response(action, intent=intent, query='An old lookup')
    choice = AIController(router, ControllerConfig()).decide(
        'Current utterance', [], addressed=True, active=False, internet_enabled=True,
    )
    assert choice.action == expected
    if expected == 'clarify':
        assert choice.speech
    assert not choice.query


def test_reordered_retry_terms_do_not_repeat_the_same_search():
    from ciel.internet import InternetResult

    router = MagicMock()
    controller, service, initial, messages = _web_fixture(router)
    service.lookup.return_value = InternetResult(attempted=True)
    router.send.return_value = _response('search', query='LN Amber S4')
    reply = controller.run_web(initial, messages[-1].content, messages, service,
                               lambda _text: None, lambda: True)
    assert 'verify' in reply.text
    assert service.lookup.call_count == 1


def test_uncertain_addressed_query_asks_for_subject_before_searching():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response('search', query='festival date', confidence=0.8)
    decision = AIController(router, ControllerConfig()).decide(
        'Please look up the festival date. I forgot which festival it was.', [],
        addressed=True, active=False, internet_enabled=True,
    )
    assert decision.action == 'clarify' and not decision.query
    assert 'name or topic' in decision.speech


def test_schema_repair_shares_total_controller_deadline():
    import time

    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    closed = Event()
    router = MagicMock()

    def decide_or_stall(_messages, **kwargs):
        if router.send.call_count == 1:
            time.sleep(0.3)
            return LLMResponse("not JSON", "test", "nim")
        kwargs["cancel_event"].wait(1)
        closed.set()
        return _response("answer")

    router.send.side_effect = decide_or_stall
    start = time.monotonic()
    result = AIController(router, ControllerConfig(timeout_seconds=0.4)).decide(
        "Could you explain that?", [], addressed=False, active=True, internet_enabled=True,
    )
    assert result.action == "clarify"
    assert time.monotonic() - start < 0.6
    assert closed.wait(0.5)
    assert router.send.call_count == 2


def test_low_confidence_personal_name_correction_does_not_request_lookup_wording():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="synthetic game title", confidence=0.8)
    decision = AIController(router, ControllerConfig()).decide(
        "The two words are connected together.",
        [ChatMessage("user", "My favorite game is Example Two.")],
        addressed=False, active=True, transcribed=True, internet_enabled=True,
    )
    assert decision.action == "clarify"
    assert "name you meant" in decision.speech
    assert "look up" not in decision.speech


def test_active_supported_correction_routes_locally_without_granting_write_permission():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="Example Two", confidence=0.8)
    decision = AIController(router, ControllerConfig()).decide(
        "No, like my my favorite game is Example Two.", [],
        addressed=False, active=True, transcribed=True, internet_enabled=True,
    )
    assert decision.action == "local"


@pytest.mark.parametrize("text", ["What do you think?", "Should I try it?"])
def test_reflective_followup_cannot_become_model_selected_web_research(text):
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response(
        "search", query="reverse engineering anti-cheat program", intent="follow_up",
    )
    decision = AIController(router, ControllerConfig()).decide(
        text, [
            ChatMessage("user", "I'm thinking of reverse engineering an anti-cheat program."),
            ChatMessage("assistant", "What interests you about it?"),
        ], addressed=True, active=True, internet_enabled=True,
    )
    assert decision.action == "answer" and decision.speech == ""
    assert decision.query == ""


def test_reflective_question_about_changing_facts_keeps_requested_research():
    from ciel.config import ControllerConfig
    from ciel.controller import AIController

    router = MagicMock()
    router.send.return_value = _response("search", query="latest Python release changes")
    decision = AIController(router, ControllerConfig()).decide(
        "What do you think about the latest Python release? Check the changes online.",
        [], addressed=True, active=True, internet_enabled=True,
    )
    assert decision.action == "search"
