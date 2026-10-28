"""Search identity and requested outcomes survive contextual corrections safely."""

import json
from threading import Event
from unittest.mock import MagicMock

import pytest

from ciel.config import ControllerConfig
from ciel.controller import _CONTEXT_MARKER, AIController, ControllerDecision, resolve_web_goal
from ciel.providers.base import ChatMessage, LLMResponse


def _router(**fields):
    router = MagicMock()
    router.send.return_value = LLMResponse(json.dumps({
        "intent": "follow_up", "action": "search", "confidence": 0.84,
        "subject": "Amber Harbor", "subject_confidence": 0.96,
        "query": "Amber Harbor anime novel continuation",
        "goal": "Find where Amber Harbor's anime continues in its light novel.",
        **fields,
    }), "test", "test")
    return router


def _history():
    return [
        ChatMessage("user", "Search where Amber Harbor's anime continues in the light novel."),
        ChatMessage("assistant", "Did you mean the light novel or the manga?"),
    ]


@pytest.mark.parametrize("text", [
    "light novel continuation", "And where does it continue in the novel?",
])
def test_grounded_subject_can_resolve_a_lower_confidence_search_route(text):
    router = _router()
    result = AIController(router, ControllerConfig()).decide(
        text, _history(), addressed=True, active=True, transcribed=True, internet_enabled=True,
    )
    assert result.action == "search"
    assert result.subject == "Amber Harbor" and result.subject_confidence == 0.96
    assert "light novel" in result.goal
    sent = router.send.call_args.args[0]
    payload = json.loads(sent[0].content.split(_CONTEXT_MARKER, 1)[1])
    assert sent[-1].content == text
    assert payload["current_utterance"] == text
    assert "Amber Harbor" in payload["previous_dialogue"][0]["text"]


def test_close_spelling_is_grounded_in_the_users_words_without_title_exceptions():
    router = _router(subject="Amber Harbour")
    result = AIController(router, ControllerConfig()).decide(
        "The light novel continuation.", _history(), addressed=True, active=True,
        internet_enabled=True, transcribed=True,
    )
    assert result.action == "search"


@pytest.mark.parametrize("history,fields,text", [
    ([ChatMessage("assistant", "Amber Harbor is an anime.")], {}, "The novel continuation."),
    (_history(), {"subject": "Naruto"}, "The light novel continuation."),
    (_history(), {"subject_confidence": 0.8}, "The light novel continuation."),
    (_history(), {"confidence": 0.49}, "The light novel continuation."),
    ([], {"subject": "festival", "query": "festival date"},
     "Search the festival date. I forgot which festival it was."),
])
def test_uncertain_or_ungrounded_identity_still_needs_clarification(history, fields, text):
    result = AIController(_router(**fields), ControllerConfig()).decide(
        text, history, addressed=True, active=True, internet_enabled=True,
    )
    assert result.action == "clarify" and not result.query


@pytest.mark.parametrize("field", ["confidence", "subject_confidence"])
def test_boolean_confidence_scores_are_invalid(field):
    fields = {"action": "search", "query": "Amber Harbor", "confidence": 0.8, field: True}
    with pytest.raises(ValueError):
        ControllerDecision(**fields)


def test_existing_decision_schema_defaults_remain_compatible():
    decision = ControllerDecision(action="search", query="Amber Harbor", confidence=0.98)
    assert decision.subject == decision.goal == "" and decision.subject_confidence == 0


def test_grounded_identity_never_overrides_private_search_policy():
    history = [ChatMessage("system", "Protected", local_only=True), *_history()]
    router = _router()
    result = AIController(router, ControllerConfig()).decide(
        "The light novel continuation.", history, addressed=True, active=True,
        internet_enabled=True,
    )
    assert result.action == "clarify" and "private" in result.speech
    assert all(message.local_only for message in router.send.call_args.args[0])


def test_canceled_grounded_followup_never_calls_the_model():
    canceled = Event()
    canceled.set()
    router = _router()
    result = AIController(router, ControllerConfig()).decide(
        "The light novel continuation.", _history(), addressed=True, active=True,
        internet_enabled=True, cancel_event=canceled,
    )
    assert result.action == "silent"
    router.send.assert_not_called()


def test_brief_medium_correction_retains_the_original_requested_outcome():
    goal = resolve_web_goal("light novel contunation", _history(), "Amber Harbor anime")
    assert "Amber Harbor" in goal and "continues" in goal and "light novel" in goal


def test_referential_followup_retains_the_user_title_instead_of_an_assistant_guess():
    history = [
        ChatMessage("user", "Search Amber Harbor anime to lightnovel continuation."),
        ChatMessage("assistant", "Violet Castle episode 92 is the latest."),
    ]
    goal = resolve_web_goal("And where does it continue in the novel?", history,
                            "Amber Harbor novel continuation")
    assert "Amber Harbor" in goal and "Violet Castle" not in goal and "92" not in goal


@pytest.mark.parametrize("text", ["Hello!", "Search the Python release notes."])
def test_greetings_and_new_requests_do_not_revive_an_old_search_goal(text):
    goal = resolve_web_goal(text, _history(), "Python release notes",
                            "Find where Amber Harbor's anime continues in its light novel.")
    assert "Amber Harbor" not in goal and "light novel" not in goal


def test_an_intervening_greeting_breaks_the_previous_task_reference():
    history = [*_history(), ChatMessage("user", "Hello!"),
               ChatMessage("assistant", "Hello, how can I help?")]
    goal = resolve_web_goal("And what about it?", history, "an unrelated query")
    assert "Amber Harbor" not in goal


def test_model_goal_cannot_add_a_season_only_the_assistant_invented():
    goal = resolve_web_goal(
        "The novel continuation.", _history(), "Amber Harbor novel",
        "Find where Amber Harbor season 4 ends in volume 12.",
    )
    assert "4" not in goal and "12" not in goal
    assert "Amber Harbor" in goal


def test_model_goal_cannot_hide_invented_scope_in_word_numbers():
    goal = resolve_web_goal(
        "Where does Amber Harbor anime end in the light novel?", [], "Amber Harbor anime novel",
        "Find Amber Harbor season four light novel continuation.",
    )
    assert "season four" not in goal.casefold()
    assert "Amber Harbor" in goal and "light novel" in goal


@pytest.mark.parametrize("text,query", [
    ("How much is this RTX5090?", "RTX5090 price"),
    ("Also, what is the weather in Paris?", "Paris weather"),
])
def test_named_new_question_does_not_inherit_an_unrelated_reading_task(text, query):
    history = [
        ChatMessage("user", "Where does Amber Harbor season 4 end in the light novel?"),
        ChatMessage("assistant", "I could not verify that online."),
    ]

    goal = resolve_web_goal(text, history, query)

    assert "Amber Harbor" not in goal and "season 4" not in goal
    assert text in goal


def test_context_mode_preserves_pending_local_and_network_policy_fields():
    router = _router(action="answer")
    AIController(router, ControllerConfig()).decide(
        "Yes.", _history(), addressed=True, active=True, internet_enabled=False,
        pending_local=True,
    )
    sent = router.send.call_args.args[0]
    payload = json.loads(sent[0].content.split(_CONTEXT_MARKER, 1)[1])
    assert sent[-1].content == "Yes."
    assert payload["pending_local_confirmation"] is True
    assert payload["internet_enabled"] is False
    assert payload["active_conversation"] is True


def test_long_task_goal_keeps_the_final_user_correction():
    history = [ChatMessage("user", "Search Amber Harbor. " + "Some background. " * 100),
               ChatMessage("assistant", "Which season did you finish?")]
    goal = resolve_web_goal("Season 4 part 2.", history, "Amber Harbor")
    assert "Amber Harbor" in goal and goal.endswith("Season 4 part 2.")


def test_initial_search_cannot_invent_an_episode_from_model_memory():
    router = _router(confidence=0.98, query="Amber Harbor novel after episode 24")
    result = AIController(router, ControllerConfig()).decide(
        "And the light novel continuation?", _history(), addressed=True, active=True,
        internet_enabled=True,
    )
    assert result.action == "clarify" and not result.query
    assert "episode" in result.speech.casefold() and "name" not in result.speech.casefold()


def test_user_provided_episode_scope_survives_a_brief_medium_correction():
    history = [ChatMessage("user", "Where does Amber Harbor ep. 24 end?"),
               ChatMessage("assistant", "Did you mean the novel or manga?")]
    controller = AIController(_router(query="Amber Harbor episode 24 novel"), ControllerConfig())
    result = controller.decide(
        "The light novel.", history, addressed=True, active=True, internet_enabled=True,
    )
    assert result.action == "search" and "24" in result.query


def test_word_number_scope_and_user_correction_do_not_revert_to_an_old_season():
    history = [ChatMessage("user", "Search Amber Harbor season four continuation."),
               ChatMessage("assistant", "Which medium?")]
    result = AIController(_router(query="Amber Harbor season 4 novel"), ControllerConfig()).decide(
        "The light novel.", history, addressed=True, active=True, internet_enabled=True,
    )
    assert result.action == "search"
    result = AIController(_router(query="Amber Harbor season 4 novel"), ControllerConfig()).decide(
        "Actually season five, the light novel.", history, addressed=True, active=True,
        internet_enabled=True,
    )
    assert result.action == "search" and "season 5" in result.query
    assert "season 4" not in result.query


def test_retained_goal_keeps_only_the_latest_explicit_season_correction():
    history = [
        ChatMessage("user", "Search Amber Harbor season four anime to light novel continuation."),
        ChatMessage("assistant", "Which medium do you want to continue?"),
    ]
    goal = resolve_web_goal(
        "Actually season five, the light novel.", history, "Amber Harbor season 4 novel",
    )
    assert "season 5" in goal.casefold()
    assert "season 4" not in goal.casefold() and "season four" not in goal.casefold()


def test_research_can_follow_an_episode_observed_in_its_tool_results():
    controller = AIController(_router(query="Amber Harbor episode 24 novel"), ControllerConfig())
    result = controller.decide(
        "And the light novel continuation?", _history(), addressed=True, active=True,
        internet_enabled=True, tool_result={"preview": "Episode 24 ends the season."},
    )
    assert result.action == "search" and "24" in result.query


@pytest.mark.parametrize("history", [[], _history()])
def test_both_prompt_templates_explicitly_request_separate_subject_and_goal_fields(history):
    router = _router()
    AIController(router, ControllerConfig()).decide(
        "Search Amber Harbor novel continuation.", history, addressed=True, active=True,
        internet_enabled=True,
    )
    prompt = router.send.call_args.args[0][0].content
    assert '"subject"' in prompt and '"subject_confidence"' in prompt and '"goal"' in prompt
