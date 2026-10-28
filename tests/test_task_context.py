"""A shared task projection preserves intent without rewriting or authorizing turns."""

import pytest

from ciel.providers.base import ChatMessage
from ciel.task_context import resolve_task_context


def _history():
    return [
        ChatMessage("user", "Search where Amber Harbor season 3 anime ends in the light novel."),
        ChatMessage("assistant", "I couldn't finish checking the answer."),
    ]


@pytest.mark.parametrize("text", [
    "Again.", "Check again.", "Can you check again?", "Search again", "Try again please",
])
def test_retry_preserves_original_question_after_a_failed_answer(text):
    result = resolve_task_context(text, _history())
    assert result.kind == "retry" and result.task_type == "web"
    assert "Amber Harbor" in result.request and "light novel" in result.request
    assert "season 3" in result.request and result.anchor_index == 0


@pytest.mark.parametrize("text", ["No, Tensura.", "I mean Tensura.", "Just Tensura.", "Tensura*"])
def test_subject_repair_changes_only_the_subject_and_retains_the_outcome(text):
    history = [ChatMessage("user", "Search TensorFlow anime to light novel continuation.")]
    result = resolve_task_context(text, history)
    assert result.kind == "correction" and result.corrected_subject == "Tensura"
    assert "Tensura" in result.request and "TensorFlow" not in result.request
    assert "anime to light novel continuation" in result.request
    assert history[0].content == "Search TensorFlow anime to light novel continuation."


def test_retry_keeps_name_and_medium_corrections_through_several_rounds():
    history = [
        ChatMessage("user", "Search TensorFlow anime continuation after season four."),
        ChatMessage("assistant", "What do you mean by TensorFlow?"),
        ChatMessage("user", "Tensura*"),
        ChatMessage("assistant", "Did you mean the manga or light novel?"),
        ChatMessage("user", "The light novel continuation."),
        ChatMessage("assistant", "I couldn't finish checking the answer."),
        ChatMessage("user", "Actually season five."),
        ChatMessage("assistant", "I couldn't finish checking the answer."),
    ]
    result = resolve_task_context("Check again.", history)
    assert result.kind == "retry" and result.corrected_subject == "Tensura"
    assert "TensorFlow" not in result.request and "Tensura" in result.request
    assert "light novel" in result.request and "season 5" in result.request
    assert "season 4" not in result.request and "season four" not in result.request


def test_continue_remains_an_explanation_when_previous_playback_was_interrupted():
    history = [
        ChatMessage("user", "Explain closures in Python."),
        ChatMessage("assistant", "A closure retains variables. [Playback was interrupted.]"),
    ]
    result = resolve_task_context("Continue", history)
    assert result.kind == "continue" and result.interrupted
    assert result.task_type == "conversation" and result.request == history[0].content


def test_a_pause_acknowledgement_does_not_erase_the_unfinished_reply():
    history = [
        ChatMessage("user", "Search Amber Harbor light novel continuation."),
        ChatMessage("assistant", "The guide starts by describing... [Reply was interrupted.]"),
        ChatMessage("user", "Wait"),
        ChatMessage("assistant", "Take your time."),
    ]
    result = resolve_task_context("Continue", history)
    assert result.kind == "continue" and result.interrupted and result.task_type == "web"
    assert "Amber Harbor" in result.request


def test_wait_retains_a_task_but_stop_and_unrelated_requests_replace_it():
    history = _history()
    waiting = resolve_task_context("Wait", history)
    assert waiting.kind == "wait" and "Amber Harbor" in waiting.request
    stopped = resolve_task_context("Stop", history)
    assert stopped.kind == "stop" and stopped.request == ""
    for text in ["Tell me a joke", "What's the weather in Paris?", "Hello!"]:
        result = resolve_task_context(text, history)
        assert result.kind == "new" and "Amber Harbor" not in result.request


@pytest.mark.parametrize("intervening", ["Stop", "Cancel", "Tell me a joke", "Thanks"])
def test_retry_cannot_revive_a_canceled_or_replaced_task(intervening):
    history = [*_history(), ChatMessage("user", intervening)]
    result = resolve_task_context("Again", history)
    assert "Amber Harbor" not in result.request


@pytest.mark.parametrize("text", [
    "No, Tensura.", "I mean Tensura.", "Just Tensura.",
    'She said "Search TensorFlow"', "Don't search TensorFlow",
    '"Search TensorFlow"', "I meant don't search TensorFlow",
])
def test_missing_or_negated_context_never_creates_a_web_task(text):
    result = resolve_task_context(text, [])
    assert result.task_type == "conversation" and result.kind == "new"


def test_current_turn_can_be_present_in_history_without_duplicating_a_detail():
    text = "The light novel."
    history = [*_history(), ChatMessage("user", text)]
    result = resolve_task_context(text, history)
    assert result.kind == "detail" and result.request.count(text) == 1


def test_assistant_claims_do_not_create_subject_or_scope():
    history = [
        ChatMessage("user", "Search Amber Harbor anime continuation."),
        ChatMessage("assistant", "Season 9 ends at novel volume 18."),
    ]
    result = resolve_task_context("Again", history)
    assert "9" not in result.request and "18" not in result.request


@pytest.mark.parametrize("text", [
    "Also, how much is this RTX5090?", "What about tensorflow?", "How is the weather in paris?",
])
def test_a_named_new_question_does_not_inherit_a_referenced_old_topic(text):
    result = resolve_task_context(text, _history())
    assert result.kind == "new" and "Amber Harbor" not in result.request


def test_screen_retry_retains_screen_focus_without_claiming_a_capture():
    history = [ChatMessage("user", "Can you check my screen?"),
               ChatMessage("assistant", "The screen model timed out.")]
    result = resolve_task_context("Again", history)
    assert result.kind == "retry" and result.task_type == "screen"
    assert result.request == history[0].content


@pytest.mark.parametrize("text", [
    "I meant the light novel.", "I mean season five.", "No, I meant the manga.",
])
def test_medium_and_scope_corrections_are_details_instead_of_replacement_names(text):
    result = resolve_task_context(text, _history())
    assert result.kind == "detail" and "Amber Harbor" in result.request
    assert not result.corrected_subject


def test_lowercase_subject_correction_preserves_lowercase_requested_outcome():
    result = resolve_task_context("no, cedarDB", [
        ChatMessage("user", "search tensorflow memory usage on linux"),
    ])
    assert result.request == "search cedarDB memory usage on linux"


@pytest.mark.parametrize("text", ["What version does it use?", "What color was it?"])
def test_dependent_question_inherits_the_subject_without_a_proper_new_name(text):
    result = resolve_task_context(text, _history())
    assert result.kind == "detail" and "Amber Harbor" in result.request


def test_no_search_removes_a_web_anchor_before_a_later_bare_retry():
    result = resolve_task_context("Again", [*_history(), ChatMessage("user", "Don't search")])
    assert result.task_type == "conversation" and "Amber Harbor" not in result.request


@pytest.mark.parametrize("text", [
    "I meant don't search TensorFlow", "No, do not search", "I meant no searching",
    '"No, Tensura"', 'She said "Just Tensura"',
])
def test_reported_and_negative_corrections_cannot_bind_an_active_search(text):
    result = resolve_task_context(text, _history())
    assert result.kind == "new" and result.task_type == "conversation"
    assert not result.corrected_subject


def test_why_keeps_the_task_available_for_a_later_missing_season_answer():
    history = [
        ChatMessage('user', 'Can you search the latest episode of Tensura?'),
        ChatMessage('assistant', "I found related pages, but I couldn't verify the answer."),
    ]
    explanation = resolve_task_context('Why?', history)
    assert explanation.kind == 'explanation' and explanation.task_type == 'web'
    assert explanation.request == history[0].content
    history.extend([
        ChatMessage('user', 'Why?'),
        ChatMessage('assistant', 'Which season of Tensura are you looking for?'),
    ])
    result = resolve_task_context('3rd chapter, episode, no, no, season 4.', history)
    assert result.task_type == 'web' and result.kind == 'detail'
    assert 'Tensura' in result.request and 'latest episode' in result.request
    assert 'season 4' in result.request
    assert 'chapter' not in result.request and '3rd' not in result.request


@pytest.mark.parametrize('text', ['Why?', 'Why not?', 'How come?', 'What went wrong?'])
def test_short_explanation_followup_does_not_discard_the_previous_task(text):
    result = resolve_task_context(text, _history())
    assert result.kind == 'explanation'
    assert result.request == _history()[0].content


@pytest.mark.parametrize('text', ['Why is CedarDB slow?', 'What went wrong with Python?'])
def test_named_explanation_question_starts_its_own_task(text):
    result = resolve_task_context(text, _history())
    assert result.kind == 'new' and 'Amber Harbor' not in result.request
