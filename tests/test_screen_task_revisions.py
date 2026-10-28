"""Revoked screen work stays revoked and speech corrections keep their task focus."""

from unittest.mock import MagicMock

import pytest

from ciel.screen.intent import ScreenIntent
from tests.test_screen_conversation_upgrade import ORIGINAL, ordinary_router
from tests.test_screen_integration import install_screen, run_callbacks, stub_runtime_internet
from tests.test_screen_service import messages
from tests.test_screen_service import service as screen_service_fixture

service = screen_service_fixture


def start_task(service):
    """Establish a real application-owned task with synthetic pixels and inference."""
    request = service.interpret("Explain this page", MagicMock(), []).request
    result = service.answer(request, messages())
    turn_id = service.conversation.add_turn("assistant", result.text)
    service.mark(turn_id, request=request)
    assert service.active()


def test_request_interpreted_before_cancel_cannot_capture_or_revive_task(service):
    start_task(service)
    pending = service.interpret("Look again", MagicMock(), []).request
    assert pending is not None
    service.clear()
    captures = service.backend.capture.call_count
    inferences = service.vision.send.call_count

    assert service.answer(pending, messages()) is None
    assert service.backend.capture.call_count == captures
    assert service.vision.send.call_count == inferences
    assert not service.active()


def test_late_visual_mark_cannot_restore_canceled_task(service):
    start_task(service)
    pending = service.interpret("Look again", MagicMock(), []).request
    result = service.answer(pending, messages())
    assert result is not None
    service.clear()

    turn_id = service.conversation.add_turn("assistant", result.text)
    service.mark(turn_id, request=pending)

    assert not service.active()
    assert service.resolve("Try again") is None


@pytest.mark.parametrize("query", [
    "Can you actually see my screen?", "What did you mean by that?", "I liked that.",
    "Why did you inspect that window?", "Could you check that answer?",
    "Recheck your reasoning about this scene.", "Check your answer about this screen.",
])
def test_native_tool_cannot_turn_screen_discussion_into_capture(service, query):
    start_task(service)

    assert service.resolve_native(query) is None


@pytest.mark.parametrize("query", [
    "Could you reassess that scene for me?", "Inspect that window again, please",
    "Please check that again", "Take another look at it", "Look again",
])
def test_native_tool_retains_explicit_visual_followups(service, query):
    start_task(service)

    request = service.resolve_native(query)

    assert request is not None and not request.previous
    assert request.original_query == "Explain this page"


def test_semantic_focus_correction_survives_another_retry(service, monkeypatch):
    start_task(service)
    monkeypatch.setattr(
        "ciel.screen.service.interpret_screen_intent",
        MagicMock(return_value=ScreenIntent("refresh", 0.99)),
    )
    correction = "The other window, not that one"
    corrected = service.interpret(correction, MagicMock(), []).request
    result = service.answer(corrected, messages())
    turn_id = service.conversation.add_turn("assistant", result.text)
    service.mark(turn_id, request=corrected)

    retry = service.interpret("Try again", MagicMock(), []).request

    assert retry is not None and not retry.previous
    assert "Explain this page" in retry.query
    assert correction in retry.query


@pytest.mark.parametrize("controller", [False, True])
def test_unreliable_active_screen_retry_clarifies_without_stale_general_reply(
    tmp_path, monkeypatch, controller,
):
    created = install_screen(monkeypatch)
    stub_runtime_internet(monkeypatch)
    router = ordinary_router()
    _, tts, _, store, turns = run_callbacks(
        tmp_path, monkeypatch, [
            (ORIGINAL, {"wake_verified": True}),
            ("Look again", {"wake_verified": True, "stt_confidence": 0.1}),
        ], router=router, controller=controller,
    )
    try:
        replies = [turn.content for turn in turns if turn.role == "assistant"]
        assert len(replies) == 2
        assert replies[-1].endswith("?") and "screen" in replies[-1].lower()
        assert "already told you" not in replies[-1]
        assert created[0].active()
        created[0].answer.assert_called_once()
        router.send.assert_not_called()
        router.stream.assert_not_called()
        assert tts.speak.call_args.args[0] == replies[-1]
    finally:
        store.close()
