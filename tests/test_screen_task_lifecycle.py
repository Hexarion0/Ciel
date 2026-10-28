"""Task cancellation cannot publish a late image response or erase an interrupted focus."""

from unittest.mock import Mock

from ciel.providers.base import LLMResponse
from tests.test_screen_service import messages
from tests.test_screen_service import service as service


def test_clearing_task_during_vision_rejects_late_result(service):
    def infer(*_args, **_kwargs):
        service.clear()
        return LLMResponse("An obsolete observation.", "test", "test")

    service.vision.send.side_effect = infer
    assert service.answer(service.resolve("Look at my screen"), messages()) is None
    assert not service.active()


def test_interrupted_screen_preparation_retains_original_question(service):
    request = service.resolve("What anime am I watching?")
    assert service.answer(request, messages(), progress=lambda *_: False) is None
    assert service.active()
    retry = service.resolve("Again.")
    assert retry is not None and retry.query == request.query
    service.backend.capture.assert_not_called()


def test_repeat_prompt_answers_original_question_and_focuses_on_changes(service):
    original = service.resolve("What anime am I watching?")
    service.answer(original, messages())
    retry = service.resolve("Check again.")
    service.answer(retry, messages())
    prompt = service.vision.send.call_args.args[0][0].content
    assert "Original question: What anime am I watching?" in prompt
    assert "relevant changes" in prompt
    assert "nothing relevant changed" in prompt


def test_clear_new_writing_task_does_not_need_intent_request(service):
    service.begin(service.resolve("Look at my screen"))
    router = Mock()
    result = service.interpret("Write a poem about a penguin.", router, messages())
    assert result.request is None and not result.clarification
    assert not service.active()
    router.send.assert_not_called()


def test_declining_screen_access_ends_task_without_model_or_capture(service):
    service.begin(service.resolve("Look at my screen"))
    router = Mock()
    result = service.interpret("Don't look at my screen again.", router, messages())
    assert result.request is None and result.clarification
    assert not service.active()
    router.send.assert_not_called()
    service.backend.capture.assert_not_called()


def test_reset_between_task_validation_and_commit_cannot_seed_new_scope(service, monkeypatch):
    save_task = service._save_task

    def reset_then_save(*args, **kwargs):
        service.conversation.clear_session()
        return save_task(*args, **kwargs)

    monkeypatch.setattr(service, "_save_task", reset_then_save)
    assert service.answer(service.resolve("Explain this page"), messages()) is None
    assert not service.active()
    service.backend.capture.assert_not_called()


def test_legacy_mark_cannot_overwrite_a_newer_visual_focus(service):
    older = service.resolve("Explain this page")
    service.answer(older, messages())
    newer = service.resolve("What anime am I watching?")
    service.answer(newer, messages())
    assert not service.mark(1, request=older)
    assert service.resolve("Try again").query == newer.query
