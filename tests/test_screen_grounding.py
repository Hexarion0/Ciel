"""The visual answer gets current image capability, not a text-only persona or old refusals."""

from ciel.providers.base import ChatMessage, LLMResponse
from tests.test_screen_service import service as screen_service_fixture

service = screen_service_fixture


def test_visual_prompt_drops_text_only_persona_and_long_unrelated_history(service):
    messages = [ChatMessage("system", "LEGACY_TEXT_ONLY_PERSONA " + "old details " * 3000)]
    messages += [ChatMessage("assistant", "OLD_REFUSAL: I can't see your screen.")]
    messages += [ChatMessage("user", "Look at my screen and identify the anime.")]
    service.answer(service.resolve(messages[-1].content), messages)
    sent = service.vision.send.call_args.args[0]
    assert "LEGACY_TEXT_ONLY_PERSONA" not in sent[0].content
    assert "OLD_REFUSAL" not in str(sent)
    assert "can see" in sent[0].content
    assert "title" in sent[0].content
    assert sum(len(m.content) for m in sent) < 5000
    assert sum(len(m.images) for m in sent) == 1


def test_visual_question_implicitly_requests_capture_but_anime_chat_does_not(service):
    assert service.resolve("What anime am I watching right now?") is not None
    assert service.resolve("What do you think about anime?") is None
    assert service.resolve("Should I watch an anime tonight?") is None
    assert service.resolve("He asked what anime am I watching") is None


def test_screen_intent_is_emitted_before_capture_and_inference(service):
    events = []
    capture = service.backend.capture.return_value
    def take(**_):
        events.append("capture")
        return capture
    def infer(*_, **__):
        events.append("vision")
        return LLMResponse("The visible title is Moonlight Orchard.", "vision-test", "nim")
    service.backend.capture.side_effect = take
    service.vision.send.side_effect = infer
    def progress(tool, query):
        events.append("ack")
        assert tool == "screen" and "anime" in query
        return True
    request = service.resolve("What anime am I watching?")
    result = service.answer(request, [ChatMessage("user", request.query)], progress=progress)
    assert result.text == "The visible title is Moonlight Orchard."
    assert events == ["ack", "capture", "vision"]
    assert not service.conversation.get_recent_turns()


def test_canceled_screen_announcement_never_starts_capture(service):
    request = service.resolve("Look at my screen")
    assert service.answer(request, [], progress=lambda *_: False) is None
    service.backend.capture.assert_not_called()
    service.vision.send.assert_not_called()
