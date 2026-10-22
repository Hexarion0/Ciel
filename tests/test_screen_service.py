"""Requested visual context stays ephemeral, owner-scoped, and independent of text routing."""

import io
import time
from threading import Event
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryStore
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.providers.vision import VisionError
from ciel.screen.cache import CaptureCache, clear_runtime_caches
from ciel.screen.capture import Capture, Cursor, Monitor, ScreenImage
from ciel.screen.service import ScreenService, resolve_request


def png(color="red", size=(600, 400)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def capture(identifier="first", color="red"):
    data = png(color)
    return Capture(identifier, time.time(),
                   (Monitor("left", -600, 0, 600, 400, focused=True),),
                   Cursor(-100, 120, time.time(), True, "left", 500, 120),
                   (ScreenImage("left", data, 600, 400, origin_x=-600),
                    ScreenImage("desktop", data, 600, 400, origin_x=-600)))


@pytest.fixture
def service(tmp_path):
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"),
                        ciel_persona_file="", internet_enabled=False)
    store = MemoryStore(settings.memory.db_path)
    conversation = ConversationManager(store=store, session_id="test-screen-owner")
    backend, vision = MagicMock(), MagicMock()
    backend.capture.return_value = capture()
    vision.send.return_value = LLMResponse("A red page is visible. What should we look at?",
                                          "vision-test", "nim")
    root = tmp_path / "screens"
    result = ScreenService(settings, conversation, backend=backend, vision=vision,
                           cache_factory=lambda scope: CaptureCache(scope, root=root))
    yield result
    result.close()
    store.close()


@pytest.mark.parametrize("query", [
    "Look at my screen", "Can u look at my screen?", "What does this error mean?",
    "What's under my cursor?", "Where is my mouse?", "Explain this page",
])
def test_clear_requests_capture(query):
    assert resolve_request(query) is not None


@pytest.mark.parametrize("query", [
    "Don't look at my screen", "Do not capture my screen", "I said look at my screen",
    'The page says "look at my screen"', "Should I look at my screen?",
    "What is an error?", "Hi", "What should I do next?", "What about now?",
    "Look at my screen, but don't take a screenshot",
    "Look at my screen but don't send it to AI",
    "Look at my screen, but no screenshots.",
    "Look at my screen, not through the cloud.",
    "Look at my screen is what she said; what should I reply?",
])
def test_other_mentions_do_not_authorize_pixels(query):
    assert resolve_request(query) is None


def messages():
    return [ChatMessage("system", "Ciel persona."), ChatMessage("user", "Look at my screen")]


def ask(service, query="Look at my screen"):
    return service.answer(service.resolve(query), messages())


def mark(service, result):
    turn_id = service.conversation.add_turn("assistant", result.text, model=result.model,
                                            provider=result.provider)
    service.mark(turn_id)


def test_one_image_payload_has_metadata_and_cursor_detail_without_mutating_history(service):
    original = messages()
    result = service.answer(service.resolve("Look at my screen"), original)
    sent = service.vision.send.call_args.args[0]
    assert result.model == "vision-test"
    assert sum(len(m.images) for m in sent) == 1
    assert '"x": -600' in sent[0].content
    assert '"logical_width": 600.0' in sent[0].content
    assert '"monitor": "left"' in sent[0].content
    assert "untrusted observations" in sent[0].content
    assert original[0].content == "Ciel persona."
    assert all(not m.images for m in original)
    mark(service, result)
    turn = service.conversation.get_recent_turns()[-1]
    assert turn.content == result.text
    assert "data:image" not in str(turn.to_dict())
    assert result.image is None


def test_contextual_followup_fresh_previous_reuse_and_topic_change(service):
    first = ask(service)
    mark(service, first)
    assert service.active()
    service.backend.capture.return_value = capture("second", "blue")
    second = ask(service, "What about now?")
    mark(service, second)
    assert service.backend.capture.call_count == 2
    assert [c.id for c in service._cache.recent()] == ["first", "second"]
    assert ask(service, "Explain the previous screenshot").capture_id == "second"
    assert service.backend.capture.call_count == 2
    service.conversation.add_turn("assistant", "Let's talk about music.")
    assert service.resolve("What about now?") is None


def test_expired_or_restarted_previous_capture_never_recaptures(service):
    first = ask(service)
    mark(service, first)
    service.close()
    result = ask(service, "Explain the previous screenshot")
    assert "no longer available" in result.text
    assert service.backend.capture.call_count == 1
    assert service.vision.send.call_count == 1


def test_screenshot_delivery_is_explicit_and_needs_no_cloud_analysis(service):
    result = ask(service, "Send me a screenshot")
    assert result.image.startswith(b"\x89PNG")
    assert result.expires_at > time.time()
    service.vision.send.assert_not_called()
    assert "image=" not in repr(result)


def test_private_history_blocks_cloud_before_capture(service):
    original = [ChatMessage("system", "Protected history.", local_only=True)]
    result = service.answer(service.resolve("Look at my screen"), original)
    assert "protected" in result.text
    service.backend.capture.assert_not_called()
    service.vision.send.assert_not_called()


def test_cancel_before_and_during_capture_does_not_infer(service):
    canceled = Event()
    canceled.set()
    assert service.answer(service.resolve("Look at my screen"), messages(),
                          cancel_event=canceled) is None
    service.backend.capture.assert_not_called()
    canceled.clear()

    def cancel(**_):
        canceled.set()
        return capture()

    service.backend.capture.side_effect = cancel
    assert service.answer(service.resolve("Look at my screen"), messages(),
                          cancel_event=canceled) is None
    service.vision.send.assert_not_called()


@pytest.mark.parametrize("invalidate", ["cache", "history"])
def test_reset_during_inference_discards_result_and_image(service, invalidate):
    def respond(*_, **__):
        if invalidate == "cache":
            clear_runtime_caches(service._cache.root)
        else:
            service.conversation.store.clear_session(service.conversation.session_id)
        return LLMResponse("Stale visual answer", "vision-test", "nim")

    service.vision.send.side_effect = respond
    assert ask(service) is None
    assert not service.conversation.get_recent_turns()


def test_model_can_request_one_bounded_native_crop(service):
    service.vision.send.side_effect = [
        LLMResponse('{"crop":{"monitor":"left","box":[10,10,100,100]}}', "v", "nim"),
        LLMResponse("The error says connection refused.", "v", "nim"),
    ]
    result = ask(service)
    assert "connection refused" in result.text
    assert service.backend.capture.call_count == 1
    assert service.vision.send.call_count == 2
    assert len(service._cache.recent()) == 1


def test_invalid_crop_does_not_read_other_images_or_retry_forever(service):
    service.vision.send.return_value = LLMResponse(
        '{"crop":{"monitor":"../../private","box":[-10,0,100,100]}}', "v", "nim",
    )
    assert "couldn't read" in ask(service).text
    assert service.vision.send.call_count == 1


def test_model_failure_returns_actionable_text_without_raw_payload(service):
    service.vision.send.side_effect = VisionError("The selected screen model timed out.")
    result = ask(service)
    assert result.text == "The selected screen model timed out."
    assert result.image is None
