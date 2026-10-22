"""Synthetic cross-process-lifetime and coordinate regressions for the screen service."""

import io
import json
import time
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from ciel.chat.state import ChatState
from ciel.config import Settings
from ciel.memory import ConversationManager, MemoryStore
from ciel.providers.base import ChatMessage, LLMResponse
from ciel.providers.vision import VisionCancelled
from ciel.screen import service as service_module
from ciel.screen.cache import CaptureCache
from ciel.screen.capture import Capture, Cursor, Monitor, ScreenImage
from ciel.screen.service import ScreenRequest, ScreenService, resolve_request


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def solid_capture(identifier, color, captured_at):
    data = encoded(Image.new("RGB", (200, 120), color))
    return Capture(
        identifier, captured_at, (Monitor("test", 0, 0, 200, 120),),
        Cursor(50, 50, captured_at, False),
        (ScreenImage("desktop", data, 200, 120), ScreenImage("test", data, 200, 120)),
    )


def mixed_capture(captured_at):
    left = Image.new("RGB", (400, 200), "red")
    right = Image.new("RGB", (200, 200), "blue")
    ImageDraw.Draw(right).rectangle((20, 40, 119, 119), fill="yellow")
    desktop = Image.new("RGB", (400, 200), "black")
    desktop.paste(left.resize((200, 100)), (0, 0))
    desktop.paste(right, (200, 0))
    return Capture(
        "mixed", captured_at,
        (Monitor("left", -200, -50, 400, 200, scale=2, focused=True),
         Monitor("right", 0, -50, 200, 200)),
        Cursor(-150, 0, captured_at, True, "left", 50, 50),
        (ScreenImage("desktop", encoded(desktop), 400, 200, origin_x=-200, origin_y=-50),
         ScreenImage("left", encoded(left), 400, 200, origin_x=-200, origin_y=-50,
                     logical_width=200, logical_height=100),
         ScreenImage("right", encoded(right), 200, 200, origin_x=0, origin_y=-50,
                     logical_width=200, logical_height=200)),
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "test.db"),
                        ciel_persona_file="", internet_enabled=False)
    store = MemoryStore(settings.memory.db_path)
    conversation = ConversationManager(store=store, session_id="synthetic-owner")
    backend, vision = MagicMock(), MagicMock()
    vision.send.return_value = LLMResponse("A desktop is visible.", "test-image-model", "nim")
    service = ScreenService(settings, conversation, backend=backend, vision=vision)
    yield service, runtime / "ciel-screen"
    service.close()
    store.close()


def ask(service, *, previous=False, current=lambda: True):
    return service.answer(
        ScreenRequest("Explain the previous screenshot" if previous else "Look at my screen",
                      previous=previous),
        [ChatMessage("system", "Explain the requested screen naturally.")],
        current=current,
    )


def spy_sheets(monkeypatch):
    calls = []
    original = service_module._sheet

    def sheet(panels):
        calls.append(tuple((name, data) for name, data in panels))
        return original(panels)

    monkeypatch.setattr(service_module, "_sheet", sheet)
    return calls


def test_previous_expiring_during_fresh_capture_is_not_sent(harness, monkeypatch):
    service, _ = harness
    now = [time.time()]
    monkeypatch.setattr("ciel.screen.cache.time.time", lambda: now[0])
    service.backend.capture.return_value = solid_capture("old-red", "red", now[0])
    assert ask(service) is not None
    sheets = spy_sheets(monkeypatch)

    def capture(**_):
        now[0] += 901
        return solid_capture("new-blue", "blue", now[0])

    service.backend.capture.side_effect = capture
    result = ask(service)
    assert result.capture_id == "new-blue"
    assert all("PREVIOUS" not in label for label, _ in sheets[-1])
    with Image.open(io.BytesIO(service.vision.send.call_args.args[0][-1].images[0].data)) as image:
        colors = image.convert("RGB").getcolors(maxcolors=image.width * image.height)
        assert not any(r > 220 and g < 20 and b < 20 for _, (r, g, b) in colors)
    assert [capture.id for capture in service._cache.recent()] == ["new-blue"]


def test_other_frontend_eviction_during_capture_never_sends_evicted_pixels(harness, monkeypatch):
    service, root = harness
    now = [time.time()]
    monkeypatch.setattr("ciel.screen.cache.time.time", lambda: now[0])
    service.backend.capture.return_value = solid_capture("old-red", "red", now[0])
    assert ask(service) is not None
    sheets = spy_sheets(monkeypatch)
    with CaptureCache(service._scope_key(), root) as other:
        def capture(**_):
            now[0] += 1
            other.add(solid_capture("intervening-green", "green", now[0]))
            now[0] += 1
            other.add(solid_capture("intervening-yellow", "yellow", now[0]))
            now[0] += 1
            return solid_capture("new-blue", "blue", now[0])

        service.backend.capture.side_effect = capture
        result = ask(service)
        assert result.capture_id == "new-blue"
        assert all("PREVIOUS" not in label for label, _ in sheets[-1])
        assert not other.contains("old-red")


def test_previous_expiry_during_sheet_build_stops_provider_transmission(harness, monkeypatch):
    service, _ = harness
    now = [time.time()]
    monkeypatch.setattr("ciel.screen.cache.time.time", lambda: now[0])
    service.backend.capture.return_value = solid_capture("old-red", "red", now[0])
    assert ask(service) is not None
    now[0] += 899
    service.backend.capture.return_value = solid_capture("current-blue", "blue", now[0])
    original = service_module._sheet
    transmitted = []

    def sheet(panels):
        data = original(panels)
        now[0] += 2
        return data

    def send(messages, *, cancel_event):
        if cancel_event.is_set():
            raise VisionCancelled("Cancelled before transmission.")
        transmitted.append(messages)
        return LLMResponse("Must not transmit expired evidence.", "test", "nim")

    monkeypatch.setattr(service_module, "_sheet", sheet)
    service.vision.send.side_effect = send
    assert ask(service) is None
    assert transmitted == []
    assert service._cache.contains("current-blue")


def test_pointer_marker_and_metadata_share_negative_origin_and_mixed_scale(harness):
    service, _ = harness
    capture = mixed_capture(time.time())
    with Image.open(io.BytesIO(service._desktop(capture))) as image:
        assert image.getpixel((66, 50)) == (0, 255, 128)
        assert image.getpixel((116, 100)) != (0, 255, 128)
    assert capture.image("left").point_to_pixel(-150, 0) == (100, 100)
    metadata = json.loads(service._metadata(capture))
    assert metadata["monitors"][0]["logical_width"] == 200
    assert metadata["monitors"][0]["logical_height"] == 100
    assert metadata["desktop_image"]["origin_x"] == -200
    assert metadata["desktop_image"]["origin_y"] == -50
    assert metadata["cursor"]["image_x"] == 50
    assert metadata["cursor"]["image_y"] == 50


def test_unstable_pointer_has_no_added_marker(harness):
    service, _ = harness
    capture = mixed_capture(time.time())
    unstable = replace(capture, cursor=replace(capture.cursor, stable=False))
    assert service._desktop(unstable) == unstable.image().data


def test_second_native_crop_retains_desktop_and_explicit_coordinate_context(harness, monkeypatch):
    service, _ = harness
    service.backend.capture.return_value = mixed_capture(time.time())
    spec = {"monitor": "right", "box": [20, 40, 100, 80]}
    responses = iter([
        LLMResponse(json.dumps({"crop": spec}), "test-image-model", "nim"),
        LLMResponse("The highlighted area is yellow.", "test-image-model", "nim"),
    ])
    sent = []

    def send(messages, **_):
        sent.append([(message.role, message.content, tuple(message.images))
                     for message in messages])
        return next(responses)

    service.vision.send.side_effect = send
    sheets = spy_sheets(monkeypatch)
    result = ask(service)
    assert result.text == "The highlighted area is yellow."
    assert len(sent) == 2
    assert sum(len(images) for _, _, images in sent[1]) == 1
    assert any("CURRENT" in label for label, _ in sheets[-1])
    label, pixels = next((label, data) for label, data in sheets[-1]
                         if label.startswith("REQUESTED DETAIL"))
    assert json.dumps(spec) in label
    assert json.dumps(spec) in sent[1][-1][1]
    assert "original monitor pixels" in sent[1][-1][1]
    assert "global logical" in sent[1][0][1]
    with Image.open(io.BytesIO(pixels)) as image:
        assert image.size == (100, 80)
        assert image.getpixel((50, 40)) == (255, 255, 0)
    assert [capture.id for capture in service._cache.recent()] == ["mixed"]


def test_real_cache_reset_invalidates_saved_results_and_caller_gate(harness):
    service, root = harness
    service.backend.capture.return_value = solid_capture("before-reset", "red", time.time())
    result = ask(service)
    assert service.still_valid(result)
    assert service.image_valid(result)
    with CaptureCache(service._scope_key(), root) as peer:
        assert peer.contains("before-reset")
        service.conversation.clear_session()
        assert not service.still_valid(result)
        assert not service.image_valid(result)
        assert list(root.glob("*/*.png")) == []
        assert not result.image
        service.backend.capture.return_value = solid_capture("after-reset", "blue", time.time())
        new_result = ask(service)
        assert service.still_valid(new_result)
        assert not service.still_valid(result)
        assert [capture.id for capture in service._cache.recent()] == ["after-reset"]


def test_caller_gate_change_during_capture_prevents_caching_or_inference(harness):
    service, root = harness
    valid = [True]

    def capture(**_):
        valid[0] = False
        return solid_capture("stale-owner", "red", time.time())

    service.backend.capture.side_effect = capture
    assert ask(service, current=lambda: valid[0]) is None
    assert list(root.glob("*/*.png")) == []
    service.vision.send.assert_not_called()


def test_pairing_changes_commit_even_when_screen_cleanup_directory_is_insecure(harness, tmp_path):
    service, root = harness
    state = ChatState(tmp_path / "pairing.db")
    state.bind(123, 42)
    service.backend.capture.return_value = solid_capture("before-unpair", "red", time.time())
    assert ask(service) is not None
    root.chmod(0o755)
    try:
        state.unpair()
        assert state.owner(123) is None
        state.bind(123, 99)
        assert state.owner(123) == 99
    finally:
        root.chmod(0o700)
        state.close()


def test_history_reset_commits_and_invalidates_results_when_screen_cleanup_fails(harness):
    service, root = harness
    service.backend.capture.return_value = solid_capture("before-reset", "red", time.time())
    result = ask(service)
    service.conversation.add_turn("user", "A synthetic screen discussion.")
    identity = service.conversation.session_id
    before_generation = service.conversation.store.get_session_generation(identity)
    root.chmod(0o755)
    try:
        assert service.conversation.clear_session() == 1
        assert service.conversation.store.get_session_generation(identity) > before_generation
        assert service.conversation.get_recent_turns() == []
        assert not service.still_valid(result)
        assert not service.image_valid(result)
    finally:
        root.chmod(0o700)


@pytest.mark.parametrize(("query", "previous"), [
    ("Send me the previous screenshot", True),
    ("Show me the earlier screen capture", True),
    ("Look at my screen and send me a screenshot", False),
])
def test_explicit_image_delivery_intent_survives_historical_and_combined_queries(query, previous):
    request = resolve_request(query, active=True)
    assert request is not None
    assert request.send_image
    assert request.previous is previous


def test_send_previous_screenshot_reuses_cached_pixels_without_cloud_analysis(harness):
    service, _ = harness
    service.backend.capture.return_value = solid_capture("cached", "red", time.time())
    assert ask(service) is not None
    service.vision.send.reset_mock()
    result = service.answer(
        resolve_request("Send me the previous screenshot", active=True),
        [ChatMessage("system", "Discuss the requested screen.")],
    )
    assert result.capture_id == "cached"
    assert result.image is not None and result.image.startswith(b"\x89PNG")
    assert service.backend.capture.call_count == 1
    service.vision.send.assert_not_called()
