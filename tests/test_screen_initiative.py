"""Observation policy is judged by when it stays silent as well as delivered hints."""

import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ciel.providers.vision import VisionError
from ciel.screen.capture import Capture, Cursor, Monitor, ScreenImage
from ciel.screen.initiative import ScreenInitiative
from ciel.screen.observation import CodingAnalyzer, CodingObservation, parse_control


def capture(color="red"):
    """Build only synthetic monitor pixels; no desktop access in these tests."""
    output = io.BytesIO()
    Image.new("RGB", (400, 240), color).save(output, format="PNG")
    image = output.getvalue()
    return Capture("test", 100.0, (Monitor("coding", 0, 0, 400, 240, focused=True),),
                   Cursor(0, 0, 100.0, True, "coding"),
                   (ScreenImage("coding", image, 400, 240),
                    ScreenImage("desktop", image, 400, 240)))


@pytest.fixture
def harness(monkeypatch):
    """Drive one scan at a time with controlled time, foreground activity and output."""
    monkeypatch.setattr(ScreenInitiative, "_run", lambda _self: None)
    state = SimpleNamespace(now=0.0, scope=("owner", 1), idle=True, private=False,
                            activity=1, image=capture(), queue=[], delivered=[])
    observation = CodingObservation("coding", "TypeError: missing argument 'name'",
                                    "Try passing the name argument.", "fake", "vision")
    backend = MagicMock()
    backend.capture.side_effect = lambda **_kwargs: state.image
    analyzer = MagicMock(vision=SimpleNamespace(provider="ollama"))
    analyzer.analyze.return_value = observation

    def deliver(observation, cancel, started):
        assert not cancel.is_set()
        if started():
            state.delivered.append(observation.hint)

    coordinator = ScreenInitiative(
        backend=backend, analyzer=analyzer, scope=lambda: state.scope,
        ready=lambda: state.idle, local_only=lambda: state.private,
        owner_activity=lambda: state.activity,
        submit=lambda callback: state.queue.append(callback) is None,
        deliver=deliver, clock=lambda: state.now,
    )

    def tick(at):
        state.now = at
        coordinator.tick()

    def repeat():
        for now in range(20, 141, 20):
            tick(now)

    state.coordinator, state.backend, state.analyzer = coordinator, backend, analyzer
    state.tick, state.repeat, state.observation = tick, repeat, observation
    yield state
    coordinator.close()


def test_disabled_by_default_and_restart_does_not_resume(harness):
    harness.tick(1000)
    harness.backend.capture.assert_not_called()
    assert "off" in harness.coordinator.status()


def test_error_must_repeat_for_two_minutes_and_pixels_are_rechecked(harness):
    harness.coordinator.control("on")
    harness.repeat()
    assert harness.analyzer.analyze.call_count == 2
    assert len(harness.queue) == 1
    assert not harness.delivered
    assert harness.backend.capture.call_count == 7
    harness.queue.pop()(threading.Event())
    assert harness.backend.capture.call_count == 8
    assert harness.delivered == [harness.observation.hint]


def test_unchanged_screen_without_error_does_not_trigger_repeated_inference(harness):
    harness.analyzer.analyze.return_value = None
    harness.coordinator.control("on")
    harness.repeat()
    assert harness.analyzer.analyze.call_count == 1
    assert not harness.queue


def test_disappearing_error_resets_the_two_minute_wait(harness):
    harness.coordinator.control("on")
    harness.tick(20)
    harness.image = capture("blue")
    harness.analyzer.analyze.return_value = None
    harness.tick(60)
    harness.image = capture()
    harness.analyzer.analyze.return_value = harness.observation
    for now in range(80, 181, 20):
        harness.tick(now)
    assert not harness.queue
    harness.tick(200)
    assert len(harness.queue) == 1


def test_different_errors_do_not_accumulate_as_one_problem(harness):
    harness.coordinator.control("on")
    for now in range(20, 181, 20):
        harness.image = capture("red" if now % 40 else "blue")
        harness.analyzer.analyze.return_value = replace(harness.observation, error=f"Error {now}")
        harness.tick(now)
    assert not harness.queue


@pytest.mark.parametrize("operation", ["pause", "off"])
def test_disable_or_pause_revokes_already_queued_delivery(harness, operation):
    harness.coordinator.control("on")
    harness.repeat()
    harness.coordinator.control(operation)
    harness.queue.pop()(threading.Event())
    assert not harness.delivered
    count = harness.backend.capture.call_count
    harness.tick(1000)
    assert harness.backend.capture.call_count == count


@pytest.mark.parametrize("change", ["scope", "foreground", "off"])
def test_inflight_analysis_yields_without_output(harness, change):
    harness.coordinator.control("on")

    def analyze(_capture, cancel, **_kwargs):
        if change == "scope":
            harness.scope = ("other-owner", 1)
        elif change == "foreground":
            harness.idle = False
        else:
            harness.coordinator.control("off")
        assert cancel.is_set()
        return harness.observation

    harness.analyzer.analyze.side_effect = analyze
    harness.tick(20)
    assert not harness.queue
    harness.tick(40)
    assert not harness.delivered


def test_owner_or_history_change_disables_session_observation(harness):
    harness.coordinator.control("on")
    harness.tick(20)
    harness.scope = ("owner", 2)
    harness.tick(40)
    assert "off" in harness.coordinator.status()
    assert harness.coordinator._observation is None


def test_busy_foreground_breaks_continuous_error_evidence(harness):
    harness.coordinator.control("on")
    harness.tick(20)
    harness.idle = False
    harness.tick(40)
    harness.idle = True
    for now in range(60, 161, 20):
        harness.tick(now)
    assert not harness.queue


@pytest.mark.parametrize("reason", ["fixed", "stale", "cancelled", "private", "chat"])
def test_delivery_rechecks_current_pixels_time_cancellation_and_privacy(harness, reason):
    harness.coordinator.control("on")
    harness.repeat()
    cancelled = threading.Event()
    if reason == "fixed":
        harness.image = capture("green")
    elif reason == "stale":
        harness.now += 16
    elif reason == "cancelled":
        cancelled.set()
    elif reason == "chat":
        harness.activity += 1
    else:
        harness.private = True
    harness.queue.pop()(cancelled)
    assert not harness.delivered


def test_tiny_error_text_change_prevents_stale_hint(harness):
    harness.coordinator.control("on")
    harness.repeat()
    output = io.BytesIO()
    image = Image.new("RGB", (400, 240), "red")
    image.putpixel((200, 100), (0, 0, 0))
    image.save(output, format="PNG")
    harness.image = replace(harness.image, images=(
        ScreenImage("coding", output.getvalue(), 400, 240),
        ScreenImage("desktop", output.getvalue(), 400, 240),
    ))
    harness.queue.pop()(threading.Event())
    assert not harness.delivered


def test_ignored_hint_stays_quiet_after_cooldown_until_owner_reengages(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.queue.pop()(threading.Event())
    count = harness.backend.capture.call_count
    harness.tick(2000)
    assert harness.backend.capture.call_count == count
    assert "quiet until" in harness.coordinator.status()
    harness.activity += 1
    harness.tick(2020)
    assert harness.backend.capture.call_count == count + 1


def test_reengagement_and_reenable_do_not_erase_fifteen_minute_gap(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.queue.pop()(threading.Event())
    harness.activity += 1
    count = harness.backend.capture.call_count
    harness.coordinator.control("off")
    harness.coordinator.control("on")
    harness.tick(1039)
    assert harness.backend.capture.call_count == count
    harness.tick(1040)
    assert harness.backend.capture.call_count == count + 1


def test_same_unresolved_error_does_not_receive_repeated_hints(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.queue.pop()(threading.Event())
    harness.activity += 1
    for now in range(1040, 1201, 20):
        harness.tick(now)
    assert not harness.queue
    assert harness.delivered == [harness.observation.hint]


def test_observing_error_disappear_allows_help_if_it_returns(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.queue.pop()(threading.Event())
    harness.activity += 1
    harness.image = capture("blue")
    harness.analyzer.analyze.return_value = None
    harness.tick(1040)
    harness.image = capture()
    harness.analyzer.analyze.return_value = harness.observation
    for now in range(1060, 1181, 20):
        harness.tick(now)
    assert len(harness.queue) == 1


def test_failed_delivery_does_not_start_cooldown(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.coordinator.deliver = lambda *_args: None
    harness.queue.pop()(threading.Event())
    assert harness.coordinator._last_delivered == float("-inf")
    assert not harness.coordinator._waiting


def test_provider_failure_backs_off_and_never_logs_screen_text(harness, caplog):
    harness.coordinator.control("on")
    harness.analyzer.analyze.side_effect = VisionError("private password on the screen")
    harness.tick(20)
    harness.tick(40)
    assert harness.backend.capture.call_count == 1
    assert "private password" not in caplog.text
    assert "connection problem" in harness.coordinator.status()
    harness.tick(60)
    assert harness.backend.capture.call_count == 2


def test_protected_context_blocks_cloud_before_capture(harness):
    harness.private = True
    harness.analyzer.vision.provider = "nim"
    harness.coordinator.control("on")
    harness.tick(20)
    harness.backend.capture.assert_not_called()
    harness.analyzer.analyze.assert_not_called()


def test_single_scan_in_flight_and_no_suggestion_backlog(harness):
    harness.coordinator.control("on")

    def analyze(*_args, **_kwargs):
        harness.coordinator.tick()
        return harness.observation

    harness.analyzer.analyze.side_effect = analyze
    harness.repeat()
    harness.tick(145)
    assert harness.analyzer.analyze.call_count == 2
    assert len(harness.queue) == 1


def test_dropped_queued_hint_expires_instead_of_blocking_future_scans(harness):
    harness.coordinator.control("on")
    harness.repeat()
    harness.tick(180)
    assert harness.analyzer.analyze.call_count == 3
    assert len(harness.queue) == 1


@pytest.mark.parametrize(("text", "operation"), [
    ("Start watching my screen", "on"), ("Please pause screen observation.", "pause"),
    ("stop watching my screen", "off"), ("resume screen awareness", "resume"),
    ("screen awareness status", "status"),
    ("can you explain screen awareness", None),
    ('the page says "start watching my screen"', None),
    ("don't start watching my screen", None),
])
def test_observation_controls_require_whole_explicit_requests(text, operation):
    assert parse_control(text) == operation


def result(**overrides):
    """A model response is data to validate, never direct authority to speak."""
    values = {"kind": "coding_error", "readable": True, "monitor": "coding",
              "error": "TypeError: missing argument 'name'", "hint": "Try passing name."}
    values.update(overrides)
    return SimpleNamespace(content=json.dumps(values), provider="fake", model="vision")


@pytest.mark.parametrize("overrides", [
    {"kind": "none"}, {"readable": False}, {"readable": "true"}, {"monitor": "unknown"},
    {"error": ""}, {"hint": "x" * 321}, {"error": ["not a string"]},
])
def test_analyzer_rejects_unclear_or_invalid_evidence(overrides):
    vision = MagicMock()
    vision.send.return_value = result(**overrides)
    assert CodingAnalyzer(vision).analyze(capture(), threading.Event()) is None


def test_analyzer_can_read_a_non_cursor_monitor_and_one_closer_view():
    vision = MagicMock()
    vision.send.side_effect = [
        SimpleNamespace(content=json.dumps({"crop": {"monitor": "coding",
                                                     "box": [0, 0, 200, 100]}})),
        result(),
    ]
    observation = CodingAnalyzer(vision).analyze(capture(), threading.Event(), local_only=True)
    assert observation.box == (0, 0, 200, 100)
    assert vision.send.call_count == 2
    messages = vision.send.call_args.args[0]
    assert messages[0].local_only
    assert sum(len(message.images) for message in messages) == 1
    assert "untrusted" in messages[0].content


def test_analyzer_never_reuses_previous_error_as_proof():
    vision = MagicMock()
    vision.send.return_value = result(kind="none")
    previous = CodingObservation("coding", "secret previous error", "Try passing name.",
                                 box=(0, 0, 200, 100))
    assert CodingAnalyzer(vision).analyze(capture(), threading.Event(), focus=previous) is None
    assert all("secret previous error" not in message.content
               for message in vision.send.call_args.args[0])
