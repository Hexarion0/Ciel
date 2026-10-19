"""Explicit microphone controls preserve work and invalidate stale voice input."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
from unittest.mock import MagicMock

import numpy as np
import pytest

from ciel.audio.listener import ListenerState, WakeListenerLoop
from ciel.audio.recorder import VoiceRecorder


@pytest.fixture
def listener() -> WakeListenerLoop:
    """Build a listener without initializing audio hardware or wake models."""
    return WakeListenerLoop(
        MagicMock(), detector=MagicMock(), recorder=VoiceRecorder(),
        stt=MagicMock(), tts=MagicMock(),
    )


def test_explicit_microphone_mute_is_idempotent(listener: WakeListenerLoop) -> None:
    assert not listener.microphone_muted
    for muted in (False, True, True, False, False, True, False):
        assert listener.set_microphone_mute(muted) is muted
        assert listener.microphone_muted is muted
    assert listener._microphone_epoch == 4
    assert listener.detector.reset.call_count == 4


@pytest.mark.parametrize("muted", [False, True])
@pytest.mark.parametrize("state", [ListenerState.RECORDING, ListenerState.PROCESSING])
def test_repeated_mute_preserves_pending_work(
    listener: WakeListenerLoop, muted: bool, state: ListenerState,
) -> None:
    listener.set_microphone_mute(muted)
    listener.detector.reset.reset_mock()
    listener._state = state
    listener._recording_generation = 7
    listener._speech_streak = 2
    listener._response_cancel = cancel = Event()
    listener._last_wake_info = wake_info = {"wake_phrase": "hey ciel"}
    frame = np.ones(1280, dtype=np.float32)
    queued_frame = (frame, 1.0, listener._microphone_epoch)
    request = (frame, {"typed_text": "keep going"}, 7)
    listener._recent_audio.append(frame)
    listener._frame_queue.put(queued_frame)
    listener._processing_queue.put(request)
    if state == ListenerState.RECORDING:
        listener.recorder.start()
        listener.recorder.add_frame(frame)
    epoch = listener._microphone_epoch

    assert listener.set_microphone_mute(muted) is muted
    assert listener.set_microphone_mute(muted) is muted

    assert listener.state == state
    assert listener._microphone_epoch == epoch
    assert listener._recording_generation == 7
    assert listener._speech_streak == 2
    assert listener._response_cancel is cancel
    assert not cancel.is_set()
    assert listener._last_wake_info is wake_info
    assert listener._recent_audio[0] is frame
    assert listener._frame_queue.unfinished_tasks == 1
    assert listener._frame_queue.get_nowait() is queued_frame
    assert listener._processing_queue.get_nowait() is request
    assert listener.recorder.is_recording() is (state == ListenerState.RECORDING)
    if state == ListenerState.RECORDING:
        np.testing.assert_array_equal(listener.recorder.get_audio(), frame)
    listener.detector.reset.assert_not_called()
    listener.tts.stop.assert_not_called()


def test_mute_transition_cancels_capture_and_discards_queued_audio(
    listener: WakeListenerLoop,
) -> None:
    listener._running = True
    listener._start_recording()
    listener.recorder.add_frame(np.ones(1280, dtype=np.float32))
    generation = listener._recording_generation
    old_epoch = listener._microphone_epoch
    listener._frame_queue.put((np.ones(1280), 1.0, old_epoch))
    listener._recent_audio.append(np.ones(1280))
    listener._speech_streak = 2

    assert listener.set_microphone_mute(True)

    assert not listener.recorder.is_recording()
    assert listener.recorder.get_audio().size == 0
    assert listener.state == ListenerState.LISTENING_WAKE
    assert listener._recording_generation == generation + 1
    assert listener._microphone_epoch == old_epoch + 1
    assert listener._frame_queue.empty()
    assert listener._frame_queue.unfinished_tasks == 0
    assert not listener._recent_audio
    assert listener._speech_streak == 0
    listener.detector.reset.assert_called_once_with(set_cooldown=False)
    listener.tts.stop.assert_not_called()


def test_mute_then_unmute_discards_voice_queued_before_transition(
    listener: WakeListenerLoop,
) -> None:
    listener._running = True
    old_epoch = listener._microphone_epoch
    assert listener.set_microphone_mute(True)
    assert not listener.set_microphone_mute(False)
    listener._processing_queue.put((
        np.ones(1280), {"microphone_epoch": old_epoch}, listener._recording_generation,
    ))
    listener._processing_queue.put((np.empty(0), {
        "input_source": "keyboard", "typed_text": "current message", "cancel_event": Event(),
    }, listener._recording_generation))
    received = []

    def respond(text: str, _info: dict, _audio: np.ndarray) -> bool:
        received.append(text)
        listener._stop_event.set()
        return False

    listener.on_transcription = respond
    listener._process_worker()

    assert received == ["current message"]
    listener.stt.transcribe_detailed.assert_not_called()
    assert listener._processing_queue.unfinished_tasks == 0


def test_mute_transition_during_stt_discards_old_voice(
    listener: WakeListenerLoop,
) -> None:
    entered, release = Event(), Event()
    listener._running = True
    listener._state = ListenerState.PROCESSING
    listener.on_transcription = MagicMock()

    def transcribe(*_args, **_kwargs) -> dict:
        entered.set()
        assert release.wait(timeout=2)
        listener._stop_event.set()
        return {"text": "stale voice"}

    listener.stt.transcribe_detailed.side_effect = transcribe
    listener._processing_queue.put((np.ones(1280), {
        "microphone_epoch": listener._microphone_epoch,
    }, listener._recording_generation))
    worker = Thread(target=listener._process_worker, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=2)
        assert listener.set_microphone_mute(True)
        assert not listener.set_microphone_mute(False)
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    listener.on_transcription.assert_not_called()
    assert listener.state == ListenerState.LISTENING_WAKE


def test_explicit_mute_preserves_ongoing_typed_reply(listener: WakeListenerLoop) -> None:
    entered, release = Event(), Event()
    listener._running = True
    received = []

    def respond(text: str, info: dict, _audio: np.ndarray) -> bool:
        received.append((text, info["cancel_event"]))
        entered.set()
        assert release.wait(timeout=2)
        listener._stop_event.set()
        return False

    listener.on_transcription = respond
    assert listener.submit_text("keep replying")
    listener.tts.stop.reset_mock()
    worker = Thread(target=listener._process_worker, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=2)
        generation = listener._recording_generation
        assert listener.set_microphone_mute(True)
        assert listener.set_microphone_mute(True)
        assert not listener.set_microphone_mute(False)
        assert not listener.set_microphone_mute(False)
        assert not received[0][1].is_set()
        assert listener._recording_generation == generation
        assert listener.state == ListenerState.PROCESSING
        listener.tts.stop.assert_not_called()
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert received[0][0] == "keep replying"
    assert not received[0][1].is_set()
    listener.stt.transcribe_detailed.assert_not_called()


def test_concurrent_mute_requests_apply_one_transition(listener: WakeListenerLoop) -> None:
    barrier = Barrier(8)

    def mute() -> bool:
        barrier.wait(timeout=2)
        return listener.set_microphone_mute(True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: mute(), range(8)))

    assert results == [True] * 8
    assert listener.microphone_muted
    assert listener._microphone_epoch == 1
    listener.detector.reset.assert_called_once_with(set_cooldown=False)


def test_concurrent_toggle_remains_atomic(listener: WakeListenerLoop) -> None:
    barrier = Barrier(8)

    def toggle() -> bool:
        barrier.wait(timeout=2)
        return listener.toggle_microphone_mute()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: toggle(), range(8)))

    assert results.count(True) == 4
    assert results.count(False) == 4
    assert not listener.microphone_muted
    assert listener._microphone_epoch == 8
    assert listener.detector.reset.call_count == 8
