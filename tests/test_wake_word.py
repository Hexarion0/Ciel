"""Tests for wake word detection, voice recording, and listener loop."""

import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from ciel.audio.listener import ListenerState, WakeListenerLoop
from ciel.audio.recorder import VoiceRecorder
from ciel.audio.wake import WakeWordDetector
from ciel.platform.base import AudioBackend, AudioDeviceInfo


class MockAudioBackend(AudioBackend):
    """Mock audio backend for simulating hardware streams in tests."""

    def __init__(self) -> None:
        self._streaming = False
        self.callback = None

    def list_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(
                index=0,
                name="Mock Mic",
                hostapi="Mock",
                max_input_channels=1,
                max_output_channels=0,
                default_samplerate=16000.0,
                is_default_input=True,
            )
        ]

    def get_default_input_device(self) -> AudioDeviceInfo | None:
        return self.list_devices()[0]

    def get_default_output_device(self) -> AudioDeviceInfo | None:
        return None

    def resolve_device(self, device: int | str | None, is_input: bool = True) -> int | None:
        return 0

    def record(
        self,
        duration: float,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> np.ndarray:
        return np.zeros((int(duration * sample_rate), channels), dtype=np.float32)

    def save_wav(self, audio_data: np.ndarray, file_path, sample_rate: int = 16000):
        return file_path

    def start_stream(
        self,
        callback,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        blocksize: int = 1024,
    ) -> None:
        self._streaming = True
        self.callback = callback

    def stop_stream(self) -> None:
        self._streaming = False

    def is_streaming(self) -> bool:
        return self._streaming


def test_wake_detector_initialization(monkeypatch):
    """Verify openWakeWord detector initializes and exposes models."""
    monkeypatch.setattr(
        WakeWordDetector, "_resolve_model_paths", staticmethod(lambda _models: ["model.onnx"])
    )
    monkeypatch.setattr(
        WakeWordDetector,
        "_load_wake_model",
        staticmethod(lambda _paths: SimpleNamespace(models={"hey_jarvis": object()})),
    )
    detector = WakeWordDetector(
        models=["hey_jarvis"], threshold=0.6, cooldown_seconds=1.5, enable_whisper_spotter=False
    )
    assert detector.threshold == 0.6
    assert detector.cooldown_seconds == 1.5
    assert not detector.is_in_cooldown()
    assert len(detector.available_models) > 0


def test_wake_detector_silent_frame():
    """Verify detector handles silent frames cleanly without false positive triggers."""
    detector = WakeWordDetector(threshold=0.5, enable_whisper_spotter=False)
    silent_frame = np.zeros(1280, dtype=np.float32)
    result = detector.process_frame(silent_frame)
    assert result is None


def test_voice_recorder():
    """Verify VoiceRecorder accumulates frames, detects speech, and detects trailing silence."""
    recorder = VoiceRecorder(
        sample_rate=16000,
        silence_threshold_rms=0.01,
        silence_duration_seconds=0.1,
        min_speech_duration_seconds=0.05,
        max_duration_seconds=2.0,
    )
    recorder.start()
    assert recorder.is_recording()

    # Feed speech chunk (sine tone)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech_frame = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 2 speech frames (~0.2s)
    cont = recorder.add_frame(speech_frame)
    assert cont is True
    cont = recorder.add_frame(speech_frame)
    assert cont is True

    # Feed silent frames to trigger silence conclusion
    silent_frame = np.zeros(1600, dtype=np.float32)
    recorder.add_frame(silent_frame)
    recorder.add_frame(silent_frame)

    audio = recorder.get_audio()
    assert len(audio) > 0


def test_wake_listener_loop_state_transitions():
    """Verify WakeListenerLoop lifecycle and mock streaming integration."""
    backend = MockAudioBackend()
    states_observed: list[ListenerState] = []

    def on_state(s: ListenerState):
        states_observed.append(s)

    loop = WakeListenerLoop(
        audio_backend=backend,
        detector=WakeWordDetector(enable_whisper_spotter=False),
        on_state_change=on_state,
    )
    assert loop.state == ListenerState.IDLE

    loop.start()
    assert loop.state == ListenerState.LISTENING_WAKE
    assert backend.is_streaming()

    loop.stop()
    assert loop.state == ListenerState.IDLE
    assert not backend.is_streaming()


def test_wake_listener_barge_in_interruption():
    """Verify a detected wake during playback triggers immediate barge-in stop."""
    backend = MockAudioBackend()
    detector = SimpleNamespace(
        process_frame=lambda frame: {"wake_phrase": "hey ciel"},
        reset=lambda **kwargs: None,
    )

    class MockTTS:
        def __init__(self):
            self._speaking = True
            self.stopped = False

        def is_speaking(self) -> bool:
            return self._speaking

        def stop(self) -> None:
            self._speaking = False
            self.stopped = True

    mock_tts = MockTTS()
    loop = WakeListenerLoop(
        audio_backend=backend,
        detector=detector,
        tts=mock_tts,
        barge_in=True,
        barge_in_threshold_rms=0.01,
    )
    loop.start()

    # Create high-energy speech frame to trigger barge-in
    t = np.linspace(0, 0.08, 1280, endpoint=False)
    speech_frame = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Trigger audio callback directly
    backend.callback(speech_frame, 1280, None, None)

    # Verify TTS was stopped and loop switched to recording
    wait_for(lambda: mock_tts.stopped)
    assert mock_tts.stopped is True
    assert loop.state == ListenerState.RECORDING

    loop.stop()


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for audio worker")
        time.sleep(0.005)


def test_keyboard_works_with_muted_microphone_and_bypasses_stt():
    from unittest.mock import MagicMock

    detector, stt, tts = MagicMock(), MagicMock(), MagicMock()
    received = []
    completed = Event()

    def respond(text, info, audio):
        received.append((text, info, audio.size))
        completed.set()
        return True

    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, stt=stt, tts=tts, on_transcription=respond,
    )
    loop.start()
    try:
        assert loop.toggle_microphone_mute()
        loop._audio_callback(np.ones(1280, dtype=np.float32), 1280, None, None)
        assert loop._frame_queue.empty()
        assert loop.submit_text("What is Ciel?")
        assert completed.wait(timeout=1)
        wait_for(lambda: loop.state == ListenerState.LISTENING_WAKE)
        assert received[0][0] == "What is Ciel?"
        assert received[0][1]["input_source"] == "keyboard"
        assert received[0][2] == 0
        stt.transcribe.assert_not_called()
        stt.transcribe_detailed.assert_not_called()
        assert not loop.recorder.is_recording()
        assert not loop.toggle_microphone_mute()
    finally:
        loop.stop()


def test_keyboard_stop_cancels_active_reply_and_listener_accepts_next_message():
    from unittest.mock import MagicMock

    entered, canceled, finished = Event(), Event(), Event()
    received = []
    tts = MagicMock()

    def respond(text, info, _audio):
        received.append(text)
        if text == "first":
            entered.set()
            assert info["cancel_event"].wait(timeout=1)
            canceled.set()
        else:
            finished.set()
        return False

    loop = WakeListenerLoop(
        MockAudioBackend(), detector=MagicMock(), tts=tts, on_transcription=respond,
    )
    loop.start()
    try:
        assert loop.submit_text("first")
        assert entered.wait(timeout=1)
        loop.cancel_response()
        assert canceled.wait(timeout=1)
        assert loop.is_running
        tts.stop.assert_called()
        assert loop.submit_text("second")
        assert finished.wait(timeout=1)
        assert received == ["first", "second"]
    finally:
        loop.stop()


def test_muting_discards_recording_and_old_queued_voice_after_unmute():
    from unittest.mock import MagicMock

    callback = MagicMock()
    loop = WakeListenerLoop(MockAudioBackend(), detector=MagicMock(), on_transcription=callback)
    loop._running = True
    loop._start_recording()
    loop.recorder.add_frame(np.ones(1280, dtype=np.float32))
    old_info = {"microphone_epoch": loop._microphone_epoch}
    assert loop.toggle_microphone_mute()
    assert not loop.recorder.is_recording()
    assert loop.recorder.get_audio().size == 0
    assert not loop.toggle_microphone_mute()
    loop._processing_queue.put((np.ones(1280), old_info, loop._recording_generation))
    loop._processing_queue.put((np.empty(0), {
        "input_source": "keyboard", "typed_text": "new message", "cancel_event": Event(),
    }, loop._recording_generation))

    def finish(text, *_args):
        assert text == "new message"
        loop._stop_event.set()
        return False

    loop.on_transcription = finish
    loop._process_worker()
    callback.assert_not_called()


def test_mute_during_stt_returns_to_standby_after_unmute():
    from unittest.mock import MagicMock

    entered, release = Event(), Event()
    stt, callback = MagicMock(), MagicMock()

    def transcribe(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=1)
        return {"text": "old voice input"}

    stt.transcribe_detailed.side_effect = transcribe
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=MagicMock(), stt=stt, on_transcription=callback,
    )
    loop.start()
    try:
        loop._set_state(ListenerState.PROCESSING)
        loop._processing_queue.put((np.ones(1280), {
            "microphone_epoch": loop._microphone_epoch,
        }, loop._recording_generation))
        assert entered.wait(timeout=1)
        assert loop.toggle_microphone_mute()
        assert not loop.toggle_microphone_mute()
        release.set()
        wait_for(lambda: loop.state == ListenerState.LISTENING_WAKE)
        callback.assert_not_called()
    finally:
        release.set()
        loop.stop()


def test_keyword_inference_does_not_block_frames_and_reset_discards_results(monkeypatch):
    entered, release = Event(), Event()

    class SlowSpotter:
        def transcribe(self, _audio, **_kwargs):
            entered.set()
            assert release.wait(2)
            return [SimpleNamespace(text="Hey Ciel")], None

    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: SlowSpotter())
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        frame = np.ones(1280, dtype=np.float32) * 0.1
        for _ in range(7):
            detector.process_frame(frame)
        assert entered.wait(1)
        assert detector.process_frame(frame) is None
        detector.reset()
        release.set()
        wait_for(lambda: detector._spotter_requests.unfinished_tasks == 0)
        assert detector.process_frame(frame) is None
    finally:
        release.set()
        detector.stop()


def test_keyword_result_triggers_wake_on_next_frame(monkeypatch):
    class Spotter:
        def transcribe(self, _audio, **_kwargs):
            return [SimpleNamespace(text="Hey Ciel")], None

    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: Spotter())
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        frame = np.ones(1280, dtype=np.float32) * 0.1
        for _ in range(7):
            detector.process_frame(frame)
        wait_for(lambda: detector._spotter_result is not None)
        result = detector.process_frame(frame)
        assert result["model"] == "whisper_keyword"
        assert result["wake_tail_samples"] == 1280
        assert detector.process_frame(frame) is None
    finally:
        detector.stop()


def test_callback_keeps_latest_audio_when_detection_is_slow():
    entered, release = Event(), Event()

    class SlowDetector:
        def process_frame(self, _frame):
            entered.set()
            assert release.wait(2)

        def reset(self, **_kwargs):
            pass

    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=SlowDetector())
    loop.start()
    try:
        frame = np.zeros(1280, dtype=np.float32)
        backend.callback(frame, 1280, None, None)
        assert entered.wait(1)
        for _ in range(100):
            backend.callback(frame, 1280, None, None)
        assert loop._frame_queue.qsize() == 8
        assert loop.dropped_frames >= 92
        frame[:] = 1
        assert not np.any(loop._frame_queue.queue[-1][0])  # Callback owns a copy.
    finally:
        release.set()
        loop.stop()


def test_slow_wake_notification_does_not_delay_recording():
    entered, release = Event(), Event()

    class Trigger:
        def process_frame(self, _frame):
            return {"model": "mock"}

        def reset(self, **_kwargs):
            pass

    def on_wake(_info):
        entered.set()
        assert release.wait(2)

    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=Trigger(), on_wake=on_wake)
    loop.start()
    try:
        frame = np.ones(1280, dtype=np.float32) * 0.1
        backend.callback(frame, 1280, None, None)
        assert entered.wait(1)
        backend.callback(frame, 1280, None, None)
        wait_for(lambda: len(loop.recorder._buffer) == 2)
        assert loop.state == ListenerState.RECORDING
    finally:
        release.set()
        loop.stop()


def test_listener_restart_and_failed_stream_start_are_clean():
    backend = MockAudioBackend()
    loop = WakeListenerLoop(backend, detector=WakeWordDetector(enable_whisper_spotter=False))
    for _ in range(2):
        loop.start()
        assert loop.is_running
        loop.stop()
        assert not loop.is_running
        assert all(not thread.is_alive() for thread in loop._threads)

    def fail(**_kwargs):
        raise RuntimeError("No microphone")

    backend.start_stream = fail
    with pytest.raises(RuntimeError, match="No microphone"):
        loop.start()
    assert loop.state == ListenerState.IDLE
    assert all(not thread.is_alive() for thread in loop._threads)


def test_old_response_cannot_cancel_barge_in():
    started, finish = Event(), Event()
    backend = MockAudioBackend()
    detector = WakeWordDetector(enable_whisper_spotter=False)

    def response(*_args):
        started.set()
        assert finish.wait(2)
        return False

    loop = WakeListenerLoop(backend, detector=detector, on_transcription=response)
    loop.start()
    try:
        with loop._lock:
            loop._start_recording()
            loop._state = ListenerState.PROCESSING
            loop._processing_queue.put((np.ones(1280), {}, loop._recording_generation))
        assert started.wait(1)
        with loop._lock:
            loop._start_recording()  # New recording started while old callback is running.
        finish.set()
        wait_for(lambda: loop._processing_queue.unfinished_tasks == 0)
        assert loop.state == ListenerState.RECORDING
    finally:
        finish.set()
        loop.stop()


def test_recorder_uses_audio_duration_instead_of_processing_speed():
    recorder = VoiceRecorder(silence_duration_seconds=0.2, min_speech_duration_seconds=0.1)
    recorder.start()
    for _ in range(4):
        assert recorder.add_frame(np.ones(1280, dtype=np.float32) * 0.1)
    assert recorder.add_frame(np.zeros(1280, dtype=np.float32))
    assert recorder.add_frame(np.zeros(1280, dtype=np.float32))
    assert not recorder.add_frame(np.zeros(1280, dtype=np.float32))


def test_keyword_spotter_filters_noise_and_low_confidence_wake(monkeypatch):
    calls = []

    class Spotter:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    text="Hey Ciel",
                    no_speech_prob=0.9,
                    avg_logprob=-1.5,
                )
            ], None

    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: Spotter())
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        frame = np.ones(1280, dtype=np.float32) * 0.1
        for _ in range(7):
            detector.process_frame(frame)
        wait_for(lambda: detector._spotter_result is not None)
        assert detector.process_frame(frame) is None
        assert calls[0]["vad_filter"] is True
        assert calls[0]["condition_on_previous_text"] is False
        assert calls[0]["initial_prompt"] != "hey ciel"
    finally:
        detector.stop()


def test_wake_timestamp_accounts_for_inference_delay(monkeypatch):
    detector = WakeWordDetector(enable_whisper_spotter=False)
    words = [
        SimpleNamespace(word="Hey", end=0.1),
        SimpleNamespace(word=" Ciel", end=0.3),
        SimpleNamespace(word=" what", end=0.5),
    ]
    assert detector._wake_word_end([SimpleNamespace(words=words)]) == 0.3
    detector._samples_seen = 8000
    detector._spotter_result = (detector._generation, "Hey Ciel what", 4800)
    result = detector.process_frame(np.zeros(1280, dtype=np.float32))
    assert result["wake_tail_samples"] == 4480


def test_wake_audio_trim_keeps_command_and_does_not_rewrite_followup():
    from unittest.mock import MagicMock

    from ciel.conversation import strip_wake_phrase

    detector = MagicMock()
    stt = SimpleNamespace(transcribe=lambda *_args, **_kwargs: "I hate Ciel.")
    outputs = []
    loop = WakeListenerLoop(
        MockAudioBackend(),
        detector=detector,
        stt=stt,
        on_transcription=lambda text, *_args: outputs.append(text) or False,
    )
    trigger = {"wake_tail_samples": 1280, "wake_phrase": "hey ciel"}
    greeting = np.full(1280, 0.2, dtype=np.float32)
    command = np.full(1280, 0.4, dtype=np.float32)
    loop._recent_audio.extend([greeting, command])
    loop._last_wake_info = trigger.copy()
    loop._start_recording()
    loop._seed_wake_audio(trigger)
    assert np.array_equal(loop.recorder.get_audio(), command)
    loop._running = True
    info = loop._last_wake_info.copy()
    loop._last_wake_info.pop("wake_prefix_trimmed")
    loop._processing_queue.put((command, info, loop._recording_generation))

    def finish(text, *_args):
        outputs.append(text)
        loop._stop_event.set()
        return False

    loop.on_transcription = finish
    loop._process_worker()
    assert strip_wake_phrase(outputs[-1]) == "I hate Ciel."
    loop._stop_event.clear()
    loop._processing_queue.put((command, loop._last_wake_info.copy(), loop._recording_generation))
    loop._process_worker()
    assert outputs[-1] == "I hate Ciel."


def test_trimmed_bare_wake_uses_known_greeting_when_stt_finds_no_command():
    detector = SimpleNamespace(reset=lambda **_kwargs: None)
    stt = SimpleNamespace(transcribe=lambda *_args, **_kwargs: "")
    outputs = []
    loop = WakeListenerLoop(MockAudioBackend(), detector=detector, stt=stt)

    def finish(text, *_args):
        outputs.append(text)
        loop._stop_event.set()
        return False

    loop.on_transcription = finish
    loop._running = True
    loop._processing_queue.put(
        (np.zeros(1280), {"wake_prefix_trimmed": True, "wake_phrase": "hey ciel"}, 0)
    )
    loop._process_worker()
    assert outputs == ["hey ciel"]


def test_mention_of_ciel_is_not_a_greeting():
    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector._spotter_result = (detector._generation, "I hate Ciel.", None)
    assert detector.process_frame(np.zeros(1280, dtype=np.float32)) is None


@pytest.mark.parametrize("greeting", ["Hey, Ciel.", "Hey, Siel!", "Hi, Ciel?"])
def test_whisper_punctuation_does_not_prevent_wake(greeting):
    from ciel.conversation import strip_wake_phrase

    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector._spotter_result = (detector._generation, greeting, None)
    assert detector.process_frame(np.zeros(1280, dtype=np.float32))["model"] == "whisper_keyword"
    assert strip_wake_phrase(greeting + " What's the time?") == "What's the time?"


def test_uncertain_trimmed_command_is_not_replaced_by_wake_acknowledgement():
    detector = SimpleNamespace(reset=lambda **_kwargs: None)
    stt = SimpleNamespace(
        transcribe_detailed=lambda *_args, **_kwargs: {
            "text": "",
            "needs_repeat": True,
            "confidence": 0.2,
        }
    )
    outputs = []
    loop = WakeListenerLoop(MockAudioBackend(), detector=detector, stt=stt)

    def finish(text, info, _audio):
        outputs.append((text, info))
        loop._stop_event.set()
        return False

    loop.on_transcription = finish
    loop._running = True
    loop._processing_queue.put(
        (np.ones(1280), {"wake_prefix_trimmed": True, "wake_phrase": "hey ciel"}, 0)
    )
    loop._process_worker()
    assert outputs[0][0] == ""
    assert outputs[0][1]["stt_needs_repeat"] is True


def test_bare_wake_does_not_decode_transient_greeting_tail():
    detector = SimpleNamespace(reset=lambda **_kwargs: None)

    def must_not_decode(*_args, **_kwargs):
        pytest.fail("The known greeting tail must not be decoded as a command")

    stt = SimpleNamespace(transcribe=must_not_decode)
    outputs = []
    loop = WakeListenerLoop(MockAudioBackend(), detector=detector, stt=stt)

    def finish(text, *_args):
        outputs.append(text)
        loop._stop_event.set()
        return False

    loop.on_transcription = finish
    loop._running = True
    loop._processing_queue.put(
        (
            np.ones(1280),
            {
                "wake_prefix_trimmed": True,
                "post_wake_speech": False,
                "wake_phrase": "hey ciel",
            },
            0,
        )
    )
    loop._process_worker()
    assert outputs == ["hey ciel"]


def test_wake_spotter_model_can_be_selected(monkeypatch):
    from unittest.mock import MagicMock

    factory = MagicMock()
    monkeypatch.setattr("faster_whisper.WhisperModel", factory)
    detector = WakeWordDetector(enable_whisper_spotter=False, spotter_model_size="small.en")
    detector._load_spotter_model()
    factory.assert_called_once_with("small.en", device="cpu", compute_type="int8", cpu_threads=2)


def test_quiet_greeting_is_submitted_without_raising_its_own_noise_floor(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.wake.time.time", lambda: clock[0])
    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector.enable_whisper_spotter = True
    detector._spotter_model = object()  # Inspect queued inference without model downloads.
    for _ in range(10):
        clock[0] += 0.08
        detector.process_frame(np.full(1280, 0.001, dtype=np.float32))
    assert detector._spotter_requests.empty()
    background = detector._ambient_rms
    for _ in range(16):
        clock[0] += 0.08
        detector.process_frame(np.full(1280, 0.008, dtype=np.float32))
    assert not detector._spotter_requests.empty()
    assert detector._ambient_rms == background


def test_finished_slow_greeting_gets_complete_snapshot_during_check_interval(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("ciel.audio.wake.time.time", lambda: clock[0])
    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector.enable_whisper_spotter = True
    detector._spotter_model = object()
    # A 1.92-second greeting exceeds the old 1.5-second buffer.
    for index in range(24):
        clock[0] += 0.08
        amplitude = 0.04 if index == 0 else 0.02
        detector.process_frame(np.full(1280, amplitude, dtype=np.float32))
    generation, audio, _end = detector._spotter_requests.get_nowait()
    detector._spotter_requests.task_done()
    assert generation == detector._generation
    assert audio[0] == pytest.approx(0.04)
    # Force the normal periodic check to be too recent; trailing silence must still
    # submit the full phrase, rather than waiting for the user to repeat it.
    detector._last_spotter_check = clock[0]
    for _ in range(2):
        clock[0] += 0.08
        detector.process_frame(np.zeros(1280, dtype=np.float32))
    _generation, complete, end = detector._spotter_requests.get_nowait()
    detector._spotter_requests.task_done()
    assert complete.size == 26 * 1280
    assert complete[0] == pytest.approx(0.04)
    assert not np.any(complete[-2560:])
    assert end == detector._samples_seen
    detector.reset()
    assert not detector._speech_pending
    detector.process_frame(np.zeros(1280, dtype=np.float32))
    assert detector._spotter_requests.empty()


def test_background_hum_below_gate_does_not_request_keyword_inference(monkeypatch):
    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector.enable_whisper_spotter = True
    detector._spotter_model = object()
    samples = np.arange(1280) / 16000
    hum = (0.003 * np.sin(2 * np.pi * 120 * samples)).astype(np.float32)
    for _ in range(100):
        assert detector.process_frame(hum) is None
    assert detector._spotter_requests.empty()


@pytest.mark.parametrize("options", [{"min_rms": 0}, {"window_seconds": 0.5}])
def test_invalid_wake_sensitivity_is_rejected(options):
    with pytest.raises(ValueError):
        WakeWordDetector(enable_whisper_spotter=False, **options)


def test_recorder_keeps_speech_across_a_thinking_pause():
    recorder = VoiceRecorder(silence_duration_seconds=1, pause_grace_seconds=0.8)
    recorder.start()
    speech = np.full(1280, 0.1, dtype=np.float32)
    silence = np.zeros(1280, dtype=np.float32)
    for _ in range(6):
        assert recorder.add_frame(speech)
    for _ in range(15):  # 1.2-second thinking pause used to end this turn.
        assert recorder.add_frame(silence)
    for _ in range(6):
        assert recorder.add_frame(speech)
    for _ in range(22):
        assert recorder.add_frame(silence)
    assert not recorder.add_frame(silence)
    assert recorder.get_audio().size == 50 * 1280


def test_recording_uses_speech_detection_instead_of_volume_alone():
    # Quiet speech is below the former RMS gate; louder steady noise is above it.
    recorder = VoiceRecorder(
        silence_duration_seconds=0.2,
        speech_detector=lambda frame: bool(np.allclose(frame, 0.002)),
    )
    recorder.start()
    quiet_speech = np.full(1280, 0.002, dtype=np.float32)
    louder_noise = np.full(1280, 0.02, dtype=np.float32)
    for _ in range(6):
        assert recorder.add_frame(quiet_speech)
    assert recorder._speech_started
    assert recorder.add_frame(louder_noise)
    assert recorder.add_frame(louder_noise)
    assert not recorder.add_frame(louder_noise)


def test_ambient_capture_starts_on_speech_without_a_wake_phrase():
    detector = WakeWordDetector(enable_whisper_spotter=False)
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, ambient=True,
        speech_detector=lambda audio: bool(np.any(audio)),
    )
    loop._state = ListenerState.LISTENING_WAKE
    loop._handle_frame(np.zeros(1280, dtype=np.float32))
    assert loop.state == ListenerState.LISTENING_WAKE
    speech = np.full(1280, 0.01, dtype=np.float32)
    loop._handle_frame(speech)
    loop._handle_frame(speech)
    assert loop.state == ListenerState.RECORDING
    assert loop._last_wake_info["ambient"]
    assert loop.recorder.get_audio().size == 3 * 1280


def test_delayed_vad_keeps_the_beginning_of_a_greeting():
    calls = [0]

    def delayed_vad(_audio):
        calls[0] += 1
        return calls[0] >= 6

    loop = WakeListenerLoop(
        MockAudioBackend(), detector=WakeWordDetector(enable_whisper_spotter=False),
        ambient=True, speech_detector=delayed_vad,
    )
    loop._state = ListenerState.LISTENING_WAKE
    greeting = np.full(1280, 0.05, dtype=np.float32)
    for _ in range(7):
        loop._handle_frame(greeting)
    assert loop.state == ListenerState.RECORDING
    # The former 240ms pre-roll lost the first four frames, potentially losing 'Hey'.
    np.testing.assert_array_equal(loop.recorder.get_audio(), np.tile(greeting, 7))


def test_stt_finishing_after_new_speech_keeps_context_but_never_replies():
    observations = []
    replies = []
    detector = WakeWordDetector(enable_whisper_spotter=False)
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector,
        on_transcript_observed=lambda text, info: observations.append((text, info.copy())),
        on_transcription=lambda *args: replies.append(args),
    )

    def transcribe(*_args, **_kwargs):
        loop._start_recording()  # A newer utterance arrived while Whisper was decoding.
        loop._stop_event.set()
        return {"text": "Hey Ciel.", "confidence": 0.8}

    loop.stt = SimpleNamespace(transcribe_detailed=transcribe)
    loop._running = True
    loop._start_recording()
    loop._processing_queue.put((np.ones(16000), {}, loop._recording_generation))
    loop._process_worker()
    assert not replies
    assert observations[0][0] == "Hey Ciel."
    assert observations[0][1]["superseded"]


def test_resumed_speech_cancels_pending_response_and_retains_onset():
    detector = WakeWordDetector(enable_whisper_spotter=False)
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, monitor_resumed_speech=True,
        speech_detector=lambda audio: bool(np.any(audio)),
    )
    loop._start_recording()
    old_cancel = loop._response_cancel
    loop._state = ListenerState.PROCESSING
    speech = np.full(1280, 0.01, dtype=np.float32)
    loop._handle_frame(speech)
    loop._handle_frame(speech)
    assert old_cancel.is_set()
    assert loop.state == ListenerState.RECORDING
    assert loop.recorder.get_audio().size == 2 * 1280


def test_ambient_ignores_speaker_playback_and_preserves_explicit_barge_in():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    tts = MagicMock()
    tts.is_speaking.return_value = True
    activity = MagicMock(return_value=True)
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, tts=tts, ambient=True,
        speech_detector=activity, monitor_resumed_speech=True,
    )
    loop._state = ListenerState.PROCESSING
    speech = np.full(1280, 0.1, dtype=np.float32)
    loop._handle_frame(speech)
    activity.assert_not_called()
    tts.stop.assert_not_called()
    detector.process_frame.return_value = {
        "model": "whisper_keyword", "wake_phrase": "hey ciel", "wake_tail_samples": 1280,
    }
    loop._handle_frame(speech)
    tts.stop.assert_called_once()
    assert loop.state == ListenerState.RECORDING
    assert loop._last_wake_info["wake_prefix_trimmed"]
    assert loop.recorder.get_audio().size == 1280


@pytest.mark.parametrize("text", ["What's up Ciel?", "So what's good Ciel?"])
def test_trailing_address_wakes_without_trimming_the_question(monkeypatch, text):
    words = [
        SimpleNamespace(word=" What's", end=0.2), SimpleNamespace(word=" up", end=0.4),
        SimpleNamespace(word=" Ciel?", end=0.9),
    ]
    spotter = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: (
            [SimpleNamespace(text=text, words=words)], None
        )
    )
    monkeypatch.setattr(WakeWordDetector, "_load_spotter_model", lambda _self: spotter)
    detector = WakeWordDetector()
    try:
        wait_for(lambda: detector._spotter_model is not None)
        detector._spotter_requests.put((detector._generation, np.ones(16000), 16000))
        wait_for(lambda: detector._spotter_result is not None)
        trigger = detector.process_frame(np.zeros(1280, dtype=np.float32))
        assert trigger["text"] == text
        assert "wake_tail_samples" not in trigger
    finally:
        detector.stop()


def test_leading_filler_before_name_is_recognized_by_keyword_detector():
    detector = WakeWordDetector(enable_whisper_spotter=False)
    detector._spotter_result = (detector._generation, "So Ciel, what's on your mind?", None)
    assert detector.poll()["text"] == "So Ciel, what's on your mind?"


def test_ambient_keyword_confirmation_does_not_restart_or_cut_off_recording():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, ambient=True,
        speech_detector=lambda audio: bool(np.any(audio)),
        recorder=VoiceRecorder(
            silence_duration_seconds=0.1, pause_grace_seconds=0,
            min_speech_duration_seconds=0.12,
        ),
    )
    loop._state = ListenerState.LISTENING_WAKE
    speech = np.full(1280, 0.1, dtype=np.float32)
    loop._handle_frame(speech)
    loop._handle_frame(speech)
    assert detector.process_frame.call_count == 2
    cancel_event = loop._response_cancel
    generation = loop._recording_generation
    detector.process_frame.return_value = {"text": "What's up Ciel?"}
    loop._handle_frame(speech)
    assert loop._last_wake_info["wake_verified"]
    assert loop._recording_generation == generation
    assert not cancel_event.is_set()
    np.testing.assert_array_equal(loop.recorder.get_audio(), np.tile(speech, 3))
    for _ in range(2):
        loop._handle_frame(np.zeros(1280, dtype=np.float32))
    _audio, info, _gen = loop._processing_queue.get_nowait()
    assert info["wake_verified"]


def test_late_ambient_wake_evidence_stays_with_queued_utterance():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, ambient=True, monitor_resumed_speech=True,
        speech_detector=lambda audio: bool(np.any(audio)),
        recorder=VoiceRecorder(
            silence_duration_seconds=0.1, pause_grace_seconds=0,
            min_speech_duration_seconds=0.12,
        ),
    )
    loop._state = ListenerState.LISTENING_WAKE
    speech = np.full(1280, 0.1, dtype=np.float32)
    silence = np.zeros(1280, dtype=np.float32)
    for frame in [speech, speech, speech, silence, silence]:
        loop._handle_frame(frame)
    _audio, old_info, _gen = loop._processing_queue.get_nowait()
    assert not old_info.get("wake_verified")
    detector.process_frame.return_value = {"wake_phrase": "hey ciel"}
    loop._handle_frame(silence)
    assert old_info["wake_verified"]
    detector.process_frame.return_value = None
    loop._handle_frame(speech)
    loop._handle_frame(speech)
    detector.reset.assert_called_once_with(set_cooldown=False)
    assert not loop._last_wake_info.get("wake_verified")
    assert old_info["cancel_event"].is_set()


def test_ambient_wake_starts_capture_even_if_vad_misses_onset():
    detector = SimpleNamespace(
        process_frame=lambda _frame: {"text": "What's up Ciel?"},
    )
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, ambient=True, speech_detector=lambda _: False,
    )
    loop._state = ListenerState.LISTENING_WAKE
    speech = np.full(1280, 0.1, dtype=np.float32)
    loop._handle_frame(speech)
    assert loop.state == ListenerState.RECORDING
    assert loop._last_wake_info["wake_verified"]
    assert not loop._last_wake_info.get("wake_prefix_trimmed")
    np.testing.assert_array_equal(loop.recorder.get_audio(), speech)


def test_keyword_completed_during_stt_authorizes_only_that_reply():
    detector = WakeWordDetector(enable_whisper_spotter=False)
    observations = []
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, ambient=True,
        speech_detector=lambda _: False,
    )
    loop._running = True
    loop._start_speech_recording()
    info = loop._last_wake_info

    def transcribe(*_args, **_kwargs):
        detector._spotter_result = (detector._generation, "What's up Ciel?", None)
        return {"text": "What's up?", "confidence": 0.7}

    def respond(text, wake_info, _audio):
        observations.append((text, wake_info.copy()))
        assert detector.poll() is None  # Old evidence cannot interrupt this reply.
        loop._stop_event.set()

    loop.stt = SimpleNamespace(transcribe_detailed=transcribe)
    loop.on_transcription = respond
    loop._processing_queue.put((np.ones(16000), info, loop._recording_generation))
    loop._process_worker()
    assert observations[0][0] == "What's up?"  # Keep the actual command transcript.
    assert observations[0][1]["wake_verified"]


@pytest.mark.parametrize("show_transcripts", [False, True])
def test_raw_ambient_transcript_logging_requires_opt_in(caplog, show_transcripts):
    detector = WakeWordDetector(enable_whisper_spotter=False)
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, show_transcripts=show_transcripts,
        stt=SimpleNamespace(
            transcribe_detailed=lambda *_args, **_kwargs: {
                "text": "background conversation", "confidence": 0.7,
            }
        ),
    )
    loop._running = True
    loop.on_transcription = lambda *_args: loop._stop_event.set()
    loop._processing_queue.put((np.ones(16000), {}, 0))
    with caplog.at_level("INFO", logger="ciel.audio.listener"):
        loop._process_worker()
    exposed = "STT candidate (diagnostic): 'background conversation'" in caplog.text
    assert exposed is show_transcripts


def test_headphone_speech_interrupts_playback_and_preserves_onset():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    speaking = [True]
    tts = MagicMock()
    tts.is_speaking.side_effect = lambda: speaking[0]

    def stop():
        if speaking[0]:
            speaking[0] = False
            return {'remaining_text': 'Continue with the next step.'}
        return None

    tts.stop.side_effect = stop
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, tts=tts, ambient=True,
        speech_detector=lambda frame: bool(np.any(frame)), barge_in_mode='speech',
    )
    loop._running = True
    loop._start_recording()
    cancel = loop._response_cancel
    loop._state = ListenerState.PROCESSING
    speech = np.full(1280, 0.01, dtype=np.float32)
    loop._handle_frame(speech)
    loop._handle_frame(speech)
    tts.stop.assert_not_called()  # Ignore shorter bursts; require sustained voice.
    loop._handle_frame(speech)
    assert not speaking[0]
    assert cancel.is_set()
    assert loop.state == ListenerState.RECORDING
    assert loop._last_wake_info['during_reply']
    assert loop._last_wake_info['supersedes_cancel_event'] is cancel
    assert not loop._last_wake_info.get('wake_verified')  # Still judge whom speech addresses.
    np.testing.assert_array_equal(loop.recorder.get_audio(), np.tile(speech, 3))


def test_headphone_barge_in_ignores_non_speech_noise():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    tts = MagicMock()
    tts.is_speaking.return_value = True
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, tts=tts, ambient=True,
        speech_detector=lambda _: False, barge_in_mode='speech',
    )
    loop._state = ListenerState.PROCESSING
    for _ in range(10):
        loop._handle_frame(np.full(1280, 0.1, dtype=np.float32))
    tts.stop.assert_not_called()
    assert loop.state == ListenerState.PROCESSING


def test_speech_during_synthesis_stops_pending_output():
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.process_frame.return_value = None
    tts = MagicMock()
    tts.is_speaking.return_value = False  # Audio synthesis has not reached playback yet.
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=detector, tts=tts, ambient=True,
        monitor_resumed_speech=True, speech_detector=lambda _: True,
    )
    loop._start_recording()
    cancel = loop._response_cancel
    loop._state = ListenerState.PROCESSING
    for _ in range(2):
        loop._handle_frame(np.full(1280, 0.1, dtype=np.float32))
    assert cancel.is_set()
    tts.stop.assert_called_once()
    assert loop.state == ListenerState.RECORDING


def test_superseded_queued_speech_is_observed_without_being_executed():
    observations, replies = [], []
    loop = WakeListenerLoop(
        MockAudioBackend(), detector=WakeWordDetector(enable_whisper_spotter=False),
        on_transcription=lambda *args: replies.append(args),
        on_transcript_observed=lambda text, info: observations.append((text, info)),
    )
    loop._running = True
    loop._start_recording()

    def transcribe(*_args, **_kwargs):
        loop._stop_event.set()
        return {'text': 'Ciel, explain ambient mode.', 'confidence': 0.8}

    loop.stt = SimpleNamespace(transcribe_detailed=transcribe)
    loop._processing_queue.put((np.ones(16000), {}, 0))
    loop._process_worker()
    assert observations[0][0] == 'Ciel, explain ambient mode.'
    assert observations[0][1]['superseded']
    assert not replies
