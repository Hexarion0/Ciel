"""Bounded audio ingestion, wake detection, recording, and conversation workers."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ciel.audio.recorder import VoiceRecorder
from ciel.audio.wake import WakeWordDetector
from ciel.latency import TurnTrace, active_trace, mark
from ciel.logging import get_logger

if TYPE_CHECKING:
    from ciel.audio.stt import SpeechToText
    from ciel.audio.tts import TextToSpeech
    from ciel.platform.base import AudioBackend

logger = get_logger("audio.listener")


class ListenerState(str, Enum):
    IDLE = "idle"
    LISTENING_WAKE = "listening_wake"
    WAKE_DETECTED = "wake_detected"
    RECORDING = "recording"
    PROCESSING = "processing"


class WakeListenerLoop:
    """Keep microphone callbacks independent of inference and user callbacks."""

    def __init__(
        self,
        audio_backend: AudioBackend,
        detector: WakeWordDetector | None = None,
        recorder: VoiceRecorder | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        on_wake: Callable[[dict[str, Any]], None] | None = None,
        on_utterance: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
        on_transcription: Callable[[str, dict[str, Any], np.ndarray], bool | None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
        on_state_change: Callable[[ListenerState], None] | None = None,
        sample_rate: int = 16000,
        device: int | str | None = None,
        barge_in: bool = True,
        barge_in_threshold_rms: float = 0.030,
        stt_beam_size: int = 3,
        ambient: bool = False,
        speech_detector: Callable[[np.ndarray], bool] | None = None,
        monitor_resumed_speech: bool = False,
        on_transcript_observed: Callable[[str, dict[str, Any]], None] | None = None,
        show_transcripts: bool = False,
        barge_in_mode: str = "wake",
        barge_in_speech_seconds: float = 0.24,
        on_interruption: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.backend = audio_backend
        self.detector = detector or WakeWordDetector()
        self.recorder = recorder or VoiceRecorder(sample_rate=sample_rate)
        self.stt, self.tts = stt, tts
        self.on_wake, self.on_utterance = on_wake, on_utterance
        self.on_transcription, self.on_barge_in = on_transcription, on_barge_in
        self.on_state_change = on_state_change
        self.sample_rate, self.device = sample_rate, device
        self.barge_in, self.barge_in_threshold_rms = barge_in, barge_in_threshold_rms
        self.stt_beam_size = stt_beam_size
        self.ambient = ambient
        self._speech_detector = speech_detector
        self.monitor_resumed_speech = monitor_resumed_speech
        self.on_transcript_observed = on_transcript_observed
        self.show_transcripts = show_transcripts
        if barge_in_mode not in {"speech", "wake"}:
            raise ValueError("Barge-in mode must be 'speech' or 'wake'")
        self.barge_in_mode = barge_in_mode
        self.barge_in_speech_seconds = barge_in_speech_seconds
        self.on_interruption = on_interruption
        self._interrupt_speech_samples = 0
        self._speech_streak = 0
        self._response_cancel: threading.Event | None = None
        self._background_cancel: threading.Event | None = None
        self._state = ListenerState.IDLE
        self._last_wake_info: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._text_only = False
        self._microphone_muted = False
        self._microphone_epoch = 0
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self._processing_queue: queue.Queue = queue.Queue(maxsize=4)
        self._notifications: queue.Queue = queue.Queue(maxsize=32)
        self._threads: list[threading.Thread] = []
        self._recent_audio: deque = deque(maxlen=max(1, round(sample_rate * 3.0 / 1280)))
        self._recording_generation = 0
        self.dropped_frames = 0
        self._frame_received_at: float | None = None
        self._last_speech_received_at: float | None = None

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def microphone_muted(self) -> bool:
        """Return whether microphone ingestion is muted."""
        with self._lock:
            return self._microphone_muted

    @staticmethod
    def _offer(target: queue.Queue, item: Any) -> bool:
        """Never block producers; replace the oldest item if the consumer falls behind."""
        try:
            target.put_nowait(item)
            return False
        except queue.Full:
            try:
                target.get_nowait()
                target.task_done()
            except queue.Empty:
                pass
            try:
                target.put_nowait(item)
            except queue.Full:
                pass
            return True

    def _notify(self, callback: Callable | None, *args: Any) -> None:
        if callback:
            self._offer(self._notifications, (callback, args))

    def _set_state(self, state: ListenerState) -> None:
        if self._state != state:
            self._state = state
            self._notify(self.on_state_change, state)

    def _standby_state(self) -> ListenerState:
        """Keep a running text session distinct from active microphone capture."""
        return ListenerState.IDLE if self._text_only else ListenerState.LISTENING_WAKE

    def _notification_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                callback, args = self._notifications.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._stop_event.is_set():
                    callback(*args)
            except Exception:
                logger.exception("Listener notification failed")
            finally:
                self._notifications.task_done()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Copy borrowed audio and enqueue it; never run inference or take the state lock."""
        microphone_epoch = self._microphone_epoch
        if self._running and not self._microphone_muted:
            if self._offer(self._frame_queue, (
                indata.copy(), time.monotonic(), microphone_epoch,
            )):
                self.dropped_frames += 1

    def cancel_response(self) -> None:
        """Cancel pending capture, generation, and playback without stopping listening."""
        with self._lock:
            self._cancel_response_locked()

    def _cancel_response_locked(self) -> None:
        if self._response_cancel is not None:
            self._response_cancel.set()
        self._recording_generation += 1
        self._microphone_epoch += 1
        self.recorder.cancel()
        self._last_wake_info = {}
        self._recent_audio.clear()
        self._speech_streak = 0
        self.detector.reset(set_cooldown=False)
        self._set_state(self._standby_state())
        self._stop_output()

    def toggle_microphone_mute(self) -> bool:
        """Toggle microphone ingestion; discard pending voice input when muting."""
        with self._lock:
            return self._set_microphone_mute_locked(not self._microphone_muted)

    def set_microphone_mute(self, muted: bool) -> bool:
        """Set microphone ingestion without disturbing work when already set."""
        with self._lock:
            return self._set_microphone_mute_locked(muted)

    def _set_microphone_mute_locked(self, muted: bool) -> bool:
        if self._text_only:
            self._microphone_muted = True
            return True
        if self._microphone_muted == muted:
            return self._microphone_muted
        self._microphone_muted = muted
        self._microphone_epoch += 1
        if muted and self._response_cancel is self._background_cancel:
            if self._background_cancel is not None:
                self._background_cancel.set()
                self._stop_output()
        while True:
            try:
                self._frame_queue.get_nowait()
                self._frame_queue.task_done()
            except queue.Empty:
                break
        self._recent_audio.clear()
        self._speech_streak = 0
        self.detector.reset(set_cooldown=False)
        if self._state == ListenerState.RECORDING:
            self.recorder.cancel()
            self._recording_generation += 1
            self._set_state(self._standby_state())
        return self._microphone_muted

    def _discard_voice_request(self, generation: int) -> None:
        """Leave standby usable after a muted request is discarded during STT."""
        with self._lock:
            if (
                self._running and generation == self._recording_generation
                and self._state == ListenerState.PROCESSING
            ):
                self._set_state(self._standby_state())

    def submit_text(self, text: str) -> bool:
        """Replace the current request with directly addressed keyboard input."""
        text = text.strip()
        if not text:
            return False
        with self._lock:
            if not self._running:
                return False
            self._cancel_response_locked()
            self._response_cancel = threading.Event()
            info = {"input_source": "keyboard", "typed_text": text,
                    "cancel_event": self._response_cancel}
            self._offer(self._processing_queue, (
                np.empty(0, dtype=np.float32), info, self._recording_generation,
            ))
            self._set_state(ListenerState.PROCESSING)
            return True

    def submit_request(self, request, audio: np.ndarray | None = None, *,
                       current=lambda: True) -> bool:
        """Serialize authenticated browser inputs without treating voice as keyboard text."""
        if not current():
            return False
        with self._lock:
            if not self._running:
                return False
            self._cancel_response_locked()
            self._response_cancel = threading.Event()
            info = {'input_source': 'web_text' if request.input_kind == 'text' else 'web_voice',
                    'typed_text': request.text if request.input_kind == 'text' else '',
                    'web_request': request, 'web_current': current, 'wake_verified': True,
                    'cancel_event': self._response_cancel}
            self._offer(self._processing_queue, (
                audio if audio is not None else np.empty(0, dtype=np.float32),
                info, self._recording_generation,
            ))
            self._set_state(ListenerState.PROCESSING)
            return True

    def _background_ready_locked(self) -> bool:
        """Background output never replaces an owner request or muted voice session."""
        return bool(
            self._running and (self._text_only or not self._microphone_muted)
            and self._state == self._standby_state() and self._processing_queue.empty()
        )

    def background_ready(self) -> bool:
        """Expose an atomic idle check without granting permission to capture."""
        with self._lock:
            return self._background_ready_locked()

    def submit_background(self, callback: Callable[[threading.Event], None]) -> bool:
        """Offer one idle suggestion to the output worker without fabricating user input."""
        with self._lock:
            if not self._background_ready_locked():
                return False
            self._recording_generation += 1
            self._response_cancel = threading.Event()
            info = {"input_source": "initiative", "background_callback": callback,
                    "cancel_event": self._response_cancel}
            self._background_cancel = self._response_cancel
            self._processing_queue.put_nowait((
                np.empty(0, dtype=np.float32), info, self._recording_generation,
            ))
            self._set_state(ListenerState.PROCESSING)
            return True

    def _start_recording(self) -> None:
        if self._response_cancel is not None:
            self._last_wake_info["supersedes_cancel_event"] = self._response_cancel
            self._response_cancel.set()
            # Also invalidate TTS still synthesizing: is_speaking() is false then.
            self._stop_output()
        self._response_cancel = threading.Event()
        self._recording_generation += 1
        self.recorder.start()
        self._last_speech_received_at = None
        self._set_state(ListenerState.RECORDING)

    def _stop_output(self) -> None:
        if self.tts:
            info = self.tts.stop()
            if isinstance(info, dict):
                self._notify(self.on_interruption, info)

    def set_ambient(self, enabled: bool) -> None:
        """Switch capture mode; background speech is never treated as a wake trigger."""
        if enabled and not self._text_only and self._speech_detector is None:
            if self.sample_rate != 16000:
                raise ValueError("Ambient speech detection requires 16000 Hz audio")
            from ciel.audio.activity import SpeechActivity

            self._speech_detector = SpeechActivity()
        with self._lock:
            self.ambient = enabled
            if self._speech_detector is not None:
                self.recorder.speech_detector = self._speech_detector
            self._speech_streak = 0
            self._recent_audio.clear()

    def _start_speech_recording(self) -> None:
        """Retain a short onset pre-roll when speech starts without a wake phrase."""
        # Neural VAD may recognize onset after the first word has already begun.
        # Keep 800 ms rather than just the two positive frames and their predecessor.
        preroll_frames = max(3, round(self.sample_rate * 0.8 / 1280))
        recent = list(self._recent_audio)[-preroll_frames:]
        resumed = self._state == ListenerState.PROCESSING
        if self.ambient and resumed:
            # A late keyword result from the previous utterance cannot authorize
            # this new one. Re-seed the detector with this recording's onset only.
            self.detector.reset(set_cooldown=False)
        speaking = bool(self.tts and self.tts.is_speaking())
        during_reply = speaking or bool(
            resumed and self._response_cancel is not None and not self._response_cancel.is_set()
        )
        if speaking:
            self._stop_output()
        self._last_wake_info = {
            "ambient": self.ambient,
            "resumed_speech": True,
            "speech_started_at": time.monotonic(),
            "during_reply": during_reply,
            "response_pending": resumed,
        }
        self._start_recording()
        for frame in recent:
            self.recorder.add_frame(frame)
            if self.ambient and resumed:
                self._check_ambient_wake(frame)
        self._recent_audio.clear()
        self._speech_streak = 0

    def _check_ambient_wake(self, audio: np.ndarray) -> dict[str, Any] | None:
        """Bind independent wake evidence to the current utterance without restarting it."""
        if self._last_wake_info.get("wake_verified"):
            return None
        trigger = self.detector.process_frame(audio)
        if trigger:
            self._confirm_ambient_wake(trigger)
        return trigger

    def _confirm_ambient_wake(self, trigger: dict[str, Any]) -> None:
        self._last_wake_info.update(
            wake_verified=True,
            wake_phrase=trigger.get(
                "wake_phrase", getattr(self.detector, "wake_phrase", "hey ciel")
            ),
        )
        self._notify(self.on_wake, trigger)

    def _seed_wake_audio(self, trigger: dict[str, Any]) -> None:
        """Keep post-greeting audio using the keyword spotter's original word timing."""
        recent = np.concatenate(list(self._recent_audio)) if self._recent_audio else np.empty(0)
        tail = trigger.get("wake_tail_samples")
        if tail is not None:
            tail = min(recent.shape[0], max(0, int(tail)))
            recent = recent[-tail:] if tail else recent[:0]
            self._last_wake_info["wake_prefix_trimmed"] = True
        for start in range(0, recent.shape[0], 1280):
            if not self.recorder.add_frame(recent[start : start + 1280]):
                break
        self._recent_audio.clear()

    def _frame_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, received_at, microphone_epoch = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if (
                        not self._running or self._microphone_muted
                        or microphone_epoch != self._microphone_epoch
                    ):
                        continue
                    self._frame_received_at = received_at
                    self._handle_frame(audio)
            except Exception:
                logger.exception("Audio frame processing failed")
                with self._lock:
                    if self._running:
                        self.detector.reset(set_cooldown=False)
                        self._set_state(self._standby_state())
            finally:
                self._frame_queue.task_done()

    def _handle_frame(self, audio: np.ndarray) -> None:
        self._recent_audio.append(audio)
        if (
            self.barge_in
            and self.tts
            and self.tts.is_speaking()
            and self._state != ListenerState.RECORDING
        ):
            trigger = self.detector.process_frame(audio)
            speech_interrupt = False
            if self.barge_in_mode == "speech" and self._speech_detector is not None:
                speech = self._speech_detector(audio)
                self._interrupt_speech_samples = (
                    self._interrupt_speech_samples + audio.size if speech else 0
                )
                speech_interrupt = (
                    self._interrupt_speech_samples >=
                    round(self.sample_rate * self.barge_in_speech_seconds)
                )
            if speech_interrupt and not trigger:
                self._start_speech_recording()
                self._notify(self.on_barge_in)
                self._interrupt_speech_samples = 0
                return
            if trigger:
                self._stop_output()
                self._last_wake_info = dict(trigger) if trigger else {}
                self._last_wake_info["ambient"] = self.ambient
                self._last_wake_info["during_reply"] = True
                self._last_wake_info["speech_started_at"] = time.monotonic()
                if trigger and self.ambient:
                    self._last_wake_info["wake_verified"] = True
                self._start_recording()
                if trigger:
                    self._seed_wake_audio(trigger)
                else:
                    self.recorder.add_frame(audio)
                self._notify(self.on_barge_in)
                return
            # With speakers select wake mode so loudspeaker speech isn't treated
            # as a user interruption. Speech mode is intended for headphones.
            return
        self._interrupt_speech_samples = 0
        if self.ambient and self._state == ListenerState.LISTENING_WAKE:
            trigger = self.detector.process_frame(audio)
            if trigger:
                self._last_wake_info = dict(trigger, ambient=True, wake_verified=True)
                self._start_recording()
                self._seed_wake_audio(trigger)
                self._notify(self.on_wake, trigger)
                return
        capture_speech = (
            self.ambient and self._state == ListenerState.LISTENING_WAKE
        ) or (self.monitor_resumed_speech and self._state == ListenerState.PROCESSING)
        if capture_speech and self._speech_detector is not None:
            speech = self._speech_detector(audio)
            self._speech_streak = self._speech_streak + 1 if speech else 0
            if self._speech_streak >= 2:
                self._start_speech_recording()
                return
        if self._state == ListenerState.LISTENING_WAKE:
            if self.ambient:
                return
            trigger = self.detector.process_frame(audio)
            if trigger:
                self._last_wake_info = trigger
                self._set_state(ListenerState.WAKE_DETECTED)
                self._start_recording()
                # Async wake inference can finish after the user starts the command.
                self._seed_wake_audio(trigger)
                self._notify(self.on_wake, trigger)
        elif self._state == ListenerState.RECORDING:
            if self.ambient:
                self._check_ambient_wake(audio)
            if not self.recorder._speech_started and not self.ambient:
                trigger = self.detector.process_frame(audio)
                if trigger:
                    self._last_wake_info = trigger
                    self._start_recording()
                    self._seed_wake_audio(trigger)
                    self._notify(self.on_wake, trigger)
                    return
            previous_speech = self.recorder._last_speech_time
            recording = self.recorder.add_frame(audio)
            if self.recorder._last_speech_time != previous_speech:
                self._last_speech_received_at = self._frame_received_at or time.monotonic()
            if not recording:
                recorded = self.recorder.get_audio()
                info = self._last_wake_info.copy()
                info["microphone_epoch"] = self._microphone_epoch
                trace = TurnTrace()
                trace.mark("capture_started", at=self.recorder.capture_started_at)
                if self.recorder.last_speech_observed_at is not None:
                    trace.mark("last_vad_speech", at=self.recorder.last_speech_observed_at)
                if self._last_speech_received_at is not None:
                    trace.mark("last_speech_frame_received", at=self._last_speech_received_at)
                if self._frame_received_at is not None:
                    trace.mark("endpoint_frame_received", at=self._frame_received_at)
                trace.mark("endpoint_detected")
                info["latency_trace"] = trace
                info["cancel_event"] = self._response_cancel
                if info.get("wake_prefix_trimmed"):
                    info["post_wake_speech"] = self.recorder._speech_started
                # The greeting belongs only to this utterance, never to a follow-up.
                self._last_wake_info.pop("wake_prefix_trimmed", None)
                if self.ambient:
                    # The keyword worker can finish during STT. Keep its evidence
                    # attached to this queued utterance, never a subsequent one.
                    self._last_wake_info = info
                self._offer(self._processing_queue, (recorded, info, self._recording_generation))
                self._set_state(ListenerState.PROCESSING)
                self._speech_streak = 0
                reset_activity = getattr(self._speech_detector, "reset", None)
                if reset_activity:
                    reset_activity()
                self._notify(self.on_utterance, recorded, info)
        elif self.ambient and self._state == ListenerState.PROCESSING:
            self._check_ambient_wake(audio)

    def _process_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, info, generation = self._processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            trace = info.get("latency_trace")
            trace_token = active_trace.set(trace)
            try:
                mark("processing_dequeued")
                if not self._running:
                    continue
                if info.get("input_source") == "initiative":
                    event = info["cancel_event"]
                    if (generation == self._recording_generation and not event.is_set()
                            and (self._text_only or not self._microphone_muted)):
                        info["background_callback"](event)
                    with self._lock:
                        if self._background_cancel is event:
                            self._background_cancel = None
                        if self._running and generation == self._recording_generation:
                            self._set_state(self._standby_state())
                    continue
                if generation != self._recording_generation:
                    logger.info("Decoding superseded speech for temporary conversation context.")
                browser = info.get('input_source') in {'web_text', 'web_voice'}
                typed = info.get("input_source") in {"keyboard", "web_text"}
                if browser and not info['web_current']():
                    continue
                voice_epoch = info.get("microphone_epoch", self._microphone_epoch)
                if typed and (
                    generation != self._recording_generation or info["cancel_event"].is_set()
                ):
                    continue
                if not typed and not browser and (
                    self._microphone_muted
                    or voice_epoch != self._microphone_epoch
                ):
                    self._discard_voice_request(generation)
                    continue
                text = info.get("typed_text", "")
                has_command_audio = not (
                    info.get("wake_prefix_trimmed") and not info.get("post_wake_speech", True)
                )
                if self.stt and audio.size and has_command_audio:
                    stt_details: dict[str, Any] = {}
                    logger.info(
                        "Transcribing %.2fs of recorded speech...", audio.size / self.sample_rate
                    )
                    detailed = getattr(self.stt, "transcribe_detailed", None)
                    mark("stt_started")
                    if callable(detailed):
                        result = detailed(audio, beam_size=self.stt_beam_size)
                        stt_details = result
                        text = result["text"]
                        info["stt_needs_repeat"] = result.get("needs_repeat", False)
                        info["stt_confidence"] = result.get("confidence", 0.0)
                        info["stt_raw_text"] = result.get("raw_text", text)
                    else:
                        text = self.stt.transcribe(audio, beam_size=self.stt_beam_size)
                    mark("stt_finished", device=self.stt.device if hasattr(
                        self.stt, "device"
                    ) else "unknown", confidence=stt_details.get("confidence"),
                         needs_repeat=stt_details.get("needs_repeat"), words=len(text.split()))
                    logger.info(
                        "STT finished: %d words, confidence=%.2f, repeat=%s.",
                        len(text.split()), info.get("stt_confidence", 0.0),
                        info.get("stt_needs_repeat", False),
                    )
                    logger.debug("STT candidate: %r", info.get("stt_raw_text", text))
                    if self.show_transcripts:
                        logger.info(
                            "STT candidate (diagnostic): %r", info.get("stt_raw_text", text)
                        )
                if info.get("wake_prefix_trimmed") and not info.get("stt_needs_repeat"):
                    greeting = info.get("wake_phrase", "hey ciel")
                    text = f"{greeting}, {text}" if text else greeting
                if not self._running or (not typed and not browser and (
                    self._microphone_muted
                    or voice_epoch != self._microphone_epoch
                )):
                    if not typed:
                        self._discard_voice_request(generation)
                    continue
                with self._lock:
                    if (not typed and not browser and self.ambient
                            and generation == self._recording_generation):
                        poll = getattr(self.detector, "poll", None)
                        trigger = poll() if callable(poll) else None
                        if trigger:
                            self._confirm_ambient_wake(trigger)
                            info["wake_verified"] = True
                        # Discard late results from this recording before playback;
                        # its own greeting must not become a fresh barge-in trigger.
                        self.detector.reset(set_cooldown=False)
                if browser and not info["web_current"]():
                    continue
                superseded = generation != self._recording_generation
                info["superseded"] = superseded
                if self.on_transcript_observed and not typed and not browser:
                    self.on_transcript_observed(text, info)
                if superseded:
                    logger.info("Speech resumed during STT; suppressed the old reply.")
                    continue
                keep_listening = (
                    bool(self.on_transcription(text, info, audio))
                    if (self.on_transcription)
                    else False
                )
                with self._lock:
                    # A completed old response cannot cancel a new barge-in recording.
                    if not self._running or generation != self._recording_generation:
                        continue
                    if (
                        keep_listening and not self.ambient and not typed and not browser
                        and not self._microphone_muted
                    ):
                        self._start_recording()
                    else:
                        self._recent_audio.clear()
                        self.detector.reset(set_cooldown=not self.ambient)
                        self._set_state(self._standby_state())
            except Exception:
                logger.exception("Utterance processing failed")
                with self._lock:
                    if self._running and generation == self._recording_generation:
                        self.detector.reset(set_cooldown=False)
                        self._set_state(self._standby_state())
            finally:
                if trace is not None:
                    trace.mark("turn_finished")
                    trace.log()
                active_trace.reset(trace_token)
                self._processing_queue.task_done()

    def start_text_only(self) -> None:
        """Start typed requests without opening capture or loading wake detection."""
        self.start(text_only=True)

    def start(self, *, text_only: bool = False) -> None:
        if self._running:
            return
        if any(thread.is_alive() for thread in self._threads):
            raise RuntimeError("Previous listener workers are still stopping")
        self._frame_queue = queue.Queue(maxsize=8)
        self._processing_queue = queue.Queue(maxsize=4)
        self._notifications = queue.Queue(maxsize=32)
        self._recent_audio.clear()
        self.dropped_frames = 0
        self._stop_event = threading.Event()
        self._text_only = text_only
        if text_only:
            self._microphone_muted = True
        else:
            if (
                self.ambient or self.monitor_resumed_speech or self.barge_in_mode == "speech"
            ) and self._speech_detector is None:
                if self.sample_rate != 16000:
                    raise ValueError("Ambient speech detection requires 16000 Hz audio")
                from ciel.audio.activity import SpeechActivity

                self._speech_detector = SpeechActivity()
            reset_activity = getattr(self._speech_detector, "reset", None)
            if reset_activity:
                reset_activity()
            if self._speech_detector is not None:
                self.recorder.speech_detector = self._speech_detector
            start_detector = getattr(self.detector, "start", None)
            if start_detector:
                start_detector()
        self._running = True
        self._set_state(self._standby_state())
        workers = (self._process_worker, self._notification_worker)
        if not text_only:
            workers = (self._frame_worker, *workers)
        self._threads = [
            threading.Thread(target=target, daemon=True)
            for target in workers
        ]
        for thread in self._threads:
            thread.start()
        if text_only:
            logger.info("Text chat active; microphone capture is unavailable.")
            return
        try:
            self.backend.start_stream(
                callback=self._audio_callback,
                sample_rate=self.sample_rate,
                channels=1,
                device=self.device,
                blocksize=1280,
            )
        except BaseException:
            self.stop()
            raise
        logger.info("Wake listener active; audio and inference run on separate workers.")

    def stop(self) -> None:
        self._running = False
        if self._response_cancel is not None:
            self._response_cancel.set()
        self._stop_event.set()
        try:
            self.backend.stop_stream()
        finally:
            if self.tts:
                self.tts.stop()
            stop_detector = getattr(self.detector, "stop", None)
            if stop_detector:
                stop_detector()
            for thread in self._threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            with self._lock:
                self._state = ListenerState.IDLE
            logger.info("Wake listener stopped (%d dropped audio frames).", self.dropped_frames)
