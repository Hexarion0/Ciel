"""Continuous wake word detection engine supporting openWakeWord and 'Hey Ciel'."""

import os
import queue
import re
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from ciel.conversation import is_direct_address, strip_wake_phrase
from ciel.logging import get_logger
from ciel.paths import resolve_asset_path

logger = get_logger("audio.wake")


class WakeWordDetector:
    """Real-time streaming wake word detection supporting openWakeWord and keyword spotting."""

    # Regex patterns to match wake phrases and natural phonetic variations
    WAKE_PATTERNS = [
        re.compile(
            r"\b(hey|hi|yo|a|okay|ok)?\s*(ciel|siel|ciell)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        wake_phrase: str = "hey ciel",
        models: list[str] | None = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 1.0,
        enable_whisper_spotter: bool = True,
        spotter_model_size: str = "base.en",
        min_rms: float = 0.006,
        window_seconds: float = 3.0,
    ) -> None:
        if not 0.0 < min_rms <= 0.05 or not 1.5 <= window_seconds <= 5.0:
            raise ValueError("Wake sensitivity or audio window is out of range")
        self.wake_phrase = wake_phrase.lower()
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self.enable_whisper_spotter = enable_whisper_spotter
        self.spotter_model_size = spotter_model_size
        self.min_rms = min_rms

        model_paths = self._resolve_model_paths(models)

        if model_paths:
            logger.info("Initializing openWakeWord detector with models: %s", model_paths)
            try:
                self.model = self._load_wake_model(model_paths)
                self.available_models = list(self.model.models.keys())
                logger.info("Loaded openWakeWord models: %s", self.available_models)
            except Exception as err:
                logger.warning(
                    "Failed to load openWakeWord model (%s). Relying on keyword spotter.",
                    err,
                )
                self.model = None
                self.available_models = []
        else:
            logger.info("No openWakeWord models configured — using Whisper keyword spotter only.")
            self.model = None
            self.available_models = []

        # Retain a complete slower greeting while asynchronous inference runs.
        self._sliding_buffer: deque = deque(maxlen=round(window_seconds * 16000))
        self._speech_frames_count = 0
        self._speech_pending = False
        self._last_speech_sample = 0
        self._last_spotter_check = 0.0
        self._ambient_rms = min_rms / 2
        self._spotter_model: Any = None
        self._spotter_lock = threading.Lock()
        self._generation = 0
        self._spotter_result: tuple[int, str, int | None] | None = None
        self._samples_seen = 0
        self._spotter_requests: queue.Queue = queue.Queue(maxsize=1)
        self._spotter_stop = threading.Event()
        self._spotter_thread: threading.Thread | None = None
        if self.enable_whisper_spotter:
            self.start()

    @staticmethod
    def _load_wake_model(paths):
        from openwakeword.model import Model

        return Model(wakeword_model_paths=paths)

    def _load_spotter_model(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.spotter_model_size, device="cpu", compute_type="int8", cpu_threads=2
        )

    def start(self) -> None:
        """Load and run the keyword spotter separately from incoming audio."""
        if not self.enable_whisper_spotter:
            return
        if self._spotter_thread and self._spotter_thread.is_alive():
            if self._spotter_stop.is_set():
                raise RuntimeError("Previous keyword spotter is still stopping")
            return
        self._spotter_stop = threading.Event()
        self._spotter_requests = queue.Queue(maxsize=1)
        self._spotter_thread = threading.Thread(target=self._spotter_worker, daemon=True)
        self._spotter_thread.start()

    def stop(self) -> None:
        self._spotter_stop.set()
        if self._spotter_thread and self._spotter_thread is not threading.current_thread():
            self._spotter_thread.join(timeout=1.0)

    def _spotter_worker(self) -> None:
        try:
            if self._spotter_model is None:
                self._spotter_model = self._load_spotter_model()
            logger.info(
                "Wake keyword spotter ready: %s (cpu/int8)", self.spotter_model_size
            )
        except Exception as err:
            logger.error("Failed to initialize keyword spotter: %s", err)
            return
        while not self._spotter_stop.is_set():
            try:
                generation, audio, snapshot_end = self._spotter_requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._spotter_lock:
                    if generation != self._generation:
                        continue
                segments, _ = self._spotter_model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    initial_prompt="Ciel is the name of the assistant.",
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 200},
                    no_speech_threshold=0.6,
                    max_new_tokens=32,
                )
                accepted = [
                    segment
                    for segment in segments
                    if getattr(segment, "no_speech_prob", 0.0) < 0.6
                    and getattr(segment, "avg_logprob", 0.0) >= -1.0
                    and getattr(segment, "compression_ratio", 0.0) <= 2.4
                ]
                text = " ".join(segment.text.strip() for segment in accepted)
                # Only a leading greeting may be trimmed. In 'what's up Ciel',
                # the question precedes the name and must remain in the recording.
                wake_end = (
                    self._wake_word_end(accepted)
                    if strip_wake_phrase(text, self.wake_phrase) != text.strip() else None
                )
                wake_sample = (
                    snapshot_end - audio.size + round(wake_end * 16000)
                    if wake_end is not None
                    else None
                )
                if text and not strip_wake_phrase(text, self.wake_phrase):
                    # A snapshot recognized as only the greeting is already consumed.
                    # Re-decoding its final syllables can invent a separate command.
                    wake_sample = snapshot_end
                with self._spotter_lock:
                    if generation == self._generation and not self._spotter_stop.is_set():
                        self._spotter_result = (generation, text, wake_sample)
            except Exception as err:
                logger.debug("Keyword spotter failed: %s", err)
            finally:
                self._spotter_requests.task_done()

    def _wake_word_end(self, segments: list[Any]) -> float | None:
        """Locate the end of the detected greeting in original audio coordinates."""
        words = [word for segment in segments for word in (getattr(segment, "words", None) or [])]
        if not words:
            return None
        text = "".join(word.word for word in words)
        configured = re.search(re.escape(self.wake_phrase), text, re.IGNORECASE)
        matches = [configured] if configured else []
        if "ciel" in self.wake_phrase:
            matches.extend(
                match for pattern in self.WAKE_PATTERNS if (match := pattern.search(text))
            )
        if not matches:
            return None
        match = min(matches, key=lambda candidate: candidate.start())
        position = 0
        for word in words:
            position += len(word.word)
            if position >= match.end():
                return float(word.end)
        return None

    @staticmethod
    def _resolve_model_paths(models: list[str] | None) -> list[str]:
        """Resolve model names to full paths. Returns [] when no models are requested."""
        if not models:
            return []  # No models → Whisper-only mode, do NOT load all pretrained models

        import openwakeword

        pretrained_paths = openwakeword.get_pretrained_model_paths()
        resolved: list[str] = []
        for requested in models:
            req_path = resolve_asset_path(requested)
            if req_path.is_file() and requested.endswith(".onnx"):
                resolved.append(str(req_path.resolve()))
                continue

            matched = False
            for p in pretrained_paths:
                base_name = os.path.basename(p).lower()
                if requested.lower() in base_name:
                    resolved.append(p)
                    matched = True
                    break

            if not matched:
                logger.warning(
                    "Wake model '%s' not found in pretrained models — skipping.",
                    requested,
                )

        return resolved

    def is_in_cooldown(self) -> bool:
        """Check whether the detector is currently within cooldown protection."""
        return (time.time() - self.last_trigger_time) < self.cooldown_seconds

    def poll(self) -> dict[str, Any] | None:
        """Consume completed keyword inference without adding audio or waiting for it."""
        if self.is_in_cooldown():
            return None
        with self._spotter_lock:
            completed = self._spotter_result
            self._spotter_result = None
        if completed is None:
            return None
        generation, transcription, wake_sample = completed
        if generation != self._generation or not is_direct_address(transcription, self.wake_phrase):
            return None
        now = time.time()
        self.last_trigger_time = now
        self.reset(set_cooldown=True)
        result = {
            "model": "whisper_keyword", "score": 1.0, "timestamp": now,
            "text": transcription, "wake_phrase": self.wake_phrase,
        }
        if wake_sample is not None:
            result["wake_tail_samples"] = max(0, self._samples_seen - wake_sample)
        return result

    def process_frame(self, audio_chunk: np.ndarray) -> dict[str, Any] | None:
        """Feed a mono 16kHz audio frame and check for wake word triggers."""
        self._samples_seen += audio_chunk.size
        if audio_chunk.size == 0 or self.is_in_cooldown():
            return None

        # Format chunk to 1D float32 and 1D int16
        if np.issubdtype(audio_chunk.dtype, np.floating):
            float_chunk = audio_chunk.squeeze().astype(np.float32)
            pcm16_chunk = (np.clip(float_chunk, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm16_chunk = audio_chunk.squeeze().astype(np.int16)
            float_chunk = (pcm16_chunk.astype(np.float32) / 32768.0).astype(np.float32)

        now = time.time()

        # 1. Check the configured openWakeWord models.
        if self.model is not None:
            predictions = self.model.predict(pcm16_chunk)
            for model_name, score in predictions.items():
                if score >= self.threshold:
                    self.last_trigger_time = now
                    logger.info(
                        "🎯 Wake word detected via openWakeWord! Model: '%s' (Confidence: %.2f)",
                        model_name,
                        score,
                    )
                    self.reset(set_cooldown=True)
                    return {
                        "model": model_name,
                        "score": float(score),
                        "timestamp": now,
                    }

        if trigger := self.poll():
            return trigger

        if self.enable_whisper_spotter:
            self._sliding_buffer.extend(float_chunk)
            rms = float(np.sqrt(np.mean(float_chunk**2))) if float_chunk.size else 0.0
            gate = max(self.min_rms, self._ambient_rms * 1.5)
            if rms > gate:
                self._speech_frames_count = min(3, self._speech_frames_count + 1)
                self._last_speech_sample = self._samples_seen
                if self._speech_frames_count >= 2:
                    self._speech_pending = True
            else:
                # Only update the noise floor with frames below the speech gate.
                # Quiet speech must not teach the detector to reject itself.
                self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * rms
                self._speech_frames_count = 0
            final_snapshot = (
                self._speech_pending
                and self._samples_seen - self._last_speech_sample >= 2560
            )
            if (
                self._spotter_model is not None
                and (self._speech_frames_count >= 2 or final_snapshot)
                and len(self._sliding_buffer) >= 8000
                and (final_snapshot or now - self._last_spotter_check >= 0.25)
            ):
                self._last_spotter_check = now
                if final_snapshot:
                    self._speech_pending = False
                request = (
                    self._generation,
                    np.array(self._sliding_buffer, dtype=np.float32),
                    self._samples_seen,
                )
                # Keep only the newest pending snapshot if inference falls behind.
                try:
                    self._spotter_requests.put_nowait(request)
                except queue.Full:
                    try:
                        self._spotter_requests.get_nowait()
                        self._spotter_requests.task_done()
                    except queue.Empty:
                        pass
                    try:
                        self._spotter_requests.put_nowait(request)
                    except queue.Full:
                        pass
        return None

    def reset(self, set_cooldown: bool = False) -> None:
        """Reset internal prediction buffers and optionally engage cooldown."""
        if self.model is not None:
            self.model.reset()
            # Flush internal openwakeword buffers with zeros
            silence = np.zeros(1280, dtype=np.int16)
            for _ in range(8):
                self.model.predict(silence)

        with self._spotter_lock:
            self._generation += 1
            self._spotter_result = None
        self._sliding_buffer.clear()
        self._speech_frames_count = 0
        self._speech_pending = False
        self._last_speech_sample = 0
        self._last_spotter_check = time.time()
        if set_cooldown:
            self.last_trigger_time = time.time()
