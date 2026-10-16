"""Voice utterance recorder with silence / VAD energy cutoff."""

import time
from collections.abc import Callable

import numpy as np

from ciel.logging import get_logger

logger = get_logger("audio.recorder")


class VoiceRecorder:
    """Records speech utterances with dynamic silence detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold_rms: float = 0.008,
        silence_duration_seconds: float = 1.0,
        min_speech_duration_seconds: float = 0.25,
        max_duration_seconds: float = 30.0,
        initial_silence_timeout: float = 3.5,
        pause_grace_seconds: float = 0.0,
        speech_detector: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold_rms = silence_threshold_rms
        self.silence_duration_seconds = silence_duration_seconds
        self.min_speech_duration_seconds = min_speech_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.initial_silence_timeout = initial_silence_timeout
        self.pause_grace_seconds = pause_grace_seconds
        self.speech_detector = speech_detector

        self._buffer: list[np.ndarray] = []
        self._is_recording = False
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._start_time = 0.0
        self.capture_started_at = 0.0
        self.last_speech_observed_at: float | None = None

    def start(self) -> None:
        """Begin accumulating speech audio frames."""
        self._buffer.clear()
        self._is_recording = True
        self._speech_started = False
        self._start_time = 0.0
        self._sample_count = 0
        self.capture_started_at = time.monotonic()
        self.last_speech_observed_at = None
        self._speech_start_time = self._start_time
        self._last_speech_time = self._start_time
        logger.info("🎙️ Utterance recording started. Listening for speech...")

    def add_frame(self, audio_frame: np.ndarray) -> bool:
        """Add an incoming audio frame and check if the utterance has concluded.

        Returns:
            True if recording should continue, False if recording is finished (silence or timeout).
        """
        if not self._is_recording:
            return False

        # Flatten frame
        frame = audio_frame.squeeze()
        self._buffer.append(frame.copy())
        self._sample_count += frame.size
        now = self._sample_count / self.sample_rate

        # Calculate frame RMS
        rms = self.calculate_rms(frame)

        # Check speech presence
        is_speech = (
            self.speech_detector(frame)
            if self.speech_detector is not None
            else rms >= self.silence_threshold_rms
        )
        if is_speech:
            self.last_speech_observed_at = time.monotonic()
            if not self._speech_started:
                self._speech_started = True
                self._speech_start_time = now
                logger.debug("Speech activity detected (RMS=%.4f).", rms)
            self._last_speech_time = now
        else:
            # Silence observed
            if self._speech_started:
                silence_elapsed = now - self._last_speech_time
                if silence_elapsed >= self.silence_duration_seconds + self.pause_grace_seconds:
                    speech_duration = self._last_speech_time - self._speech_start_time
                    if speech_duration < self.min_speech_duration_seconds:
                        # Transient sound (< min_speech_duration) — reset and keep listening
                        logger.debug(
                            "Transient sound ignored (%.2fs) — resuming wait for speech.",
                            speech_duration,
                        )
                        self._speech_started = False
                        self._start_time = now
                    else:
                        logger.info(
                            "Silence detected after speech (%.1fs). Concluding utterance.",
                            silence_elapsed,
                        )
                        self._is_recording = False
                        return False
            else:
                # User has not spoken since recording started
                if (now - self._start_time) >= self.initial_silence_timeout:
                    logger.debug(
                        "Initial silence timeout reached (%.1fs). Concluding utterance.",
                        self.initial_silence_timeout,
                    )
                    self._is_recording = False
                    return False

        # Hard timeout check
        if now >= self.max_duration_seconds:
            logger.info(
                "Maximum recording duration reached (%.1fs). Concluding utterance.",
                self.max_duration_seconds,
            )
            self._is_recording = False
            return False

        return True

    def cancel(self) -> None:
        """Discard an unfinished utterance without returning its audio."""
        self._is_recording = False
        self._buffer.clear()

    def get_audio(self) -> np.ndarray:
        """Return the accumulated audio as a single 1D float32 numpy array."""
        if not self._buffer:
            return np.empty((0,), dtype=np.float32)
        combined = np.concatenate(self._buffer)
        if np.issubdtype(combined.dtype, np.floating):
            return combined.astype(np.float32)
        return (combined.astype(np.float32) / 32768.0).astype(np.float32)

    def is_recording(self) -> bool:
        """Return True if currently active."""
        return self._is_recording

    @staticmethod
    def calculate_rms(frame: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy level of an audio frame."""
        sq = frame.squeeze()
        if np.issubdtype(sq.dtype, np.floating):
            float_frame = sq
        else:
            float_frame = sq.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(float_frame**2))) if float_frame.size > 0 else 0.0
