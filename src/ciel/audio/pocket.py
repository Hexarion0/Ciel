"""Lazy CPU-only adapter for the verified Pocket TTS 3.3 Python API."""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ciel.audio.voice_style import TTSRequest, VoiceStyle
from ciel.logging import get_logger

logger = get_logger("audio.pocket")


class _GenerationStop(threading.Event):
    """Upstream cleanup can cancel inference without canceling fallback/playback."""

    def __init__(self, parent: threading.Event) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set()


class PocketInitializationError(RuntimeError):
    """Assets, dependencies, or model access require setup before another attempt."""


class PocketTTS:
    """Reuse one model and at most six reference states; serialize upstream inference.

    This module deliberately imports neither torch nor pocket_tts until first use,
    and never probes or initializes CUDA. Upstream loads directly on CPU.
    """

    def __init__(
        self, references: Path, *, neutral_reference: Path | None = None,
        quantization: str = "none", language: str = "english_2026-09",
        builtin_voice: str = "alba", gain_db: float = 0.0,
    ) -> None:
        if quantization not in {"none", "int8"}:
            raise ValueError("Pocket quantization must be none or int8")
        if not math.isfinite(gain_db) or not -60 <= gain_db <= 0:
            raise ValueError("Pocket gain must be finite and between -60 and 0 dB")
        self.references = references
        self.neutral_reference = neutral_reference
        self.quantization = quantization
        self.effective_quantization = "none"
        self.language = language
        self.builtin_voice = builtin_voice
        self._gain = np.float32(10 ** (gain_db / 20))
        self._model: Any = None
        self._states: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._warned: set[str] = set()

    def _reference_file(self, style: str) -> Path | None:
        # Prefer portable recordings; exported states must match this model version.
        for suffix in (".wav", ".flac", ".ogg", ".safetensors"):
            path = self.references / f"{style}{suffix}"
            if path.is_file():
                return path
        return None

    def _reference(self, style: VoiceStyle) -> Path | str:
        neutral = self.neutral_reference or self._reference_file("neutral")
        if neutral is not None and not neutral.is_file():
            raise FileNotFoundError(f"Pocket neutral reference is missing: {neutral}")
        if neutral is None:
            if "builtin" not in self._warned:
                logger.info("Pocket voice: built-in '%s'.", self.builtin_voice)
                self._warned.add("builtin")
            return self.builtin_voice
        reference = self._reference_file(style) if style != "neutral" else neutral
        if reference is None:
            if style not in self._warned:
                logger.warning("Pocket style '%s' missing; using Ciel neutral reference.", style)
                self._warned.add(style)
            return neutral
        return reference

    def _load(self) -> Any:
        if self._closed:
            raise RuntimeError("Pocket TTS is closed")
        if self._model is None:
            from pocket_tts import TTSModel

            quantize = self.quantization == "int8"
            try:
                model = TTSModel.load_model(language=self.language, quantize=quantize)
                self.effective_quantization = self.quantization
            except (ImportError, AttributeError, RuntimeError, NotImplementedError) as err:
                if not quantize:
                    raise
                logger.warning("Pocket INT8 loading failed; retrying stable CPU FP32: %s", err)
                model = TTSModel.load_model(language=self.language, quantize=False)
                self.effective_quantization = "none"
            if str(model.device) != "cpu":
                raise RuntimeError(f"Pocket must run on CPU, got {model.device}")
            if self._closed:
                raise RuntimeError("Pocket TTS closed during initialization")
            self._model = model
            logger.info("Pocket ready: CPU, %s, %s.", self.effective_quantization, self.language)
        return self._model

    def _state(self, model: Any, style: VoiceStyle) -> Any:
        reference = self._reference(style)
        key = str(reference)
        if key not in self._states:
            self._states[key] = model.get_state_for_audio_prompt(reference, truncate=True)
        return self._states[key]

    def warmup(self, style: VoiceStyle = "neutral") -> None:
        """Load model and reference once without generating placeholder speech."""
        with self._lock:
            try:
                model = self._load()
                self._state(model, style)
            except Exception:
                self._states.clear()
                self._model = None
                raise

    def stream(
        self, request: TTSRequest, stop: threading.Event,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield float32 mono chunks; drain a canceled iterator to join upstream workers."""
        while not self._lock.acquire(timeout=0.02):
            if stop.is_set():
                return
        iterator = None
        inference_stop = _GenerationStop(stop)
        try:
            if stop.is_set():
                return
            try:
                model = self._load()
                state = self._state(model, request.style)
            except Exception as err:
                self._states.clear()
                self._model = None
                raise PocketInitializationError(str(err)) from err
            if stop.is_set():
                return
            iterator = model.generate_audio_stream(
                state, request.text, copy_state=True, stop=inference_stop,
            )
            for tensor in iterator:
                if stop.is_set():
                    continue
                wave = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
                if not np.isfinite(wave).all():
                    raise RuntimeError("Pocket produced non-finite audio")
                if wave.size:
                    if self._gain != 1:
                        wave = wave * self._gain
                    yield wave, model.sample_rate
        finally:
            try:
                if iterator is not None:
                    inference_stop.set()
                    # Closing at a yield skips upstream joins; draining runs them.
                    for _ in iterator:
                        pass
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
            finally:
                if self._closed:
                    self._states.clear()
                    self._model = None
                self._lock.release()

    def close(self) -> None:
        """Drop model/reference tensors after active inference has stopped."""
        self._closed = True
        if not self._lock.acquire(timeout=1):
            return  # Active stream releases its tensors in its finally block.
        try:
            self._states.clear()
            self._model = None
        finally:
            self._lock.release()
