"""Speech-to-text conversion engine using faster-whisper with VAD and anti-hallucination."""

import gc
import math
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from ciel.config import AudioConfig
from ciel.conversation import strip_wake_phrase
from ciel.logging import get_logger

logger = get_logger("audio.stt")
_DEFAULT_AUDIO = AudioConfig()


def free_cuda_memory_mb() -> int | None:
    """Read current free memory on the default GPU, with a bounded optional probe."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=0.5, check=True,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

# Common Whisper hallucination patterns on silence or ambient noise
_HALLUCINATION_PATTERNS = [
    re.compile(
        r"^\s*(thank\s+you(\s+very\s+much|\s+for\s+watching)?|thanks\s+for\s+watching)[.?!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(subtitles?\s+by|subscribe|like\s+and\s+subscribe)[.?!]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(\[[^\]]+\]|\([^\)]+\))\s*$"),  # [Music], (bell rings), etc.
    re.compile(r"^\s*(\.|\?|!|,|-|_)+\s*$"),  # lone punctuation
    re.compile(r"^\s*(you|bye|okay|oh)\.?\s*$", re.IGNORECASE),  # single phantom syllables on noise
]


def _preload_cuda_libraries() -> None:
    """Preload only the CUDA 12/cuDNN 9 libraries required by faster-whisper."""
    try:
        import ctypes
        import sys

        site_pkgs = (
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
        )
        # cuBLAS depends on cuBLASLt, so load it first. Never glob all NVIDIA
        # libraries: NVBLAS exports CPU BLAS symbols and intercepts Whisper's
        # CPU operations, causing native failures without an NVBLAS config.
        libraries = (
            site_pkgs / "cublas/lib/libcublasLt.so.12",
            site_pkgs / "cublas/lib/libcublas.so.12",
            site_pkgs / "cudnn/lib/libcudnn.so.9",
        )
        for so_file in libraries:
            if so_file.is_file():
                try:
                    ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except Exception:
        pass


class SpeechToText:
    """Fast, local speech-to-text transcriber powered by faster-whisper / CTranslate2."""

    def __init__(
        self,
        model_size: str = _DEFAULT_AUDIO.stt_model,
        device: str = _DEFAULT_AUDIO.stt_device,
        compute_type: str = _DEFAULT_AUDIO.stt_compute_type,
        language: str = _DEFAULT_AUDIO.stt_language,
        initial_prompt: str = "Ciel is the name of the assistant.",
        min_confidence: float = _DEFAULT_AUDIO.stt_min_confidence,
        retry_confidence: float = _DEFAULT_AUDIO.stt_retry_confidence,
        retry_model: str | None = _DEFAULT_AUDIO.stt_retry_model,
        wake_phrase: str = _DEFAULT_AUDIO.wake_word,
        retry_min_free_mb: int = _DEFAULT_AUDIO.stt_retry_min_free_mb,
        load_timeout: float = _DEFAULT_AUDIO.stt_startup_timeout_seconds,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0 or not 0.0 <= retry_confidence <= 1.0:
            raise ValueError("STT confidence thresholds must be between zero and one")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self.min_confidence = min_confidence
        self.retry_confidence = retry_confidence
        self.retry_model_size = retry_model or None
        self.retry_min_free_mb = retry_min_free_mb
        if not math.isfinite(load_timeout) or load_timeout <= 0:
            raise ValueError("STT loading timeout must be finite and positive")
        self.load_timeout = load_timeout
        self.wake_phrase = wake_phrase
        self._retry_model: WhisperModel | None = None
        self._retry_blocked_devices: set[str] = set()

        self.model: WhisperModel | None = None
        self._ready = threading.Event()
        self._load_error: Exception | None = None

        _preload_cuda_libraries()
        # Load model in background thread — startup continues immediately
        self._loader = threading.Thread(target=self._load_model_bg, daemon=True)
        self._loader.start()

    def _load_model_bg(self) -> None:
        """Background thread: load model then set the ready event."""
        try:
            self.model = self._load_model(self.model_size, self.device, self.compute_type)
            logger.info(
                "STT ready: %s (%s/%s)", self.model_size, self.device, self.compute_type
            )
        except Exception as err:
            self._load_error = err
            logger.error("STT model failed to load: %s", err)
        finally:
            self._ready.set()

    def is_ready(self) -> bool:
        """Return True only after the model has loaded successfully."""
        return self._ready.is_set() and self._load_error is None and self.model is not None

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until model is loaded (or timeout seconds). Returns True if loaded."""
        return self._ready.wait(
            timeout=self.load_timeout if timeout is None else timeout,
        ) and self.is_ready()

    def _load_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        # Resolve 'auto' device
        resolved_device = device
        resolved_compute = compute_type

        if device == "auto":
            try:
                import ctranslate2

                if ctranslate2.get_cuda_device_count() > 0:
                    resolved_device = "cuda"
                    resolved_compute = (
                        "float16" if compute_type in ("default", "auto") else compute_type
                    )
                else:
                    resolved_device = "cpu"
                    resolved_compute = (
                        "int8" if compute_type in ("default", "auto") else compute_type
                    )
            except Exception:
                resolved_device = "cpu"
                resolved_compute = "int8"
        elif compute_type in ("default", "auto"):
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper STT model: '%s' (device=%s, compute_type=%s)",
            model_size,
            resolved_device,
            resolved_compute,
        )

        try:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=resolved_device,
                compute_type=resolved_compute,
            )
            if resolved_device == "cuda":
                # CTranslate2 loads some CUDA libraries lazily on first inference.
                # Validate them in the loader, before claiming GPU STT is ready.
                segments, _info = model.transcribe(
                    np.zeros(16000, dtype=np.float32), language=self.language,
                    beam_size=1, vad_filter=False, condition_on_previous_text=False,
                )
                list(segments)
            self.device = resolved_device
            self.compute_type = resolved_compute
            return model
        except Exception as err:
            logger.warning(
                "Whisper initialization on %s/%s failed (%s). Falling back to CPU/int8.",
                resolved_device,
                resolved_compute,
                err,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            return WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
            )

    @staticmethod
    def _is_hallucination(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
        """Check if a segment appears to be a Whisper hallucination on noise/silence."""
        if not text:
            return True
        if not any(character.isalnum() for character in text):
            return True
        # High probability of silence / no speech
        if no_speech_prob > 0.65:
            return True
        # Very low confidence decoding
        if avg_logprob < -1.3:
            return True
        # Matches common hallucination regexes
        for pattern in _HALLUCINATION_PATTERNS:
            if pattern.match(text) and (no_speech_prob > 0.35 or avg_logprob < -0.7):
                return True
        return False

    def transcribe(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        beam_size: int = _DEFAULT_AUDIO.stt_beam_size,
    ) -> str:
        """Transcribe an audio numpy array or file path to plain text."""
        result = self.transcribe_detailed(
            audio=audio,
            language=language or self.language,
            beam_size=beam_size,
        )
        return result["text"]

    def transcribe_detailed(
        self,
        audio: np.ndarray | str | Path,
        language: str | None = None,
        beam_size: int = _DEFAULT_AUDIO.stt_beam_size,
    ) -> dict[str, Any]:
        """Transcribe audio with Silero VAD filtering and anti-hallucination safeguards."""
        if isinstance(audio, np.ndarray) and (audio.size == 0 or not np.any(audio)):
            return {
                "text": "",
                "language": language or self.language,
                "segments": [],
                "duration": 0.0,
                "latency": 0.0,
                "confidence": 0.0,
                "needs_repeat": False,
                "retried": False,
            }
        # Wait for background model load if it hasn't finished yet
        if not self._ready.is_set():
            logger.info("⏳ STT model still loading — waiting...")
            if not self._ready.wait(timeout=self.load_timeout):
                raise TimeoutError("STT model is still loading; please try again later")
        if self._load_error or self.model is None:
            raise RuntimeError(f"STT model failed to load: {self._load_error}")

        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return {
                    "text": "",
                    "language": language or self.language,
                    "segments": [],
                    "duration": 0.0,
                    "latency": 0.0,
                }
            audio_input = audio.squeeze().astype(np.float32)
        else:
            audio_input = str(Path(audio).resolve())

        target_lang = language or self.language
        start_t = time.time()

        # Decode deterministically without penalizing natural repetitions and corrections.
        decode_kwargs: dict[str, Any] = {
            "language": target_lang,
            "beam_size": beam_size,
            "temperature": 0,  # Greedy decoding — no random word sampling
            "best_of": 1,  # With temperature=0, only one candidate needed
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": False,  # Avoid cascading hallucinations
            "vad_filter": True,  # Silero VAD strips silence before inference
            "vad_parameters": dict(
                min_silence_duration_ms=200,
                speech_pad_ms=400,  # Extra padding so word edges aren't clipped
            ),
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "compression_ratio_threshold": 2.2,  # Tighter: discard garbled/repetition-heavy output
            "log_prob_threshold": -0.7,  # Tighter: drop low-confidence segments
            "no_speech_threshold": 0.55,  # Slightly tighter silence filter
            "word_timestamps": True,
        }

        def decode(options: dict[str, Any], model: WhisperModel | None = None) -> dict[str, Any]:
            """Consume lazy inference and expose quality without treating it as certainty."""
            generator, info = (model or self.model).transcribe(audio_input, **options)
            segments_list = []
            text_parts = []
            raw_parts = []
            suspected_speech = False
            for segment in generator:
                cleaned = segment.text.strip()
                if not cleaned:
                    continue
                suspected_speech |= segment.no_speech_prob < 0.65
                if segment.no_speech_prob < 0.65:
                    raw_parts.append(cleaned)
                if self._is_hallucination(cleaned, segment.no_speech_prob, segment.avg_logprob):
                    continue
                if getattr(segment, "compression_ratio", 0.0) > 2.4:
                    continue
                words = getattr(segment, "words", None) or []
                word_score = sum(word.probability for word in words) / len(words) if words else 1.0
                score = float(min(math.exp(min(0.0, segment.avg_logprob)), word_score))
                text_parts.append(cleaned)
                segments_list.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": cleaned,
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                        "confidence": score,
                    }
                )
            weight = sum(len(segment["text"]) for segment in segments_list)
            confidence = (
                sum(segment["confidence"] * len(segment["text"]) for segment in segments_list)
                / weight
                if weight
                else 0.0
            )
            return {
                "text": " ".join(text_parts).strip(),
                "raw_text": " ".join(raw_parts).strip(),
                "language": getattr(info, "language", target_lang),
                "segments": segments_list,
                "duration": getattr(info, "duration", 0.0),
                "confidence": confidence,
                "suspected_speech": suspected_speech,
            }

        try:
            result = decode(decode_kwargs)
        except Exception as err:
            logger.warning("Whisper transcription failed (%s). Retrying on CPU/int8.", err)
            self._retry_model = None  # A cached CUDA retry must not follow the CPU fallback.
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.device = "cpu"
            self.compute_type = "int8"
            result = decode(decode_kwargs)

        duration = (
            audio_input.size / 16000 if isinstance(audio_input, np.ndarray) else result["duration"]
        )
        retried = False
        clear_greeting = (
            bool(result["text"])
            and result["confidence"] >= self.min_confidence
            and not strip_wake_phrase(result["text"], self.wake_phrase)
        )
        if (
            self.device not in self._retry_blocked_devices
            and not clear_greeting and result["suspected_speech"]
            and result["confidence"] < max(self.min_confidence, self.retry_confidence)
        ):
            # Spend extra decoding work only on ambiguous short utterances.
            if duration <= 12.0:
                retried = True
                logger.info(
                    "Retrying unclear STT (confidence=%.2f) with %s...",
                    result["confidence"], self.retry_model_size or self.model_size,
                )
                try:
                    retry_size = self.retry_model_size
                    # Reuse the primary model when the requested retry is identical.
                    if retry_size == self.model_size:
                        retry_size = None
                    if retry_size and self.device == "cuda" and self.retry_min_free_mb:
                        free_mb = free_cuda_memory_mb()
                        needed_mb = self.retry_min_free_mb if self._retry_model is None else 512
                        if free_mb is None or free_mb < needed_mb:
                            self._retry_model = None
                            retry_size = None
                            gc.collect()
                            logger.info(
                                "Using primary STT for quality retry: free VRAM=%s MB, "
                                "required=%d MB.", free_mb, needed_mb,
                            )
                    if retry_size and self._retry_model is None:
                        self._retry_model = WhisperModel(
                            retry_size,
                            device=self.device,
                            compute_type=self.compute_type,
                        )
                    candidate = decode(
                        {
                            **decode_kwargs,
                            "beam_size": max(8, beam_size),
                            "patience": 1.5,
                            "max_new_tokens": 128,
                        },
                        model=self._retry_model if retry_size else self.model,
                    )
                    if candidate["confidence"] > result["confidence"]:
                        result = candidate
                except Exception as err:
                    logger.warning("STT quality retry failed: %s", err)
                    if "out of memory" in str(err).casefold():
                        self._retry_model = None
                        self._retry_blocked_devices.add(self.device)
                        logger.warning(
                            "Disabled STT quality retries on %s after memory exhaustion; "
                            "keeping primary transcription.", self.device,
                        )
                if self.device == "cuda" and self._retry_model is not None:
                    # A temporary quality upgrade should not occupy training VRAM
                    # after this utterance. CPU retries can retain their cache.
                    self._retry_model = None
                    gc.collect()
                elif self.device in self._retry_blocked_devices:
                    # The exception traceback has now been released as well.
                    gc.collect()
        result["needs_repeat"] = (
            result.pop("suspected_speech") and result["confidence"] < self.min_confidence
        )
        if result["needs_repeat"]:
            # Keep diagnostics but never pass the uncertain text off as a command.
            result["text"] = ""
        result["retried"] = retried
        result["latency"] = time.time() - start_t
        logger.debug(
            "STT confidence=%.2f retry=%s repeat=%s latency=%.2fs",
            result["confidence"],
            retried,
            result["needs_repeat"],
            result["latency"],
        )
        return result
