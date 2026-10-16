import argparse
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig
from piper.download_voices import download_voice

from ciel.audio.alignment import CharacterTimeline, character_timeline, estimated_timeline
from ciel.audio.chatterbox_worker import ChatterboxTurboWorker, ChatterboxWorkerError
from ciel.audio.chunk_playback import ChunkPlayback
from ciel.audio.native import sd
from ciel.audio.pocket import PocketInitializationError, PocketTTS
from ciel.audio.speech_events import (
    SpeechEvent,
    SpeechEventLimiter,
    SpeechExpressiveness,
    split_speech_event,
    strip_speech_events,
    strip_stage_directions,
)
from ciel.audio.voice_style import TTSRequest, VoiceStyle, select_voice_style
from ciel.config import get_settings, normalize_audio_device
from ciel.conversation import strip_internal_reply_notes
from ciel.logging import get_logger
from ciel.paths import bundled_worker_path, resolve_asset_path, voice_profile_path

logger = get_logger("audio.tts")

def resolve_tts_engine(voice_name: str, engine: str = "auto") -> str:
    """An explicit engine wins; otherwise choose the engine for the voice."""
    selected = engine.lower()
    if selected == "auto":
        return "piper"
    if selected in {"chatterbox", "hq"}:
        selected = "chatterbox_turbo"
    if selected not in {"piper", "pocket", "chatterbox_turbo"}:
        raise ValueError(f"Unknown TTS engine: {engine}")
    return selected


def resolve_piper_voice_paths(voice_name: str, models_dir: str | Path) -> tuple[Path, Path]:
    """Resolve a Piper ONNX + config pair without starting any TTS server."""
    models = resolve_asset_path(models_dir)
    direct_path = resolve_asset_path(voice_name)
    if direct_path.is_file() and direct_path.suffix == ".onnx":
        json_file = direct_path.with_suffix(".onnx.json")
        if not json_file.is_file():
            json_file = direct_path.with_name(f"{direct_path.stem}.json")
        return direct_path, json_file

    models.mkdir(parents=True, exist_ok=True)
    onnx_file = models / f"{voice_name}.onnx"
    json_file = models / f"{voice_name}.onnx.json"
    if not json_file.is_file():
        alt_json = models / f"{voice_name}.json"
        if alt_json.is_file():
            json_file = alt_json

    if not onnx_file.is_file() or not json_file.is_file():
        logger.info("Downloading Piper TTS voice model '%s'...", voice_name)
        download_voice(voice_name, models)
        logger.info("Successfully downloaded TTS voice '%s'.", voice_name)
    return onnx_file, json_file


class TextToSpeech:
    """CPU Pocket/Piper speech with optional persistent Chatterbox Turbo HQ mode."""

    def __init__(
        self,
        voice_name: str | None = None,
        engine: str | None = None,
        models_dir: str | Path = "models/tts",
        speed: float | None = None,
        output_device: int | str | None = None,
        enabled: bool = True,
        include_alignments: bool = False,
        tts_fallback_voice: str | None = None,
        voice_profiles_dir: str | Path | None = None,
        chatterbox_model_dir: str | Path | None = None,
        chatterbox_python: str | Path | None = None,
        min_free_vram_mib: int | None = None,
        audio_queue_size: int | None = None,
        playback_latency: float | None = None,
        expressiveness: SpeechExpressiveness | None = None,
    ) -> None:
        settings = get_settings().audio

        self.voice_name = voice_name if voice_name is not None else settings.tts_voice
        raw_engine = engine if engine is not None else settings.tts_engine
        self.engine = resolve_tts_engine(self.voice_name, raw_engine)
        self.expressiveness = (
            expressiveness if expressiveness is not None else settings.tts_expressiveness
        )
        if self.expressiveness not in {"expressive", "natural", "off"}:
            raise ValueError("expressiveness must be expressive, natural, or off")

        self.models_dir = resolve_asset_path(models_dir)
        self.speed = speed if speed is not None else settings.tts_speed
        self.output_device = normalize_audio_device(
            output_device if output_device is not None else settings.output_device
        )
        self.enabled = enabled
        self.include_alignments = include_alignments
        self._synthesis_timing = threading.local()
        self.tts_fallback_voice = tts_fallback_voice or settings.tts_fallback_voice
        self.voice_profiles_dir = Path(voice_profiles_dir or settings.tts_voice_profiles)
        self.chatterbox_model_dir = Path(
            chatterbox_model_dir or settings.tts_chatterbox_model
        )
        self.chatterbox_python = Path(chatterbox_python or settings.tts_chatterbox_python)
        self.min_free_vram_mib = (
            min_free_vram_mib
            if min_free_vram_mib is not None else settings.tts_min_free_vram_mb
        )
        self.audio_queue_size = audio_queue_size or settings.tts_audio_queue_size
        self.playback_latency = (
            playback_latency if playback_latency is not None else settings.tts_playback_latency
        )
        self._chatterbox: ChatterboxTurboWorker | None = None
        self._chatterbox_failed = False
        self._profile_data: dict | None = None
        self._pocket: PocketTTS | None = None
        self._pocket_failed = False
        self._pocket_retry_at = 0.0
        self._pocket_stops: set[threading.Event] = set()
        self._pocket_stops_lock = threading.Lock()
        self._pocket_streaming = settings.tts_pocket_streaming
        self.last_engine: str | None = None

        self._voice: PiperVoice | None = None
        self._fallback_voice: PiperVoice | None = None
        self._is_playing = False
        self._playback_lock = threading.Lock()
        self._playback_generation = 0
        self._playback_serial = 0
        self._stop_event = threading.Event()  # signals stop() to unblock speak(block=True)
        self._pending_text = ""
        self._play_started_at = 0.0
        self._play_duration = 0.0
        self._play_stopped_at = 0.0
        self._stream_text = ""
        self._stream_spoken: list[str] | None = None
        self._stream_played_seconds = 0.0
        self._synthesis_lock = threading.Lock()
        self._synthesis_slots = threading.BoundedSemaphore(2)

        if self.enabled:
            logger.info(
                "Initializing TTS engine '%s' with voice '%s'...", self.engine, self.voice_name
            )
            if self.engine == "piper":
                self._load_voice()
            elif self.engine == "chatterbox_turbo":
                logger.info(
                    "Chatterbox Turbo voice '%s' will load on its first request.", self.voice_name
                )
            elif self.engine == "pocket":
                logger.info("Pocket TTS will load on first request, on CPU only.")

    def _ensure_model_files(self) -> tuple[Path, Path]:
        """Ensure voice model .onnx and .onnx.json files exist locally, downloading if necessary."""
        return resolve_piper_voice_paths(self.voice_name, self.models_dir)

    def _load_voice(self) -> None:
        """Load the Piper neural voice model."""
        try:
            onnx_path, json_path = self._ensure_model_files()
            logger.info("Loading Piper TTS voice: %s", self.voice_name)
            self._voice = PiperVoice.load(
                model_path=str(onnx_path),
                config_path=str(json_path),
                use_cuda=False,
                **({"include_alignments": True} if self.include_alignments else {}),
            )
            logger.info("Piper TTS engine initialized successfully.")
        except Exception as err:
            logger.warning("Failed to initialize Piper TTS: %s. Audio output disabled.", err)
            self._voice = None
            self.enabled = False

    @property
    def supports_sentence_pipeline(self) -> bool:
        """Turbo synthesis is faster than playback and benefits from bounded prefetch."""
        return self.engine == "chatterbox_turbo"

    def _load_pocket(self) -> PocketTTS:
        """Construct the CPU adapter lazily; no torch/CUDA import during Ciel startup."""
        if self._pocket is None:
            settings = get_settings().audio
            self._pocket = PocketTTS(
                self._resolve_local_path(settings.tts_pocket_references),
                neutral_reference=(
                    self._resolve_local_path(settings.tts_pocket_neutral_reference)
                    if settings.tts_pocket_neutral_reference else None
                ),
                quantization=settings.tts_pocket_quantization,
                language=settings.tts_pocket_language,
                builtin_voice=settings.tts_pocket_builtin_voice,
            )
        return self._pocket

    def _pocket_chunks(
        self, request: TTSRequest, stop: threading.Event,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Fallback before first audio; never repeat a partially spoken sentence."""
        yielded = False
        if (
            self._pocket_failed and self._pocket_retry_at
            and time.monotonic() >= self._pocket_retry_at
        ):
            self._pocket_failed = False
        if not self._pocket_failed:
            try:
                for wave, rate in self._load_pocket().stream(request, stop):
                    if stop.is_set():
                        return
                    yielded = True
                    self.last_engine = "pocket"
                    yield wave, rate
                if yielded or stop.is_set():
                    return
                raise RuntimeError("Pocket returned no audio")
            except Exception as err:
                if stop.is_set():
                    return
                self._pocket_failed = True
                self._pocket_retry_at = (
                    0.0 if isinstance(err, PocketInitializationError) else time.monotonic() + 30
                )
                logger.warning("Pocket failed; selecting local Piper fallback: %s", err)
                if yielded:
                    raise RuntimeError("Pocket failed after audio started; reply stopped") from err
        if not stop.is_set():
            result = self._synthesize_piper_fallback(request.text)
            if result is not None:
                self.last_engine = "piper"
                yield result
            else:
                logger.error("No local Piper fallback could synthesize speech.")

    def synthesize_stream(
        self, text: str, *, style: VoiceStyle | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Public chunk interface, also used by the benchmark without audio devices."""
        selected_style = select_voice_style(text, style)
        if not self.enabled:
            return
        clean = SpeechEventLimiter(self.expressiveness).apply(self.clean_text_for_speech(text))
        if not clean:
            return
        stop = cancel_event if cancel_event is not None else threading.Event()
        with self._pocket_stops_lock:
            self._pocket_stops.add(stop)
        previous_cancelled = getattr(self._synthesis_timing, "cancelled", lambda: False)
        self._synthesis_timing.cancelled = lambda: stop.is_set() or previous_cancelled()
        try:
            if stop.is_set():
                return
            if self.engine == "chatterbox_turbo" and not self._chatterbox_failed:
                result = self._synthesize_chatterbox(clean, selected_style, fallback=False)
                if result is not None:
                    self.last_engine = "chatterbox_turbo"
                    yield result
                    return
            words = strip_speech_events(clean)
            if not words or stop.is_set():
                return
            if self.engine in {"pocket", "chatterbox_turbo"}:
                yield from self._pocket_chunks(TTSRequest(words, selected_style), stop)
            else:
                result = self._synthesize_piper(words)
                if result is not None and not stop.is_set():
                    self.last_engine = "piper"
                    yield result
        finally:
            self._synthesis_timing.cancelled = previous_cancelled
            with self._pocket_stops_lock:
                self._pocket_stops.discard(stop)

    def _synthesize_pocket(
        self, text: str, style: VoiceStyle = "neutral",
    ) -> tuple[np.ndarray, int] | None:
        """Collect chunks for callers that explicitly need a complete waveform."""
        stop = threading.Event()
        canceled = getattr(self._synthesis_timing, "cancelled", lambda: False)
        waves = []
        rate = 24000
        with self._pocket_stops_lock:
            self._pocket_stops.add(stop)
        iterator = self._pocket_chunks(TTSRequest(text, style), stop)
        try:
            for wave, rate in iterator:
                if canceled():
                    stop.set()
                    return None
                waves.append(wave)
            return (np.concatenate(waves), rate) if waves and not canceled() else None
        except Exception as err:
            if canceled() or stop.is_set():
                return None
            logger.warning("Pocket batch synthesis failed before playback; trying Piper: %s", err)
            result = self._synthesize_piper_fallback(text)
            if result is not None:
                self.last_engine = "piper"
            return result
        finally:
            stop.set()
            iterator.close()
            with self._pocket_stops_lock:
                self._pocket_stops.discard(stop)

    def _resolve_local_path(self, path: str | Path) -> Path:
        """Resolve configured voice assets from the writable runtime root or an absolute path."""
        return resolve_asset_path(path)

    def _load_chatterbox_worker(self) -> ChatterboxTurboWorker:
        """Read one voice profile and start the persistent, offline pinned-model worker."""
        if self._chatterbox is not None:
            return self._chatterbox
        manifest = voice_profile_path(self.voice_name, self.voice_profiles_dir)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("name") != self.voice_name or data.get("engine") != "chatterbox_turbo":
            raise ChatterboxWorkerError(f"Invalid Chatterbox voice profile: {manifest}")
        expected = sorted(event.value for event in SpeechEvent)
        if sorted(data.get("supported_events", [])) != expected:
            raise ChatterboxWorkerError(
                "Voice profile event list does not match native Turbo events"
            )
        reference = self._resolve_local_path(data["reference_audio"])
        transcript = self._resolve_local_path(data["reference_transcript"])
        if not transcript.is_file() or not transcript.read_text(encoding="utf-8").strip():
            raise ChatterboxWorkerError(f"Reference transcript is missing or empty: {transcript}")
        model_dir = self._resolve_local_path(self.chatterbox_model_dir)
        python = resolve_asset_path(self.chatterbox_python, preserve_symlinks=True)
        script = bundled_worker_path()
        self._profile_data = data
        self._chatterbox = ChatterboxTurboWorker(
            python,
            script,
            model_dir,
            reference,
            transcript.read_text(encoding="utf-8"),
            min_free_vram_mib=self.min_free_vram_mib,
            synthesis_timeout=get_settings().audio.tts_synthesis_timeout_seconds,
            event_reference=(
                self._resolve_local_path(data["event_reference_audio"])
                if data.get("event_reference_audio") else None
            ),
            cue_audio={key: self._resolve_local_path(path)
                       for key, path in data.get("event_audio", {}).items()},
        )
        return self._chatterbox

    def _synthesize_chatterbox(
        self, text: str, style: VoiceStyle = "neutral", *, fallback: bool = True,
    ) -> tuple[np.ndarray, int] | None:
        """Generate with cached Turbo conditionals; latch failures into local Piper fallback."""
        event_text, spoken_text = split_speech_event(text)
        # Turbo accepts inline and standalone events. Piper must receive only words.
        model_text = f"{event_text} {spoken_text}".strip()
        spoken_text = strip_speech_events(spoken_text)
        if not model_text:
            return None
        if self._chatterbox_failed:
            return self._synthesize_pocket(spoken_text, style) if spoken_text and fallback else None
        canceled = getattr(self._synthesis_timing, "cancelled", lambda: False)
        try:
            worker = self._load_chatterbox_worker()
            result = worker.synthesize(model_text, canceled=canceled)
            if result is None:
                if canceled():
                    return None
                raise ChatterboxWorkerError("Chatterbox returned no audio")
            self.last_engine = "chatterbox_turbo"
            logger.debug("Chatterbox Turbo synthesis completed (%d characters).", len(spoken_text))
            return result
        except Exception as err:
            self._chatterbox_failed = True
            logger.warning(
                "Chatterbox Turbo failed; switching to CPU Pocket, then local Piper: %s",
                err,
            )
            if self._chatterbox is not None:
                self._chatterbox.close(force=True)
                self._chatterbox = None
            return self._synthesize_pocket(spoken_text, style) if spoken_text and fallback else None

    def warmup(self) -> bool:
        """Load the cached voice in the background without synthesizing placeholder speech."""
        if not self.enabled:
            return False
        if self.engine == "pocket":
            try:
                self._load_pocket().warmup()
                return True
            except Exception as err:
                self._pocket_failed = True
                self._pocket_retry_at = 0.0
                logger.warning("Pocket initialization failed; selecting local Piper: %s", err)
                return False
        if self.engine != "chatterbox_turbo" or self._chatterbox_failed:
            return False
        try:
            self._load_chatterbox_worker().start()
            return True
        except Exception as err:
            self._chatterbox_failed = True
            logger.warning(
                "Chatterbox Turbo initialization failed; fallback is Pocket then Piper: %s", err
            )
            if self._chatterbox is not None:
                self._chatterbox.close(force=True)
                self._chatterbox = None
            return False

    def _synthesize_piper_fallback(self, text: str) -> tuple[np.ndarray, int] | None:
        """Use the selected CIEL Piper voice, then Amy, without a network fallback."""
        if self._fallback_voice is not None:
            try:
                return self._synthesize_piper(
                    strip_speech_events(text), voice=self._fallback_voice
                )
            except Exception as err:
                logger.warning("Cached Piper fallback failed; trying another local voice: %s", err)
                self._fallback_voice = None
        configured = (
            self._profile_data.get("fallback_voice")
            if self._profile_data else self.tts_fallback_voice
        )
        for name in dict.fromkeys([configured, "en_US-amy-medium"]):
            onnx = self.models_dir / f"{name}.onnx"
            config = onnx.with_suffix(".onnx.json")
            if not config.is_file():
                config = onnx.with_suffix(".json")
            if not onnx.is_file() or not config.is_file():
                logger.warning("Local Piper fallback '%s' is not installed.", name)
                continue
            try:
                voice = PiperVoice.load(
                    model_path=str(onnx), config_path=str(config), use_cuda=False,
                    **({"include_alignments": True} if self.include_alignments else {}),
                )
                self._fallback_voice = voice
                logger.info("Using local Piper fallback voice '%s'.", name)
                return self._synthesize_piper(strip_speech_events(text), voice=voice)
            except Exception as err:
                self._fallback_voice = None
                logger.warning("Local Piper fallback '%s' failed: %s", name, err)
        return None

    def close(self) -> None:
        """Release the persistent Turbo process during orderly CIEL shutdown."""
        self.stop()
        worker, self._chatterbox = self._chatterbox, None
        if worker is not None:
            worker.close()
        if self._pocket is not None:
            self._pocket.close()
            self._pocket = None

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Clean markdown, code blocks, reasoning tags, emojis, and symbols for spoken speech."""
        text = strip_internal_reply_notes(text)
        # Strip reasoning / thinking tags <think>...</think> and <thought>...</thought>
        clean = re.sub(r"<(think|thought)>[\s\S]*?</\1>", "", text, flags=re.IGNORECASE)
        clean = re.sub(r"^<(think|thought)>[\s\S]*", "", clean, flags=re.IGNORECASE)
        # Remove code blocks
        clean = re.sub(r"```[\s\S]*?```", " [code omitted] ", clean)
        # Remove inline backticks
        clean = re.sub(r"`([^`]+)`", r"\1", clean)
        # Remove markdown links [text](url) -> text
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
        # Remove markdown headers (#, ##, etc.)
        clean = re.sub(r"^\s*#{1,6}\s+", "", clean, flags=re.MULTILINE)
        # Remove markdown bold/italics
        clean = re.sub(r"[*_~]{1,2}([^*_~]+)[*_~]{1,2}", r"\1", clean)
        # Remove remaining stray asterisks/underscores
        clean = re.sub(r"[*_~]", "", clean)
        # Remove bullets / list markers
        clean = re.sub(r"^\s*[-*+]\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"^\s*\d+\.\s+", "", clean, flags=re.MULTILINE)
        # Remove emojis (surrogate pairs and emoji blocks)
        clean = re.sub(r"[\U00010000-\U0010ffff]", "", clean)
        clean = strip_stage_directions(clean)
        # Collapse multiple spaces / newlines
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _synthesize_piper(
        self,
        text: str,
        voice: PiperVoice | None = None,
    ) -> tuple[np.ndarray, int] | None:
        """Synthesize using local Piper ONNX model."""
        if voice is None:
            if self._voice is None:
                self._load_voice()
            voice = self._voice
        if voice is None:
            return None

        syn_config = SynthesisConfig(
            length_scale=1.0 / max(0.2, min(self.speed, 3.0)),
        )

        audio_chunks: list[np.ndarray] = []
        aligned_chunks = []
        sample_rate = 22050

        for chunk in voice.synthesize(
            text, syn_config=syn_config,
            **({"include_alignments": True} if self.include_alignments else {}),
        ):
            sample_rate = chunk.sample_rate
            if chunk.audio_float_array is not None and chunk.audio_float_array.size > 0:
                audio_chunks.append(chunk.audio_float_array)
                aligned_chunks.append(chunk)

        if not audio_chunks:
            return None

        combined_audio = np.concatenate(audio_chunks)
        if self.include_alignments:
            timeline = character_timeline(
                text, aligned_chunks, sample_rate, combined_audio.size, phonemize=voice.phonemize,
            )
            self._synthesis_timing.result = combined_audio, timeline
        return combined_audio, sample_rate

    def synthesize(
        self, text: str, *, style: VoiceStyle | None = None,
    ) -> tuple[np.ndarray, int] | None:
        """Synthesize text to a 1D float32 audio numpy array and sample rate."""
        if not self.enabled:
            return None
        selected_style = select_voice_style(text, style)

        clean_text = SpeechEventLimiter(self.expressiveness).apply(self.clean_text_for_speech(text))
        if not clean_text:
            return None

        if self.engine == "chatterbox_turbo":
            return self._synthesize_chatterbox(clean_text, selected_style)
        clean_text = strip_speech_events(clean_text)
        if not clean_text:
            return None
        if self.engine == "pocket":
            return self._synthesize_pocket(clean_text, selected_style)
        self.last_engine = "piper"
        return self._synthesize_piper(clean_text)

    def begin_stream(self) -> int:
        """Start cumulative playback tracking for a sentence-streamed reply."""
        with self._playback_lock:
            self._playback_generation += 1
            self._pending_text = ""
            self._play_started_at = 0.0
            self._play_duration = 0.0
            self._stream_text = ""
            self._stream_spoken = []
            self._stream_played_seconds = 0.0
            return self._playback_generation

    def update_stream(self, generation: int, text: str) -> None:
        """Track known generated text without reviving a canceled stream."""
        with self._playback_lock:
            if generation == self._playback_generation:
                self._stream_text = strip_speech_events(self.clean_text_for_speech(text))

    def end_stream(self, generation: int) -> None:
        """Clear completed stream state; an old worker cannot clear a newer reply."""
        with self._playback_lock:
            if generation == self._playback_generation:
                self._stream_text = ""
                self._stream_spoken = None
                self._stream_played_seconds = 0.0

    def _synthesize_cancelable(
        self, text: str, cancel_event: threading.Event, generation: int,
        style: VoiceStyle | None = None,
    ) -> tuple[np.ndarray, int, CharacterTimeline] | None:
        """Let a canceled caller leave while a bounded synthesis job unwinds.

        Voice inference is serialized: an old Piper/Fish job cannot race a new
        job against the same voice object. Its late audio never reaches playback.
        """
        def current() -> bool:
            return not cancel_event.is_set() and generation == self._playback_generation

        while not self._synthesis_slots.acquire(timeout=0.02):
            if not current():
                return None
        result: queue.Queue = queue.Queue(maxsize=1)

        def synthesize() -> None:
            acquired = False
            try:
                while current():
                    acquired = self._synthesis_lock.acquire(timeout=0.02)
                    if acquired:
                        break
                if acquired and current():
                    self._synthesis_timing.cancelled = lambda: not current()
                    result.put(
                        self._prepare_playback(text) if style is None
                        else self._prepare_playback(text, style)
                    )
                else:
                    result.put(None)
            except Exception as err:
                result.put(err)
            finally:
                self._synthesis_timing.cancelled = lambda: False
                if acquired:
                    self._synthesis_lock.release()
                self._synthesis_slots.release()

        worker = threading.Thread(target=synthesize, name="ciel-synthesis", daemon=True)
        try:
            worker.start()
        except Exception:
            self._synthesis_slots.release()
            raise
        while current():
            try:
                audio = result.get(timeout=0.02)
                if isinstance(audio, Exception):
                    raise audio
                return audio
            except queue.Empty:
                continue
        return None

    def _prepare_playback(
        self, text: str, style: VoiceStyle | None = None,
    ) -> tuple[np.ndarray, int, CharacterTimeline] | None:
        """Keep timing attached to the exact audio from this synthesis job."""
        self._synthesis_timing.result = None
        result = self.synthesize(text) if style is None else self.synthesize(text, style=style)
        if result is None:
            return None
        audio, sample_rate = result
        aligned = self._synthesis_timing.result
        display_text = strip_speech_events(self.clean_text_for_speech(text))
        timeline = (
            aligned[1] if aligned is not None and aligned[0] is audio
            else estimated_timeline(display_text, audio.size / sample_rate)
        )
        return audio, sample_rate, timeline

    def prepare_sentence(
        self, text: str, cancel_event: threading.Event, generation: int,
    ) -> tuple[np.ndarray, int, CharacterTimeline] | None:
        """Synthesize a sentence ahead of playback, honoring the reply generation."""
        if not self.enabled or cancel_event.is_set():
            return None
        return self._synthesize_cancelable(text, cancel_event, generation)

    def _speak_streaming(
        self, text: str, generation: int, cancel_event: threading.Event | None,
        on_start: Callable[[], None] | None,
        on_progress: Callable[[str, bool, bool], None] | None,
        style: VoiceStyle | None,
    ) -> bool:
        """Play Pocket chunks immediately, with existing generation/interruption tracking."""
        stop = threading.Event()
        serial = None

        def current() -> bool:
            return (
                generation == self._playback_generation
                and not stop.is_set() and not (cancel_event and cancel_event.is_set())
                and (serial is None or serial == self._playback_serial)
            )

        while not self._synthesis_slots.acquire(timeout=0.02):
            if not current():
                return False
        playback = ChunkPlayback(
            stop, queue_size=max(4, self.audio_queue_size * 4),
            timeout=get_settings().audio.tts_synthesis_timeout_seconds,
        )
        with self._pocket_stops_lock:
            self._pocket_stops.add(stop)

        def produce() -> None:
            acquired = False
            try:
                while current():
                    acquired = self._synthesis_lock.acquire(timeout=0.02)
                    if acquired:
                        break
                if acquired and current():
                    self._synthesis_timing.cancelled = lambda: not current()
                    playback.produce(self.synthesize_stream(text, style=style, cancel_event=stop))
            finally:
                self._synthesis_timing.cancelled = lambda: False
                if acquired:
                    self._synthesis_lock.release()
                self._synthesis_slots.release()

        worker = threading.Thread(target=produce, name="ciel-pocket-audio", daemon=True)
        display = strip_speech_events(self.clean_text_for_speech(text))
        visible = 0
        started = False
        progress = on_progress

        def emit(count: int, finished: bool = False, interrupted: bool = False) -> None:
            nonlocal visible, progress
            if progress is None or serial != self._playback_serial:
                return
            visible = max(visible, count)
            try:
                progress(display[:visible], finished, interrupted)
            except Exception as err:
                logger.warning("Speech caption output failed: %s", err)
                progress = None

        def start() -> None:
            nonlocal serial, started
            with self._playback_lock:
                if not current():
                    stop.set()
                    return
                self._stop_event.clear()
                self._play_stopped_at = 0.0
                self._playback_serial += 1
                serial = self._playback_serial
                self._is_playing = True
                self._play_started_at = time.monotonic()
                self._play_duration = max(
                    len(display) / 12, playback.total_samples / playback.sample_rate,
                )
                started = True
            if on_start:
                on_start()

        def tick(played: float, generated: float) -> None:
            with self._playback_lock:
                if current() and serial == self._playback_serial:
                    self._play_duration = max(generated, len(display) / 12)
            # Streaming has no final duration/alignment yet. Reveal conservatively
            # from played samples; mark the complete sentence only after draining.
            emit(min(max(0, len(display) - 1), int(max(0, played - self.playback_latency) * 12)))

        played = False
        try:
            worker.start()
            played = playback.play(
                device=self.output_device, latency=self.playback_latency, current=current,
                on_start=start, on_tick=tick,
            )
            return played
        except Exception as err:
            logger.error("Streaming TTS failed: %s", err)
            return False
        finally:
            stop.set()
            if worker.ident is None:
                self._synthesis_slots.release()
            else:
                worker.join(timeout=1)
            with self._pocket_stops_lock:
                self._pocket_stops.discard(stop)
            if started:
                emit(len(display) if played else visible, True, not played)
            with self._playback_lock:
                if generation == self._playback_generation:
                    if serial is None or serial == self._playback_serial:
                        self._is_playing = False
                        if played and self._stream_spoken is not None:
                            self._stream_spoken.append(display)
                            self._stream_played_seconds += (
                                playback.played_samples / playback.sample_rate
                            )
                        self._pending_text = ""

    def speak(
        self,
        text: str,
        block: bool = True,
        *,
        cancel_event: threading.Event | None = None,
        generation: int | None = None,
        on_start: Callable[[], None] | None = None,
        on_progress: Callable[[str, bool, bool], None] | None = None,
        _prepared: tuple[np.ndarray, int, CharacterTimeline] | None = None,
        style: VoiceStyle | None = None,
    ) -> bool:
        """Synthesize and play speech audio through speakers."""
        if not self.enabled:
            return False
        select_voice_style(text, style)

        with self._playback_lock:
            if generation is None:
                generation = self._playback_generation
            if generation != self._playback_generation or (cancel_event and cancel_event.is_set()):
                return False
            self._pending_text = strip_speech_events(self.clean_text_for_speech(text))
            self._play_started_at = 0.0
            self._play_duration = 0.0
        if self._pocket_streaming and _prepared is None and (
            self.engine == "pocket"
            or (self.engine == "chatterbox_turbo" and self._chatterbox_failed)
        ):
            if block:
                return self._speak_streaming(
                    text, generation, cancel_event, on_start, on_progress, style,
                )
            threading.Thread(
                target=self._speak_streaming,
                args=(text, generation, cancel_event, on_start, on_progress, style),
                name="ciel-pocket-playback", daemon=True,
            ).start()
            return True
        if _prepared is not None:
            synth_result = _prepared
        elif cancel_event is not None:
            synth_result = self._synthesize_cancelable(text, cancel_event, generation, style)
        else:
            with self._synthesis_lock:
                synth_result = (
                    self._prepare_playback(text) if style is None
                    else self._prepare_playback(text, style)
                )
        if synth_result is None:
            with self._playback_lock:
                if generation == self._playback_generation:
                    self._pending_text = ""
            return False

        audio, sample_rate, timeline = synth_result

        try:
            with self._playback_lock:
                if (
                    generation != self._playback_generation
                    or (cancel_event and cancel_event.is_set())
                ):
                    return False
                self._stop_event.clear()
                self._play_stopped_at = 0.0
                self._playback_serial += 1
                serial = self._playback_serial
                self._is_playing = True
                self._play_started_at = time.monotonic()
                self._play_duration = audio.size / sample_rate
                # ALSA's automatic callback period can be under 9 ms for Turbo's
                # 24 kHz audio. Give Python workers enough scheduling headroom.
                sd.play(
                    audio, samplerate=sample_rate, device=self.output_device,
                    latency=self.playback_latency, blocksize=1024,
                )
                self._play_started_at = time.monotonic()
                started = self._play_started_at
            try:
                latency = float(getattr(sd.get_stream(), "latency", 0.0))
                latency = latency if np.isfinite(latency) and latency >= 0 else 0.0
            except Exception:
                latency = 0.0
            if on_start:
                on_start()

            def monitor() -> None:
                """Reveal audio-timed characters and leave a partial caption on interruption."""
                callback = on_progress
                visible = 0

                def emit(*, finished: bool = False, interrupted: bool = False) -> None:
                    nonlocal visible, callback
                    if callback is None or serial != self._playback_serial:
                        return
                    now = time.monotonic()
                    if interrupted and self._play_stopped_at:
                        now = min(now, self._play_stopped_at)
                    elapsed = max(0.0, now - started - latency)
                    count = (
                        len(timeline.text) if finished and not interrupted
                        else timeline.visible_count(elapsed)
                    )
                    try:
                        while visible < count:
                            if serial != self._playback_serial:
                                return
                            if not interrupted and (
                                generation != self._playback_generation
                                or (cancel_event is not None and cancel_event.is_set())
                            ):
                                return
                            visible += 1
                            callback(timeline.text[:visible], False, False)
                        if finished and serial == self._playback_serial:
                            callback(timeline.text[:visible], True, interrupted)
                    except Exception as err:
                        logger.warning("Speech caption output failed: %s", err)
                        callback = None

                interrupted = False
                try:
                    # Poll instead of sd.wait(), which can stall after Linux barge-in.
                    while serial == self._playback_serial:
                        if (
                            generation != self._playback_generation or self._stop_event.is_set()
                            or (cancel_event is not None and cancel_event.is_set())
                        ):
                            interrupted = True
                            if generation == self._playback_generation:
                                self.stop()
                            break
                        try:
                            active = sd.get_stream().active
                        except Exception:
                            active = False
                        if not active:
                            break
                        emit()
                        self._stop_event.wait(timeout=0.01 if callback is not None else 0.02)
                finally:
                    emit(finished=True, interrupted=interrupted)
                    underflow = False
                    with self._playback_lock:
                        if (
                            generation == self._playback_generation
                            and serial == self._playback_serial
                        ):
                            try:
                                status = sd.get_status()
                                underflow = bool(status and status.output_underflow)
                            except Exception:
                                # Some audio backends do not expose callback flags.
                                pass
                            self._is_playing = False
                            if self._stream_spoken is not None:
                                self._stream_spoken.append(self._pending_text)
                                self._stream_played_seconds += self._play_duration
                            self._pending_text = ""
                    if underflow:
                        logger.warning(
                            "TTS playback underrun: the output buffer ran dry "
                            "(requested latency %.3fs). Increase TTS_PLAYBACK_LATENCY "
                            "or reduce competing CPU load.", self.playback_latency,
                        )

            if block:
                monitor()
            else:
                threading.Thread(
                    target=monitor, name="ciel-playback-progress", daemon=True,
                ).start()
            return generation == self._playback_generation and serial == self._playback_serial
        except Exception as err:
            logger.error("Error playing TTS audio: %s", err)
            self._is_playing = False
            return False

    def stop(self) -> dict | None:
        """Immediately stop audio playback (barge-in support)."""
        with self._pocket_stops_lock:
            for event in self._pocket_stops:
                event.set()
        with self._playback_lock:
            self._play_stopped_at = time.monotonic()
            interruption = None
            if self._pending_text or self._stream_text:
                elapsed = (
                    max(0.0, time.monotonic() - self._play_started_at)
                    if self._pending_text and self._play_started_at else 0.0
                )
                duration = self._play_duration if self._pending_text else 0.0
                fraction = min(1.0, elapsed / duration) if duration else 0.0
                words = self._pending_text.split()
                heard = int(len(words) * fraction)
                spoken = " ".join([*(self._stream_spoken or []), " ".join(words[:heard])]).strip()
                full = self._stream_text or self._pending_text
                remaining = " ".join(full.split()[len(spoken.split()):])
                interruption = {
                    "full_text": full[:4000],
                    "estimated_spoken_text": spoken[:4000],
                    "remaining_text": remaining[:4000],
                    "played_seconds": round(self._stream_played_seconds + elapsed, 2),
                    "duration_seconds": round(self._stream_played_seconds + duration, 2),
                }
            self._pending_text = ""
            self._stream_text = ""
            self._stream_spoken = None
            self._stream_played_seconds = 0.0
            self._playback_generation += 1
            self._stop_event.set()
            was_playing = self._is_playing
            self._is_playing = False
        if not was_playing:
            return interruption
        try:
            sd.stop()
        except Exception as err:
            logger.debug("Error stopping sounddevice: %s", err)
        finally:
            self._is_playing = False
        return interruption

    def is_speaking(self) -> bool:
        """Check whether audio is currently playing."""
        return self._is_playing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CIEL Text-to-Speech")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! I am Ciel, your desktop companion.",
        help="Text to speak",
    )
    parser.add_argument("--engine", default=None, help="TTS engine: pocket, piper, chatterbox, hq")
    parser.add_argument("--voice", default=None, help="TTS voice name")
    parser.add_argument("--style", choices=["neutral", "warm", "amused", "firm", "focused", "soft"])
    parser.add_argument("--output", type=Path, help="Write a WAV instead of speaker playback")
    parser.add_argument("--require-engine", action="store_true", help="Fail if fallback was used")
    parser.add_argument(
        "--warmup-only", action="store_true", help="Load model and neutral reference",
    )
    args = parser.parse_args()

    app_settings = get_settings()
    selected_engine = args.engine or app_settings.audio.tts_engine
    selected_voice = args.voice or app_settings.audio.tts_voice

    tts = TextToSpeech(
        voice_name=selected_voice,
        engine=selected_engine,
        speed=app_settings.audio.tts_speed,
        output_device=app_settings.audio.output_device,
    )
    print(f"Synthesizing: '{args.text}' [Engine: {tts.engine}, Voice: {tts.voice_name}]")
    try:
        if args.warmup_only:
            success = tts.warmup()
        elif args.output:
            import soundfile as sf

            result = tts.synthesize(args.text, style=args.style)
            success = result is not None
            if result is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(args.output, result[0], result[1])
        else:
            success = tts.speak(args.text, block=True, style=args.style)
        if args.require_engine and not args.warmup_only:
            success = success and tts.last_engine == tts.engine
        print(f"Speech finished (success={success}, actual_engine={tts.last_engine}).")
    finally:
        tts.close()
    raise SystemExit(0 if success else 1)
