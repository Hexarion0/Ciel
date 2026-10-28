"""Speak sentence-sized replies while provider tokens continue arriving."""

import queue
import re
import threading
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass

from ciel.audio.speech_events import SpeechEventLimiter, is_stage_direction, strip_speech_events
from ciel.conversation import INTERNAL_REPLY_NOTES
from ciel.latency import active_trace, mark
from ciel.logging import get_logger
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk

logger = get_logger("audio.streaming")
_READERS = threading.BoundedSemaphore(2)
_TAGS = ("think", "thought", "reasoning", "reflection")
_TOKENS = tuple(f"<{tag}>" for tag in _TAGS) + tuple(f"</{tag}>" for tag in _TAGS)
_STATUS_TOKENS = tuple(note.lower() for note in INTERNAL_REPLY_NOTES)


class VisibleText:
    """Discard reasoning and fenced code even when delimiters cross token boundaries."""

    def __init__(self, *, omit_code: bool = True) -> None:
        self.pending = ""
        self.hidden: list[str] = []
        self.code = False
        self.omit_code = omit_code

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return only newly available visible text."""
        self.pending += text
        output = []
        while self.pending:
            lowered = self.pending.lower()
            tokens = ("```",) if self.code else (*_TOKENS, *_STATUS_TOKENS, "```")
            token = next((item for item in tokens if lowered.startswith(item)), None)
            if token:
                self.pending = self.pending[len(token):]
                if token in _STATUS_TOKENS:
                    continue
                if token == "```":
                    self.code = not self.code
                    if not self.hidden:
                        if not self.omit_code:
                            output.append(token)
                        elif not self.code:
                            output.append(" [code omitted] ")
                elif token.startswith("</"):
                    if self.hidden and self.hidden[-1] == token[2:-1]:
                        self.hidden.pop()
                else:
                    self.hidden.append(token[1:-1])
                continue
            if any(item.startswith(lowered) for item in tokens):
                if final:
                    self.pending = ""
                break
            if not self.hidden and not self.code and self.pending.startswith("["):
                end = self.pending.find("]")
                if end < 0:
                    if not final:
                        break
                    if is_stage_direction(self.pending[1:]):
                        self.pending = ""
                        break
                elif is_stage_direction(self.pending[1:end]):
                    self.pending = self.pending[end + 1:]
                    continue
            char, self.pending = self.pending[0], self.pending[1:]
            if not self.hidden and (not self.code or not self.omit_code):
                output.append(char)
        return "".join(output)


class SentenceBuffer:
    """Keep decimals and common abbreviations together; bound unpunctuated chunks."""

    def __init__(self, max_chars: int = 240) -> None:
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, text: str, *, final: bool = False) -> list[str]:
        """Emit complete sentences, or the final unfinished sentence."""
        self.pending += text
        output = []
        while self.pending:
            end = None
            for match in re.finditer(r'[.!?]["\u201d\u2019\x27]*(?=\s)|\n', self.pending):
                prefix = self.pending[:match.end()]
                if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|e\.g|i\.e)\.$", prefix, re.I):
                    continue
                if re.fullmatch(r"\s*\d+\.", prefix):
                    continue
                end = match.end()
                break
            if end is None and len(self.pending) > self.max_chars:
                end = self.pending.rfind(" ", 0, self.max_chars)
                if end <= 0:
                    end = self.max_chars
            if end is None:
                if not final:
                    break
                end = len(self.pending)
            segment, self.pending = self.pending[:end].strip(), self.pending[end:].lstrip()
            if segment:
                output.append(segment)
        return output


@dataclass
class StreamedReply:
    """Generated text and playback outcome, including a canceled partial reply."""

    response: LLMResponse
    canceled: bool = False
    spoken: bool = False
    first_token_seconds: float | None = None
    first_audio_seconds: float | None = None
    total_seconds: float = 0.0
    error: Exception | None = None
    speech_pipeline: dict | None = None


class _SentencePipeline:
    """Bounded sentence synthesis queue and sequential audio playback for fast TTS."""

    _END = object()

    def __init__(
        self, tts, generation: int, current: Callable[[], bool],
        cancel_event: threading.Event | None, started: float,
        on_audio_started: Callable[[str], None],
        on_progress: Callable[[str, bool, bool], None] | None,
    ) -> None:
        self.tts = tts
        self.generation = generation
        self.current = current
        self.cancel_event = cancel_event
        self.started = started
        self.on_audio_started = on_audio_started
        self.on_progress = on_progress
        self.trace = active_trace.get()
        self.cancelled = threading.Event()
        limit = max(1, int(getattr(tts, "audio_queue_size", 2)))
        self.sentences: queue.Queue = queue.Queue(maxsize=1)
        self.audio: queue.Queue = queue.Queue(maxsize=limit)
        self.metrics: dict = {
            "sentences": [], "max_audio_queue_depth": 0, "playback_gaps": [], "errors": []
        }
        self._metrics_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self.spoken = False
        self.errors: list[Exception] = []
        self._previous_play_end: float | None = None
        self._synthesizer = threading.Thread(
            target=self._synthesize, name="ciel-tts-prefetch", daemon=True
        )
        self._player = threading.Thread(
            target=self._play, name="ciel-audio-queue", daemon=True
        )
        self._synthesizer.start()
        self._player.start()

    def _current(self) -> bool:
        return not self.cancelled.is_set() and self.current() and not (
            self.cancel_event is not None and self.cancel_event.is_set()
        )

    def submit(self, text: str, available: float) -> bool:
        """Apply backpressure instead of retaining a full long response in memory."""
        while self._current():
            try:
                self.sentences.put((text, available), timeout=0.02)
                return True
            except queue.Full:
                continue
        return False

    def _synthesize(self) -> None:
        try:
            while self._current():
                try:
                    item = self.sentences.get(timeout=0.03)
                except queue.Empty:
                    continue
                if item is self._END:
                    break
                text, available = item
                row = {"text": text, "text_available_seconds": available}
                started = time.monotonic()
                row["synthesis_started_seconds"] = started - self.started
                if self.trace is not None:
                    self.trace.mark("tts_synthesis_started")
                try:
                    prepared = self.tts.prepare_sentence(text, self.cancelled, self.generation)
                except Exception as err:
                    self.errors.append(err)
                    self.metrics["errors"].append(str(err))
                    logger.exception("Sentence synthesis failed")
                    continue
                finished = time.monotonic()
                if self.trace is not None:
                    self.trace.mark("tts_synthesis_finished")
                row["synthesis_seconds"] = finished - started
                row["synthesis_finished_seconds"] = finished - self.started
                if prepared is None or not self._current():
                    with self._metrics_lock:
                        self.metrics["sentences"].append(row)
                    continue
                message = (text, prepared, row)
                while self._current():
                    try:
                        self.audio.put(message, timeout=0.02)
                        with self._metrics_lock:
                            self.metrics["max_audio_queue_depth"] = max(
                                self.metrics["max_audio_queue_depth"], self.audio.qsize()
                            )
                            self.metrics["sentences"].append(row)
                        break
                    except queue.Full:
                        continue
        finally:
            while self._current():
                try:
                    self.audio.put(self._END, timeout=0.02)
                    break
                except queue.Full:
                    continue

    def _play(self) -> None:
        while self._current():
            try:
                item = self.audio.get(timeout=0.03)
            except queue.Empty:
                continue
            if item is self._END:
                return
            text, prepared, row = item
            if not self._current():
                return
            controls = {}
            if self.on_progress is not None:
                def progress(visible: str, finished: bool, interrupted: bool) -> None:
                    if self._current() or finished or interrupted:
                        self.on_progress(visible, finished, interrupted)

                controls["on_progress"] = progress

            def started() -> None:
                playback_start = time.monotonic()
                if self.trace is not None:
                    self.trace.mark("first_audio_playback")
                row["playback_started_seconds"] = playback_start - self.started
                if self._previous_play_end is not None:
                    gap = max(0.0, playback_start - self._previous_play_end)
                    row["gap_before_seconds"] = gap
                    self.metrics["playback_gaps"].append(gap)
                self.on_audio_started(text)

            try:
                played = self.tts.speak(
                    text,
                    block=True,
                    cancel_event=self.cancelled,
                    generation=self.generation,
                    on_start=started,
                    _prepared=prepared,
                    **controls,
                )
                with self._result_lock:
                    self.spoken |= bool(played)
            except Exception as err:
                self.errors.append(err)
                self.metrics["errors"].append(str(err))
                logger.exception("Queued sentence playback failed")
            self._previous_play_end = time.monotonic()

    def finish(self, *, cancel: bool) -> None:
        """Drain valid audio or discard every stale sentence after barge-in."""
        deadline = time.monotonic() + (0.1 if cancel else 180)

        def discard_queued() -> None:
            self.cancelled.set()
            for target in [self.sentences, self.audio]:
                while True:
                    try:
                        target.get_nowait()
                    except queue.Empty:
                        break

        def join_worker(worker: threading.Thread) -> bool:
            nonlocal deadline
            while worker.is_alive():
                now = time.monotonic()
                if not self._current():
                    discard_queued()
                    deadline = min(deadline, now + 0.1)
                remaining = deadline - now
                if remaining <= 0:
                    return False
                worker.join(timeout=min(0.02, remaining))
            return True

        if cancel:
            discard_queued()
        else:
            while self._current():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    discard_queued()
                    raise TimeoutError("Queued CIEL synthesis did not finish")
                try:
                    self.sentences.put(self._END, timeout=min(0.02, remaining))
                    break
                except queue.Full:
                    continue
            if not self._current():
                discard_queued()
        synthesized = join_worker(self._synthesizer)
        if not synthesized and self._current():
            discard_queued()
            raise TimeoutError("Queued CIEL synthesis did not finish")
        if not self._current():
            discard_queued()
        played = join_worker(self._player)
        if not played and self._current():
            discard_queued()
            raise TimeoutError("Queued CIEL audio did not finish playing")


def stream_reply(
    router,
    messages: list[ChatMessage],
    tts,
    current: Callable[[], bool],
    *,
    max_tokens: int = 400,
    cancel_event: threading.Event | None = None,
    on_sentence_start: Callable[[str], None] | None = None,
    on_progress: Callable[[str, bool, bool], None] | None = None,
) -> StreamedReply:
    """Read tokens concurrently with playback and return promptly on interruption.

    ``on_sentence_start`` receives intended sentence text after playback starts;
    it does not provide word alignment or verify synthesized pronunciation.

    Two bounded readers cap resource use if a provider cannot immediately abort
    a blocked read. Every reader owns and closes its own generator.
    """
    def still_current() -> bool:
        return current() and not (cancel_event is not None and cancel_event.is_set())

    started = time.monotonic()
    trace = active_trace.get()
    mark("reply_stream_started")
    result = StreamedReply(LLMResponse("", "", ""))
    if not still_current():
        result.canceled = True
        return result
    if not _READERS.acquire(blocking=False):
        raise RuntimeError("Previous provider streams are still closing; try again shortly")
    halted = threading.Event()
    events: queue.Queue = queue.Queue(maxsize=32)
    end = object()
    generation_elapsed: list[float | None] = [None]

    def offer(item) -> None:
        while not halted.is_set():
            try:
                events.put(item, timeout=0.02)
                return
            except queue.Full:
                continue

    def read() -> None:
        iterator = None
        try:
            iterator = router.stream(
                messages, temperature=0.7, max_tokens=max_tokens, cancel_event=halted,
            )
            for chunk in iterator:
                if halted.is_set():
                    break
                offer(chunk)
        except Exception as err:
            offer(err)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close:
                    close()
            finally:
                generation_elapsed[0] = time.monotonic() - started
                offer(end)
                _READERS.release()

    context = copy_context()
    worker = threading.Thread(
        target=lambda: context.run(read), name="ciel-reply-stream", daemon=True
    )
    try:
        generation = tts.begin_stream()
        worker.start()
    except Exception:
        _READERS.release()
        raise
    visible, audible, sentences = VisibleText(omit_code=False), VisibleText(), SentenceBuffer()
    speech_text = ""
    pipeline: _SentencePipeline | None = None
    expressiveness = getattr(tts, "expressiveness", "expressive")
    if not isinstance(expressiveness, str):
        expressiveness = "expressive"
    event_limiter = SpeechEventLimiter(expressiveness)

    def audio_started(sentence: str) -> None:
        if result.first_audio_seconds is None:
            if trace is not None:
                trace.mark("first_audio_playback")
            result.first_audio_seconds = time.monotonic() - started
            logger.info("Reply audio started after %.2fs.", result.first_audio_seconds)
        if on_sentence_start is not None and still_current():
            on_sentence_start(sentence)

    if getattr(tts, "supports_sentence_pipeline", False):
        pipeline = _SentencePipeline(
            tts, generation, still_current, cancel_event, started, audio_started, on_progress
        )

    def speak(parts: list[str]) -> None:
        parts = [clean for part in parts if (clean := event_limiter.apply(part))]
        if parts:
            mark("first_tts_chunk", chars=len(parts[0]))
        if pipeline is not None:
            available = time.monotonic() - started
            for part in parts:
                if not pipeline.submit(part, available):
                    return
            return
        for part in parts:
            if not still_current():
                return
            controls = {}
            if on_progress is not None:
                def progress(visible: str, finished: bool, interrupted: bool) -> None:
                    if still_current() or finished or interrupted:
                        on_progress(visible, finished, interrupted)

                controls["on_progress"] = progress
            played = tts.speak(
                part, block=True, cancel_event=cancel_event or halted, generation=generation,
                on_start=lambda sentence=part: audio_started(sentence),
                **controls,
            )
            result.spoken |= bool(played)

    try:
        while still_current():
            try:
                item = events.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is end or isinstance(item, Exception):
                result.error = item if isinstance(item, Exception) else None
                tail = visible.feed("", final=True)
                result.response.content += tail
                speech_tail = audible.feed(tail, final=True)
                speech_text += speech_tail
                tts.update_stream(generation, speech_text)
                speak(sentences.feed(speech_tail, final=True))
                break
            if not isinstance(item, LLMStreamChunk):
                raise TypeError("Provider emitted an invalid stream chunk")
            result.response.provider, result.response.model = item.provider, item.model
            delta = visible.feed(item.delta)
            if delta and result.first_token_seconds is None:
                mark("first_visible_text")
                result.first_token_seconds = time.monotonic() - started
                logger.info("First reply text arrived after %.2fs.", result.first_token_seconds)
            result.response.content += delta
            speech_delta = audible.feed(delta)
            if speech_delta.strip():
                mark("first_usable_spoken_text")
            speech_text += speech_delta
            tts.update_stream(generation, speech_text)
            speak(sentences.feed(speech_delta))
        if pipeline is not None:
            was_canceled = not still_current()
            if was_canceled:
                halted.set()
                tts.stop()
            pipeline.finish(cancel=was_canceled)
            result.spoken = pipeline.spoken
            result.speech_pipeline = pipeline.metrics
            if pipeline.errors:
                logger.warning("Speech pipeline had %d errors.", len(pipeline.errors))
        result.canceled = not still_current()
        if result.canceled and pipeline is None:
            # stop() also invalidates synthesis that has not reached playback yet.
            halted.set()
            tts.stop()
        result.response.content = strip_speech_events(result.response.content)
        result.total_seconds = time.monotonic() - started
        result.response.latency = generation_elapsed[0] or result.total_seconds
        logger.info(
            "Reply timing: generation=%.2fs, total=%.2fs, canceled=%s.",
            result.response.latency, result.total_seconds, result.canceled,
        )
        if result.error and not result.response.content and not result.canceled:
            raise result.error
        if not result.response.content.strip() and not result.canceled:
            raise RuntimeError("Provider stream ended without a visible reply")
        return result
    finally:
        if pipeline is not None and (
            pipeline._synthesizer.is_alive() or pipeline._player.is_alive()
        ):
            pipeline.finish(cancel=True)
        halted.set()
        tts.end_stream(generation)
        worker.join(timeout=0.1)
