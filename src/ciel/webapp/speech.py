"""Origin-only browser speech with a bounded producer queue and single-tab claim."""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ciel.audio.alignment import estimated_timeline
from ciel.audio.text_reply import StreamingExcerpt, spoken_text_reply
from ciel.latency import TurnTrace, active_trace
from ciel.logging import SecretMaskingFilter, get_logger
from ciel.webapp.audio import encode_audio_wav
from ciel.webapp.requests import WebRequest

logger = get_logger("webapp.speech")

CAPTION_CHARS_PER_SECOND = 14.0
"""Speaking pace used to place captions when an engine streams several audio segments."""
ECHO_MARGIN_SECONDS = 0.8
"""After handed-out audio should have finished, keep treating Ciel as audible this long.

Covers network delivery, decoding, and device output latency, so a speaker's tail is not
picked up by any microphone, in this tab, another tab, or another device.
"""
MAX_JOBS = 8
_END = object()
_TIMED_STAGES = ("web_request_received", "stt_finished", "first_visible_text",
                 "reply_text_complete")
"""Turn stages reported beside each reply's first audio; never text or audio content."""


class _Cancellation(threading.Event):
    """Let the voice engine observe parent cancellation and revocation immediately."""

    def __init__(self, parent: threading.Event, current: Callable[[], bool]) -> None:
        super().__init__()
        self.parent, self.current = parent, current

    def is_set(self) -> bool:
        return super().is_set() or self.parent.is_set() or not self.current()


@dataclass
class _Playback:
    request: WebRequest
    text: str
    current: Callable[[], bool]
    canceled: threading.Event
    parent: threading.Event
    chunks: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=4))
    client: str = ""
    caption: str = ""
    message_id: int | None = None
    done: bool = False
    error: str = ""
    expires: float = field(default_factory=lambda: time.monotonic() + 120)
    excerpt: bool = False
    sentences: queue.Queue | None = None
    """A streamed reply's (text, caption start, caption end) items, ended by ``_END``."""
    closed: bool = True
    """Whether the complete written answer is known; streamed replies close at the end."""
    trace: TurnTrace | None = None
    opened: float = field(default_factory=time.monotonic)
    ready: float | None = None
    served: bool = False


class AudioChunk(bytes):
    """One WAV chunk with its length and the caption characters it speaks.

    ``caption_from``/``caption_to`` index the job's caption text, so the browser can
    reveal the words in step with playback, the same schedule terminal captions use.
    """

    seconds: float
    caption_from: int
    caption_to: int

    def __new__(cls, data: bytes, seconds: float, caption_from: int, caption_to: int):
        chunk = super().__new__(cls, data)
        chunk.seconds, chunk.caption_from, chunk.caption_to = seconds, caption_from, caption_to
        return chunk


def _browser_offset(text: str, index: int) -> int:
    """Convert a code-point index to the UTF-16 offset JavaScript strings use (emoji count 2)."""
    return len(text[:index].encode("utf-16-le")) // 2


def _segment_chunks(
    audio: Any, rate: int, caption: str, first: int, last: int,
) -> list[AudioChunk]:
    """Split one synthesized segment into one-second chunks covering caption[first:last]."""
    seconds = len(audio) / rate
    timeline = estimated_timeline(caption[first:last], seconds)
    chunks = []
    for start in range(0, len(audio), rate):
        samples = audio[start : start + rate]
        end = start + len(samples)
        reveal_from = first + timeline.visible_count(start / rate)
        reveal_to = last if end >= len(audio) else first + timeline.visible_count(end / rate)
        chunks.append(AudioChunk(encode_audio_wav(samples, rate), len(samples) / rate,
                                 _browser_offset(caption, reveal_from),
                                 _browser_offset(caption, reveal_to)))
    return chunks


class BrowserSpeech:
    """Synthesize through CIEL's configured voice, without opening PC audio output."""

    def __init__(self, tts: Any, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.tts = tts
        self.clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, _Playback] = {}
        self._audible_until = 0.0
        self._epoch = 0

    def audible(self) -> bool:
        """Whether some browser is, or may still be, playing Ciel's voice."""
        with self._lock:
            return self.clock() < self._audible_until + ECHO_MARGIN_SECONDS

    def overlaps_playback(self, clip_seconds: float) -> bool:
        """Whether a just-finished recording of this length began while Ciel was audible."""
        with self._lock:
            started = self.clock() - max(0.0, clip_seconds)
            return started < self._audible_until + ECHO_MARGIN_SECONDS

    def speak(
        self,
        text: str,
        request: WebRequest,
        *,
        cancel_event: threading.Event,
        current: Callable[[], bool],
        full: bool = False,
        message_id: int | None = None,
    ) -> str:
        """Start bounded synthesis; playback success is acknowledged separately by the browser."""
        with self._lock:
            handle, job = self._open(request, text[:64000], cancel_event, current,
                                     message_id=message_id)
            excerpt = spoken_text_reply(job.text)
            audible = job.text if full else excerpt
            job.caption = SecretMaskingFilter._redact(audible)
            job.excerpt = excerpt != job.text
        self._start(job, self._produce, audible)
        return handle

    def stream(
        self, request: WebRequest, *, cancel_event: threading.Event, current: Callable[[], bool],
    ) -> SpeechStream:
        """Prepare a streaming reply's voice; it starts with the first finished sentence."""
        with self._lock:
            epoch = self._epoch
        return SpeechStream(self, request, cancel_event, current, epoch, trace=active_trace.get())

    def _open(
        self, request: WebRequest, text: str, cancel_event: threading.Event,
        current: Callable[[], bool], *, message_id: int | None = None, streamed: bool = False,
        trace: TurnTrace | None = None,
    ) -> tuple[str, _Playback]:
        """Register a job under the lock; the caller starts its producer."""
        with self._lock:
            self._prune()
            if len(self._jobs) >= MAX_JOBS:
                self._drop_oldest_finished()
            if len(self._jobs) >= MAX_JOBS:
                raise ValueError("Finish or stop the current reply before requesting more audio.")
            handle = secrets.token_urlsafe(18)
            job = _Playback(
                request, text, current, _Cancellation(cancel_event, current), cancel_event,
                message_id=message_id, trace=trace if streamed else active_trace.get(),
            )
            if streamed:
                job.sentences, job.closed = queue.Queue(), False
            self._jobs[handle] = job
        if job.trace is not None:
            job.trace.mark("browser_speech_started", streamed=streamed)
        return handle, job

    def _start(self, job: _Playback, target: Callable[..., None], *args: Any) -> None:
        def produce() -> None:
            try:
                target(job, *args)
            except Exception:
                job.error = "Audio is unavailable. You can read the answer below."
            finally:
                job.done = True

        threading.Thread(target=produce, name="ciel-browser-speech", daemon=True).start()

    def _produce(self, job: _Playback, audible: str) -> None:
        """Speak a complete reply, its caption revealed across all of its audio."""
        self._speak_text(job, audible, 0, len(job.caption))

    def _produce_stream(self, job: _Playback) -> None:
        """Speak a streamed reply's sentences as they arrive, until it ends or stops.

        A new sentence wakes the producer at once; the timeout only paces validity checks,
        which consult the conversation store, while it waits for the reply to continue.
        """
        while self._valid(job):
            try:
                item = job.sentences.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is _END:
                return
            text, start, end = item
            if not self._speak_text(job, text, start, end):
                return

    def _speak_text(self, job: _Playback, text: str, start: int, end: int) -> bool:
        """Queue one-second chunks of ``text``, each labelled with the caption it speaks.

        The chunks reveal caption[start:end]. Looking one segment ahead tells whether the
        current one is the last: a single segment (Turbo, Piper) spreads the whole range over
        its audio; streamed segments advance at a speaking pace and the final one finishes
        the range. If synthesizing the next segment fails, the current one still plays first.
        Returns False once the job stopped.
        """
        caption = job.caption
        segments = iter(self.tts.synthesize_stream(text, cancel_event=job.canceled))
        current, produced, first = next(segments, None), 0.0, start
        while current is not None:
            failure: Exception | None = None
            try:
                following = next(segments, None)
            except Exception as error:  # noqa: BLE001 - re-raised after queuing current audio
                following, failure = None, error
            audio, rate = current
            if not 8000 <= rate <= 96000:
                raise ValueError("Unsupported voice sample rate.")
            produced += len(audio) / rate
            last = end if following is None else max(
                first, min(end, start + round(produced * CAPTION_CHARS_PER_SECOND)))
            for chunk in _segment_chunks(audio, rate, caption, first, last):
                if not self._queue(job, chunk):
                    return False
            if failure is not None:
                raise failure
            current, first = following, last
        return True

    def _queue(self, job: _Playback, chunk: AudioChunk) -> bool:
        """Wait for the browser to drain the bounded queue; give up after 30 seconds."""
        deadline = time.monotonic() + 30
        while self._valid(job):
            try:
                job.chunks.put(chunk, timeout=0.1)
            except queue.Full:
                if time.monotonic() > deadline:
                    job.canceled.set()
                continue
            if job.ready is None:
                job.ready = time.monotonic()
                if job.trace is not None:
                    job.trace.mark("browser_first_chunk_ready")
            return True
        return False

    def _add(self, job: _Playback, sentences: list[str]) -> None:
        """Append streamed sentences to the caption and the synthesis queue, in order."""
        with self._lock:
            for sentence in sentences:
                spoken = SecretMaskingFilter._redact(sentence)
                start = len(job.caption) + (1 if job.caption else 0)
                job.caption = f"{job.caption} {spoken}" if job.caption else spoken
                job.sentences.put((spoken, start, len(job.caption)))

    def _close(self, job: _Playback, text: str) -> None:
        """Record a streamed reply's complete answer for Read full answer."""
        with self._lock:
            job.text = text[:64000]
            job.excerpt = spoken_text_reply(job.text) != job.text
            job.closed = True

    @staticmethod
    def _valid(job: _Playback) -> bool:
        valid = (
            not job.canceled.is_set()
            and not job.parent.is_set()
            and job.expires > time.monotonic()
            and job.current()
        )
        if not valid:
            job.canceled.set()
        return valid

    def _prune(self) -> None:
        self._jobs = {h: j for h, j in self._jobs.items() if self._valid(j)}

    def _drop_oldest_finished(self) -> None:
        """Free one slot from a reply whose audio was fully handed out. Caller holds the lock.

        Finished replies stay for replay until they expire, so without this a hands-free
        session would lose its voice after MAX_JOBS replies within that window.
        """
        for handle, job in self._jobs.items():  # Insertion order: oldest first.
            if job.done and job.chunks.empty():
                del self._jobs[handle]
                return

    def list(self, device_id: str, revision: int) -> list[dict]:
        """Advertise only this device's current, unexpired jobs; never old chat audio."""
        with self._lock:
            self._prune()
            return [
                {"id": h, "error": j.error, "excerpt": j.excerpt,
                 "caption": j.caption, "message": j.message_id}
                for h, j in self._jobs.items()
                if j.request.device_id == device_id and j.request.revision == revision
            ]

    def chunk(
        self, handle: str, device_id: str, revision: int, client: str, *, wait: float = 0.0,
    ) -> AudioChunk | None:
        """Claim once across tabs and drain one chunk; None means synthesis is still pending.

        ``wait`` holds the request briefly for a chunk that is nearly ready, so the browser
        need not sleep and ask again (saving most of its retry delay before the first word).
        """
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            with self._lock:
                job = self._jobs.get(handle)
                if (
                    job is None
                    or not self._valid(job)
                    or job.request.device_id != device_id
                    or job.request.revision != revision
                    or job.client
                    and job.client != client
                ):
                    raise ValueError("This playback is no longer available.")
                job.client = client
                try:
                    chunk = job.chunks.get_nowait()
                except queue.Empty:
                    if job.done:
                        raise EOFError(job.error) from None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    chunk = None
                else:
                    break
            # Wait outside the lock, in short steps so Stop or a switch ends it quickly.
            try:
                chunk = job.chunks.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            with self._lock:
                if handle in self._jobs and self._valid(job):
                    break
            raise ValueError("This playback is no longer available.")
        with self._lock:
            # Browsers play chunks back to back, so audibility extends from the later of now
            # and the end of the audio already handed out.
            self._audible_until = max(self.clock(), self._audible_until) + chunk.seconds
            first, job.served = not job.served, True
        if first:
            self._log_first_audio(job)
        return chunk

    @staticmethod
    def _log_first_audio(job: _Playback) -> None:
        """Report when a reply's first audio reached the browser, relative to its request."""
        trace = job.trace
        if trace is None:
            return
        trace.mark("browser_first_chunk_served")
        served = time.monotonic()
        events = trace.report()["events"]
        timing: dict[str, Any] = {
            stage: round(events[stage]["seconds"], 3) for stage in _TIMED_STAGES if stage in events
        }
        timing.update({
            "speech_started": round(job.opened - trace.origin, 3),
            "first_chunk_ready": round((job.ready or served) - trace.origin, 3),
            "first_chunk_served": round(served - trace.origin, 3),
        })
        logger.info(
            "Browser reply audio (%s, turn %s): %s",
            "streamed" if job.sentences is not None else "complete reply",
            trace.turn_id, json.dumps(timing, sort_keys=True),
        )

    def replay(self, handle: str, device_id: str, revision: int) -> str:
        """Read the already generated full answer; never make another model request."""
        with self._lock:
            job = self._jobs.get(handle)
            if (
                job is None
                or not self._valid(job)
                or not job.closed
                or job.request.device_id != device_id
                or job.request.revision != revision
            ):
                raise ValueError("This answer is no longer available for playback.")
            return self.speak(
                job.text, job.request, cancel_event=job.parent, current=job.current, full=True,
                message_id=job.message_id,
            )

    def abandon(self, request: WebRequest) -> None:
        """Silence a streamed reply whose turn ended without completing its answer."""
        with self._lock:
            for handle, job in list(self._jobs.items()):
                if job.sentences is not None and not job.closed and job.request == request:
                    job.canceled.set()
                    del self._jobs[handle]

    def cancel(self) -> None:
        """Invalidate queued chunks and stop producers on Stop, switch, or shutdown."""
        with self._lock:
            self._epoch += 1
            for job in self._jobs.values():
                job.canceled.set()
            self._jobs.clear()
            # Stop silences the browser at once, so the mic is free immediately afterwards.
            self._audible_until = min(self._audible_until, self.clock() - ECHO_MARGIN_SECONDS)


class SpeechStream:
    """Speak one streaming browser reply sentence by sentence, then close it.

    Feed it the reply's cumulative masked text as it streams. Its first finished sentence
    opens the reply's audio while the rest is still being generated; ``finish`` speaks the
    remainder of the same excerpt the complete reply would have had. Stop, a conversation
    switch, or the end of the turn silences it, and no later text reopens it.
    """

    def __init__(
        self, speech: BrowserSpeech, request: WebRequest, cancel_event: threading.Event,
        current: Callable[[], bool], epoch: int, *, trace: TurnTrace | None,
    ) -> None:
        self._speech = speech
        self._request = request
        self._cancel_event = cancel_event
        self._current = current
        self._epoch = epoch
        self._trace = trace
        self._excerpt = StreamingExcerpt()
        self._lock = threading.Lock()
        self._handle: str | None = None
        self._job: _Playback | None = None
        self._error: ValueError | None = None
        self._ended = False
        self._closed = False

    def _live(self, *, check_current: bool = True) -> bool:
        return (
            not self._cancel_event.is_set()
            and self._speech._epoch == self._epoch
            and (self._job is None or not self._job.canceled.is_set())
            and (not check_current or self._current())
        )

    def update(self, text: str) -> None:
        """Speak newly finished sentences of the cumulative reply; never raises.

        Callers feed text only while the turn is current; audio re-checks that itself.
        """
        with self._lock:
            if (self._closed or self._error is not None
                    or not self._live(check_current=False)):
                return
            try:
                sentences = self._excerpt.update(text)
                if sentences:
                    self._send(sentences)
                if self._excerpt.settled:
                    self._end_audio()
            except ValueError as error:
                self._error = error
            except Exception:  # noqa: BLE001 - speech must never interrupt the written reply
                logger.debug("Streamed reply speech stopped.", exc_info=True)
                self._error = ValueError("Audio is unavailable.")

    def finish(self, text: str) -> str | None:
        """Close with the complete reply; return its audio handle, or None when silent.

        Raises ValueError when no audio could be prepared, like ``BrowserSpeech.speak``.
        """
        with self._lock:
            if self._closed:
                return self._handle
            self._closed = True
            if not self._live():
                self._discard()
                return None
            if self._error is not None:
                self._discard()
                raise self._error
            remainder = self._excerpt.finish(SecretMaskingFilter._redact(text))
            if remainder:
                self._send(remainder)
            if self._job is None:
                return None
            self._end_audio()
            self._speech._close(self._job, text)
            return self._handle

    def abandon(self) -> None:
        """Silence the reply if it never finished; a finished reply keeps playing."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._discard()

    def _send(self, sentences: list[str]) -> None:
        if self._job is None:
            self._handle, self._job = self._speech._open(
                self._request, "", self._cancel_event, self._current, streamed=True,
                trace=self._trace,
            )
            self._speech._start(self._job, self._speech._produce_stream)
        self._speech._add(self._job, sentences)

    def _end_audio(self) -> None:
        if self._job is not None and not self._ended:
            self._ended = True
            self._job.sentences.put(_END)

    def _discard(self) -> None:
        if self._job is not None:
            self._job.canceled.set()
            with self._speech._lock:
                self._speech._jobs.pop(self._handle, None)
