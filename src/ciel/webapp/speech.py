"""Origin-only browser speech with a bounded producer queue and single-tab claim."""

import queue
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ciel.audio.alignment import estimated_timeline
from ciel.audio.text_reply import spoken_text_reply
from ciel.logging import SecretMaskingFilter
from ciel.webapp.audio import encode_audio_wav
from ciel.webapp.requests import WebRequest

CAPTION_CHARS_PER_SECOND = 14.0
"""Speaking pace used to place captions when an engine streams several audio segments."""
ECHO_MARGIN_SECONDS = 0.8
"""After handed-out audio should have finished, keep treating Ciel as audible this long.

Covers network delivery, decoding, and device output latency, so a speaker's tail is not
picked up by any microphone, in this tab, another tab, or another device.
"""

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
            self._prune()
            if len(self._jobs) >= 8:
                raise ValueError("Finish or stop the current reply before requesting more audio.")
            handle = secrets.token_urlsafe(18)
            job = _Playback(
                request, text[:64000], current, _Cancellation(cancel_event, current), cancel_event
            )
            audible = job.text if full else spoken_text_reply(job.text)
            job.caption, job.message_id = SecretMaskingFilter._redact(audible), message_id
            self._jobs[handle] = job

        def produce() -> None:
            try:
                self._produce(job, audible)
            except Exception:
                job.error = "Audio is unavailable. You can read the answer below."
            finally:
                job.done = True

        threading.Thread(target=produce, name="ciel-browser-speech", daemon=True).start()
        return handle

    def _produce(self, job: _Playback, audible: str) -> None:
        """Queue one-second chunks, each labelled with the caption characters it speaks.

        Looking one segment ahead tells whether the current one is the last: a single
        segment (Turbo, Piper) spreads the whole caption over its audio; streamed segments
        advance at a speaking pace and the final one finishes the caption. Single-segment
        engines finish at once, so the lookahead only delays streamed engines by one segment.
        If synthesizing the next segment fails, the current one still plays first.
        """
        caption = job.caption
        segments = iter(self.tts.synthesize_stream(audible, cancel_event=job.canceled))
        current, produced, first = next(segments, None), 0.0, 0
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
            last = len(caption) if following is None else max(
                first, min(len(caption), round(produced * CAPTION_CHARS_PER_SECOND)))
            for chunk in _segment_chunks(audio, rate, caption, first, last):
                if not self._queue(job, chunk):
                    return
            if failure is not None:
                raise failure
            current, first = following, last

    def _queue(self, job: _Playback, chunk: AudioChunk) -> bool:
        """Wait for the browser to drain the bounded queue; give up after 30 seconds."""
        deadline = time.monotonic() + 30
        while self._valid(job):
            try:
                job.chunks.put(chunk, timeout=0.1)
                return True
            except queue.Full:
                if time.monotonic() > deadline:
                    job.canceled.set()
        return False

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

    def list(self, device_id: str, revision: int) -> list[dict]:
        """Advertise only this device's current, unexpired jobs; never old chat audio."""
        with self._lock:
            self._prune()
            return [
                {"id": h, "error": j.error, "excerpt": spoken_text_reply(j.text) != j.text,
                 "caption": j.caption, "message": j.message_id}
                for h, j in self._jobs.items()
                if j.request.device_id == device_id and j.request.revision == revision
            ]

    def chunk(
        self, handle: str, device_id: str, revision: int, client: str,
    ) -> AudioChunk | None:
        """Claim once across tabs and drain one chunk; None means synthesis is still pending."""
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
                return None
            # Browsers play chunks back to back, so audibility extends from the later of now
            # and the end of the audio already handed out.
            self._audible_until = max(self.clock(), self._audible_until) + chunk.seconds
            return chunk

    def replay(self, handle: str, device_id: str, revision: int) -> str:
        """Read the already generated full answer; never make another model request."""
        with self._lock:
            job = self._jobs.get(handle)
            if (
                job is None
                or not self._valid(job)
                or job.request.device_id != device_id
                or job.request.revision != revision
            ):
                raise ValueError("This answer is no longer available for playback.")
            return self.speak(
                job.text, job.request, cancel_event=job.parent, current=job.current, full=True,
                message_id=job.message_id,
            )

    def cancel(self) -> None:
        """Invalidate queued chunks and stop producers on Stop, switch, or shutdown."""
        with self._lock:
            for job in self._jobs.values():
                job.canceled.set()
            self._jobs.clear()
            # Stop silences the browser at once, so the mic is free immediately afterwards.
            self._audible_until = min(self._audible_until, self.clock() - ECHO_MARGIN_SECONDS)
