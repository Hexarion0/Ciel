"""Stream visible text without constructing speech or audio services."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

from ciel.audio.streaming import VisibleText
from ciel.providers.base import ChatMessage, LLMResponse, LLMStreamChunk

_READERS = threading.BoundedSemaphore(2)


@dataclass
class TextStreamReply:
    """Visible answer and completion outcome; canceled replies must not become history."""

    response: LLMResponse
    canceled: bool = False
    error: Exception | None = None
    published: bool = False


class _Cancellation(threading.Event):
    """Combine reader shutdown, caller cancellation, and current request ownership."""

    def __init__(
        self, parent: threading.Event | None, current: Callable[[], bool],
    ) -> None:
        super().__init__()
        self.parent, self.current = parent, current

    def is_set(self) -> bool:
        return (super().is_set() or (self.parent is not None and self.parent.is_set())
                or not self.current())

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            super().wait(0.02 if remaining is None else min(remaining, 0.02))
        return self.is_set()


def stream_text_reply(
    router: Any,
    messages: list[ChatMessage],
    current: Callable[[], bool],
    *,
    on_delta: Callable[[str], None] | None = None,
    max_tokens: int = 4096,
    cancel_event: threading.Event | None = None,
) -> TextStreamReply:
    """Publish visible deltas from one bounded reader, retaining code and partial answers.

    The caller owns history and display. Check ``canceled`` before saving the response.
    A reader closes its own iterator, so a provider blocked in a read cannot delay a
    canceled turn. At most two unfinished provider readers are allowed.
    """
    started = time.monotonic()
    result = TextStreamReply(LLMResponse("", "", ""))
    halted = _Cancellation(cancel_event, current)
    if halted.is_set():
        result.canceled = True
        return result
    if not _READERS.acquire(blocking=False):
        raise RuntimeError("Previous provider streams are still closing; try again shortly")
    events: queue.Queue = queue.Queue(maxsize=32)
    end = object()
    generation_elapsed: list[float | None] = [None]

    def offer(item: Any) -> None:
        while not halted.is_set():
            try:
                events.put(item, timeout=0.02)
                return
            except queue.Full:
                continue

    def read() -> None:
        iterator = None
        try:
            if halted.is_set():
                return
            iterator = router.stream(
                messages, temperature=0.7, max_tokens=max_tokens, cancel_event=halted,
            )
            for chunk in iterator:
                if halted.is_set():
                    break
                offer(chunk)
        except Exception as error:
            offer(error)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except Exception as error:
                offer(error)
            finally:
                generation_elapsed[0] = time.monotonic() - started
                offer(end)
                _READERS.release()

    context = copy_context()
    worker = threading.Thread(
        target=lambda: context.run(read), name="ciel-text-stream", daemon=True,
    )
    try:
        worker.start()
    except Exception:
        _READERS.release()
        raise
    visible = VisibleText(omit_code=False)

    def publish(delta: str) -> None:
        if not delta or halted.is_set():
            return
        result.response.content += delta
        if on_delta is not None and not halted.is_set():
            result.published = True
            on_delta(delta)
        # A callback may reset the conversation or cancel the request.
        if halted.is_set():
            result.canceled = True

    try:
        while not halted.is_set():
            try:
                item = events.get(timeout=0.02)
            except queue.Empty:
                continue
            if halted.is_set():
                break
            if item is end or isinstance(item, Exception):
                if isinstance(item, Exception):
                    result.error = item
                publish(visible.feed("", final=True))
                break
            if not isinstance(item, LLMStreamChunk):
                raise TypeError("Provider emitted an invalid stream chunk")
            result.response.provider, result.response.model = item.provider, item.model
            publish(visible.feed(item.delta))
    except Exception as error:
        result.error = error
    finally:
        result.canceled = result.canceled or halted.is_set()
        halted.set()
        worker.join(timeout=0.1)
        result.canceled = result.canceled or not current() or (
            cancel_event is not None and cancel_event.is_set()
        )
        result.response.latency = generation_elapsed[0] or time.monotonic() - started
    if result.error and not result.response.content.strip() and not result.canceled:
        raise result.error
    if not result.response.content.strip() and not result.canceled:
        raise RuntimeError("Provider stream ended without a visible reply")
    return result
