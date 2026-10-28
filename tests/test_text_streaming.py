"""Text streaming preserves code and request ownership without initializing audio."""

import subprocess
import sys
import threading
import time

import pytest

from ciel.conversation import INTERNAL_REPLY_NOTES
from ciel.providers.base import ChatMessage, LLMStreamChunk, ToolCallDelta
from ciel.text_streaming import stream_text_reply


class Router:
    def __init__(self, *deltas):
        self.deltas = deltas
        self.calls = []
        self.closed = threading.Event()

    def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        try:
            for delta in self.deltas:
                if isinstance(delta, Exception):
                    raise delta
                yield delta if isinstance(delta, LLMStreamChunk) else LLMStreamChunk(
                    delta, "synthetic-model", "synthetic",
                )
        finally:
            self.closed.set()


def test_visible_text_preserves_code_and_filters_split_internal_text():
    router = Router(
        "<thi", "nk>private analysis</thi", "nk>", INTERNAL_REPLY_NOTES[0],
        "[pa", "uses]Answer.\n`", "``python\nx = [1, 2]\n", "```",
    )
    deltas = []
    messages = [ChatMessage("user", "Explain with code", local_only=True)]
    result = stream_text_reply(router, messages, lambda: True, on_delta=deltas.append)
    assert result.response.content == "Answer.\n```python\nx = [1, 2]\n```"
    assert "".join(deltas) == result.response.content
    assert result.response.provider == "synthetic"
    assert result.response.model == "synthetic-model"
    assert result.response.latency >= 0
    assert result.published and not result.canceled and result.error is None
    assert len(router.calls) == 1
    assert router.calls[0][0] is messages
    assert router.calls[0][0][0].local_only
    assert router.calls[0][1]["max_tokens"] == 4096
    assert router.closed.is_set()


def test_native_metadata_never_becomes_visible_text():
    router = Router(LLMStreamChunk("", "model", "synthetic", tool_calls=(
        ToolCallDelta(0, "call", "memory_search", '{"query":"private note"}'),
    )), "The answer.")
    deltas = []
    result = stream_text_reply(router, [], lambda: True, on_delta=deltas.append, max_tokens=1234)
    assert result.response.content == "The answer."
    assert "".join(deltas) == "The answer."
    assert router.calls[0][1]["max_tokens"] == 1234


def test_partial_provider_failure_retains_visible_answer_without_retry():
    error = RuntimeError("synthetic provider failure")
    router = Router("A useful partial answer.", error)
    result = stream_text_reply(router, [], lambda: True)
    assert result.response.content == "A useful partial answer."
    assert result.error is error
    assert not result.canceled and not result.published
    assert len(router.calls) == 1
    assert router.closed.is_set()


def test_empty_provider_failure_is_raised():
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        stream_text_reply(Router("<think>hidden</think>", RuntimeError(
            "synthetic provider failure",
        )), [], lambda: True)


def test_empty_success_is_reported_as_unusable():
    with pytest.raises(RuntimeError, match="without a visible reply"):
        stream_text_reply(Router("<reasoning>hidden</reasoning>"), [], lambda: True)


@pytest.mark.parametrize("reset", [False, True])
def test_cancel_or_reset_inside_callback_discards_queued_deltas(reset):
    router = Router("First.", "Must not appear.")
    canceled = threading.Event()
    active = [True]
    deltas = []

    def publish(delta):
        deltas.append(delta)
        if reset:
            active[0] = False
        else:
            canceled.set()

    result = stream_text_reply(router, [], lambda: active[0], on_delta=publish,
                               cancel_event=canceled)
    assert deltas == ["First."]
    assert result.response.content == "First."
    assert result.canceled and result.published
    assert router.calls[0][1]["cancel_event"].is_set()


@pytest.mark.parametrize("reset", [False, True])
def test_cancel_or_reset_while_provider_blocked_returns_promptly(reset):
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    cancel = threading.Event()
    active = [True]
    deltas, results = [], []

    class BlockedRouter:
        def stream(self, messages, **kwargs):
            try:
                entered.set()
                release.wait(3)
                yield LLMStreamChunk("Stale answer.", "model", "synthetic")
            finally:
                closed.set()

    worker = threading.Thread(target=lambda: results.append(stream_text_reply(
        BlockedRouter(), [], lambda: active[0], on_delta=deltas.append, cancel_event=cancel,
    )))
    worker.start()
    try:
        assert entered.wait(1)
        started = time.monotonic()
        if reset:
            active[0] = False
        else:
            cancel.set()
        worker.join(0.5)
        assert not worker.is_alive()
        assert time.monotonic() - started < 0.5
        assert results[0].canceled
        assert not results[0].published
        assert not deltas
    finally:
        release.set()
        assert closed.wait(1)
        worker.join(1)
    assert not deltas


def test_already_canceled_turn_does_not_start_provider():
    router = Router("Should not run.")
    cancel = threading.Event()
    cancel.set()
    result = stream_text_reply(router, [], lambda: True, cancel_event=cancel)
    assert result.canceled and not result.published
    assert not router.calls


def test_stuck_readers_are_bounded_and_release_capacity():
    entered = [threading.Event(), threading.Event()]
    closed = [threading.Event(), threading.Event()]
    release = threading.Event()
    canceled = [threading.Event(), threading.Event()]
    results = []

    class BlockedRouter:
        def __init__(self, index):
            self.index = index

        def stream(self, messages, **kwargs):
            try:
                entered[self.index].set()
                release.wait(3)
                yield LLMStreamChunk("Stale.", "model", "synthetic")
            finally:
                closed[self.index].set()

    workers = [threading.Thread(target=lambda index=index: results.append(stream_text_reply(
        BlockedRouter(index), [], lambda: True, cancel_event=canceled[index],
    ))) for index in range(2)]
    try:
        for worker in workers:
            worker.start()
        assert all(event.wait(1) for event in entered)
        for event in canceled:
            event.set()
        for worker in workers:
            worker.join(0.5)
            assert not worker.is_alive()
        assert len(results) == 2
        assert all(result.canceled for result in results)
        third = Router("Must not start while readers are blocked.")
        with pytest.raises(RuntimeError, match="still closing"):
            stream_text_reply(third, [], lambda: True)
        assert not third.calls
    finally:
        release.set()
        assert all(event.wait(1) for event in closed)
        for worker in workers:
            worker.join(1)


def test_import_and_reply_do_not_load_native_audio_or_speech_models():
    program = """
import sys
from ciel.providers.base import LLMStreamChunk
from ciel.text_streaming import stream_text_reply
class Router:
    def stream(self, *args, **kwargs):
        yield LLMStreamChunk('Hello.', 'synthetic', 'synthetic')
assert stream_text_reply(Router(), [], lambda: True).response.content == 'Hello.'
assert not any(name in sys.modules for name in (
    'ciel.audio.native', 'ciel.audio.stt', 'ciel.audio.tts', 'sounddevice',
    'faster_whisper', 'pocket_tts',
))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr
