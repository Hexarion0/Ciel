"""Reference setup and idle cleanup must not retain training graphs or interrupt speech."""

import contextlib
import io
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ciel.audio.turbo_worker import idle_requests, prepare_voice_reference, release_unused_memory


@pytest.mark.parametrize("with_style", [False, True])
def test_normal_and_style_reference_preparation_run_in_inference_mode(tmp_path, with_style):
    active = []

    @contextlib.contextmanager
    def inference_mode():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def prepare(*_args, **_kwargs):
        assert active, "Reference preparation retained a training graph"

    def create(*_args):
        assert active, "Event reference preparation retained a training graph"
        return "renderer"

    model = SimpleNamespace(prepare_conditionals=Mock(side_effect=prepare))
    reference = tmp_path / "reference.wav"
    reference.touch()
    factory = Mock(side_effect=create)
    result = prepare_voice_reference(
        model, SimpleNamespace(inference_mode=inference_mode), reference,
        reference if with_style else None, {}, factory,
    )
    model.prepare_conditionals.assert_called_once_with(str(reference), exaggeration=0.0)
    assert result == ("renderer" if with_style else None)
    assert not active
    assert factory.call_count == int(with_style)


def test_optional_style_failure_keeps_normal_voice_prepared(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.touch()
    model = SimpleNamespace(prepare_conditionals=Mock())
    result = prepare_voice_reference(
        model, SimpleNamespace(inference_mode=contextlib.nullcontext), reference, reference, {},
        Mock(side_effect=ValueError("Optional style is invalid")),
    )
    assert result is None
    model.prepare_conditionals.assert_called_once()


def test_cleanup_reports_unused_cache_without_changing_live_allocation(monkeypatch):
    monkeypatch.setattr("ciel.audio.turbo_worker.sys.platform", "other")
    reserved = [512 * 2**20]

    def empty():
        reserved[0] = 64 * 2**20

    cuda = SimpleNamespace(
        memory_reserved=lambda: reserved[0], memory_allocated=lambda: 48 * 2**20,
        empty_cache=Mock(side_effect=empty),
    )
    result = release_unused_memory(SimpleNamespace(cuda=cuda))
    assert result == {"cuda_allocated_mib": 48, "cuda_reserved_mib": 64,
                      "released_cuda_cache_mib": 448}
    cuda.empty_cache.assert_called_once()


def test_buffered_requests_are_not_delayed_or_cleaned_during_synthesis():
    cleanup = Mock()
    requests = idle_requests(io.StringIO("first\nsecond\n"), cleanup, timeout=0.01)
    assert next(requests) == "first\n"
    # Holding the yielded request represents the main thread doing synthesis.
    time.sleep(0.03)
    assert list(requests) == ["second\n"]
    cleanup.assert_not_called()


def test_optional_cache_failure_does_not_fail_voice_startup(monkeypatch):
    monkeypatch.setattr("ciel.audio.turbo_worker.sys.platform", "other")
    cuda = SimpleNamespace(memory_reserved=lambda: 512,
                           empty_cache=Mock(side_effect=RuntimeError("Synthetic cache error")))
    assert release_unused_memory(SimpleNamespace(cuda=cuda)) == {
        "memory_cleanup_error": "RuntimeError",
    }


def test_idle_cleanup_runs_once_then_waits_without_polling():
    incoming = queue.Queue()
    cleaned, delivered = threading.Event(), threading.Event()
    calls, lines = [], []

    def stream():
        while True:
            line = incoming.get()
            if line is None:
                return
            yield line

    def cleanup():
        calls.append(True)
        cleaned.set()

    def consume():
        for line in idle_requests(stream(), cleanup, timeout=0.01):
            lines.append(line)
            delivered.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    try:
        assert cleaned.wait(2)
        time.sleep(0.03)
        assert len(calls) == 1
        incoming.put("hello\n")
        assert delivered.wait(2)
        incoming.put(None)
        thread.join(timeout=2)
        assert not thread.is_alive() and lines == ["hello\n"]
    finally:
        incoming.put(None)


def test_idle_cleanup_failure_does_not_drop_next_request():
    incoming = queue.Queue()
    failed = threading.Event()

    def stream():
        while (line := incoming.get()) is not None:
            yield line

    def cleanup():
        failed.set()
        raise RuntimeError("Synthetic cleanup failure")

    lines = []
    thread = threading.Thread(target=lambda: lines.extend(
        idle_requests(stream(), cleanup, timeout=0.01),
    ), daemon=True)
    thread.start()
    try:
        assert failed.wait(2)
        incoming.put("request\n")
        incoming.put(None)
        thread.join(timeout=2)
        assert not thread.is_alive() and lines == ["request\n"]
    finally:
        incoming.put(None)
