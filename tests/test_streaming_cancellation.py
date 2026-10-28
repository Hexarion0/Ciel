"""A full audio queue must never hold up an interrupted reply's finalization."""

import queue
import threading
import time

from ciel.audio.streaming import _SentencePipeline


def test_cancel_while_audio_sentinel_waits_for_full_queue_returns_promptly(monkeypatch):
    """Stop after synthesis ends but before its final sentinel has queue space."""
    playing, queued, sentinel, release = (threading.Event() for _ in range(4))
    canceled, finished = threading.Event(), threading.Event()
    real_queue = queue.Queue
    created = []

    class ObservedAudioQueue(real_queue):
        def put(self, item, block=True, timeout=None):
            if item is _SentencePipeline._END:
                sentinel.set()
            result = super().put(item, block=block, timeout=timeout)
            if item is not _SentencePipeline._END and item[0] == "Second.":
                queued.set()
            return result

    def make_queue(*args, **kwargs):
        target = (ObservedAudioQueue if created else real_queue)(*args, **kwargs)
        created.append(target)
        return target

    monkeypatch.setattr("ciel.audio.streaming.queue.Queue", make_queue)

    class Speech:
        audio_queue_size = 1

        def prepare_sentence(self, text, _cancel_event, _generation):
            return text

        def speak(self, _text, **_kwargs):
            playing.set()
            assert release.wait(2)
            return False

    pipeline = _SentencePipeline(
        Speech(), 1, lambda: not canceled.is_set(), None,
        time.monotonic(), lambda _text: None, None,
    )
    assert pipeline.submit("First.", time.monotonic())
    assert playing.wait(1)
    assert pipeline.submit("Second.", time.monotonic())
    assert queued.wait(1)

    def finish():
        pipeline.finish(cancel=False)
        finished.set()

    finisher = threading.Thread(target=finish, daemon=True)
    finisher.start()
    try:
        assert sentinel.wait(1), "did not reach the full audio queue's final sentinel"
        canceled.set()
        release.set()
        assert finished.wait(0.5), "canceled producer waited for an absent audio consumer"
    finally:
        canceled.set()
        release.set()
        # Unblock the original implementation after the failing assertion too.
        while True:
            try:
                pipeline.audio.get_nowait()
            except queue.Empty:
                break
        finisher.join(1)
        pipeline._synthesizer.join(1)
        pipeline._player.join(1)
    assert not pipeline._synthesizer.is_alive()
    assert not pipeline._player.is_alive()


def test_cancel_during_finish_leaves_inflight_synthesis_owned_until_it_returns(monkeypatch):
    """A slow inference job keeps its lock without blocking the next user request."""
    preparing, release, joining, finished = (threading.Event() for _ in range(4))
    canceled, synthesis_lock = threading.Event(), threading.Lock()
    played = []

    class Speech:
        audio_queue_size = 1

        def prepare_sentence(self, text, _cancel_event, _generation):
            with synthesis_lock:
                preparing.set()
                assert release.wait(2)
                return text

        def speak(self, text, **_kwargs):
            played.append(text)
            return True

    pipeline = _SentencePipeline(
        Speech(), 1, lambda: not canceled.is_set(), None,
        time.monotonic(), lambda _text: None, None,
    )
    assert pipeline.submit("Stale.", time.monotonic())
    assert preparing.wait(1)
    original_join = pipeline._synthesizer.join

    def joined(*args, **kwargs):
        joining.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(pipeline._synthesizer, "join", joined)

    def finish():
        pipeline.finish(cancel=False)
        finished.set()

    finisher = threading.Thread(target=finish, daemon=True)
    finisher.start()
    try:
        assert joining.wait(1)
        canceled.set()
        assert finished.wait(0.5), "finish retained its original long wait after cancellation"
        assert pipeline._synthesizer.is_alive()
        assert not synthesis_lock.acquire(blocking=False), "released an active inference lock"
        assert not played
    finally:
        canceled.set()
        release.set()
        finisher.join(1)
        original_join(1)
        pipeline._player.join(1)
    assert not pipeline._synthesizer.is_alive()
    assert not played
    assert synthesis_lock.acquire(blocking=False)
    synthesis_lock.release()
