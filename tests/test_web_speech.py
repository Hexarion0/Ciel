"""Browser speech is origin-scoped, claimed once, and canceled before late chunks."""

import threading
import time

import numpy as np
import pytest

from ciel.webapp.requests import WebRequest


def test_browser_speech_uses_synthesis_without_pc_playback():
    from ciel.webapp.speech import BrowserSpeech

    class TTS:
        def synthesize_stream(self, text, **kwargs):
            yield np.zeros(160, dtype=np.float32), 16000

    service = BrowserSpeech(TTS())
    request = WebRequest("r", "phone", "chat", 2, "text", "hello")
    job = service.speak("Hello", request, cancel_event=threading.Event(), current=lambda: True)
    assert service.list("other", 2) == []
    assert service.list("phone", 2)[0]["id"] == job
    chunk = None
    for _ in range(100):
        chunk = service.chunk(job, "phone", 2, "tab-one")
        if chunk:
            break
        time.sleep(0.01)
    assert chunk.startswith(b"RIFF")
    with pytest.raises(ValueError):
        service.chunk(job, "phone", 2, "tab-two")
    service.cancel()
    with pytest.raises(ValueError):
        service.chunk(job, "phone", 2, "tab-one")


def test_synthesis_cancel_event_observes_parent_cancellation_immediately():
    from ciel.webapp.speech import BrowserSpeech

    entered, release = threading.Event(), threading.Event()
    parent = threading.Event()
    observed = []

    class TTS:
        def synthesize_stream(self, text, *, cancel_event):
            entered.set()
            assert release.wait(2)
            observed.append(cancel_event.is_set())
            return
            yield

    service = BrowserSpeech(TTS())
    request = WebRequest("r", "phone", "chat", 1, "text", "Hello")
    service.speak("Hello", request, cancel_event=parent, current=lambda: True)
    assert entered.wait(2)
    parent.set()
    release.set()
    for _ in range(100):
        if observed:
            break
        time.sleep(0.01)
    assert observed == [True]
    service.cancel()
