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


POINTER = "The full answer is in the chat."
SENTENCE = "This is one complete sentence with ten simple ordinary words."


class SentenceTTS:
    """Half a second of audio per call; records each text it was asked to say."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls: list[str] = []
        self.gate = gate
        self.entered = threading.Event()

    def synthesize_stream(self, text, **kwargs):
        self.calls.append(text)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(2)
        yield np.zeros(8000, dtype=np.float32), 16000


def wait_for(condition, message="condition was not reached"):
    for _ in range(400):
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError(message)


def drain(service, job, device="phone", revision=1):
    chunks = []
    for _ in range(800):
        try:
            chunk = service.chunk(job, device, revision, "tab-one")
        except EOFError:
            return chunks
        if chunk:
            chunks.append(chunk)
        else:
            time.sleep(0.005)
    raise AssertionError("speech job did not finish")


def open_stream(tts, *, cancel_event=None, current=lambda: True):
    from ciel.webapp.speech import BrowserSpeech

    service = BrowserSpeech(tts)
    request = WebRequest("request", "phone", "chat", 1, "voice", "")
    stream = service.stream(request, cancel_event=cancel_event or threading.Event(),
                            current=current)
    return service, request, stream


def test_streamed_reply_speaks_its_first_sentence_before_the_answer_finishes():
    tts = SentenceTTS()
    service, _request, stream = open_stream(tts)

    stream.update("Sure thing.")
    assert service.list("phone", 1) == []
    stream.update("Sure thing. Here")
    job = service.list("phone", 1)[0]["id"]
    wait_for(lambda: service._jobs[job].chunks.qsize(), "first sentence audio was not ready")
    assert tts.calls == ["Sure thing."]
    first = service.chunk(job, "phone", 1, "tab-one")
    assert first.startswith(b"RIFF")

    stream.update("Sure thing. Here is the rest")
    handle = stream.finish("Sure thing. Here is the rest.")

    assert handle == job
    assert len(drain(service, job)) == 1
    assert tts.calls == ["Sure thing.", "Here is the rest."]
    listed = service.list("phone", 1)
    assert len(listed) == 1 and listed[0]["excerpt"] is False
    assert listed[0]["caption"] == "Sure thing. Here is the rest."


def test_long_streamed_reply_settles_early_and_replays_the_full_answer():
    tts = SentenceTTS()
    service, _request, stream = open_stream(tts)
    answer = " ".join([SENTENCE] * 12)

    stream.update(" ".join([SENTENCE] * 9) + " More")
    job = service.list("phone", 1)[0]["id"]
    drain(service, job)  # the excerpt ended before the answer did

    assert tts.calls == [SENTENCE] * 7 + [POINTER]
    assert service.list("phone", 1)[0]["excerpt"] is False
    stream.update(answer)
    assert stream.finish(answer) == job
    assert service.list("phone", 1)[0]["excerpt"] is True
    replay = service.replay(job, "phone", 1)
    wait_for(lambda: tts.calls[-1] == answer, "replay did not read the full answer")
    assert replay != job


def test_short_reply_without_a_released_sentence_is_spoken_at_finish():
    tts = SentenceTTS()
    service, _request, stream = open_stream(tts)

    stream.update("Okay.")
    assert service.list("phone", 1) == []
    job = stream.finish("Okay.")

    assert drain(service, job) and tts.calls == ["Okay."]


def test_stop_during_a_streamed_reply_drops_pending_sentences():
    gate = threading.Event()
    tts = SentenceTTS(gate)
    canceled = threading.Event()
    service, _request, stream = open_stream(tts, cancel_event=canceled)

    stream.update("One is first. Two is next. Three follows. Four")
    job = service.list("phone", 1)[0]["id"]
    assert tts.entered.wait(2)
    canceled.set()
    service.cancel()
    gate.set()
    stream.update("One is first. Two is next. Three follows. Four is last. Five")
    assert stream.finish("One is first. Two is next. Three follows. Four is last.") is None

    time.sleep(0.05)
    assert tts.calls == ["One is first."]
    assert service.list("phone", 1) == []
    with pytest.raises(ValueError):
        service.chunk(job, "phone", 1, "tab-one")


def test_stream_canceled_before_its_first_sentence_never_opens_audio():
    tts = SentenceTTS()
    canceled = threading.Event()
    service, _request, stream = open_stream(tts, cancel_event=canceled)
    canceled.set()

    stream.update("Hello there. General")
    assert stream.finish("Hello there. General Kenobi.") is None

    assert service.list("phone", 1) == [] and tts.calls == []


def test_stop_and_switch_silence_a_stream_even_without_its_cancel_event():
    tts = SentenceTTS()
    service, _request, stream = open_stream(tts)
    stream.update("Hello there. General")
    service.cancel()

    stream.update("Hello there. General Kenobi. More")
    assert stream.finish("Hello there. General Kenobi. More.") is None
    assert service.list("phone", 1) == []


def test_turn_end_abandons_an_unclosed_stream_but_keeps_a_closed_one():
    tts = SentenceTTS()
    service, request, stream = open_stream(tts)
    stream.update("Hello there. General")
    job = service.list("phone", 1)[0]["id"]

    service.abandon(request)

    assert service.list("phone", 1) == []
    with pytest.raises(ValueError):
        service.chunk(job, "phone", 1, "tab-one")

    service, request, stream = open_stream(SentenceTTS())
    stream.update("Hello there. General")
    closed = stream.finish("Hello there. General Kenobi.")
    service.abandon(request)
    assert service.list("phone", 1)[0]["id"] == closed
    assert len(drain(service, closed)) == 2


def test_a_full_speech_queue_does_not_break_the_text_stream():
    from ciel.webapp.speech import BrowserSpeech

    tts = SentenceTTS(threading.Event())
    service = BrowserSpeech(tts)
    for index in range(8):
        service.speak("Hello.", WebRequest(f"busy{index}", "phone", "chat", 1, "text", ""),
                      cancel_event=threading.Event(), current=lambda: True)
    stream = service.stream(WebRequest("request", "phone", "chat", 1, "voice", ""),
                            cancel_event=threading.Event(), current=lambda: True)

    stream.update("Hello there. General")  # must not raise inside text streaming
    with pytest.raises(ValueError):
        stream.finish("Hello there. General Kenobi.")
    service.cancel()


def test_first_served_chunk_logs_request_timing_without_text(caplog):
    import logging

    from ciel.latency import TurnTrace, active_trace

    trace = TurnTrace()
    token = active_trace.set(trace)
    try:
        tts = SentenceTTS()
        service, _request, stream = open_stream(tts)
        with caplog.at_level(logging.INFO, logger="ciel"):
            stream.update("Private words here. More")
            job = service.list("phone", 1)[0]["id"]
            stream.finish("Private words here. More private words.")
            drain(service, job)
    finally:
        active_trace.reset(token)

    events = trace.report()["events"]
    assert {"browser_speech_started", "browser_first_chunk_ready",
            "browser_first_chunk_served"} <= set(events)
    assert events["browser_speech_started"]["streamed"] is True
    timing = [record.getMessage() for record in caplog.records
              if "Browser reply audio" in record.getMessage()]
    assert len(timing) == 1 and trace.turn_id in timing[0]
    assert "Private" not in caplog.text
