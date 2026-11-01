"""Ciel must not hear itself: while any browser plays its voice, every page's mic is held."""

import io
import threading
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ciel.webapp.requests import WebRequest


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class OneSecondTTS:
    def synthesize_stream(self, text, **kwargs):
        yield np.zeros(16000 * 2, dtype=np.float32), 16000  # two one-second chunks


def drain_one(service, job, client="tab-one"):
    for _ in range(200):
        chunk = service.chunk(job, "phone", 1, client)
        if chunk:
            return chunk
        threading.Event().wait(0.005)
    raise AssertionError("no audio chunk was produced")


def speaking_service(clock):
    from ciel.webapp.speech import BrowserSpeech

    service = BrowserSpeech(OneSecondTTS(), clock=clock)
    request = WebRequest("r", "phone", "chat", 1, "text", "hello")
    job = service.speak("Hello there", request, cancel_event=threading.Event(),
                        current=lambda: True)
    return service, job


def test_handing_out_audio_marks_ciel_audible_for_its_duration_plus_margin():
    from ciel.webapp.speech import ECHO_MARGIN_SECONDS

    clock = Clock()
    service, job = speaking_service(clock)
    assert not service.audible()

    drain_one(service, job)
    drain_one(service, job)  # prefetched while the first plays: playback runs two seconds
    clock.now += 2 + ECHO_MARGIN_SECONDS - 0.05
    assert service.audible()
    clock.now += 0.1
    assert not service.audible()


def test_stop_ends_audibility_immediately():
    clock = Clock()
    service, job = speaking_service(clock)
    drain_one(service, job)

    service.cancel()

    assert not service.audible()


def test_a_clip_that_started_during_playback_is_echo():
    from ciel.webapp.speech import ECHO_MARGIN_SECONDS

    clock = Clock()
    service, job = speaking_service(clock)
    drain_one(service, job)  # audible until 1001 + margin
    clock.now += 3
    assert service.overlaps_playback(clip_seconds=2.5)  # started at 1000.5
    assert not service.overlaps_playback(clip_seconds=3 - 1 - ECHO_MARGIN_SECONDS - 0.1)


def test_state_reports_speaking_to_every_page_while_any_browser_plays():
    from ciel.web import DesktopWebUI

    clock = Clock()
    ui = DesktopWebUI("owner", "hey ciel")
    service, job = speaking_service(clock)
    ui.channels.speech = service
    assert ui.snapshot()["state"] != "speaking"

    drain_one(service, job)

    assert ui.snapshot()["state"] == "speaking"


def _voice_upload(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * int(16000 * seconds))
    return output.getvalue()


def _post_voice(ui, data: bytes):
    handler = SimpleNamespace(
        path="/api/voice", command="POST", client_address=("127.0.0.1", 1),
        headers={"Content-Type": "audio/wav", "Content-Length": str(len(data)),
                 "X-Request-ID": "a" * 32, "X-Revision": "-1", "X-Conversation": ""},
        rfile=io.BytesIO(data), _json=MagicMock(), response_headers={},
    )
    assert ui.channels.authorize(handler)
    ui.channels.post(handler)
    return handler._json.call_args.args


def test_voice_heard_while_ciel_was_speaking_is_dropped_not_answered():
    from ciel.web import DesktopWebUI

    clock = Clock()
    ui = DesktopWebUI("owner", "hey ciel")
    submit = MagicMock(return_value=True)
    ui.channels.configure(submit, OneSecondTTS())
    service, job = speaking_service(clock)
    ui.channels.speech = service
    drain_one(service, job)
    clock.now += 2

    status, body = _post_voice(ui, _voice_upload(2.5))

    assert status == 200 and body == {"accepted": False, "ignored": True}
    submit.assert_not_called()


def test_voice_after_playback_finished_is_accepted():
    from ciel.web import DesktopWebUI

    clock = Clock()
    ui = DesktopWebUI("owner", "hey ciel")
    submit = MagicMock(return_value=True)
    ui.channels.configure(submit, OneSecondTTS())
    service, job = speaking_service(clock)
    ui.channels.speech = service
    drain_one(service, job)
    clock.now += 10

    status, body = _post_voice(ui, _voice_upload(1.5))

    assert status == 200 and body == {"accepted": True}
    submit.assert_called_once()
