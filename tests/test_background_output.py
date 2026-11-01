"""Unsolicited output uses the serialized listener worker and yields to owner input."""

import threading
from unittest.mock import MagicMock

import pytest

from ciel.audio.listener import ListenerState, WakeListenerLoop
from ciel.audio.recorder import VoiceRecorder


@pytest.fixture
def listener():
    loop = WakeListenerLoop(MagicMock(), detector=MagicMock(), recorder=VoiceRecorder(),
                            stt=MagicMock(), tts=MagicMock(), on_transcription=MagicMock())
    loop._running = True
    loop._state = ListenerState.LISTENING_WAKE
    return loop


def run_worker_until_queue_drained(loop):
    worker = threading.Thread(target=loop._process_worker, daemon=True)
    worker.start()
    loop._processing_queue.join()
    loop._stop_event.set()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_background_output_never_enters_transcription_or_stt(listener):
    called = []
    assert listener.submit_background(lambda event: called.append(not event.is_set()))
    assert not listener.background_ready()
    assert not listener.submit_background(MagicMock())
    run_worker_until_queue_drained(listener)
    assert called == [True]
    assert listener.state == ListenerState.LISTENING_WAKE
    listener.on_transcription.assert_not_called()
    listener.stt.transcribe_detailed.assert_not_called()


@pytest.mark.parametrize("operation", ["typed", "stop", "mute", "voice"])
def test_foreground_input_and_controls_cancel_queued_suggestions(listener, operation):
    callback = MagicMock()
    assert listener.submit_background(callback)
    if operation == "typed":
        listener.submit_text("help me instead")
    elif operation == "stop":
        listener.cancel_response()
    elif operation == "voice":
        with listener._lock:
            listener._start_recording()
    else:
        listener.set_microphone_mute(True)
    run_worker_until_queue_drained(listener)
    callback.assert_not_called()
    if operation == "typed":
        assert listener.on_transcription.call_args.args[0] == "help me instead"


@pytest.mark.parametrize("state", [ListenerState.RECORDING, ListenerState.PROCESSING])
def test_background_output_cannot_displace_foreground_work(listener, state):
    listener._state = state
    assert not listener.background_ready()
    assert not listener.submit_background(MagicMock())
    assert listener._processing_queue.empty()


def test_muting_interrupts_background_speech_but_preserves_ordinary_speech(listener):
    events = []
    assert listener.submit_background(lambda event: events.append(event))
    event = listener._response_cancel
    listener.tts.stop.reset_mock()
    listener.set_microphone_mute(True)
    assert event.is_set()
    listener.tts.stop.assert_called_once()
    assert not listener.background_ready()


def test_background_callback_rechecks_cancellation_during_output(listener):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def output(event):
        entered.set()
        assert release.wait(2)
        assert event.is_set()
        done.set()

    assert listener.submit_background(output)
    worker = threading.Thread(target=listener._process_worker, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        listener.submit_text("new request")
        release.set()
        assert done.wait(2)
        listener._processing_queue.join()
    finally:
        release.set()
        listener._stop_event.set()
        worker.join(timeout=2)
    assert listener.on_transcription.call_args.args[0] == "new request"


def test_text_only_session_can_receive_visible_background_hints(listener):
    listener._text_only = True
    listener._microphone_muted = True
    listener._state = ListenerState.IDLE
    assert listener.background_ready()
    assert listener.submit_background(MagicMock())


def test_browser_voice_is_direct_and_survives_pc_microphone_mute(listener):
    import numpy as np

    from ciel.webapp.requests import WebRequest

    listener._microphone_muted = True
    listener.stt = None
    request = WebRequest('request', 'phone', 'chat', 1, 'voice')
    assert listener.submit_request(request, np.ones(160, dtype=np.float32))
    run_worker_until_queue_drained(listener)
    info = listener.on_transcription.call_args.args[1]
    assert info['input_source'] == 'web_voice' and info['wake_verified']
    assert listener.state == ListenerState.LISTENING_WAKE


def test_revoked_browser_input_is_discarded_before_transcription(listener):
    from ciel.webapp.requests import WebRequest

    valid = [True]
    request = WebRequest('request', 'phone', 'chat', 1, 'text', 'Hello')
    assert listener.submit_request(request, current=lambda: valid[0])
    valid[0] = False
    run_worker_until_queue_drained(listener)
    listener.on_transcription.assert_not_called()


def test_browser_requests_report_when_handled_even_without_a_reply():
    """A noise clip with no words still ends the browser's Thinking state."""
    from ciel.webapp.requests import WebRequest

    finished = []
    loop = WakeListenerLoop(MagicMock(), detector=MagicMock(), recorder=VoiceRecorder(),
                            stt=MagicMock(), tts=MagicMock(), on_transcription=MagicMock(),
                            on_request_done=finished.append)
    loop._running = True
    loop._state = ListenerState.LISTENING_WAKE
    skipped = WebRequest('skipped', 'phone', 'chat', 1, 'text', 'Hello')
    assert loop.submit_request(skipped, current=lambda: False) is False
    request = WebRequest('noise', 'phone', 'chat', 1, 'text', 'Hello')
    valid = [True]
    assert loop.submit_request(request, current=lambda: valid[0])
    valid[0] = False  # dropped before transcription, like an empty clip
    run_worker_until_queue_drained(loop)
    while not loop._notifications.empty():  # delivered by the notifier thread in the app
        callback, args = loop._notifications.get_nowait()
        callback(*args)
    assert finished == [request]
