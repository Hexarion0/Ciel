"""Microphone failures remain actionable without switching capture devices."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.platform.base import AudioInputError
from ciel.platform.linux import LinuxAudioBackend


class FakePortAudioError(Exception):
    """Represent a native capture error without initializing PortAudio."""


def test_failed_microphone_can_keep_real_text_workers_running_without_recapturing():
    from ciel.audio.listener import ListenerState, WakeListenerLoop

    backend = MagicMock()
    backend.start_stream.side_effect = AudioInputError("Microphone unavailable")
    detector, stt = MagicMock(), MagicMock()
    received = []
    answered = Event()

    def on_text(text, info, audio):
        received.append((text, info["input_source"], audio.size))
        answered.set()
        return True

    loop = WakeListenerLoop(
        audio_backend=backend, detector=detector, recorder=MagicMock(), stt=stt,
        on_transcription=on_text,
    )
    with pytest.raises(AudioInputError):
        loop.start()
    assert not loop.is_running
    try:
        loop.start_text_only()
        assert loop.is_running and loop.microphone_muted
        assert loop.state == ListenerState.IDLE
        assert loop.set_microphone_mute(False) is True
        assert loop.submit_text("We can still chat.")
        assert answered.wait(2)
        assert received == [("We can still chat.", "keyboard", 0)]
        backend.start_stream.assert_called_once()
        detector.start.assert_called_once()
        stt.transcribe.assert_not_called()
        stt.transcribe_detailed.assert_not_called()
    finally:
        loop.stop()
    assert not loop.is_running
    assert all(not thread.is_alive() for thread in loop._threads)


@pytest.mark.parametrize("stage", ["construct", "start"])
def test_portaudio_failure_closes_stream_and_allows_same_device_retry(monkeypatch, stage):
    backend = LinuxAudioBackend()
    failed, replacement = MagicMock(), MagicMock()
    failed.active = False
    native_error = FakePortAudioError("ALSA error -2: No such file or directory")
    if stage == "construct":
        streams = [native_error, replacement]
    else:
        failed.start.side_effect = native_error
        streams = [failed, replacement]
    constructor = MagicMock(side_effect=streams)
    monkeypatch.setattr(backend, "resolve_device", lambda device, **kwargs: device)
    monkeypatch.setattr("ciel.platform.linux.sd", SimpleNamespace(
        InputStream=constructor, PortAudioError=FakePortAudioError,
    ))

    with pytest.raises(AudioInputError, match="microphone") as caught:
        backend.start_stream(lambda *args: None, device=7)
    assert caught.value.__cause__ is native_error
    assert backend._stream is None
    assert constructor.call_count == 1
    if stage == "start":
        failed.close.assert_called_once()

    backend.start_stream(lambda *args: None, device=7)
    assert [call.kwargs["device"] for call in constructor.call_args_list] == [7, 7]
    replacement.start.assert_called_once()
    backend.stop_stream()
    replacement.close.assert_called_once()


@pytest.mark.parametrize("error", [RuntimeError("unrelated failure"), KeyboardInterrupt()])
def test_stream_preserves_other_failures_and_cancellation(monkeypatch, error):
    backend = LinuxAudioBackend()
    stream = MagicMock(active=False)
    stream.start.side_effect = error
    monkeypatch.setattr(backend, "resolve_device", lambda *args, **kwargs: 7)
    monkeypatch.setattr("ciel.platform.linux.sd", SimpleNamespace(
        InputStream=MagicMock(return_value=stream), PortAudioError=FakePortAudioError,
    ))
    with pytest.raises(type(error)) as caught:
        backend.start_stream(lambda *args: None)
    assert caught.value is error
    assert backend._stream is None
    stream.close.assert_called_once()


def _prepare_launch(monkeypatch, error):
    """Keep launch boundary tests independent of hardware, models, and services."""
    from ciel import config, logging, platform, runtime, terminal_ui
    from ciel.chat import autostart

    settings = config.Settings(_env_file=None)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(autostart, "reconcile_chat_startup", MagicMock(
        return_value=SimpleNamespace(success=True),
    ))
    monkeypatch.setattr(platform, "get_audio_backend", MagicMock())
    monkeypatch.setattr(terminal_ui, "ConsoleUI", MagicMock())
    logger = MagicMock()
    monkeypatch.setattr(logging, "setup_logging", lambda *args, **kwargs: logger)
    monkeypatch.setattr(runtime, "run_desktop", MagicMock(side_effect=error))
    return logger


@pytest.mark.parametrize("arguments", [[], ["start", "dev"]])
def test_launch_reports_microphone_guidance_and_retains_debug_details(
    monkeypatch, capsys, arguments,
):
    from ciel import __main__

    logger = _prepare_launch(monkeypatch, AudioInputError("microphone unavailable"))
    assert __main__.main(arguments) == 1
    output = capsys.readouterr()
    assert "microphone" in output.err
    assert "system sound settings" in output.err
    assert "ciel devices --input" in output.err
    assert "ciel setup" in output.err
    assert "Traceback" not in output.out + output.err
    logger.debug.assert_called_once()
    assert logger.debug.call_args.kwargs["exc_info"] is True


@pytest.mark.parametrize("error", [RuntimeError("unrelated failure"), KeyboardInterrupt()])
def test_launch_does_not_hide_other_errors_or_cancellation(monkeypatch, error):
    from ciel import __main__

    _prepare_launch(monkeypatch, error)
    with pytest.raises(type(error)) as caught:
        __main__.main([])
    assert caught.value is error
