"""Unit tests for Text-to-Speech (TTS) engine."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ciel.audio.tts import TextToSpeech


def test_internal_status_is_not_spoken_by_direct_tts():
    assert TextToSpeech.clean_text_for_speech("[Playback was interrupted.]") == ""
    assert TextToSpeech.clean_text_for_speech(
        "[Playback was interrupted.] Let's continue."
    ) == "Let's continue."


def test_clean_text_for_speech():
    raw_text = """
    <think>Internal reasoning here that should be stripped</think>
    # Header 1
    Here is some `code inline` and a link [OpenAI](https://openai.com).
    ```python
    def test():
        pass
    ```
    * Bullet 1
    1. Numbered item
    **Bold text** and *italic* and emojis 🚀🔥
    """
    cleaned = TextToSpeech.clean_text_for_speech(raw_text)
    assert "Internal reasoning" not in cleaned
    assert "#" not in cleaned
    assert "https://openai.com" not in cleaned
    assert "🚀" not in cleaned
    assert "🔥" not in cleaned
    assert "code inline" in cleaned
    assert "Bold text" in cleaned


def test_turbo_events_are_explicit_and_limited_to_the_pinned_native_set():
    from ciel.audio.speech_events import (
        SpeechEvent,
        add_speech_event,
        split_speech_event,
        strip_speech_events,
    )

    plain = "You're still awake."
    assert add_speech_event(plain, None) == plain
    assert add_speech_event(plain, SpeechEvent.SIGH) == f"[sigh] {plain}"
    assert split_speech_event("[chuckle] I had a feeling.") == (
        "[chuckle]", "I had a feeling."
    )
    assert split_speech_event("[whisper] Speak softly.") == ("", "Speak softly.")
    assert strip_speech_events("[chuckle] A thought. [sigh] A pause.") == (
        "A thought.  A pause."
    )


def test_explicit_auto_engine_preserves_piper():
    tts = TextToSpeech(voice_name="en_US-amy-medium", enabled=False, engine="auto")
    assert tts.engine == "piper"


def test_fresh_tts_defaults_use_pocket_without_loading_models(monkeypatch):
    from ciel.config import Settings

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    monkeypatch.setattr("ciel.audio.tts.get_settings", lambda: settings)
    tts = TextToSpeech(enabled=False)
    assert (tts.engine, tts.voice_name) == ("pocket", "ciel")
    assert tts._pocket is None
    tts.close()


def test_default_pocket_missing_dependency_uses_local_amy(tmp_path, monkeypatch):
    from ciel.config import Settings

    with patch.dict(os.environ, {}, clear=True):
        settings = Settings(_env_file=None)
    monkeypatch.setattr("ciel.audio.tts.get_settings", lambda: settings)
    monkeypatch.setitem(sys.modules, "pocket_tts", None)
    for name in ("en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"):
        (tmp_path / name).write_text("local test asset")
    load = MagicMock()
    monkeypatch.setattr("ciel.audio.tts.PiperVoice.load", load)
    download = MagicMock(side_effect=AssertionError("Fallback must stay offline"))
    monkeypatch.setattr("ciel.audio.tts.download_voice", download)
    tts = TextToSpeech(models_dir=tmp_path)
    expected = (np.ones(100, dtype=np.float32), 22050)
    monkeypatch.setattr(tts, "_synthesize_piper", lambda *_args, **_kwargs: expected)
    try:
        audio, rate = tts.synthesize("Hello.")
        assert np.array_equal(audio, expected[0]) and rate == expected[1]
        assert tts.last_engine == "piper"
        assert tts._pocket_failed and tts._pocket_retry_at == 0
        load.assert_called_once()
        assert "en_US-amy-medium.onnx" in str(load.call_args)
        download.assert_not_called()
    finally:
        tts.close()


@pytest.mark.integration
def test_piper_initialization_and_synthesis():
    tts = TextToSpeech(voice_name="en_US-amy-medium", engine="piper", enabled=True)
    assert tts.enabled is True
    assert tts.voice_name == "en_US-amy-medium"

    result = tts.synthesize("Hello sir, CIEL system is online.")
    assert result is not None
    audio, sample_rate = result
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.size > 0
    assert sample_rate == 22050


def test_tts_disabled():
    tts = TextToSpeech(enabled=False)
    assert tts.synthesize("Hello") is None
    assert tts.speak("Hello") is False


def test_stop_during_synthesis_prevents_late_playback(monkeypatch):
    import threading
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    played = []

    def synthesize(_text):
        entered.set()
        assert release.wait(2)
        return np.zeros(16000, dtype=np.float32), 16000

    monkeypatch.setattr(tts, "synthesize", synthesize)
    monkeypatch.setattr(
        "ciel.audio.tts.sd", SimpleNamespace(play=lambda *_a, **_k: played.append(1))
    )
    results = []
    worker = threading.Thread(target=lambda: results.append(tts.speak("hello")))
    worker.start()
    try:
        assert entered.wait(1)
        interruption = tts.stop()
    finally:
        release.set()
        worker.join(timeout=2)
    assert results == [False]
    assert played == []
    assert interruption['remaining_text'] == 'hello'
    assert interruption['estimated_spoken_text'] == ''
    assert interruption['played_seconds'] == 0


def test_interrupted_playback_reports_estimated_unheard_text(monkeypatch):
    from types import SimpleNamespace

    clock = [100.0]
    monkeypatch.setattr('ciel.audio.tts.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('ciel.audio.tts.sd', SimpleNamespace(
        play=lambda *_args, **_kwargs: None,
        get_stream=lambda: SimpleNamespace(active=True), stop=lambda: None,
    ))
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, 'synthesize', lambda _: (np.ones(80000), 8000))
    assert tts.speak('one two three four five six seven eight nine ten', block=False)
    clock[0] += 5
    interruption = tts.stop()
    assert interruption['estimated_spoken_text'] == 'one two three four five'
    assert interruption['remaining_text'] == 'six seven eight nine ten'
    assert interruption['played_seconds'] == 5
    assert interruption['duration_seconds'] == 10
    assert not tts.is_speaking()
    assert tts.stop() is None  # Do not replay an old interruption on another stop.


def test_completed_playback_does_not_report_an_interruption(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr('ciel.audio.tts.sd', SimpleNamespace(
        play=lambda *_args, **_kwargs: None, get_stream=lambda: SimpleNamespace(active=False),
        stop=lambda: None,
    ))
    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, 'synthesize', lambda _: (np.ones(8000), 8000))
    assert tts.speak('This reply finished.')
    assert tts.stop() is None


def test_omitted_engine_uses_selected_piper_voice(monkeypatch):
    from ciel.config import Settings

    settings = Settings(_env_file=None, tts_voice="en_US-amy-medium", tts_engine="auto")
    monkeypatch.setattr("ciel.audio.tts.get_settings", lambda: settings)
    tts = TextToSpeech(
        voice_name=settings.audio.tts_voice, engine=settings.audio.tts_engine, enabled=False
    )
    tts.enabled = True
    expected = np.ones(100, dtype=np.float32), 22050
    with (
        patch.object(tts, "_synthesize_piper", return_value=expected) as local,
    ):
        assert tts.synthesize("I'm here.") is expected
    local.assert_called_once()


@pytest.mark.parametrize(
    "voice, engine, expected",
    [
        ("en_US-amy-medium", "auto", "piper"),
        ("en_GB-alan-medium", "auto", "piper"),
        ("ciel", "chatterbox_turbo", "chatterbox_turbo"),
        ("ciel", "chatterbox", "chatterbox_turbo"),
        ("ciel", "piper", "piper"),
    ],
)
def test_engine_resolution_respects_explicit_selection(voice, engine, expected):
    from ciel.audio.tts import resolve_tts_engine

    assert resolve_tts_engine(voice, engine) == expected


def test_removed_engines_are_rejected():
    from ciel.audio.tts import resolve_tts_engine

    with pytest.raises(ValueError, match="Unknown TTS engine"):
        resolve_tts_engine("ciel", "fish_speech")


def test_chatterbox_backend_starts_lazily_and_falls_back_offline(monkeypatch):
    from ciel.audio.tts import ChatterboxWorkerError, resolve_tts_engine

    tts = TextToSpeech(voice_name="ciel", engine="chatterbox_turbo", enabled=True)
    tts._pocket_failed = True  # This test exercises the final offline fallback.
    assert resolve_tts_engine("ciel", "chatterbox") == "chatterbox_turbo"
    assert tts._chatterbox is None
    monkeypatch.setattr(
        tts, "_load_chatterbox_worker",
        lambda: (_ for _ in ()).throw(ChatterboxWorkerError("model unavailable")),
    )
    expected = (np.ones(120, dtype=np.float32), 24000)
    monkeypatch.setattr(tts, "_synthesize_piper_fallback", lambda _text: expected)
    result = tts.synthesize("Of course.")
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1] == expected[1]
    assert tts._chatterbox_failed
    result = tts.synthesize("[sigh] You're still awake.")
    np.testing.assert_array_equal(result[0], expected[0])
    assert result[1] == expected[1]
    assert tts.enabled


def test_chatterbox_worker_preserves_virtualenv_python_symlink(tmp_path):
    import sys

    from ciel.audio.chatterbox_worker import ChatterboxTurboWorker

    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    worker = ChatterboxTurboWorker(
        launcher, tmp_path / "worker.py", tmp_path / "model", tmp_path / "ref.wav", "text"
    )
    assert worker.python == launcher.absolute()
    assert worker.python != launcher.resolve()


def test_speech_event_prompt_is_optional_and_limited_to_native_events():
    from ciel.audio.speech_events import SpeechEvent, speech_event_instruction

    assert speech_event_instruction(enabled=False) == ""
    instruction = speech_event_instruction(enabled=True)
    assert all(event.value in instruction for event in SpeechEvent)
    assert "Use them sparingly" in instruction
    assert "[angry]" not in instruction


def test_chatterbox_fallback_uses_amy_when_ciel_piper_is_missing(tmp_path, monkeypatch):
    from ciel.audio.tts import PiperVoice

    amy = tmp_path / "en_US-amy-medium.onnx"
    amy.write_bytes(b"")
    amy.with_suffix(".onnx.json").write_text("{}")
    tts = TextToSpeech(
        voice_name="ciel", engine="chatterbox_turbo", enabled=False,
        models_dir=tmp_path, tts_fallback_voice="en_US-not-installed",
    )
    piper = object()
    expected = (np.ones(100, dtype=np.float32), 22050)
    monkeypatch.setattr(PiperVoice, "load", lambda **_kwargs: piper)
    monkeypatch.setattr(
        tts, "_synthesize_piper",
        lambda _text, voice: expected if voice is piper else None,
    )
    assert tts._synthesize_piper_fallback("Normal speech.") == expected
    assert tts._fallback_voice is piper


def test_explicit_output_device_numeric_string_and_name():
    for value, expected in [("3", 3), ("USB Speaker", "USB Speaker")]:
        tts = TextToSpeech(enabled=False, output_device=value)
        assert tts.output_device == expected


@pytest.mark.parametrize("latency", [None, 0.2])
def test_playback_buffers_audio_without_changing_waveform(monkeypatch, latency):
    from types import SimpleNamespace

    from ciel.config import Settings

    monkeypatch.setattr(
        "ciel.audio.tts.get_settings",
        lambda: Settings(_env_file=None, tts_engine="piper", tts_playback_latency=0.12),
    )
    options = {} if latency is None else {"playback_latency": latency}
    tts = TextToSpeech(enabled=False, output_device="USB Speaker", **options)
    tts.enabled = True
    audio = np.ones(2400, dtype=np.float32)
    monkeypatch.setattr(tts, "synthesize", lambda _text: (audio, 24000))
    play = patch("ciel.audio.tts.sd", SimpleNamespace(
        play=lambda data, **kwargs: calls.append((data, kwargs)),
        get_stream=lambda: SimpleNamespace(active=False, latency=0.128),
    ))
    calls = []
    with play:
        assert tts.speak("Buffered speech.")
    assert calls[0][0] is audio
    assert calls[0][1] == {
        "samplerate": 24000, "device": "USB Speaker",
        "latency": latency or 0.12, "blocksize": 1024,
    }


@pytest.mark.parametrize("underflow", [False, True])
def test_playback_reports_underruns_without_failing_speech(monkeypatch, caplog, underflow):
    from types import SimpleNamespace

    tts = TextToSpeech(enabled=False)
    tts.enabled = True
    monkeypatch.setattr(tts, "synthesize", lambda _text: (np.ones(100), 24000))
    monkeypatch.setattr("ciel.audio.tts.sd", SimpleNamespace(
        play=lambda *_args, **_kwargs: None,
        get_stream=lambda: SimpleNamespace(active=False),
        get_status=lambda: SimpleNamespace(output_underflow=underflow),
    ))
    assert tts.speak("Test speech.")
    assert ("TTS playback underrun" in caplog.text) is underflow
