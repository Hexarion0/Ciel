"""Check truthful startup state and its ordering around audio and keyboard input."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import Settings
from ciel.runtime import _terminal_status_rows


@pytest.mark.parametrize(
    "prepared,enabled,ready,expected",
    [
        (True, True, True, "Ciel Turbo"),
        (False, True, True, "Ciel Turbo · fallback pending"),
        (None, True, False, "Ciel Turbo · deferred"),
        (None, False, False, "Off · text replies"),
    ],
)
def test_panel_reflects_live_state_and_cli_choices(prepared, enabled, ready, expected):
    settings = Settings(_env_file=None, stt_device="cuda", tts_engine="chatterbox_turbo")
    stt = SimpleNamespace(model_size="small.en", device="cpu", is_ready=lambda: ready)
    tts = SimpleNamespace(engine="chatterbox_turbo", enabled=enabled)
    rows = dict(_terminal_status_rows(
        settings, stt, tts, ambient=False, captions=False, voice_prepared=prepared,
    ))
    assert rows["● Listening"] == "Wake-word mode"
    assert rows["Live captions"] == "Off"
    assert rows["Voice"] == expected
    assert rows["Recognition"] == "small.en · CPU" + ("" if ready else " · not ready")


@pytest.mark.parametrize("preset", ["azelma", "cosette"])
def test_panel_names_selected_pocket_preset_without_loading_model(tmp_path, preset):
    settings = Settings(
        _env_file=None, tts_engine="pocket", tts_pocket_references=str(tmp_path),
        tts_pocket_builtin_voice=preset, tts_pocket_neutral_reference=None,
    )
    # A style recording without neutral does not override the built-in identity.
    (tmp_path / "warm.wav").touch()
    stt = SimpleNamespace(is_ready=lambda: True)
    tts = SimpleNamespace(engine="pocket", enabled=True)
    rows = dict(_terminal_status_rows(
        settings, stt, tts, ambient=True, captions=True, voice_prepared=True,
    ))
    assert rows["Voice"] == f"{preset.title()} · Pocket"


@pytest.mark.parametrize("reference", ["neutral.wav", "neutral.flac", "neutral.ogg",
                                       "neutral.safetensors", "explicit.wav", "missing.wav"])
def test_panel_preserves_custom_pocket_identity_and_explicit_reference_precedence(
    tmp_path, reference,
):
    explicit = reference in {"explicit.wav", "missing.wav"}
    path = tmp_path / reference
    if reference != "missing.wav":
        path.touch()
    settings = Settings(
        _env_file=None, tts_engine="pocket", tts_pocket_references=str(tmp_path),
        tts_pocket_builtin_voice="azelma",
        tts_pocket_neutral_reference=str(path) if explicit else None,
    )
    stt = SimpleNamespace(is_ready=lambda: True)
    tts = SimpleNamespace(engine="pocket", enabled=True)
    prepared = reference != "missing.wav"
    rows = dict(_terminal_status_rows(
        settings, stt, tts, ambient=True, captions=True, voice_prepared=prepared,
    ))
    assert rows["Voice"] == "Ciel Pocket" + ("" if prepared else " · fallback pending")


@pytest.mark.parametrize("failure", [False, True])
def test_header_precedes_models_panel_follows_stream_and_precedes_keyboard(
    tmp_path, monkeypatch, failure,
):
    from ciel import __main__, audio, config, platform, providers, terminal, terminal_ui

    events = []
    settings = Settings(_env_file=None, memory_db_path=str(tmp_path / "memory.db"))
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(providers, "get_model_router", MagicMock())
    backend = MagicMock()

    def get_backend():
        events.append("backend")
        return backend

    monkeypatch.setattr(platform, "get_audio_backend", get_backend)
    ui = MagicMock()
    ui.banner.side_effect = lambda *_a, **_k: events.append("banner")
    ui.panel.side_effect = lambda *_a: events.append("panel")
    monkeypatch.setattr(terminal_ui, "ConsoleUI", lambda **_kw: ui)
    for component in ("WakeWordDetector", "VoiceRecorder"):
        monkeypatch.setattr(audio, component, MagicMock())
    stt = SimpleNamespace(
        wait_ready=lambda **_kw: True, is_ready=lambda: True,
        model_size="small.en", device="cpu",
    )

    def make_stt(**_kw):
        events.append("models")
        return stt

    monkeypatch.setattr(audio, "SpeechToText", make_stt)
    tts = MagicMock(engine="pocket", enabled=True)
    monkeypatch.setattr(audio, "TextToSpeech", lambda **_kw: tts)
    loop = MagicMock(is_running=False)

    def start_stream():
        events.append("stream")
        if failure:
            raise RuntimeError("Audio device unavailable")

    loop.start.side_effect = start_stream
    monkeypatch.setattr(audio, "WakeListenerLoop", lambda **_kw: loop)
    keyboard = MagicMock()
    keyboard.start.side_effect = lambda: events.append("keyboard")
    monkeypatch.setattr(terminal, "TerminalInput", lambda **_kw: keyboard)
    monkeypatch.setattr(sys, "argv", ["ciel", "--text-input", "--no-show-ai-transcripts"])

    if failure:
        with pytest.raises(RuntimeError, match="Audio device unavailable"):
            __main__.main()
        assert events == ["banner", "backend", "models", "stream"]
        ui.panel.assert_not_called()
        keyboard.start.assert_not_called()
    else:
        assert __main__.main() == 0
        assert events == ["banner", "backend", "models", "stream", "panel", "keyboard"]
        rows = dict(ui.panel.call_args.args[0])
        assert rows["● Listening"] == "Ambient conversation"
        assert rows["Live captions"] == "Off"
        assert rows["Recognition"] == "small.en · CPU"
    loop.stop.assert_called_once()
    tts.close.assert_called_once()
