import json
from unittest.mock import MagicMock

import pytest
from dotenv import dotenv_values

from ciel import setup_wizard


@pytest.fixture(autouse=True)
def hidden_key_answers(monkeypatch):
    """Use each test's input sequence for synthetic secret prompts as well."""
    monkeypatch.setattr(setup_wizard, "getpass", lambda prompt: input(prompt))


def test_selected_voice_persists_matching_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "", "2", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    settings = MagicMock()
    monkeypatch.setattr(setup_wizard, "get_settings", settings)
    tts = MagicMock()
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    assert saved["TTS_VOICE"] == "en_US-amy-medium"
    assert saved["TTS_ENGINE"] == "piper"
    settings.cache_clear.assert_called_once()
    tts.assert_called_once_with(voice_name="en_US-amy-medium", engine="piper", enabled=True)
    tts.return_value.close.assert_called_once()


def test_setup_preserves_custom_settings_and_reports_playback_failure(
    tmp_path, monkeypatch, capsys
):
    from ciel.config import Settings

    monkeypatch.chdir(tmp_path)
    original = (
        "# Keep my settings\n"
        "stt_model=small.en\nSTT_BEAM_SIZE=5\nMEMORY_MAX_SHORT_TERM_TURNS=20\n"
        "TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n"
        'AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n'
        'CUSTOM_VALUE="literal ${NOT_AN_ENV_VAR}"\n'
    )
    (tmp_path / ".env").write_text(original)
    answers = iter([""] * 6)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    tts = MagicMock()
    tts.return_value.speak.return_value = False
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
    setup_wizard.run_setup_wizard()
    saved_text = (tmp_path / ".env").read_text()
    assert saved_text.startswith(original)
    output = capsys.readouterr().out
    assert "Audio test failed" in output
    assert "Audio test successful" not in output
    settings = Settings(_env_file=tmp_path / ".env")
    assert settings.audio.tts_voice == "en_US-amy-medium"
    assert settings.audio.tts_engine == "piper"
    assert settings.stt_beam_size == 5
    assert settings.memory.max_short_term_turns == 20
    assert settings.audio.input_device == 3
    assert settings.audio.output_device == "USB Speaker"


def test_setup_can_reset_devices_and_change_voice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\nAUDIO_INPUT_DEVICE=3\n"
    )
    answers = iter(["", "", "", "2", "default", "USB Speaker"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    monkeypatch.setattr(setup_wizard, "TextToSpeech", MagicMock())
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_ENGINE"] == "piper"
    assert saved["TTS_VOICE"] == "en_US-amy-medium"
    assert "AUDIO_INPUT_DEVICE" not in saved
    assert saved["AUDIO_OUTPUT_DEVICE"] == "USB Speaker"


@pytest.mark.parametrize(
    ("existing", "choice", "voice", "engine"),
    [
        ("", "", "ciel", "pocket"),
        ("TTS_VOICE=en_US-amy-medium\n", "", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=piper\n", "", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=pocket\n", "", "ciel", "pocket"),
        ("TTS_VOICE=en_US-custom-medium\nTTS_ENGINE=auto\n", "",
         "en_US-custom-medium", "auto"),
        ("TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n", "1", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\n", "", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\n", "", "ciel", "chatterbox_turbo"),
        ("", "3", "ciel", "pocket"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=pocket\n", "", "ciel", "pocket"),
    ],
)
def test_setup_voice_choices(tmp_path, monkeypatch, existing, choice, voice, engine):
    monkeypatch.chdir(tmp_path)
    if existing:
        (tmp_path / ".env").write_text(existing)
    answers = iter(["", "", "", choice, "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    assets = MagicMock(return_value=[])
    monkeypatch.setattr(setup_wizard, "custom_voice_asset_problems", assets)
    tts = MagicMock()
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)

    setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_VOICE"] == voice
    assert saved["TTS_ENGINE"] == engine
    tts.assert_called_once_with(voice_name=voice, engine=engine, enabled=True)
    assert assets.call_count == (1 if engine == "chatterbox_turbo" else 0)
    tts.return_value.close.assert_called_once()


def test_missing_custom_assets_save_choice_without_claiming_success(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    answers = iter(["", "", "", "invalid", "1", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    monkeypatch.setattr(
        setup_wizard, "custom_voice_asset_problems",
        lambda _config: ["reference_audio: /private/reference.wav"],
    )
    tts = MagicMock()
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)

    setup_wizard.run_setup_wizard()

    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_VOICE"] == "ciel"
    assert saved["TTS_ENGINE"] == "chatterbox_turbo"
    tts.assert_not_called()
    output = capsys.readouterr().out
    assert "Please enter 1 for Turbo HQ, 2 for Amy, or 3 for CPU Pocket" in output
    assert "reference_audio: /private/reference.wav" in output
    assert "choose Amy" in output
    assert "Setup Complete!" not in output
    assert "Audio test successful" not in output


def test_custom_asset_check_uses_configured_paths(tmp_path):
    config = {
        "TTS_VOICE": "ciel",
        "TTS_VOICE_PROFILES": str(tmp_path / "profiles"),
        "TTS_CHATTERBOX_MODEL": str(tmp_path / "model"),
        "TTS_CHATTERBOX_PYTHON": str(tmp_path / "env/bin/python"),
    }
    problems = setup_wizard.custom_voice_asset_problems(config)
    assert any("Turbo Python environment" in problem for problem in problems)
    assert any("voice profile" in problem for problem in problems)
    assert any("Turbo model file" in problem for problem in problems)
    profile = tmp_path / "profiles/ciel"
    profile.mkdir(parents=True)
    (profile / "voice.json").write_text(json.dumps({
        "engine": "chatterbox_turbo",
        "reference_audio": str(tmp_path / "reference.wav"),
        "reference_transcript": str(tmp_path / "reference.txt"),
    }))
    problems = setup_wizard.custom_voice_asset_problems(config)
    assert any("reference_audio" in problem for problem in problems)
    assert any("reference_transcript" in problem for problem in problems)
    for problem in problems:
        path = tmp_path / problem.split(": ", 1)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test asset")
    assert setup_wizard.custom_voice_asset_problems(config) == []
    (tmp_path / "reference.txt").write_text("")
    assert len(setup_wizard.custom_voice_asset_problems(config)) == 1
    (profile / "voice.json").write_text("invalid json")
    assert "voice profile" in setup_wizard.custom_voice_asset_problems(config)[0]
