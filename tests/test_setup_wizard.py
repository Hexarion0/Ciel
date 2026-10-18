"""Setup tests isolate terminal input, secrets, devices, and playback."""

import json
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from dotenv import dotenv_values

from ciel import setup_wizard
from ciel.terminal_ui import ConsoleUI


class ScriptedUI:
    """Exercise wizard decisions independently of terminal menu rendering."""

    def __init__(self, choices, prompts=()):
        self.choices = iter(choices)
        self.prompts = iter(prompts)
        self.summaries = []
        self.on_choose = None

    def banner(self, text):
        print(text)

    step = note = finish = banner

    def summary(self, rows):
        self.summaries.append(rows)
        for label, value in rows:
            print(f"{label}: {value}")

    def choose(self, title, choices, default=None):
        if self.on_choose is not None:
            self.on_choose(title)
        answer = next(self.choices)
        if isinstance(answer, BaseException):
            raise answer
        selected = default if answer is None else answer
        assert selected in {value for value, _label in choices}, (title, selected)
        return selected

    def prompt(self, label):
        answer = next(self.prompts)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def wizard_factory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def create(choices, *, secrets=(), prompts=(), devices=()):
        ui = ScriptedUI(choices, prompts)
        secret_answers = iter(secrets)

        def secret_input(prompt):
            print(prompt)
            answer = next(secret_answers)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(setup_wizard, "ConsoleUI", lambda: ui)
        monkeypatch.setattr(setup_wizard, "getpass", secret_input)
        backend = MagicMock()
        backend.list_devices.return_value = list(devices)
        monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
        settings = MagicMock()
        monkeypatch.setattr(setup_wizard, "get_settings", settings)
        tts = MagicMock()
        tts.return_value.speak.return_value = True
        tts.return_value.last_engine = "pocket"
        monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)
        assets = MagicMock(return_value=[])
        monkeypatch.setattr(setup_wizard, "custom_voice_asset_problems", assets)
        return SimpleNamespace(ui=ui, tts=tts, settings=settings, assets=assets, backend=backend)

    return create


def test_quick_start_reviews_before_saving_default_profile(tmp_path, wizard_factory, capsys):
    harness = wizard_factory([None, None, None], secrets=[""])
    env_path = tmp_path / ".env"

    def check_review(title):
        if title == "Ready to save?":
            assert not env_path.exists()

    harness.ui.on_choose = check_review
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(env_path)
    assert env_path.stat().st_mode & 0o777 == 0o600
    assert saved["TTS_VOICE"] == "ciel" and saved["TTS_ENGINE"] == "pocket"
    assert (saved["STT_MODEL"], saved["STT_DEVICE"], saved["STT_COMPUTE_TYPE"]) == (
        "small.en", "cuda", "int8_float16",
    )
    assert saved["STT_BEAM_SIZE"] == "5" and saved["STT_RETRY_MODEL"] == "medium.en"
    assert saved["AMBIENT_LISTENING"] == "true" and saved["SHOW_AI_TRANSCRIPTS"] == "true"
    assert saved["BARGE_IN_MODE"] == "speech"
    harness.backend.list_devices.assert_not_called()
    harness.settings.cache_clear.assert_called_once()
    harness.tts.assert_called_once_with(
        voice_name="ciel", engine="pocket", enabled=True, output_device=None,
    )
    harness.tts.return_value.close.assert_called_once()
    output = capsys.readouterr().out
    assert "local tools remain available" in output
    assert "No cloud key; local tools available" in output
    assert "Run: ciel" in output and "connection verified" not in output.lower()


@pytest.mark.parametrize("configured", [False, True])
def test_quick_start_keeps_new_and_existing_secrets_hidden(
    tmp_path, wizard_factory, capsys, configured,
):
    secret = "synthetic-private-key-for-setup"
    if configured:
        (tmp_path / ".env").write_text(f'NIM_API_KEY="{secret}"\n')
    harness = wizard_factory(["quick", "save", "yes"], secrets=[] if configured else [secret])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["NIM_API_KEY"] == secret
    output = capsys.readouterr().out
    assert secret not in output and "key configured" in output
    assert "connection hasn't been tested yet" in output
    assert len(harness.ui.summaries) == 1


def test_existing_settings_comments_case_and_literals_survive_quick_start(
    tmp_path, wizard_factory, capsys,
):
    from ciel.config import Settings

    original = (
        "# Keep my settings\n"
        "stt_model=base.en\nSTT_DEVICE=cpu\nSTT_COMPUTE_TYPE=int8\nSTT_BEAM_SIZE=1\n"
        "STT_RETRY_MODEL=\nAMBIENT_LISTENING=false\nSHOW_AI_TRANSCRIPTS=false\n"
        "BARGE_IN_MODE=wake\nMEMORY_MAX_SHORT_TERM_TURNS=20\n"
        "TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n"
        'AUDIO_INPUT_DEVICE=3\nAUDIO_OUTPUT_DEVICE="USB Speaker"\n'
        'CUSTOM_VALUE="literal ${NOT_AN_ENV_VAR}"\n'
    )
    (tmp_path / ".env").write_text(original)
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.tts.return_value.speak.return_value = False
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").read_text().startswith(original)
    output = capsys.readouterr().out
    assert "Audio test failed" in output and "Voice confirmed" not in output
    settings = Settings(_env_file=tmp_path / ".env")
    assert settings.audio.tts_voice == "en_US-amy-medium" and settings.audio.tts_engine == "piper"
    assert (settings.stt_model, settings.stt_device, settings.stt_compute_type) == (
        "base.en", "cpu", "int8",
    )
    assert settings.stt_beam_size == 1 and not settings.stt_retry_model
    assert not settings.ambient_listening and not settings.show_ai_transcripts
    assert settings.barge_in_mode == "wake" and settings.memory.max_short_term_turns == 20
    assert settings.audio.input_device == 3 and settings.audio.output_device == "USB Speaker"


def test_customize_can_reset_devices_and_change_speech_settings(tmp_path, wizard_factory):
    (tmp_path / ".env").write_text(
        "TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\nAUDIO_INPUT_DEVICE=3\n"
    )
    harness = wizard_factory([
        "customize", "amy", "default", "custom", "wake", "off", "save", "yes",
    ], secrets=["", "", ""], prompts=["USB Speaker"])
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_ENGINE"] == "piper" and saved["TTS_VOICE"] == "en_US-amy-medium"
    assert "AUDIO_INPUT_DEVICE" not in saved and saved["AUDIO_OUTPUT_DEVICE"] == "USB Speaker"
    assert saved["AMBIENT_LISTENING"] == "false" and saved["SHOW_AI_TRANSCRIPTS"] == "false"
    harness.tts.assert_called_once_with(
        voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device="USB Speaker",
    )


@pytest.mark.parametrize(
    ("existing", "choice", "voice", "engine"),
    [
        ("", "keep", "ciel", "pocket"),
        ("TTS_VOICE=en_US-amy-medium\n", "keep", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=piper\n", "keep", "en_US-amy-medium", "piper"),
        ("TTS_ENGINE=pocket\n", "keep", "ciel", "pocket"),
        ("TTS_VOICE=en_US-custom-medium\nTTS_ENGINE=auto\n", "keep",
         "en_US-custom-medium", "auto"),
        ("TTS_VOICE=en_US-amy-medium\nTTS_ENGINE=piper\n", "turbo", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\n", "keep", "ciel", "chatterbox_turbo"),
        ("TTS_VOICE=ciel\n", "keep", "ciel", "chatterbox_turbo"),
        ("", "pocket", "ciel", "pocket"),
        ("TTS_VOICE=ciel\nTTS_ENGINE=pocket\n", "keep", "ciel", "pocket"),
    ],
)
def test_setup_voice_choices(tmp_path, wizard_factory, existing, choice, voice, engine):
    if existing:
        (tmp_path / ".env").write_text(existing)
    harness = wizard_factory([
        "customize", choice, "keep", "keep", None, None, "save", "yes",
    ], secrets=["", "", ""])
    setup_wizard.run_setup_wizard()
    saved = dotenv_values(tmp_path / ".env")
    assert saved["TTS_VOICE"] == voice and saved["TTS_ENGINE"] == engine
    harness.tts.assert_called_once_with(
        voice_name=voice, engine=engine, enabled=True, output_device=None,
    )
    assert harness.assets.call_count == (1 if engine == "chatterbox_turbo" else 0)
    harness.tts.return_value.close.assert_called_once()


def test_review_can_return_to_customize(tmp_path, wizard_factory):
    harness = wizard_factory([
        "quick", "customize", "amy", "keep", "keep", "ambient", "on", "save", "yes",
    ], secrets=["", "", "", ""])
    setup_wizard.run_setup_wizard()
    assert len(harness.ui.summaries) == 2
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "piper"


def test_numbered_terminal_menu_completes_customize_flow(tmp_path, wizard_factory, monkeypatch):
    harness = wizard_factory([], secrets=["", "", ""])
    output = StringIO()
    # Customize, Amy, keep devices, ambient, captions on, save, confirm playback.
    terminal_input = StringIO("2\n3\n1\n1\n1\n1\n1\n1\n")
    monkeypatch.setattr(
        setup_wizard, "ConsoleUI", lambda: ConsoleUI(stream=output, input_stream=terminal_input),
    )
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_VOICE"] == "en_US-amy-medium"
    assert "Review your setup" in output.getvalue()
    assert "Voice confirmed" in output.getvalue()
    harness.tts.return_value.close.assert_called_once()


def test_customize_enter_keeps_unchanged_boolean_spellings_and_comments(tmp_path, wizard_factory):
    original = (
        "# Retain exact spelling and inline notes\n"
        "ambient_listening=YES # My listening preference\n"
        "show_ai_transcripts=ON # My captions preference\n"
    )
    (tmp_path / ".env").write_text(original)
    wizard_factory([
        "customize", "keep", "keep", "keep", None, None, "save", "yes",
    ], secrets=["", "", ""])
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").read_text().startswith(original)


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_review_leaves_configuration_untouched(tmp_path, wizard_factory, existing, capsys):
    env_path = tmp_path / ".env"
    original = '# Keep this unchanged\nNIM_API_KEY="synthetic-preserved"\n'
    if existing:
        env_path.write_text(original)
        env_path.chmod(0o640)
    harness = wizard_factory(["quick", "cancel"], secrets=[] if existing else [""])
    setup_wizard.run_setup_wizard()
    if existing:
        assert env_path.read_text() == original
        assert env_path.stat().st_mode & 0o777 == 0o640
    else:
        assert not env_path.exists()
    harness.settings.cache_clear.assert_not_called()
    harness.tts.assert_not_called()
    assert "No changes were saved" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interrupted_secret_prompt_does_not_create_configuration(
    tmp_path, wizard_factory, interruption, capsys,
):
    harness = wizard_factory(["quick"], secrets=[interruption()])
    setup_wizard.run_setup_wizard()
    assert not (tmp_path / ".env").exists()
    harness.tts.assert_not_called()
    assert "No changes were saved" in capsys.readouterr().out


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interrupted_customization_preserves_existing_configuration(
    tmp_path, wizard_factory, interruption,
):
    env_path = tmp_path / ".env"
    original = '# original\nNIM_API_KEY="synthetic-original"\n'
    env_path.write_text(original)
    harness = wizard_factory(["customize", interruption()], secrets=["replacement", "", ""])
    setup_wizard.run_setup_wizard()
    assert env_path.read_text() == original
    harness.settings.cache_clear.assert_not_called()
    assert list(tmp_path.glob(".ciel-setup-*")) == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
def test_interruption_after_save_keeps_configuration(
    tmp_path, wizard_factory, interruption, capsys,
):
    harness = wizard_factory(["quick", "save", interruption()], secrets=[""])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "pocket"
    harness.tts.return_value.close.assert_called_once()
    assert "Configuration is saved" in capsys.readouterr().out


def test_play_again_retries_then_requires_user_confirmation(wizard_factory, capsys):
    harness = wizard_factory(["quick", "save", "again", "yes"], secrets=[""])
    harness.tts.return_value.speak.side_effect = [False, True]
    setup_wizard.run_setup_wizard()
    assert harness.tts.call_count == 2 and harness.tts.return_value.close.call_count == 2
    output = capsys.readouterr().out
    assert "Audio test failed" in output and "Voice confirmed" in output


def test_output_device_retry_saves_and_uses_the_new_device(tmp_path, wizard_factory):
    device = SimpleNamespace(index=9, name="USB Speaker", max_output_channels=2)
    harness = wizard_factory([
        "quick", "save", "output", "device:9", "yes",
    ], secrets=[""], devices=[device])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["AUDIO_OUTPUT_DEVICE"] == "9"
    assert harness.settings.cache_clear.call_count == 2
    assert harness.tts.call_args_list == [
        call(voice_name="ciel", engine="pocket", enabled=True, output_device=None),
        call(voice_name="ciel", engine="pocket", enabled=True, output_device=9),
    ]


def test_retry_can_reset_a_newly_saved_output_device_to_system_default(tmp_path, wizard_factory):
    device = SimpleNamespace(index=9, name="USB Speaker", max_output_channels=2)
    harness = wizard_factory([
        "customize", "amy", "keep", "device:9", None, None, "save",
        "output", "default", "yes",
    ], secrets=["", "", ""], devices=[device])
    setup_wizard.run_setup_wizard()
    assert "AUDIO_OUTPUT_DEVICE" not in dotenv_values(tmp_path / ".env")
    assert harness.settings.cache_clear.call_count == 2
    assert harness.tts.call_args_list == [
        call(voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device=9),
        call(voice_name="en_US-amy-medium", engine="piper", enabled=True, output_device=None),
    ]


def test_playback_exception_keeps_saved_configuration_without_claiming_success(
    tmp_path, wizard_factory, capsys,
):
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.tts.return_value.speak.side_effect = RuntimeError("speaker unavailable")
    setup_wizard.run_setup_wizard()
    assert (tmp_path / ".env").exists()
    harness.tts.return_value.close.assert_called_once()
    output = capsys.readouterr().out
    assert "Audio test failed" in output
    assert "Voice confirmed" not in output and "Audio is not confirmed" in output


def test_missing_custom_assets_save_choice_without_claiming_success(
    tmp_path, wizard_factory, capsys,
):
    (tmp_path / ".env").write_text("TTS_VOICE=ciel\nTTS_ENGINE=chatterbox_turbo\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    harness.assets.return_value = ["reference_audio: /private/reference.wav"]
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == "chatterbox_turbo"
    harness.tts.assert_not_called()
    output = capsys.readouterr().out
    assert "reference_audio: /private/reference.wav" in output and "choose Amy" in output
    assert "Voice confirmed" not in output and "Audio is not confirmed" in output


@pytest.mark.parametrize("engine", ["", "unsupported-engine"])
def test_invalid_saved_engine_reports_failure_without_overwriting_the_choice(
    tmp_path, wizard_factory, capsys, engine,
):
    (tmp_path / ".env").write_text(f"TTS_VOICE=ciel\nTTS_ENGINE={engine}\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    setup_wizard.run_setup_wizard()
    assert dotenv_values(tmp_path / ".env")["TTS_ENGINE"] == engine
    harness.tts.assert_not_called()
    output = capsys.readouterr().out
    assert "Audio test failed (ValueError)" in output
    assert "Audio is not confirmed" in output


def test_failed_save_leaves_original_configuration_intact(tmp_path, wizard_factory, monkeypatch):
    env_path = tmp_path / ".env"
    original = '# original\nNIM_API_KEY="synthetic-original"\n'
    env_path.write_text(original)
    harness = wizard_factory(["quick", "save"])
    real_set_key = setup_wizard.set_key
    writes = 0

    def fail_after_first_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        return real_set_key(*args, **kwargs)

    monkeypatch.setattr(setup_wizard, "set_key", fail_after_first_write)
    with pytest.raises(OSError, match="disk full"):
        setup_wizard.run_setup_wizard()
    assert env_path.read_text() == original
    harness.settings.cache_clear.assert_not_called()
    assert list(tmp_path.glob(".ciel-setup-*")) == []


def test_disabled_voice_output_does_not_play_a_sample(tmp_path, wizard_factory, capsys):
    (tmp_path / ".env").write_text("TTS_ENABLED=false\n")
    harness = wizard_factory(["quick", "save", None], secrets=[""])
    setup_wizard.run_setup_wizard()
    harness.tts.assert_not_called()
    assert dotenv_values(tmp_path / ".env")["TTS_ENABLED"] == "false"
    assert "Voice output is disabled" in capsys.readouterr().out


def test_playback_errors_do_not_echo_a_saved_secret(tmp_path, wizard_factory, capsys):
    secret = "synthetic-never-echo-this"
    (tmp_path / ".env").write_text(f'NIM_API_KEY="{secret}"\n')
    harness = wizard_factory(["quick", "save", None])
    harness.tts.return_value.speak.side_effect = RuntimeError(f"backend failure with {secret}")
    setup_wizard.run_setup_wizard()
    output = capsys.readouterr().out
    assert secret not in output
    assert "RuntimeError" in output


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
