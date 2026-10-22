"""Regression coverage for configuration and portable local voice assets."""

import os
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ciel.config import AudioConfig, Settings


@pytest.mark.parametrize("configuration", [Settings, AudioConfig])
def test_voice_only_custom_ciel_keeps_legacy_turbo_engine(configuration):
    with patch.dict(os.environ, {}, clear=True):
        options = {"tts_voice": "ciel"}
        if configuration is Settings:
            options["_env_file"] = None
        selected = configuration(**options)
    audio = selected.audio if configuration is Settings else selected
    assert audio.tts_engine == "chatterbox_turbo"
    assert audio.tts_voice == "ciel"


def test_wizard_key_prompts_never_include_saved_secret_material(tmp_path, monkeypatch):
    from ciel import setup_wizard

    secrets = {
        "NIM_API_KEY": "nv-secret-value", "GROQ_API_KEY": "xK7", "OPENROUTER_API_KEY": "or-key",
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("".join(f"{key}={value}\n" for key, value in secrets.items()))
    prompts = []
    monkeypatch.setattr(setup_wizard, "getpass", lambda prompt: prompts.append(prompt) or "")
    output = StringIO()
    ui = setup_wizard.ConsoleUI(
        stream=output, input_stream=StringIO("\n".join([
            "2", "", "", "",  # Customize and keep personal defaults.
            "1", "2", "2", "2", "3", "1",  # Select each provider, then continue.
            "1", "1", "1", "2", "1", "3", "1", "1", "1", "1", "1", "1",
            # Free search, NIM screen/default model, skip chat, personality, Amy, save.
        ])),
    )
    monkeypatch.setattr(setup_wizard, "ConsoleUI", lambda: ui)
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    monkeypatch.setattr(setup_wizard, "TextToSpeech", MagicMock())

    setup_wizard.run_setup_wizard()

    assert len(prompts) == 3
    assert all("configured" in prompt for prompt in prompts)
    assert all(value[:6] not in " ".join(prompts) for value in secrets.values())
    assert all(value not in output.getvalue() for value in secrets.values())


def test_installed_custom_wake_path_is_independent_of_working_directory(tmp_path, monkeypatch):
    import openwakeword

    import ciel.paths as paths
    from ciel.audio.wake import WakeWordDetector

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "assets"))
    model = tmp_path / "assets/ciel/models/custom/hey_ciel.onnx"
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(openwakeword, "get_pretrained_model_paths", lambda: [])
    assert WakeWordDetector._resolve_model_paths(["models/custom/hey_ciel.onnx"]) == [str(model)]


def test_installed_wake_samples_use_same_writable_asset_root(tmp_path, monkeypatch):
    import ciel.paths as paths
    from ciel.audio import trainer

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "assets"))
    monkeypatch.chdir(tmp_path)
    backend = MagicMock()
    backend.record.return_value = np.ones((16000, 1), dtype=np.float32)
    backend.analyze_audio.return_value = {"is_silent": False}
    monkeypatch.setattr(trainer, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(trainer.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "")

    result = trainer.record_voice_samples(count=1)

    expected = tmp_path / "assets/ciel/data/wake_samples/sample_01.wav"
    assert result == [expected]
    assert backend.save_wav.call_args.args[1] == expected
    assert not (tmp_path / "data").exists()


def test_pre_canceled_stream_never_starts_provider_or_playback():
    from threading import Event

    from ciel.audio.streaming import stream_reply
    from ciel.providers.base import LLMStreamChunk

    canceled = Event()
    canceled.set()
    router = MagicMock()
    router.stream.return_value = iter([LLMStreamChunk("Hello.", "model", "provider")])
    tts = MagicMock(supports_sentence_pipeline=False)

    result = stream_reply(router, [], tts, lambda: True, cancel_event=canceled)

    assert result.canceled
    router.stream.assert_not_called()
    tts.begin_stream.assert_not_called()


def test_cancel_event_alone_releases_blocked_provider_stream():
    import threading

    from ciel.audio.streaming import stream_reply

    entered, release, canceled, finished = (threading.Event() for _ in range(4))
    router = MagicMock()

    def provider(*_args, **_kwargs):
        entered.set()
        release.wait(2)
        yield from ()

    router.stream.side_effect = provider
    tts = MagicMock(supports_sentence_pipeline=False)
    results = []
    errors = []

    def run():
        try:
            results.append(stream_reply(router, [], tts, lambda: True, cancel_event=canceled))
        except Exception as err:
            errors.append(err)
        finally:
            finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(1)
        canceled.set()
        assert finished.wait(0.5)
        assert not errors
        assert results[0].canceled
        tts.speak.assert_not_called()
        tts.stop.assert_called_once()
    finally:
        release.set()
        worker.join(timeout=2)


def test_setup_checks_hq_alias_assets_before_testing_audio(tmp_path, monkeypatch):
    from ciel import setup_wizard

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TTS_VOICE=ciel\nTTS_ENGINE=hq\n")
    monkeypatch.setattr(setup_wizard, "getpass", lambda _: "")
    output = StringIO()
    ui = setup_wizard.ConsoleUI(stream=output, input_stream=StringIO("\n" * 11))
    monkeypatch.setattr(setup_wizard, "ConsoleUI", lambda: ui)
    backend = MagicMock()
    backend.list_devices.return_value = []
    monkeypatch.setattr(setup_wizard, "get_audio_backend", lambda: backend)
    monkeypatch.setattr(setup_wizard, "get_settings", MagicMock())
    assets = MagicMock(return_value=["Missing custom reference"])
    monkeypatch.setattr(setup_wizard, "custom_voice_asset_problems", assets)
    tts = MagicMock()
    monkeypatch.setattr(setup_wizard, "TextToSpeech", tts)

    setup_wizard.run_setup_wizard()

    assets.assert_called_once()
    tts.assert_not_called()
    assert "Missing custom reference" in output.getvalue()
