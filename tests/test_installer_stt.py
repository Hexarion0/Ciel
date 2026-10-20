"""Selected speech models download after configuration without initializing inference."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel import installation
from ciel.config import AudioConfig

INSTALLER = Path(__file__).resolve().parents[1] / "setup.sh"


@pytest.fixture
def downloads(monkeypatch):
    """Supply download-only speech packages without network or model initialization."""
    whisper = ModuleType("faster_whisper")
    whisper.download_model = MagicMock()
    whisper.WhisperModel = MagicMock(side_effect=AssertionError("Inference initialized"))
    piper = ModuleType("piper")
    voices = ModuleType("piper.download_voices")
    voices.download_voice = MagicMock()
    piper.download_voices = voices
    monkeypatch.setitem(sys.modules, "faster_whisper", whisper)
    monkeypatch.setitem(sys.modules, "piper", piper)
    monkeypatch.setitem(sys.modules, "piper.download_voices", voices)
    pocket = MagicMock()
    monkeypatch.setattr("ciel.audio.pocket.PocketTTS", pocket)
    return SimpleNamespace(
        whisper=whisper.download_model, inference=whisper.WhisperModel,
        piper=voices.download_voice, pocket=pocket, notify=MagicMock(),
    )


@pytest.mark.parametrize("command,wake", [
    ("small.en", "base.en"), ("base.en", "tiny.en"), ("tiny.en", "tiny.en"),
])
def test_installer_downloads_configured_command_and_distinct_wake_model(downloads, command, wake):
    audio = AudioConfig(
        stt_model=command, wake_stt_model=wake, stt_retry_model="medium.en", tts_enabled=False,
    )

    installation.prepare_models(audio, downloads.notify)

    assert [call.args[0] for call in downloads.whisper.call_args_list] == list(
        dict.fromkeys([command, wake]),
    )
    downloads.inference.assert_not_called()
    downloads.piper.assert_not_called()


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("exists", [False, True])
def test_installer_keeps_explicit_local_model_without_download(
    tmp_path, monkeypatch, downloads, relative, exists,
):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "local-command-model"
    if exists:
        local.mkdir()
    configured_path = f"./{local.name}" if relative else str(local)
    audio = AudioConfig(stt_model=configured_path, wake_stt_model="base.en", tts_enabled=False)

    if exists:
        installation.prepare_models(audio, downloads.notify)
        downloads.whisper.assert_called_once_with("base.en")
    else:
        with pytest.raises(FileNotFoundError, match="speech model directory is missing"):
            installation.prepare_models(audio, downloads.notify)
        downloads.whisper.assert_not_called()
    downloads.inference.assert_not_called()


def test_installer_keeps_existing_local_directory_without_explicit_path_prefix(
    tmp_path, monkeypatch, downloads,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local-model").mkdir()
    audio = AudioConfig(stt_model="local-model", wake_stt_model="local-model", tts_enabled=False)

    installation.prepare_models(audio, downloads.notify)

    downloads.whisper.assert_not_called()
    downloads.inference.assert_not_called()


@pytest.mark.parametrize("engine", ["pocket", "chatterbox_turbo", "hq"])
def test_installer_does_not_fetch_unselected_piper_fallback(downloads, engine):
    installation.prepare_models(AudioConfig(tts_engine=engine), downloads.notify)

    downloads.piper.assert_not_called()
    downloads.inference.assert_not_called()


def test_installer_prepares_selected_pocket_model_with_configured_options(
    tmp_path, monkeypatch, downloads,
):
    monkeypatch.setattr("ciel.paths.runtime_data_root", lambda: tmp_path)
    audio = AudioConfig(
        tts_engine="pocket", tts_pocket_references="custom/references",
        tts_pocket_neutral_reference="custom/reference.wav", tts_pocket_quantization="int8",
        tts_pocket_language="french_2026-09", tts_pocket_builtin_voice="cosette",
        tts_pocket_gain_db=-11.35,
    )

    installation.prepare_models(audio, downloads.notify)

    downloads.pocket.assert_called_once_with(
        tmp_path / "custom/references", neutral_reference=tmp_path / "custom/reference.wav",
        quantization="int8", language="french_2026-09", builtin_voice="cosette", gain_db=-11.35,
    )
    downloads.pocket.return_value.warmup.assert_called_once_with()
    downloads.pocket.return_value.close.assert_called_once_with()
    downloads.piper.assert_not_called()


def test_installer_prepares_pocket_builtin_without_an_explicit_reference(downloads):
    installation.prepare_models(AudioConfig(tts_engine="pocket"), downloads.notify)

    assert downloads.pocket.call_args.kwargs["neutral_reference"] is None
    downloads.pocket.return_value.warmup.assert_called_once_with()
    downloads.pocket.return_value.close.assert_called_once_with()


def test_pocket_model_download_failure_propagates_without_piper_fallback(downloads):
    failure = RuntimeError("Pocket model download failed")
    downloads.pocket.return_value.warmup.side_effect = failure

    with pytest.raises(RuntimeError, match="Pocket model download failed") as raised:
        installation.prepare_models(AudioConfig(tts_engine="pocket"), downloads.notify)

    assert raised.value is failure
    downloads.pocket.return_value.close.assert_called_once_with()
    downloads.piper.assert_not_called()


def test_disabled_voice_does_not_prepare_pocket_models(downloads):
    installation.prepare_models(
        AudioConfig(tts_engine="pocket", tts_enabled=False), downloads.notify,
    )

    downloads.pocket.assert_not_called()


def test_installer_downloads_only_selected_piper_voice_to_resolved_assets(
    tmp_path, monkeypatch, downloads,
):
    monkeypatch.setattr("ciel.paths.runtime_data_root", lambda: tmp_path)
    audio = AudioConfig(tts_engine="piper", tts_voice="en_US-lessac-medium")

    installation.prepare_models(audio, downloads.notify)

    downloads.piper.assert_called_once_with("en_US-lessac-medium", tmp_path / "models/tts")
    downloads.inference.assert_not_called()


def test_installer_reuses_selected_piper_model_and_configuration(tmp_path, monkeypatch, downloads):
    monkeypatch.setattr("ciel.paths.runtime_data_root", lambda: tmp_path)
    models = tmp_path / "models/tts"
    models.mkdir(parents=True)
    for suffix in (".onnx", ".onnx.json"):
        (models / f"en_US-lessac-medium{suffix}").write_bytes(b"local model")

    installation.prepare_models(
        AudioConfig(tts_engine="piper", tts_voice="en_US-lessac-medium"), downloads.notify,
    )

    downloads.piper.assert_not_called()


def test_shell_bootstraps_only_core_then_runs_setup_with_environment_python():
    source = INSTALLER.read_text(encoding="utf-8")
    assert 'VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"' in source
    install = '"$VENV_PYTHON" -m pip install -e .'
    wizard = '"$VENV_PYTHON" -m ciel setup'
    assert install in source and wizard in source
    assert source.index(install) < source.index(wizard)
    assert "download_model" not in source and "download_voice" not in source
    assert "pocket-tts,gpu" not in source and "pip install torch" not in source
