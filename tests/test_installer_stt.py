"""Installer speech-model preparation follows configuration without initializing inference."""

import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ciel.config import AudioConfig

INSTALLER = Path(__file__).resolve().parents[1] / "setup.sh"


def _prefetch_code() -> str:
    source = INSTALLER.read_text(encoding="utf-8")
    heredoc = re.search(r"python - <<'PY'\n(.*?)\nPY", source, re.S)
    legacy = re.search(r'python -c "\n(.*?)\n"', source, re.S)
    match = heredoc or legacy
    assert match is not None, "Installer model-prefetch block is missing"
    return match.group(1)


@pytest.mark.parametrize("command,wake", [
    ("small.en", "base.en"),
    ("base.en", "tiny.en"),
    ("tiny.en", "tiny.en"),
])
def test_installer_downloads_configured_command_and_distinct_wake_model(
    tmp_path, monkeypatch, command, wake,
):
    import faster_whisper
    import faster_whisper.utils
    import piper.download_voices

    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(audio=AudioConfig(
        stt_model=command, wake_stt_model=wake, stt_device="cpu", stt_compute_type="int8",
    ))
    monkeypatch.setattr("ciel.config.get_settings", lambda: settings)
    download = MagicMock()
    inference = MagicMock(side_effect=AssertionError("Prefetch must not initialize Whisper"))
    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    monkeypatch.setattr(faster_whisper, "download_model", download)
    monkeypatch.setattr(faster_whisper, "WhisperModel", inference)
    monkeypatch.setattr(piper.download_voices, "download_voice", MagicMock())

    exec(compile(_prefetch_code(), str(INSTALLER), "exec"), {})

    assert [call.args[0] for call in download.call_args_list] == list(
        dict.fromkeys([command, wake]),
    )
    inference.assert_not_called()


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("exists", [False, True])
def test_installer_keeps_explicit_local_model_without_download(
    tmp_path, monkeypatch, relative, exists,
):
    import faster_whisper
    import faster_whisper.utils
    import piper.download_voices

    monkeypatch.chdir(tmp_path)
    local = tmp_path / "local-command-model"
    if exists:
        local.mkdir()
    configured_path = f"./{local.name}" if relative else str(local)
    monkeypatch.setattr("ciel.config.get_settings", lambda: SimpleNamespace(audio=AudioConfig(
        stt_model=configured_path, wake_stt_model="base.en",
    )))
    download = MagicMock()
    inference = MagicMock(side_effect=AssertionError("Local prefetch must not initialize Whisper"))
    monkeypatch.setattr(faster_whisper.utils, "download_model", download)
    monkeypatch.setattr(faster_whisper, "download_model", download)
    monkeypatch.setattr(faster_whisper, "WhisperModel", inference)
    monkeypatch.setattr(piper.download_voices, "download_voice", MagicMock())

    if exists:
        exec(compile(_prefetch_code(), str(INSTALLER), "exec"), {})
        download.assert_called_once_with("base.en")
    else:
        with pytest.raises(FileNotFoundError, match="speech model directory is missing"):
            exec(compile(_prefetch_code(), str(INSTALLER), "exec"), {})
        download.assert_not_called()
    inference.assert_not_called()


def test_installer_adds_whisper_gpu_libraries_and_preserves_cpu_pocket_torch():
    source = INSTALLER.read_text(encoding="utf-8")
    install_lines = [shlex.split(line) for line in source.splitlines()
                     if line.lstrip().startswith("python -m pip install")]
    package = next(parts[-1] for parts in install_lines if "-e" in parts)
    assert set(package.removeprefix(".[").removesuffix("]").split(",")) >= {"pocket-tts", "gpu"}
    assert ["python", "-m", "pip", "install", "torch", "--index-url",
            "https://download.pytorch.org/whl/cpu"] in install_lines
