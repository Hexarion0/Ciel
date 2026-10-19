"""Regression coverage for inspection commands that must not launch voice sessions."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from ciel import __main__, config
from ciel.config import Settings

SOURCE = str(Path(__file__).resolve().parents[1] / "src")
KEYS = {
    "nim_api_key": "test-nim-private-key",
    "groq_api_key": "test-groq-private-key",
    "openrouter_api_key": "test-openrouter-private-key",
}
DEPENDENCY_FREE_CLI = """
import importlib.abc
import sys

blocked = {
    "sounddevice", "faster_whisper", "torch", "piper", "openwakeword",
    "onnxruntime", "pysilero_vad", "pocket_tts", "chatterbox",
    "ciel.audio", "ciel.runtime", "ciel.platform",
}
attempts = []

class NoVoiceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            attempts.append(fullname)
            raise AssertionError("Voice dependency imported: " + fullname)

sys.meta_path.insert(0, NoVoiceImports())
from ciel.__main__ import main
try:
    result = main(sys.argv[1:])
except SystemExit as error:
    result = error.code
assert not attempts, attempts
raise SystemExit(result)
"""


def run_without_voice_dependencies(tmp_path, arguments, **overrides):
    """Run the console entry point independently of pytest's audio-importing fixtures."""
    settings_names = set(Settings.model_fields)
    environment = {
        name: value for name, value in os.environ.items() if name.lower() not in settings_names
    }
    environment.update(PYTHONPATH=SOURCE, **overrides)
    return subprocess.run(
        [sys.executable, "-c", DEPENDENCY_FREE_CLI, *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.fixture
def configured_settings(tmp_path, monkeypatch):
    """Supply public configuration examples and distinct dummy secrets."""
    settings = Settings(
        _env_file=None,
        **KEYS,
        ambient_listening=False,
        wake_word="hello ciel",
        tts_enabled=False,
        tts_engine="piper",
        tts_voice="en_US-amy-medium",
        stt_model="tiny.en",
        stt_device="cpu",
        stt_compute_type="int8",
        memory_db_path=str(tmp_path / "private-memory.db"),
    )
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    return settings


def test_status_json_describes_configuration(configured_settings, capsys):
    assert __main__.main(["status", "--json"]) == 0

    output = capsys.readouterr()
    status = json.loads(output.out)
    assert status["mode"] == "wake-word"
    assert status["wake_word"] == "hello ciel"
    assert status["voice"] == {
        "enabled": False, "engine": "piper", "profile": "en_US-amy-medium",
    }
    assert status["recognition"] == {
        "model": "tiny.en", "device": "cpu", "compute_type": "int8",
    }
    assert all(status["providers"][name]["configured"] for name in ("nim", "groq", "openrouter"))
    assert status["providers"]["ollama"]["connectivity"] == "not checked"
    assert status["paths"]["memory_database"] == configured_settings.memory_db_path
    assert not Path(configured_settings.memory_db_path).exists()
    assert all(secret not in output.out + output.err for secret in KEYS.values())


@pytest.mark.parametrize("arguments", [
    ["config"], ["config", "show"], ["config", "--json"],
    ["config", "show", "--json"], ["config", "--json", "show"],
])
@pytest.mark.parametrize("configured", [True, False])
def test_config_reports_secret_states_instead_of_values(
    monkeypatch, capsys, arguments, configured,
):
    settings = Settings(_env_file=None, **{
        name: secret if configured else None for name, secret in KEYS.items()
    })
    monkeypatch.setattr(config, "get_settings", lambda: settings)

    assert __main__.main(arguments) == 0

    output = capsys.readouterr()
    assert all(secret not in output.out + output.err for secret in KEYS.values())
    expected = "configured" if configured else "unconfigured"
    if "--json" in arguments:
        values = json.loads(output.out)
        assert all(values[name.upper()] == expected for name in KEYS)
    else:
        assert all(f'{name.upper()}="{expected}"' in output.out for name in KEYS)


@pytest.mark.parametrize("command", ["status", "config", "doctor"])
@pytest.mark.parametrize("as_json", [False, True])
def test_configuration_errors_never_echo_exception_values(monkeypatch, capsys, command, as_json):
    def invalid_settings():
        raise ValueError("Invalid environment value: " + KEYS["nim_api_key"])

    monkeypatch.setattr(config, "get_settings", invalid_settings)
    arguments = [command, *(["--json"] if as_json else [])]

    assert __main__.main(arguments) == 1

    output = capsys.readouterr()
    assert KEYS["nim_api_key"] not in output.out + output.err
    message = json.loads(output.out)["error"] if as_json else output.err
    assert "ciel setup" in message


@pytest.mark.parametrize("arguments", [
    ["status", "--json"], ["--", "status", "--json"], ["config", "--json"],
])
def test_settings_tools_work_without_voice_dependencies_or_local_files(tmp_path, arguments):
    result = run_without_voice_dependencies(tmp_path, arguments)

    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    if "status" in arguments:
        assert values["paths"]["config_file"] == str(tmp_path / ".env")
        assert values["paths"]["config_file_exists"] is False
    else:
        assert all(values[name.upper()] == "unconfigured" for name in KEYS)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command", [
    None, "status", "doctor", "devices", "config", "memory", "start", "setup", "help",
])
def test_help_remains_available_without_voice_dependencies(tmp_path, command):
    arguments = ["help", *([command] if command else [])]
    result = run_without_voice_dependencies(tmp_path, arguments)

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--help" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [
    ["help", "missing"], ["help", "status", "extra"], ["status", "--listen"],
    ["config", "set"], ["devices", "--input", "--output"],
    ["--ambient", "status"], ["--listen", "doctor"],
    ["--ambient", "help"], ["--listen", "help"],
])
def test_invalid_toolkit_arguments_do_not_start_runtime(tmp_path, arguments):
    result = run_without_voice_dependencies(tmp_path, arguments)

    assert result.returncode == 2, result.stderr
    assert "error:" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def device_probe(monkeypatch):
    from ciel import device_probe
    from ciel.platform.base import AudioDeviceInfo

    devices = [
        AudioDeviceInfo(0, "Microphone", "ALSA", 1, 0, 16000, is_default_input=True),
        AudioDeviceInfo(2, "Speakers", "ALSA", 0, 2, 48000, is_default_output=True),
        AudioDeviceInfo(5, "Headset", "ALSA", 1, 2, 48000),
    ]
    inspect = Mock(return_value=devices)
    monkeypatch.setattr(device_probe, "list_devices", inspect)
    monkeypatch.setattr(config, "get_settings", Mock(side_effect=AssertionError("Settings loaded")))
    return inspect, devices


@pytest.mark.parametrize("direction, indexes", [(None, [0, 2, 5]), ("input", [0, 5]), (
    "output", [2, 5],
)])
def test_devices_filters_metadata_without_loading_settings(
    device_probe, capsys, direction, indexes,
):
    inspect, devices = device_probe
    arguments = ["devices", "--json", *([f"--{direction}"] if direction else [])]

    assert __main__.main(arguments) == 0

    assert json.loads(capsys.readouterr().out) == [
        asdict(device) for device in devices if device.index in indexes
    ]
    inspect.assert_called_once_with()


def test_device_text_labels_default_input_and_output(device_probe, capsys):
    assert __main__.main(["devices"]) == 0

    output = capsys.readouterr().out
    assert "0: Microphone [input default]" in output
    assert "2: Speakers [output default]" in output
    assert "5: Headset [input, output]" in output


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("failure", ["empty", "filtered", "backend"])
def test_devices_reports_empty_or_unavailable_audio_with_failure_status(
    device_probe, capsys, as_json, failure,
):
    inspect, devices = device_probe
    if failure == "backend":
        inspect.side_effect = RuntimeError("Private backend exception")
    else:
        inspect.return_value = [devices[1]] if failure == "filtered" else []
    arguments = ["devices", "--input", *(["--json"] if as_json else [])]

    assert __main__.main(arguments) == 1

    output = capsys.readouterr()
    assert "Private backend exception" not in output.out + output.err
    if as_json:
        data = json.loads(output.out)
        assert "error" in data if failure == "backend" else data == []
    else:
        assert "audio" in (output.out + output.err).lower()
    inspect.assert_called_once_with()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("has_error", [False, True])
def test_doctor_aggregates_checks_and_only_errors_fail(
    configured_settings, monkeypatch, capsys, as_json, has_error,
):
    from ciel import diagnostics
    from ciel.diagnostics import DiagnosticCheck

    checks = [
        DiagnosticCheck("Configuration", "ok", "Settings loaded."),
        DiagnosticCheck("Voice assets", "warning", "Optional voice absent.", "Run ciel setup."),
    ]
    if has_error:
        checks.append(DiagnosticCheck("Microphone", "error", "No input device.", "Connect a mic."))
    inspect = Mock(return_value=checks)
    monkeypatch.setattr(diagnostics, "run_checks", inspect)

    assert __main__.main(["doctor", *(["--json"] if as_json else [])]) == int(has_error)

    inspect.assert_called_once_with(configured_settings)
    output = capsys.readouterr()
    if as_json:
        report = json.loads(output.out)
        assert report["checks"] == [asdict(check) for check in checks]
        assert report["summary"] == {"ok": 1, "warning": 1, "error": int(has_error)}
    else:
        assert f"1 passed, 1 warnings, {int(has_error)} errors" in output.out
        assert "Fix: Run ciel setup." in output.out
        if has_error:
            assert "[ERROR] Microphone: No input device." in output.out
