"""Offline diagnostics inspect local prerequisites without starting the assistant."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ciel import diagnostics
from ciel.config import Settings


@pytest.fixture
def local_install(tmp_path, monkeypatch):
    """Provide an isolated audio listing and readable Piper files without real probes."""
    monkeypatch.chdir(tmp_path)

    def asset_path(path, **_kwargs):
        value = Path(path)
        return value if value.is_absolute() else tmp_path / value

    monkeypatch.setattr(diagnostics, "resolve_asset_path", asset_path)
    monkeypatch.setattr(diagnostics, "voice_profile_path", lambda voice, directory: (
        asset_path(directory) / voice / "voice.json"
    ))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _module: object())
    microphone = SimpleNamespace(
        index=2, name="USB Microphone", max_input_channels=1, max_output_channels=0,
        is_default_input=True, is_default_output=False,
    )
    speaker = SimpleNamespace(
        index=7, name="Headset Speakers", max_input_channels=0, max_output_channels=2,
        is_default_input=False, is_default_output=True,
    )
    monkeypatch.setattr(
        diagnostics, "_list_audio_devices", Mock(return_value=[microphone, speaker]),
    )
    models = tmp_path / "models/tts"
    models.mkdir(parents=True)
    (models / "en_US-amy-medium.onnx").write_bytes(b"not loaded")
    (models / "en_US-amy-medium.onnx.json").write_text("{}", encoding="utf-8")
    return Settings(
        _env_file=None, tts_engine="piper", tts_voice="en_US-amy-medium", tts_enabled=True,
        audio_input_device=None, audio_output_device=None,
        memory_db_path=str(tmp_path / "private" / "memory.db"),
    )


def check_named(settings, name):
    """Return one named result and ensure results keep their documented status values."""
    checks = diagnostics.run_checks(settings)
    assert all(check.status in {"ok", "warning", "error"} for check in checks)
    return next(check for check in checks if check.name == name)


def test_results_are_immutable():
    check = diagnostics.DiagnosticCheck("test", "ok", "observed")
    assert check.hint == ""
    with pytest.raises(FrozenInstanceError):
        check.status = "error"


def test_healthy_install_checks_files_and_permissions_without_loading(local_install, tmp_path):
    checks = diagnostics.run_checks(local_install)
    assert checks
    assert all(check.status == "ok" for check in checks)
    assert "#2 USB Microphone" in checks[0].detail
    assert "#7 Headset Speakers" in checks[1].detail
    assert "not loaded" in check_named(local_install, "Voice assets").detail
    assert not (tmp_path / "private").exists()
    diagnostics._list_audio_devices.assert_called_with()


@pytest.mark.parametrize(
    "field,value,name",
    [
        ("audio_input_device", 91, "Microphone"),
        ("audio_output_device", "missing headset", "Speakers"),
        ("audio_input_device", 7, "Microphone"),
        ("audio_output_device", 2, "Speakers"),
    ],
)
def test_invalid_explicit_device_is_reported_instead_of_falling_back(
    local_install, field, value, name,
):
    setattr(local_install, field, value)
    check = check_named(local_install, name)
    assert check.status == "error"
    assert "unavailable" in check.detail
    assert "ciel devices" in check.hint


def test_explicit_device_name_matching_and_nondefault_fallback(local_install, monkeypatch):
    local_install.audio_input_device = "usb mic"
    assert check_named(local_install, "Microphone").status == "ok"
    devices = diagnostics._list_audio_devices()
    devices[1].is_default_output = False
    monkeypatch.setattr(diagnostics, "_list_audio_devices", lambda: devices)
    assert "First available" in check_named(local_install, "Speakers").detail


def test_audio_probe_failure_does_not_skip_dependencies_or_memory(local_install, monkeypatch):
    monkeypatch.setattr(
        diagnostics, "_list_audio_devices", Mock(side_effect=ImportError("unavailable")),
    )
    checks = {check.name: check for check in diagnostics.run_checks(local_install)}
    assert checks["Microphone"].status == "error"
    assert checks["Speakers"].status == "error"
    assert checks["Speech recognition dependencies"].status == "ok"
    assert checks["Memory location"].status == "ok"


def test_empty_audio_listing_reports_both_missing_devices(local_install, monkeypatch):
    monkeypatch.setattr(diagnostics, "_list_audio_devices", lambda: [])
    assert check_named(local_install, "Microphone").status == "error"
    assert check_named(local_install, "Speakers").status == "error"


def test_missing_and_broken_package_probes_are_independent(local_install, monkeypatch):
    def package(module):
        if module == "piper":
            raise ValueError("bad package metadata")
        return None if module == "faster_whisper" else object()

    monkeypatch.setattr(diagnostics, "find_spec", package)
    checks = {check.name: check for check in diagnostics.run_checks(local_install)}
    assert checks["Speech recognition dependencies"].status == "error"
    assert "faster_whisper" in checks["Speech recognition dependencies"].detail
    assert checks["Speech output dependencies"].status == "error"
    assert "ValueError" in checks["Speech output dependencies"].detail
    assert checks["Voice assets"].status == "ok"
    assert checks["Memory location"].status == "ok"


def test_piper_missing_or_empty_files_are_reported_without_downloading(
    local_install, tmp_path,
):
    model = tmp_path / "models/tts/en_US-amy-medium.onnx"
    model.write_bytes(b"")
    check = check_named(local_install, "Voice assets")
    assert check.status == "error"
    assert str(model) in check.detail
    assert "does not download" in check.hint


def test_piper_absolute_model_and_alternate_config(local_install, tmp_path):
    model = tmp_path / "private.onnx"
    model.write_bytes(b"model")
    model.with_suffix(".json").write_text("{}", encoding="utf-8")
    local_install.tts_voice = str(model)
    assert check_named(local_install, "Voice assets").status == "ok"


def test_pocket_builtin_and_explicit_missing_reference(local_install, tmp_path):
    local_install.tts_engine = "pocket"
    local_install.tts_pocket_references = str(tmp_path / "references")
    local_install.tts_pocket_neutral_reference = None
    check = check_named(local_install, "Voice assets")
    assert check.status == "warning"
    assert "built-in" in check.detail
    local_install.tts_pocket_neutral_reference = str(tmp_path / "missing.wav")
    assert check_named(local_install, "Voice assets").status == "error"
    neutral = tmp_path / "neutral.wav"
    neutral.write_bytes(b"not decoded")
    local_install.tts_pocket_neutral_reference = str(neutral)
    assert check_named(local_install, "Voice assets").status == "ok"


def test_pocket_checks_its_dependency_and_optional_fallback(local_install, monkeypatch, tmp_path):
    local_install.tts_engine = "pocket"
    monkeypatch.setattr(diagnostics, "find_spec", lambda module: (
        None if module == "pocket_tts" else object()
    ))
    assert check_named(local_install, "Speech output dependencies").status == "error"
    assert check_named(local_install, "Piper fallback").status == "ok"
    (tmp_path / "models/tts/en_US-amy-medium.onnx").unlink()
    assert check_named(local_install, "Piper fallback").status == "warning"


def test_turbo_missing_environment_profile_and_model_are_separate(local_install, tmp_path):
    local_install.tts_engine = "chatterbox_turbo"
    local_install.tts_voice = "ciel"
    local_install.tts_chatterbox_python = str(tmp_path / "missing/python")
    local_install.tts_chatterbox_model = str(tmp_path / "missing/model")
    checks = {check.name: check for check in diagnostics.run_checks(local_install)}
    for name in ("Turbo environment", "Voice assets", "Turbo model files"):
        assert checks[name].status == "error"
    assert checks["Memory location"].status == "ok"


def test_turbo_profile_reference_checks_and_corrupt_manifest(local_install, tmp_path):
    local_install.tts_engine = "hq"
    local_install.tts_voice = "ciel"
    profile = tmp_path / "voice_profiles/ciel/voice.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("invalid JSON", encoding="utf-8")
    assert "JSONDecodeError" in check_named(local_install, "Voice assets").detail
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"not decoded")
    transcript = tmp_path / "reference.txt"
    transcript.write_text("Private reference text.", encoding="utf-8")
    manifest = {
        "name": "ciel", "engine": "chatterbox_turbo", "reference_audio": str(reference),
        "reference_transcript": str(transcript),
    }
    profile.write_text(json.dumps(manifest), encoding="utf-8")
    assert check_named(local_install, "Voice assets").status == "ok"
    transcript.unlink()
    assert "reference_transcript" in check_named(local_install, "Voice assets").detail


def test_turbo_local_files_do_not_claim_separate_runtime_readiness(local_install, tmp_path):
    local_install.tts_engine = "chatterbox"
    local_install.tts_voice = "ciel"
    python = tmp_path / "turbo-python"
    python.symlink_to(sys.executable)
    local_install.tts_chatterbox_python = str(python)
    model = tmp_path / "turbo-model"
    model.mkdir()
    local_install.tts_chatterbox_model = str(model)
    for filename in (
        "inventory.json", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
        "ve.safetensors", "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json",
    ):
        (model / filename).write_bytes(b"not loaded")
    checks = {check.name: check for check in diagnostics.run_checks(local_install)}
    assert checks["Turbo environment"].status == "ok"
    assert str(python) in checks["Turbo environment"].detail
    assert "CUDA were not tested" in checks["Turbo environment"].hint
    assert checks["Turbo model files"].status == "ok"
    assert "Not loaded" in checks["Turbo model files"].detail


def test_disabled_speech_checks_piper_but_skips_pocket_and_assets(local_install, monkeypatch):
    local_install.tts_enabled = False
    local_install.tts_engine = "pocket"
    looked_up = []
    monkeypatch.setattr(
        diagnostics, "find_spec", lambda module: looked_up.append(module) or object(),
    )
    checks = {check.name: check for check in diagnostics.run_checks(local_install)}
    assert checks["Voice assets"].status == "ok"
    assert checks["Speech output dependencies"].status == "ok"
    assert "piper" in looked_up
    assert "pocket_tts" not in looked_up


def test_memory_directory_collision_and_permission_failure(local_install, monkeypatch, tmp_path):
    path = tmp_path / "file"
    path.write_text("keep this", encoding="utf-8")
    local_install.memory_db_path = str(path / "memory.db")
    assert check_named(local_install, "Memory location").status == "error"
    local_install.memory_db_path = str(path)
    real_access = diagnostics.os.access
    monkeypatch.setattr(diagnostics.os, "access", lambda target, mode: (
        False if Path(target) == path else real_access(target, mode)
    ))
    assert check_named(local_install, "Memory location").status == "error"


def test_memory_in_memory_and_existing_file_without_sqlite_changes(local_install, tmp_path):
    local_install.memory_db_path = ":memory:"
    assert check_named(local_install, "Memory location").status == "warning"
    database = tmp_path / "memory.db"
    database.write_bytes(b"a database is never opened by this probe")
    local_install.memory_db_path = str(database)
    before = database.stat()
    assert check_named(local_install, "Memory location").status == "ok"
    after = database.stat()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size == before.st_size
    assert not database.with_name("memory.db-wal").exists()


def test_module_and_checks_do_not_load_models_write_or_contact_network(tmp_path):
    script = r'''
import sys
from pathlib import Path
from ciel.config import Settings

attempts = []
def guard(event, args):
    if event == "open":
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
            isinstance(flags, int) and flags & (64 | 512 | 1024 | 1 | 2)
        ):
            attempts.append(event)
            raise AssertionError("write attempted")
    if event in {"os.mkdir", "os.remove", "os.rename", "subprocess.Popen", "socket.connect"}:
        attempts.append(event)
        raise AssertionError(event)
sys.addaudithook(guard)
from ciel import diagnostics
assert not any(module in sys.modules for module in (
    "sounddevice", "faster_whisper", "pocket_tts", "piper", "torch",
    "ciel.audio.tts", "ciel.setup_wizard", "ciel.providers", "ciel.memory.store",
))
diagnostics._list_audio_devices = lambda: []
diagnostics.find_spec = lambda module: object()
settings = Settings(_env_file=None, tts_engine="piper", tts_voice="missing",
                    memory_db_path=str(Path(sys.argv[1]) / "missing" / "memory.db"))
checks = diagnostics.run_checks(settings)
assert any(check.name == "Memory location" for check in checks)
assert not attempts, attempts
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=tmp_path, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
