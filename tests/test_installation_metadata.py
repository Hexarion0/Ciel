"""A fresh installation can choose features before downloading voice dependencies."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
VOICE_PACKAGES = {"faster-whisper", "piper-tts", "pysilero-vad", "onnxruntime"}


def _names(requirements: list[str]) -> set[str]:
    """Compare declared distribution names independently of version constraints and extras."""
    return {re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0] for requirement in requirements}


def test_base_install_keeps_setup_dependencies_and_excludes_optional_engines():
    base = _names(PROJECT["dependencies"])
    assert {"pydantic", "pydantic-settings", "python-dotenv", "packaging", "httpx"} <= base
    assert {"numpy", "sounddevice", "soundfile"} <= base
    assert not base & (VOICE_PACKAGES | {"openwakeword", "torch", "pocket-tts"})
    assert PROJECT["description"] == "A personal desktop voice assistant"


def test_feature_extras_cover_runtime_training_and_existing_unit_tests():
    extras = PROJECT["optional-dependencies"]
    assert VOICE_PACKAGES <= _names(extras["voice"])
    assert "openwakeword" in _names(extras["wake"])
    assert {"openwakeword", "onnx", "scikit-learn"} <= _names(extras["train"])
    assert VOICE_PACKAGES | {"openwakeword", "pytest", "ruff"} <= _names(extras["dev"])
    assert "pocket-tts" in _names(extras["pocket-tts"])
    assert not _names(extras["voice"]) & {"torch", "pocket-tts", "nvidia-cublas-cu12"}


@pytest.mark.parametrize("module", ["ciel.audio.engines", "ciel.setup_wizard"])
def test_setup_modules_import_without_optional_speech_dependencies(tmp_path, module):
    script = """
import importlib
import importlib.abc
import sys

blocked = {
    "faster_whisper", "torch", "piper", "openwakeword", "onnxruntime",
    "pysilero_vad", "pocket_tts", "chatterbox",
}
attempts = []

class NoSpeechImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            attempts.append(fullname)
            raise AssertionError("Optional speech dependency imported: " + fullname)

sys.meta_path.insert(0, NoSpeechImports())
importlib.import_module(sys.argv[1])
assert not attempts, attempts
"""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script, module], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("selected,expected", [
    ("auto", "piper"), ("PIPER", "piper"), ("pocket", "pocket"),
    ("hq", "chatterbox_turbo"), ("chatterbox", "chatterbox_turbo"),
    ("chatterbox_turbo", "chatterbox_turbo"),
])
def test_lightweight_engine_resolver_preserves_runtime_selections(selected, expected):
    from ciel.audio.engines import resolve_tts_engine
    from ciel.audio.tts import resolve_tts_engine as runtime_resolver

    assert resolve_tts_engine("ciel", selected) == expected
    assert runtime_resolver is resolve_tts_engine


def test_lightweight_engine_resolver_rejects_unknown_engines():
    from ciel.audio.engines import resolve_tts_engine

    with pytest.raises(ValueError, match="Unknown TTS engine"):
        resolve_tts_engine("ciel", "missing")
