"""Voice resources and mutable assets remain usable outside a source checkout."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from ciel.audio.tts import TextToSpeech


def test_installed_voice_uses_user_assets_and_bundled_worker(tmp_path, monkeypatch):
    import ciel.paths as paths

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    asset_root = tmp_path / "user-data" / "ciel"
    transcript = asset_root / "data/voice/references/ciel/reference-1.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("A user-provided reference.", encoding="utf-8")
    python = asset_root / "data/voice/envs/clone/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)

    tts = TextToSpeech(voice_name="ciel", engine="chatterbox_turbo", enabled=False)
    worker = tts._load_chatterbox_worker()

    assert worker.script == paths.bundled_worker_path()
    assert worker.script.is_file()
    assert worker.reference == transcript.with_suffix(".wav")
    assert worker.model == asset_root / "data/voice/models/chatterbox-turbo"
    assert worker.python == python
    assert worker.python != python.resolve()
    assert tts.models_dir == asset_root / "models/tts"
    assert paths.voice_profile_path("ciel").is_file()
    assert not paths.voice_profile_path("ciel").is_relative_to(asset_root)


def test_configured_profile_and_absolute_assets_override_bundle(tmp_path, monkeypatch):
    import ciel.paths as paths

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "assets"))
    profiles = tmp_path / "profiles"
    profile = profiles / "ciel" / "voice.json"
    profile.parent.mkdir(parents=True)
    data = json.loads(paths.voice_profile_path("ciel").read_text(encoding="utf-8"))
    transcript = tmp_path / "custom-reference.txt"
    transcript.write_text("An explicit custom reference.", encoding="utf-8")
    data.update(reference_transcript=str(transcript), reference_audio=str(tmp_path / "custom.wav"))
    profile.write_text(json.dumps(data), encoding="utf-8")

    tts = TextToSpeech(
        voice_name="ciel", engine="chatterbox_turbo", enabled=False,
        voice_profiles_dir=profiles, chatterbox_model_dir=tmp_path / "models",
        chatterbox_python=tmp_path / "python", models_dir=tmp_path / "piper",
    )
    worker = tts._load_chatterbox_worker()
    assert worker.reference == tmp_path / "custom.wav"
    assert worker.reference_text == "An explicit custom reference."
    assert worker.model == tmp_path / "models"
    assert worker.python == tmp_path / "python"
    assert tts.models_dir == tmp_path / "piper"
    assert paths.voice_profile_path("ciel", profiles) == profile
    with pytest.raises(FileNotFoundError):
        TextToSpeech(
            voice_name="ciel", engine="chatterbox_turbo", enabled=False,
            voice_profiles_dir=tmp_path / "missing-profiles",
        )._load_chatterbox_worker()


def test_checkout_keeps_existing_asset_locations(tmp_path, monkeypatch):
    import ciel.paths as paths

    checkout = Path(__file__).resolve().parents[1]
    assert paths.development_root() == checkout
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_asset_path("data/voices/ciel") == checkout / "data/voices/ciel"
    assert paths.voice_profile_path("ciel") == checkout / "voice_profiles/ciel/voice.json"


def test_local_default_profile_overrides_bundled_profile(tmp_path, monkeypatch):
    import ciel.paths as paths

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    manifest = tmp_path / "ciel/voice_profiles/ciel/voice.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    assert paths.voice_profile_path("ciel") == manifest


def test_pocket_reference_paths_use_installed_data_root(tmp_path, monkeypatch):
    import ciel.paths as paths

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    pocket = Mock()
    monkeypatch.setattr("ciel.audio.tts.PocketTTS", pocket)
    TextToSpeech(engine="pocket", enabled=False)._load_pocket()
    assert pocket.call_args.args == (tmp_path / "ciel/data/voices/ciel",)


def test_packaged_worker_runs_without_app_installed_in_alternate_python(tmp_path):
    from ciel.paths import bundled_worker_path

    # Copy only the runtime files, so no source checkout can satisfy worker imports.
    worker = tmp_path / "audio/turbo_worker.py"
    worker.parent.mkdir()
    shutil.copyfile(bundled_worker_path(), worker)
    shutil.copyfile(bundled_worker_path().with_name("vocal_cues.py"),
                    worker.with_name("vocal_cues.py"))
    reference = tmp_path / "reference.wav"
    reference.touch()
    probe = """
import runpy, sys, types
numpy = types.ModuleType('numpy')
torch = types.ModuleType('torch')
torch.set_num_threads = lambda count: None
torch.cuda = types.SimpleNamespace(mem_get_info=lambda: (0, 0))
chatterbox = types.ModuleType('chatterbox')
turbo = types.ModuleType('chatterbox.tts_turbo')
turbo.ChatterboxTurboTTS = object
sys.modules.update(numpy=numpy, torch=torch, chatterbox=chatterbox)
sys.modules['chatterbox.tts_turbo'] = turbo
script, model, reference = sys.argv[1:]
sys.argv = [script, '--model', model, '--reference', reference]
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(worker), str(tmp_path), str(reference)],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert result.returncode == 1
    message = json.loads(result.stdout)
    assert message["ready"] is False
    assert "only 0 MiB is available" in message["error"]
    assert "ModuleNotFoundError" not in result.stderr


def test_packaged_manifest_matches_safe_checkout_manifest():
    from ciel.paths import voice_profile_path

    checkout = Path(__file__).resolve().parents[1]
    packaged = checkout / "src/ciel/voice_profiles/ciel/voice.json"
    assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
        voice_profile_path("ciel").read_text(encoding="utf-8")
    )


def test_installed_package_detects_resources_without_checkout(tmp_path):
    import ciel.paths as paths

    source_package = Path(paths.__file__).parent
    installed = tmp_path / "site-packages/ciel"
    installed.mkdir(parents=True)
    for resource in (
        "__init__.py", "paths.py", "audio/turbo_worker.py", "voice_profiles/ciel/voice.json",
    ):
        target = installed / resource
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_package / resource, target)
    probe = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ['XDG_DATA_HOME'] = sys.argv[2]
from ciel.paths import bundled_worker_path, development_root, runtime_data_root, voice_profile_path
print(json.dumps({
    'checkout': development_root() is not None,
    'data': str(runtime_data_root()),
    'worker': str(bundled_worker_path()),
    'voice': json.loads(voice_profile_path('ciel').read_text())['name'],
}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(installed.parent), str(tmp_path / "data")],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    assert json.loads(result.stdout) == {
        "checkout": False,
        "data": str(tmp_path / "data/ciel"),
        "worker": str(installed / "audio/turbo_worker.py"),
        "voice": "ciel",
    }


@pytest.mark.parametrize("xdg", ["", "relative/path"])
def test_user_data_root_has_home_fallback(tmp_path, monkeypatch, xdg):
    import ciel.paths as paths

    monkeypatch.setattr(paths, "development_root", lambda: None)
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert paths.runtime_data_root() == tmp_path / ".local/share/ciel"
