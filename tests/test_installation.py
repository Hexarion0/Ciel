"""Feature preparation installs only missing selected requirements in the active environment."""

import subprocess
from unittest.mock import MagicMock, call

import pytest
from packaging.requirements import Requirement

from ciel import installation
from ciel.config import AudioConfig


@pytest.mark.parametrize("platform,settings,expected", [
    ("linux", {"stt_device": "cpu", "tts_engine": "piper"}, ("voice",)),
    ("linux", {"stt_device": "cuda", "tts_engine": "piper"}, ("voice", "gpu")),
    ("linux", {"stt_device": "auto", "tts_enabled": False}, ("voice",)),
    ("darwin", {"stt_device": "cuda", "tts_enabled": False}, ("voice",)),
    ("win32", {"stt_device": "cuda", "tts_enabled": False}, ("voice",)),
    ("linux", {"stt_device": "cpu", "wake_models": ["models/custom/hey_ciel.onnx"],
               "tts_engine": "piper"}, ("voice", "wake")),
    ("linux", {"stt_device": "cpu", "tts_engine": "pocket"}, ("voice", "pocket-tts")),
    ("linux", {"stt_device": "cpu", "tts_engine": "pocket",
               "tts_pocket_quantization": "int8"}, ("voice", "pocket-int8")),
    ("linux", {"stt_device": "cpu", "tts_engine": "pocket",
               "tts_enabled": False}, ("voice",)),
    ("linux", {"stt_device": "cpu", "tts_engine": "chatterbox_turbo"}, ("voice",)),
])
def test_selected_features_follow_saved_voice_and_recognition_choices(
    monkeypatch, platform, settings, expected,
):
    monkeypatch.setattr(installation.sys, "platform", platform)
    assert installation.selected_features(AudioConfig(**settings)) == expected


def test_prepare_components_installs_dependencies_before_downloading_models(monkeypatch):
    events = []
    audio = AudioConfig(stt_device="cpu", tts_engine="pocket")
    notify = MagicMock()
    monkeypatch.setattr(installation, "ensure_features", lambda features, report: events.append(
        ("dependencies", features, report),
    ))
    monkeypatch.setattr(installation, "prepare_models", lambda settings, report: events.append(
        ("models", settings, report),
    ))

    installation.prepare_components(audio, notify)

    assert events == [("dependencies", ("voice", "pocket-tts"), notify),
                      ("models", audio, notify)]


def test_feature_requirements_use_installed_metadata_and_applicable_extra_markers(monkeypatch):
    monkeypatch.setattr(installation.metadata, "requires", lambda name: [
        "packaging>=23.0",
        'faster-whisper>=1.0; extra == "voice"',
        'piper-tts[alignment]>=1.8; extra == "voice"',
        'missing-old-python>=1; python_version < "3.0" and extra == "voice"',
        'openwakeword>=0.4; extra == "wake"',
    ])

    requirements = installation._feature_requirements("voice")

    assert {requirement.name for requirement in requirements} == {"faster-whisper", "piper-tts"}
    assert next(req for req in requirements if req.name == "piper-tts").extras == {"alignment"}
    assert all(requirement.marker is None for requirement in requirements)


@pytest.mark.parametrize("version,expected", [(None, False), ("3.2.0", False), ("3.3.0", True)])
def test_requirement_checks_installed_version(monkeypatch, version, expected):
    def installed_version(name):
        assert name == "pocket-tts"
        if version is None:
            raise installation.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(installation.metadata, "version", installed_version)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: [])
    assert installation._requirement_satisfied(Requirement("pocket-tts==3.3.0")) is expected


def test_installed_pocket_with_missing_transitive_torch_is_prepared_again(monkeypatch):
    requirements = {
        "ciel": ['pocket-tts==3.3.0; extra == "pocket-tts"'],
        "pocket-tts": ["torch>=2.5.0"],
    }

    def installed_version(name):
        if name == "pocket-tts":
            return "3.3.0"
        raise installation.metadata.PackageNotFoundError(name)

    install = MagicMock()
    monkeypatch.setattr(installation, "_pip_install", install)
    monkeypatch.setattr(installation.metadata, "version", installed_version)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: requirements.get(name, []))

    installation.ensure_features(("pocket-tts",), MagicMock())

    assert install.call_args_list == [
        call(["torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu"]),
        call(["pocket-tts==3.3.0"]),
    ]


@pytest.mark.parametrize("extras,expected", [("", True), ("[quantize]", False)])
def test_dependency_checks_respect_environment_markers_and_requested_extras(
    monkeypatch, extras, expected,
):
    requirements = {
        "parent": [
            "always>=1",
            'missing-platform>=1; sys_platform == "nonexistent-platform"',
            'missing-unused-extra>=1; extra == "unused"',
            'missing-quantizer>=1; extra == "quantize"',
        ],
    }

    def installed_version(name):
        if name in {"parent", "always"}:
            return "1.0"
        raise installation.metadata.PackageNotFoundError(name)

    version = MagicMock(side_effect=installed_version)
    monkeypatch.setattr(installation.metadata, "version", version)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: requirements.get(name, []))

    assert installation._requirement_satisfied(Requirement(f"parent{extras}>=1")) is expected
    checked = {invocation.args[0] for invocation in version.call_args_list}
    assert "always" in checked
    assert not checked & {"missing-platform", "missing-unused-extra"}


@pytest.mark.parametrize("required_parent,expected", [("parent>=1", True), ("parent>=2", False)])
def test_circular_dependencies_terminate_and_still_enforce_versions(
    monkeypatch, required_parent, expected,
):
    requirements = {"parent": ["child>=1"], "child": [required_parent]}
    version = MagicMock(return_value="1.0")
    monkeypatch.setattr(installation.metadata, "version", version)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: requirements.get(name, []))

    assert installation._requirement_satisfied(Requirement("parent>=1")) is expected
    assert version.call_count <= 4


def test_satisfied_features_skip_all_package_installs(monkeypatch):
    install = MagicMock()
    monkeypatch.setattr(installation, "_pip_install", install)
    monkeypatch.setattr(installation, "_feature_requirements", lambda feature: [
        Requirement("pocket-tts==3.3.0"),
    ])
    monkeypatch.setattr(installation, "_requirement_satisfied", lambda requirement: True)

    installation.ensure_features(("pocket-tts",), MagicMock())

    install.assert_not_called()


@pytest.mark.parametrize("torch_ready", [False, True])
def test_pocket_preinstalls_cpu_torch_only_when_required_and_strips_extra_markers(
    monkeypatch, torch_ready,
):
    install = MagicMock()
    monkeypatch.setattr(installation, "_pip_install", install)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: [
        'pocket-tts==3.3.0; extra == "pocket-tts"',
    ])
    monkeypatch.setattr(
        installation, "_requirement_satisfied",
        lambda requirement: requirement.name == "torch" and torch_ready,
    )

    installation.ensure_features(("pocket-tts",), MagicMock())

    expected = [] if torch_ready else [call([
        "torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu",
    ])]
    expected.append(call(["pocket-tts==3.3.0"]))
    assert install.call_args_list == expected


def test_non_pocket_feature_installs_missing_packages_without_torch(monkeypatch):
    install = MagicMock()
    monkeypatch.setattr(installation, "_pip_install", install)
    monkeypatch.setattr(installation.metadata, "requires", lambda name: [
        'faster-whisper>=1.0; extra == "voice"',
        'piper-tts[alignment]>=1.8; extra == "voice"',
    ])
    monkeypatch.setattr(
        installation, "_requirement_satisfied", lambda requirement: requirement.name == "piper-tts",
    )

    installation.ensure_features(("voice",), MagicMock())

    install.assert_called_once_with(["faster-whisper>=1.0"])


def test_package_installs_use_current_environment_interpreter(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(installation.subprocess, "run", run)
    monkeypatch.setattr(installation.sys, "prefix", "/test/ciel/.venv")
    monkeypatch.setattr(installation.sys, "base_prefix", "/test/python")

    installation._pip_install(["faster-whisper>=1.0"])

    assert run.call_args.args[0] == [
        installation.sys.executable, "-m", "pip", "install", "faster-whisper>=1.0",
    ]
    assert run.call_args.kwargs["check"] is True


def test_package_installs_refuse_global_python_changes(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(installation.subprocess, "run", run)
    monkeypatch.setattr(installation.sys, "prefix", "/test/python")
    monkeypatch.setattr(installation.sys, "base_prefix", "/test/python")

    with pytest.raises(RuntimeError, match=r"[Vv]irtual environment|[Vv]irtualenv|\.venv"):
        installation._pip_install(["faster-whisper>=1.0"])

    run.assert_not_called()


def test_failed_package_install_does_not_claim_success(monkeypatch):
    monkeypatch.setattr(installation.sys, "prefix", "/test/ciel/.venv")
    monkeypatch.setattr(installation.sys, "base_prefix", "/test/python")
    monkeypatch.setattr(installation.subprocess, "run", MagicMock(
        side_effect=subprocess.CalledProcessError(1, ["pip"]),
    ))

    with pytest.raises(subprocess.CalledProcessError):
        installation._pip_install(["faster-whisper>=1.0"])
