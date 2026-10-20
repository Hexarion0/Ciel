"""Prepare explicitly selected speech features after setup has saved its choices."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

from ciel.audio.engines import resolve_tts_engine
from ciel.config import AudioConfig
from ciel.paths import resolve_asset_path

_FEATURES = {"voice", "gpu", "wake", "pocket-tts", "pocket-int8", "train"}


def selected_features(audio: AudioConfig) -> tuple[str, ...]:
    """Select only the packages needed by the configured recognition and voice engines."""
    features = ["voice"]
    if audio.stt_device.lower() == "cuda" and sys.platform.startswith("linux"):
        features.append("gpu")
    if audio.wake_models:
        features.append("wake")
    if audio.tts_enabled and resolve_tts_engine(audio.tts_voice, audio.tts_engine) == "pocket":
        features.append("pocket-int8" if audio.tts_pocket_quantization == "int8" else "pocket-tts")
    return tuple(features)


def _feature_requirements(feature: str) -> list[Requirement]:
    """Use this installed release's declared extras rather than another package release."""
    if feature not in _FEATURES:
        raise ValueError(f"Unknown installation feature: {feature}")
    requirements = []
    for value in metadata.requires("ciel") or []:
        requirement = Requirement(value)
        marker = requirement.marker
        if marker is not None and marker.evaluate({"extra": feature}) and not marker.evaluate(
            {"extra": ""},
        ):
            requirement.marker = None
            requirements.append(requirement)
    # Linux-only features legitimately have no requirements on other platforms.
    if not requirements and feature != "gpu":
        raise RuntimeError("CIEL package metadata is out of date. Rerun setup.sh.")
    return requirements


def _requirement_satisfied(
    requirement: Requirement, _seen: set[tuple[str, frozenset[str]]] | None = None,
) -> bool:
    """Check installed versions and requested optional dependencies without importing models."""
    try:
        installed = metadata.version(requirement.name)
        if not requirement.specifier.contains(installed, prereleases=True):
            return False
        seen = set() if _seen is None else _seen
        key = (canonicalize_name(requirement.name), frozenset(requirement.extras))
        if key in seen:
            return True
        seen.add(key)
        for value in metadata.requires(requirement.name) or []:
            dependency = Requirement(value)
            marker = dependency.marker
            if marker is None or marker.evaluate({"extra": ""}) or any(
                marker.evaluate({"extra": extra}) for extra in requirement.extras
            ):
                if not _requirement_satisfied(dependency, seen):
                    return False
    except (metadata.PackageNotFoundError, InvalidVersion):
        return False
    return True


def _pip_install(arguments: list[str]) -> None:
    """Install in the current isolated environment, passing arguments without a shell."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run bash setup.sh to install components in CIEL's virtual environment.")
    subprocess.run([sys.executable, "-m", "pip", "install", *arguments], check=True)
    importlib.invalidate_caches()


def ensure_features(features: Iterable[str], notify: Callable[[str], None]) -> None:
    """Install missing selected extras; existing compatible components remain in place."""
    missing: dict[str, Requirement] = {}
    selected = tuple(dict.fromkeys(features))
    for feature in selected:
        for requirement in _feature_requirements(feature):
            if not _requirement_satisfied(requirement):
                missing[str(requirement)] = requirement
    if not missing:
        notify("The selected components are already installed.")
        return
    if any(feature.startswith("pocket-") for feature in selected) and not _requirement_satisfied(
        Requirement("torch>=2.5.0"),
    ):
        notify("Installing CPU PyTorch for the Pocket voice.")
        _pip_install(["torch>=2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
    notify("Installing the missing components for your choices.")
    _pip_install(list(missing))


def prepare_models(audio: AudioConfig, notify: Callable[[str], None]) -> None:
    """Cache the selected voice and speech models, without loading Whisper inference."""
    models = []
    for name in dict.fromkeys((audio.stt_model, audio.wake_stt_model)):
        model_path = Path(name).expanduser()
        if model_path.is_dir():
            notify(f"Using local speech model: {model_path}")
        elif model_path.is_absolute() or name.startswith(("./", "../", "~")):
            raise FileNotFoundError(f"Configured speech model directory is missing: {model_path}")
        else:
            models.append(name)
    from faster_whisper import download_model

    for name in models:
        notify(f"Checking speech model: {name} (cached files are reused).")
        download_model(name)
    if not audio.tts_enabled:
        return
    engine = resolve_tts_engine(audio.tts_voice, audio.tts_engine)
    if engine == "pocket":
        notify("Preparing the selected Pocket voice model. Cached files are reused.")
        _prepare_pocket(audio)
    elif engine == "piper":
        voice = audio.tts_voice
        directory = resolve_asset_path("models/tts")
        direct = resolve_asset_path(voice)
        model = direct if direct.suffix == ".onnx" else directory / f"{voice}.onnx"
        configs = (model.with_suffix(".onnx.json"), model.with_suffix(".json"))
        if model.is_file() and model.stat().st_size and any(
            path.is_file() and path.stat().st_size for path in configs
        ):
            notify(f"Using installed Piper voice: {voice}")
            return
        if direct.suffix == ".onnx":
            raise FileNotFoundError("The selected local Piper model or its config is missing.")
        from piper.download_voices import download_voice

        directory.mkdir(parents=True, exist_ok=True)
        notify(f"Downloading the selected Piper voice: {voice}")
        download_voice(voice, directory)
    else:
        notify("Turbo uses its separate environment and private reference; "
               "setup checks those next.")


def _prepare_pocket(audio: AudioConfig) -> None:
    """Check the selected Pocket model and reference without allowing a fallback voice."""
    from ciel.audio.pocket import PocketTTS

    pocket = PocketTTS(
        resolve_asset_path(audio.tts_pocket_references),
        neutral_reference=(resolve_asset_path(audio.tts_pocket_neutral_reference)
                           if audio.tts_pocket_neutral_reference else None),
        quantization=audio.tts_pocket_quantization,
        language=audio.tts_pocket_language,
        builtin_voice=audio.tts_pocket_builtin_voice,
        gain_db=audio.tts_pocket_gain_db,
    )
    try:
        pocket.warmup()
    finally:
        pocket.close()


def prepare_components(audio: AudioConfig, notify: Callable[[str], None]) -> None:
    """Install selected missing packages, then prepare the models used by this configuration."""
    ensure_features(selected_features(audio), notify)
    prepare_models(audio, notify)
