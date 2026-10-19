"""Read-only, offline installation checks without loading speech models."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ciel.paths import resolve_asset_path, voice_profile_path

if TYPE_CHECKING:
    from ciel.config import AudioConfig, Settings


@dataclass(frozen=True)
class DiagnosticCheck:
    """One installation observation; success does not establish runtime readiness."""

    name: str
    status: str
    detail: str
    hint: str = ""


def _probe(name: str, inspect: Callable[[], DiagnosticCheck]) -> DiagnosticCheck:
    """Keep an unavailable dependency or inaccessible path from stopping other checks."""
    try:
        return inspect()
    except Exception as error:
        return DiagnosticCheck(
            name, "error", f"Inspection failed ({type(error).__name__}).",
            "Check the local installation and file permissions.",
        )


def _list_audio_devices() -> list[Any]:
    """Use the bounded native probe only when doctor explicitly inspects hardware."""
    from ciel.device_probe import list_devices

    return list_devices()


def _audio_check(
    name: str, devices: list[Any], configured: int | str | None, *, is_input: bool,
) -> DiagnosticCheck:
    """Validate the configured selection directly, without the backend's fallback."""
    channel_field = "max_input_channels" if is_input else "max_output_channels"
    candidates = [device for device in devices if getattr(device, channel_field) > 0]
    kind = "input" if is_input else "output"
    if configured is not None:
        matches = [device for device in candidates if (
            device.index == configured if isinstance(configured, int)
            else configured.casefold() in device.name.casefold()
        )]
        if not matches:
            return DiagnosticCheck(
                name, "error", f"Configured {kind} device {configured!r} is unavailable.",
                "Run 'ciel devices' and choose a matching device in 'ciel setup'.",
            )
        selected = matches[0]
        selection = "Configured"
    elif candidates:
        default_flag = "is_default_input" if is_input else "is_default_output"
        selected = next((device for device in candidates if getattr(device, default_flag)),
                        candidates[0])
        selection = "Default" if getattr(selected, default_flag) else "First available"
    else:
        return DiagnosticCheck(
            name, "error", f"No {kind} device was reported by the audio backend.",
            "Connect an audio device and check your system's sound settings.",
        )
    return DiagnosticCheck(
        name, "ok", f"{selection}: #{selected.index} {selected.name}. Capture/playback not tested.",
    )


def _dependencies(name: str, modules: tuple[str, ...], hint: str) -> DiagnosticCheck:
    """Look up top-level module metadata without importing packages or models."""
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        return DiagnosticCheck(name, "error", "Missing packages: " + ", ".join(missing), hint)
    return DiagnosticCheck(
        name, "ok", "Packages found: " + ", ".join(modules) + ". Imports/models not tested.",
    )


def _usable_file(path: Path) -> bool:
    """Inspect only regular, readable, nonempty files, without opening private assets."""
    return path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK)


def _piper_paths(voice: str) -> tuple[Path, Path]:
    """Locate Piper files without invoking the runtime's automatic download helper."""
    direct = resolve_asset_path(voice)
    model = direct if direct.suffix == ".onnx" else resolve_asset_path("models/tts") / (
        voice + ".onnx"
    )
    configuration = model.with_suffix(".onnx.json")
    alternate = model.with_suffix(".json")
    if not configuration.is_file() and alternate.is_file():
        configuration = alternate
    return model, configuration


def _piper_assets(voice: str) -> DiagnosticCheck:
    """Report selected local Piper files, without downloading or parsing a model."""
    missing = [str(path) for path in _piper_paths(voice) if not _usable_file(path)]
    if missing:
        return DiagnosticCheck(
            "Voice assets", "error", "Missing, empty, or unreadable Piper files: "
            + ", ".join(missing),
            "Run 'ciel setup' to install the selected voice; doctor does not download files.",
        )
    return DiagnosticCheck(
        "Voice assets", "ok", f"Piper files found for {voice}. Model not loaded.",
    )


def _pocket_assets(audio: AudioConfig) -> DiagnosticCheck:
    """Inspect the optional custom reference while allowing a built-in Pocket voice."""
    if audio.tts_pocket_neutral_reference:
        reference = resolve_asset_path(audio.tts_pocket_neutral_reference)
    else:
        references = resolve_asset_path(audio.tts_pocket_references)
        reference = next((references / ("neutral" + suffix) for suffix in (
            ".wav", ".flac", ".ogg", ".safetensors",
        ) if (references / ("neutral" + suffix)).is_file()), None)
    if reference is None:
        return DiagnosticCheck(
            "Voice assets", "warning",
            f"No custom Pocket reference; using built-in '{audio.tts_pocket_builtin_voice}'.",
            "Pocket model cache/access and playback have not been tested.",
        )
    if not _usable_file(reference):
        return DiagnosticCheck(
            "Voice assets", "error",
            f"Pocket reference is missing, empty, or unreadable: {reference}",
            "Restore the configured reference or select another voice in 'ciel setup'.",
        )
    return DiagnosticCheck(
        "Voice assets", "ok", f"Pocket reference found: {reference}. Model/audio not validated.",
    )


def _turbo_python(audio: AudioConfig) -> DiagnosticCheck:
    """Preserve virtual-environment symlinks and inspect, without running, its Python."""
    python = resolve_asset_path(audio.tts_chatterbox_python, preserve_symlinks=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        return DiagnosticCheck(
            "Turbo environment", "error",
            f"Python executable is missing or not executable: {python}",
            "Install or restore the separate Turbo environment; see the custom voice guide.",
        )
    return DiagnosticCheck(
        "Turbo environment", "ok", f"Python executable found: {python}.",
        "Separate-environment packages and CUDA were not tested.",
    )


def _turbo_profile(audio: AudioConfig) -> DiagnosticCheck:
    """Read the small manifest and check reference paths without reading private audio."""
    manifest = voice_profile_path(audio.tts_voice, audio.tts_voice_profiles)
    if not _usable_file(manifest):
        return DiagnosticCheck("Voice assets", "error", f"Voice profile unavailable: {manifest}")
    if manifest.stat().st_size > 1024 * 1024:
        return DiagnosticCheck("Voice assets", "error", f"Voice profile is too large: {manifest}")
    profile = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("name") != audio.tts_voice or (
        profile.get("engine") != "chatterbox_turbo"
    ):
        return DiagnosticCheck("Voice assets", "error", f"Invalid Turbo voice profile: {manifest}")
    missing = []
    for field in ("reference_audio", "reference_transcript"):
        value = profile.get(field)
        if not isinstance(value, str) or not value or not _usable_file(resolve_asset_path(value)):
            missing.append(field)
    if missing:
        return DiagnosticCheck(
            "Voice assets", "error", "Missing, empty, or unreadable Turbo references: "
            + ", ".join(missing), "Restore your private reference files.",
        )
    return DiagnosticCheck(
        "Voice assets", "ok", "Turbo profile/reference files found. Audio/model not validated.",
    )


def _turbo_model(audio: AudioConfig) -> DiagnosticCheck:
    """Check expected model files without loading checkpoint contents."""
    model = resolve_asset_path(audio.tts_chatterbox_model)
    expected = (
        "inventory.json", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
        "ve.safetensors", "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json",
    )
    missing = [filename for filename in expected if not _usable_file(model / filename)]
    if missing:
        return DiagnosticCheck(
            "Turbo model files", "error", f"Missing, empty, or unreadable files in {model}: "
            + ", ".join(missing), "Restore the pinned Turbo model; see the custom voice guide.",
        )
    return DiagnosticCheck(
        "Turbo model files", "ok", f"Expected files found in {model}. Not loaded.",
    )


def _fallback_assets(audio: AudioConfig) -> DiagnosticCheck:
    """Check whether at least one offline Piper fallback has its local file pair."""
    for voice in dict.fromkeys((audio.tts_fallback_voice, "en_US-amy-medium")):
        if all(_usable_file(path) for path in _piper_paths(voice)):
            return DiagnosticCheck(
                "Piper fallback", "ok", f"Fallback files found for {voice}. Model not loaded.",
            )
    return DiagnosticCheck(
        "Piper fallback", "warning", "No complete local Piper fallback voice was found.",
        "Select Amy in 'ciel setup' to install a fallback voice.",
    )


def _memory_location(db_path: str) -> DiagnosticCheck:
    """Inspect filesystem permissions without opening SQLite or creating directories."""
    if db_path == ":memory:":
        return DiagnosticCheck(
            "Memory location", "warning", "Memory database is temporary (:memory:).",
            "Choose a file path to retain memory after restart.",
        )
    path = Path(db_path).absolute()
    if path.exists() and (not path.is_file() or not os.access(path, os.R_OK | os.W_OK)):
        return DiagnosticCheck(
            "Memory location", "error", f"Database path is not a readable, writable file: {path}",
            "Choose a writable file path or correct its permissions.",
        )
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        return DiagnosticCheck(
            "Memory location", "error",
            f"Database directory cannot be created or written: {path.parent}",
            "Choose a writable database directory or correct its permissions.",
        )
    detail = f"Filesystem permissions allow the configured location: {path}."
    if not path.exists():
        detail += " Database not created yet."
    return DiagnosticCheck(
        "Memory location", "ok", detail,
        "Permission check only; database integrity and free space were not tested.",
    )


def run_checks(settings: Settings) -> list[DiagnosticCheck]:
    """Inspect local prerequisites without network, downloads, recording, or writes."""
    audio = settings.audio
    checks = []
    try:
        devices = _list_audio_devices()
    except Exception as error:
        for name in ("Microphone", "Speakers"):
            checks.append(DiagnosticCheck(
                name, "error", f"Audio device discovery failed ({type(error).__name__}).",
                "Check audio dependencies, PortAudio, and the system's sound settings.",
            ))
    else:
        for name, configured, is_input in (
            ("Microphone", audio.input_device, True),
            ("Speakers", audio.output_device, False),
        ):
            checks.append(_probe(name, lambda n=name, c=configured, i=is_input: _audio_check(
                n, devices, c, is_input=i,
            )))
    for name, modules in (
        ("Audio dependencies", ("numpy", "sounddevice", "soundfile")),
        ("Speech recognition dependencies", ("faster_whisper", "ctranslate2")),
    ):
        checks.append(_probe(name, lambda n=name, m=modules: _dependencies(
            n, m, "Run 'ciel setup' to install the selected speech components."
            if n == "Speech recognition dependencies"
            else "Install CIEL's required dependencies again.",
        )))
    engine = {"auto": "piper", "hq": "chatterbox_turbo",
              "chatterbox": "chatterbox_turbo"}.get(audio.tts_engine, audio.tts_engine)
    # The shared desktop runtime imports Piper even with speech output disabled.
    pocket_enabled = audio.tts_enabled and engine == "pocket"
    dependencies = ("piper", "pocket_tts") if pocket_enabled else ("piper",)
    checks.append(_probe("Speech output dependencies", lambda: _dependencies(
        "Speech output dependencies", dependencies,
        "Run 'ciel setup' to install the selected voice components.",
    )))
    if audio.tts_enabled:
        if engine == "piper":
            checks.append(_probe("Voice assets", lambda: _piper_assets(audio.tts_voice)))
        elif engine == "pocket":
            checks.append(_probe("Voice assets", lambda: _pocket_assets(audio)))
        else:
            for name, inspect in (
                ("Turbo environment", lambda: _turbo_python(audio)),
                ("Voice assets", lambda: _turbo_profile(audio)),
                ("Turbo model files", lambda: _turbo_model(audio)),
            ):
                checks.append(_probe(name, inspect))
        if engine != "piper":
            checks.append(_probe("Piper fallback", lambda: _fallback_assets(audio)))
    else:
        checks.append(DiagnosticCheck("Voice assets", "ok", "Speech output is disabled."))
    checks.append(_probe("Memory location", lambda: _memory_location(settings.memory.db_path)))
    return checks
