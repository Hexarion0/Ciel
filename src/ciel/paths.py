"""Separate bundled runtime resources from writable, locally supplied voice assets."""

from __future__ import annotations

import os
from pathlib import Path


def development_root() -> Path | None:
    """Return the source root only for an actual src-layout development checkout."""
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    if package == root / "src/ciel" and (root / "pyproject.toml").is_file():
        return root
    return None


def runtime_data_root() -> Path:
    """Keep checkout assets in place; installed apps use the user's XDG data directory."""
    checkout = development_root()
    if checkout is not None:
        return checkout
    configured = os.environ.get("XDG_DATA_HOME", "")
    data_home = Path(configured).expanduser() if configured else Path.home() / ".local/share"
    if not data_home.is_absolute():
        data_home = Path.home() / ".local/share"
    return data_home / "ciel"


def resolve_asset_path(path: str | Path, *, preserve_symlinks: bool = False) -> Path:
    """Resolve relative configured assets under the writable root, preserving overrides."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = runtime_data_root() / value
    # Python venv launchers must retain their symlink to select the correct environment.
    return Path(os.path.abspath(value)) if preserve_symlinks else value.resolve()


def voice_profile_path(voice: str, profiles_dir: str | Path = "voice_profiles") -> Path:
    """Prefer local profiles, with bundled defaults only for the standard profile directory."""
    local = resolve_asset_path(profiles_dir) / voice / "voice.json"
    if local.is_file() or Path(profiles_dir) != Path("voice_profiles"):
        return local
    bundled = Path(__file__).resolve().parent / "voice_profiles" / voice / "voice.json"
    return bundled if bundled.is_file() else local


def bundled_worker_path() -> Path:
    """Return the installed worker script for execution by the isolated Turbo environment."""
    return Path(__file__).resolve().parent / "audio/turbo_worker.py"
