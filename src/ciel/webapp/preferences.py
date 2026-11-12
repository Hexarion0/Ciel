"""Save a group of host preferences: secrets to .env, everything else to config.json."""

from pathlib import Path

from ciel.config_store import save_configuration


def save_preferences(path: Path, changes: dict[str, str]) -> None:
    """Atomically save each change to the file that owns it (``path`` is the .env)."""
    save_configuration(path, changes)
