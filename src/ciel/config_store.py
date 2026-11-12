"""Two configuration files: config.json for ordinary settings, .env for secrets only.

``config.json`` sits beside ``.env`` and holds every setting that is not a
credential, in readable JSON types. ``.env`` keeps API keys and tokens. Both are
parsed with the same rules, and a setting in ``.env`` (or the process
environment) still overrides ``config.json``, so older all-in-one ``.env`` files
keep working.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import types
import typing
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, set_key, unset_key
from pydantic import SecretStr
from pydantic_settings import BaseSettings, EnvSettingsSource

CONFIG_FILENAME = "config.json"
_LOCK = threading.RLock()


def config_path_for(env_path: Path | str) -> Path:
    """The config.json that belongs with a given .env file."""
    return Path(env_path).with_name(CONFIG_FILENAME)


def _as_text(value: Any) -> str | None:
    """Turn a JSON value into the text an environment variable would hold."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def read_config(path: Path) -> dict[str, str]:
    """Read config.json as upper-case names and text values; missing file is empty."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object of settings")
    values = {str(key).upper(): _as_text(value) for key, value in data.items()}
    return {key: value for key, value in values.items() if value is not None}


class JsonConfigSource(EnvSettingsSource):
    """Feed config.json through the same parsing as environment variables."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        self._config_path = path
        super().__init__(settings_cls)

    def _load_env_vars(self) -> Mapping[str, str | None]:
        values = read_config(self._config_path)
        if self.case_sensitive:
            return values
        return {key.lower(): value for key, value in values.items()}


def _env_file(source: Any) -> Path | None:
    """The single .env path a dotenv source reads, or None when files are disabled."""
    env_file = getattr(source, "env_file", None)
    if isinstance(env_file, (list, tuple)):
        env_file = env_file[-1] if env_file else None
    return Path(env_file) if env_file else None


def json_source(settings_cls: type[BaseSettings], dotenv_source: Any) -> JsonConfigSource | None:
    """config.json beside the .env file; tests that pass ``_env_file=None`` skip both."""
    env_file = _env_file(dotenv_source)
    return None if env_file is None else JsonConfigSource(settings_cls, config_path_for(env_file))


@lru_cache(maxsize=1)
def secret_names() -> frozenset[str]:
    """Upper-case names of every credential setting (the SecretStr fields)."""
    from ciel.config import Settings

    return frozenset(name.upper() for name, field in Settings.model_fields.items()
                     if _mentions(field.annotation, SecretStr))


def _mentions(annotation: Any, target: type) -> bool:
    if annotation is target:
        return True
    return any(_mentions(arg, target) for arg in typing.get_args(annotation))


def _json_value(name: str, text: str) -> Any:
    """Store a setting in its natural JSON type when the field's type is plain."""
    from ciel.config import Settings

    field = Settings.model_fields.get(name.lower())
    if field is None:
        return text
    annotation = field.annotation
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        options = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        annotation = options[0] if len(options) == 1 else annotation
        origin = typing.get_origin(annotation)
    try:
        if annotation is bool:
            lowered = text.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        elif annotation is int:
            return int(text)
        elif annotation is float:
            return float(text)
        elif origin in (list, dict, tuple, set):
            return json.loads(text)
    except ValueError:
        pass
    return text


def load_configuration(env_path: Path) -> dict[str, str]:
    """Both files merged as upper-case text, .env winning, for setup screens."""
    merged = read_config(config_path_for(env_path))
    if env_path.is_file():
        for key, value in dotenv_values(env_path, interpolate=False).items():
            if value is not None:
                merged[key.upper()] = value
    return merged


def stored_names(env_path: Path) -> dict[str, str]:
    """Upper-case name to the spelling it is saved under (.env keeps its own casing).

    ``${var}`` references in .env are case-sensitive, so setup must resolve them
    with the names exactly as written.
    """
    names = {key: key for key in read_config(config_path_for(env_path))}
    if env_path.is_file():
        names.update({key.upper(): key for key in dotenv_values(env_path, interpolate=False)})
    return names


def _publish(path: Path, write: Any) -> None:
    """Write a private file atomically: temp copy, edit, chmod 600, replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".ciel-config-", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        temporary.chmod(0o600)
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_configuration(env_path: Path, changes: Mapping[str, str | None]) -> None:
    """Save settings to the right file; ``None`` removes a setting from both.

    Secrets (and ``${VAR}`` references) go to .env. Everything else goes to
    config.json and is removed from .env, where it would otherwise keep
    overriding the new value.
    """
    env_path = env_path.absolute()
    config_path = config_path_for(env_path)
    secrets = secret_names()
    with _LOCK:
        env_values = (dotenv_values(env_path, interpolate=False)
                      if env_path.is_file() else {})
        env_names = {key.upper(): key for key in env_values}
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            config = {}
        if not isinstance(config, dict):
            raise ValueError(f"{config_path} must contain a JSON object of settings")
        config_names = {str(key).upper(): key for key in config}

        env_set: dict[str, str] = {}
        env_unset: set[str] = set()
        config_changed = False
        for raw_name, value in changes.items():
            name = raw_name.upper()
            # ${VAR} references only expand in .env, so they stay there.
            in_env = name in secrets or (value is not None and "${" in value)
            if value is None or not in_env:
                if name in env_names:
                    env_unset.add(env_names[name])
            if value is not None and in_env:
                if env_values.get(env_names.get(name, name)) != value:
                    env_set[env_names.get(name, name)] = value
                if name in config_names:
                    config.pop(config_names.pop(name))
                    config_changed = True
                continue
            if name in config_names:
                config.pop(config_names.pop(name))
                config_changed = True
            if value is not None:
                config[name.lower()] = _json_value(name, value)
                config_names[name] = name.lower()
                config_changed = True

        if env_set or env_unset:
            original = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""

            def write_env(temporary: Path) -> None:
                temporary.write_text(original, encoding="utf-8")
                for key in env_unset:
                    unset_key(temporary, key)
                for key, value in env_set.items():
                    set_key(temporary, key, value, quote_mode="always")

            _publish(env_path, write_env)
        if config_changed:
            text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
            _publish(config_path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    _clear_settings_cache()


def split_env_file(env_path: Path) -> Path:
    """Write the full config.json from the current settings and keep only secrets in .env.

    Every non-secret setting is listed, defaults included, so config.json is the
    one place to read and edit them. Returns the config.json path.
    """
    from ciel.config import Settings

    env_path = env_path.absolute()
    config_path = config_path_for(env_path)
    secrets = {name.lower() for name in secret_names()}
    current = Settings(_env_file=env_path).model_dump(mode="json")
    full = {name: value for name, value in current.items()
            if name not in secrets and value is not None}
    with _LOCK:
        text = json.dumps(full, indent=2, ensure_ascii=False) + "\n"
        _publish(config_path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
        if env_path.is_file():
            keep = {key for key in dotenv_values(env_path, interpolate=False)
                    if key.upper() in secret_names()}
            drop = [key for key in dotenv_values(env_path, interpolate=False) if key not in keep]
            if drop:
                original = env_path.read_text(encoding="utf-8")

                def write_env(temporary: Path) -> None:
                    temporary.write_text(original, encoding="utf-8")
                    for key in drop:
                        unset_key(temporary, key)

                _publish(env_path, write_env)
    _clear_settings_cache()
    return config_path


def _clear_settings_cache() -> None:
    from ciel.config import get_settings

    get_settings.cache_clear()
