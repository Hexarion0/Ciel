"""Validated browser settings with persistent preferences and no credential readback."""

import re
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr, ValidationError

from ciel.config import Settings
from ciel.persona import load_persona_preferences, update_setup_persona_file
from ciel.webapp.preferences import save_preferences

FIELDS = (
    "ciel_preferred_name",
    "primary_provider",
    "nim_model",
    "nim_complex_model",
    "nim_fallback_model",
    "nim_reasoning",
    "groq_model",
    "openrouter_model",
    "ollama_model",
    "ollama_host",
    "nim_api_key",
    "groq_api_key",
    "openrouter_api_key",
    "screen_enabled",
    "screen_provider",
    "screen_model",
    "tts_enabled",
    "tts_engine",
    "tts_voice",
    "tts_speed",
    "tts_pocket_builtin_voice",
    "stt_model",
    "stt_device",
    "stt_compute_type",
    "stt_language",
    "memory_learning_mode",
    "memory_recall_enabled",
    "internet_enabled",
)
SECRETS = frozenset({"nim_api_key", "groq_api_key", "openrouter_api_key"})
RESTART_FIELDS = frozenset(
    {
        "tts_enabled",
        "tts_engine",
        "tts_voice",
        "tts_speed",
        "tts_pocket_builtin_voice",
        "stt_model",
        "stt_device",
        "stt_compute_type",
        "stt_language",
        "screen_provider",
        "screen_model",
    }
)


class WebSettings:
    """Keep settings owned by the server while exposing only explicit UI preferences."""

    def __init__(
        self,
        settings: Settings,
        env_path: Path,
        *,
        apply: Callable[[set[str]], None],
        recall: Callable[[], bool] | None = None,
    ) -> None:
        self.settings, self.env_path, self.apply = settings, env_path, apply
        self._lock = threading.RLock()
        self.needs_restart = False
        self.recall = recall

    def snapshot(self) -> dict:
        """Return editable values and key-presence flags, never saved credentials."""
        with self._lock:
            values = {key: getattr(self.settings, key) for key in FIELDS if key not in SECRETS}
            if self.recall is not None:
                values["memory_recall_enabled"] = self.recall()
            configured = {
                key: bool(
                    getattr(self.settings, key)
                    and getattr(self.settings, key).get_secret_value().strip()
                )
                for key in SECRETS
            }
            return {
                "values": values,
                "configured": configured,
                "persona": load_persona_preferences(self.settings.ciel_persona_file),
                "needs_restart": self.needs_restart,
            }

    def save(self, changes: dict) -> dict:
        """Validate a whole change before persistence; reject arbitrary files and fields."""
        if not isinstance(changes, dict) or not changes or set(changes) - {*FIELDS, "persona"}:
            raise ValueError("Choose a supported setting to change.")
        with self._lock:
            persona = changes.get("persona")
            if persona is not None and (not isinstance(persona, str) or len(persona) > 4000):
                raise ValueError("Keep personality instructions within 4000 characters.")
            fields = {key: value for key, value in changes.items() if key != "persona"}
            for key, value in fields.items():
                if isinstance(value, str) and (
                    len(value) > (4096 if key in SECRETS else 256)
                    or any(ord(c) < 32 for c in value)
                ):
                    raise ValueError("A setting contains invalid text.")
            if "tts_voice" in fields and not re.fullmatch(r"[\w.-]{1,100}", fields["tts_voice"]):
                raise ValueError("Choose a voice name, not a file path.")
            try:
                candidate = Settings.model_validate({**self.settings.model_dump(), **fields})
            except (ValueError, ValidationError) as error:
                raise ValueError("Check the selected values and try again.") from error
            updates = {}
            for key in fields:
                value = getattr(candidate, key)
                encoded = (
                    value.get_secret_value()
                    if isinstance(value, SecretStr)
                    else ""
                    if value is None
                    else str(value).lower()
                    if isinstance(value, bool)
                    else str(value)
                )
                updates[key.upper()] = encoded
            if persona is not None:
                if not candidate.ciel_persona_file:
                    raise ValueError(
                        "Set a personality file on the host before editing this section."
                    )
                if not update_setup_persona_file(candidate.ciel_persona_file, persona):
                    raise ValueError("Personality preferences could not be saved.")
            save_preferences(self.env_path, updates)
            for key in fields:
                setattr(self.settings, key, getattr(candidate, key))
            self.needs_restart |= bool(set(fields) & RESTART_FIELDS)
            self.apply(set(fields))
            return self.snapshot()
