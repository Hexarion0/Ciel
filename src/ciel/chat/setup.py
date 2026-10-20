"""Configure chat channels without importing speech engines or downloading models."""

from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import dotenv_values, set_key

from ciel.terminal_ui import ConsoleUI


def _enabled(configuration: dict[str, str], key: str) -> bool:
    return configuration.get(key, "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _services(configuration: dict[str, str]) -> list[str]:
    return [value.strip().lower() for value in configuration.get("CHAT_SERVICES", "").split(",")
            if value.strip()]


def configure_chat(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Select a chat service before requesting its token, retaining saved secrets on skip."""
    ui.step("Chat connections")
    ui.note("Telegram is available now. Discord and Signal can be added later.")
    ui.note("Chat runs separately from voice. This PC must be on and connected.")
    services = _services(configuration)
    token_saved = bool(configuration.get("TELEGRAM_BOT_TOKEN", "").strip())
    label = "Telegram"
    if "telegram" in services:
        label += " · enabled" + (" · token configured" if token_saved else " · token needed")
    elif token_saved:
        label += " · saved token"
    choices = [("telegram", label), ("done", "Keep chat connections and continue" if services
                                           else "Skip for now")]
    if services:
        choices.append(("disable", "Disable chat connections; keep saved tokens"))
    selected = ui.choose("Chat service", choices, default="done")
    if selected == "disable":
        configuration["CHAT_SERVICES"] = ""
        configuration["CHAT_AUTOSTART"] = "false"
        return
    if selected == "telegram":
        ui.note("Create a bot with @BotFather in Telegram, then enter its token here. "
                "The token stays in your local .env file.")
        answer = getpass("Telegram bot token [" + (
            "configured; Enter to keep" if token_saved else "optional; Enter to skip"
        ) + "]: ").strip()
        if answer:
            configuration["TELEGRAM_BOT_TOKEN"] = answer
            token_saved = True
        if token_saved:
            if "telegram" not in services:
                services.append("telegram")
            configuration["CHAT_SERVICES"] = ",".join(services)
        else:
            ui.note("Telegram was skipped because no token was entered.")
    if _services(configuration):
        ui.note("Login autostart keeps chat available after you sign in; "
                "it cannot send reminders while your PC is off.")
        start = ui.choose("Chat at login", [
            ("off", "Start chat manually"), ("on", "Start chat when I log in · Linux systemd"),
        ], default="on" if _enabled(configuration, "CHAT_AUTOSTART") else "off")
        configuration["CHAT_AUTOSTART"] = "true" if start == "on" else "false"


def chat_review_rows(configuration: dict[str, str]) -> list[tuple[str, str]]:
    """Summarize selected channels without implying a live connection or owner pairing."""
    services = _services(configuration)
    if not services:
        return [("Chat", "Off"), ("Chat at login", "Off")]
    labels = []
    for service in services:
        if service == "telegram":
            labels.append("Telegram · " + (
                "token configured; pairing checked when connecting"
                if configuration.get("TELEGRAM_BOT_TOKEN") else "token needed"
            ))
        else:
            labels.append(f"{service} · not supported in this version")
    return [("Chat", "; ".join(labels)),
            ("Chat at login", "Requested; applied after saving" if _enabled(
                configuration, "CHAT_AUTOSTART",
            ) else "Off")]


def apply_chat_autostart(
    ui: ConsoleUI, configuration: dict[str, str], *, project_dir: Path,
) -> bool:
    """Apply only the saved chat login preference and report real systemd outcomes."""
    from ciel.chat.autostart import set_autostart

    enabled = bool(_services(configuration)) and _enabled(configuration, "CHAT_AUTOSTART")
    result = set_autostart(enabled, project_dir)
    if enabled or not result.success:
        ui.note(result.message)
    return result.success


def _save_configuration(env_path: Path, configuration: dict[str, str]) -> None:
    """Atomically preserve non-chat settings, comments, and original key spelling."""
    original = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
    names = {key.upper(): key for key in original}
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=env_path.parent, prefix=".ciel-chat-", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(env_path.read_text(encoding="utf-8") if env_path.exists()
                            else "# CIEL Configuration File\n")
        for key, value in configuration.items():
            original_key = names.get(key, key)
            if original.get(original_key) != value:
                set_key(temporary_path, original_key, value, quote_mode="always")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, env_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    from ciel.config import get_settings

    get_settings.cache_clear()


def run_chat_setup() -> bool:
    """Run a small chat-only configure/review/save wizard without audio initialization."""
    ui = ConsoleUI()
    saved = False
    try:
        ui.banner("Connect Ciel to your chat.")
        ui.note("Review before saving. Press Ctrl+C to leave setup.")
        env_path = Path(".env")
        raw = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
        configuration = {key.upper(): value for key, value in raw.items() if value is not None}
        configuration.setdefault("CHAT_SERVICES", "")
        configuration.setdefault("CHAT_STATE_PATH", "data/chat.db")
        configuration.setdefault("CHAT_AUTOSTART", "false")
        while True:
            configure_chat(ui, configuration)
            ui.step("Review chat setup")
            ui.summary(chat_review_rows(configuration))
            selected = ui.choose("Save chat setup?", [
                ("save", "Save"), ("edit", "Change choices"), ("cancel", "Cancel"),
            ], default="save")
            if selected == "cancel":
                ui.finish("Chat setup cancelled. No changes were saved.")
                return False
            if selected == "save":
                break
        _save_configuration(env_path, configuration)
        saved = True
        success = apply_chat_autostart(ui, configuration, project_dir=env_path.parent.resolve())
        if "telegram" in _services(configuration):
            ui.note("Pair your Telegram account with: ciel chat pair telegram")
            ui.note("Then start chat with: ciel chat run")
        ui.finish("Chat settings saved." + (
            " Autostart needs another step." if not success else ""
        ))
        return success
    except (KeyboardInterrupt, EOFError):
        ui.finish("Chat settings were saved. Rerun ciel chat setup to finish." if saved
                  else "Chat setup cancelled. No changes were saved.")
        return False
    except (OSError, UnicodeError):
        ui.finish("Chat setup could not finish. Check the .env file permissions and try again.")
        return False
