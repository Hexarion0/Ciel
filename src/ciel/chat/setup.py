"""Configure chat channels without importing speech engines or downloading models."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path

from ciel.config_store import load_configuration, save_configuration
from ciel.terminal_ui import ConsoleUI


def _services(configuration: dict[str, str]) -> list[str]:
    return [value.strip().lower() for value in configuration.get("CHAT_SERVICES", "").split(",")
            if value.strip()]


def configure_chat(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Select a chat service before requesting its token, retaining saved secrets on skip."""
    ui.step("Chat connections")
    ui.note("Telegram is available now. Discord and Signal can be added later.")
    ui.note("Chat starts automatically and stays available after login. "
            "This PC must be on and connected.")
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
        ui.note("Chat starts automatically after saving and whenever you log in. "
                "Pair Telegram once to connect your account.")


def chat_review_rows(configuration: dict[str, str]) -> list[tuple[str, str]]:
    """Summarize selected channels without implying a live connection or owner pairing."""
    services = _services(configuration)
    if not services:
        return [("Chat", "Off")]
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
            ("Chat startup", "Automatic · applied after saving")]


def apply_chat_autostart(
    ui: ConsoleUI, configuration: dict[str, str], *, project_dir: Path,
) -> bool:
    """Always start selected, configured services and report the actual systemd result."""
    from ciel.chat.autostart import reconcile_chat_startup
    from ciel.config import ChatConfig

    chat = ChatConfig(services=_services(configuration),
                      telegram_bot_token=configuration.get("TELEGRAM_BOT_TOKEN") or None,
                      state_path=configuration.get("CHAT_STATE_PATH", "data/chat.db"))
    result = reconcile_chat_startup(chat, project_dir=project_dir, restart=True)
    if chat.services or not result.success:
        ui.note(result.message)
    return result.success


def _save_configuration(env_path: Path, configuration: dict[str, str]) -> None:
    """Save only what changed: the bot token to .env, chat options to config.json."""
    original = load_configuration(env_path)
    changes = {key: value for key, value in configuration.items() if original.get(key) != value}
    if changes:
        save_configuration(env_path, changes)


def run_chat_setup() -> bool:
    """Run a small chat-only configure/review/save wizard without audio initialization."""
    ui = ConsoleUI()
    saved = False
    try:
        ui.banner("Connect Ciel to your chat.")
        ui.note("Review before saving. Press Ctrl+C to leave setup.")
        env_path = Path(".env")
        configuration = load_configuration(env_path)
        configuration.setdefault("CHAT_SERVICES", "")
        configuration.setdefault("CHAT_STATE_PATH", "data/chat.db")
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
        ui.finish("Chat settings saved." + (
            " Automatic startup needs attention; Ciel will retry when opened."
            if not success else ""
        ))
        return success
    except (KeyboardInterrupt, EOFError):
        ui.finish("Chat settings were saved. Rerun ciel chat setup to finish." if saved
                  else "Chat setup cancelled. No changes were saved.")
        return False
    except (OSError, UnicodeError):
        ui.finish("Chat setup could not finish. Check the .env file permissions and try again.")
        return False
