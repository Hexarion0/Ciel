"""Manage Ciel's optional Linux login service without storing credentials in it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UNIT_NAME = "ciel-chat.service"
_MANAGED_HEADER = "# Managed by Ciel chat setup\n"


@dataclass(frozen=True)
class AutostartResult:
    """Report whether the requested login service change actually succeeded."""

    success: bool
    message: str


def _unit_path() -> Path:
    """Use the user's systemd configuration directory, honoring XDG settings."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "systemd" / "user" / UNIT_NAME


def _quoted(value: str, *, command: bool = False) -> str:
    """Quote a literal systemd argument, including its percent specifiers."""
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Service paths must not contain control characters")
    if command:
        value = value.replace("$", "$$")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render_unit(project_dir: Path, python: Path | None = None) -> str:
    """Build a chat-only service using the current interpreter and saved setup directory."""
    directory = project_dir.expanduser().resolve()
    if any(ord(character) < 32 or ord(character) == 127 for character in str(directory)):
        raise ValueError("Service paths must not contain control characters")
    # WorkingDirectory takes one literal path, not a shell/ExecStart argument.
    # A trailing slash protects trailing whitespace/backslashes in directory names.
    working_directory = str(directory).replace("%", "%%").rstrip("/") + "/"
    # Resolving a venv interpreter symlink would bypass its installed dependencies.
    interpreter = (python or Path(sys.executable)).expanduser().absolute()
    command = " ".join(_quoted(value, command=True) for value in (
        str(interpreter), "-m", "ciel", "chat", "run",
    ))
    return (
        _MANAGED_HEADER
        + "[Unit]\nDescription=Ciel private chat\n\n"
        + "[Service]\nType=simple\n"
        + f"WorkingDirectory={working_directory}\n"
        + f"ExecStart={command}\n"
        + "Environment=PYTHONUNBUFFERED=1\nRestart=on-failure\nRestartSec=10\n\n"
        + "[Install]\nWantedBy=default.target\n"
    )


def _run_systemctl(*arguments: str) -> subprocess.CompletedProcess[str] | None:
    """Run a bounded user service command without exposing its diagnostic output."""
    try:
        return subprocess.run(
            ["systemctl", "--user", *arguments], capture_output=True, text=True, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _systemctl(*arguments: str) -> bool:
    result = _run_systemctl(*arguments)
    return result is not None and result.returncode == 0


def autostart_status() -> str:
    """Return a verified enabled state, or an honest local/unavailable status."""
    path = _unit_path()
    try:
        if not path.exists():
            return "off"
        if path.is_symlink() or not path.read_text(encoding="utf-8").startswith(_MANAGED_HEADER):
            return "unmanaged service"
        if sys.platform != "linux" or shutil.which("systemctl") is None:
            return "unavailable"
        if not _systemctl("show-environment"):
            return "user session unavailable"
        result = _run_systemctl("is-enabled", UNIT_NAME)
        if result is None:
            return "unavailable"
        state = result.stdout.strip()
        if result.returncode == 0 and state == "enabled":
            return "enabled"
        if state in {"disabled", "masked", "masked-runtime", "static", "not-found"}:
            return "not enabled"
        return "enablement could not be confirmed"
    except (OSError, UnicodeError):
        return "unavailable"


def set_autostart(enabled: bool, project_dir: Path) -> AutostartResult:
    """Enable at next login or remove only Ciel's own managed chat service."""
    unit_path = _unit_path()
    try:
        if unit_path.is_symlink():
            return AutostartResult(False, f"Autostart was not changed: {unit_path} is a symlink.")
        existing = unit_path.read_text(encoding="utf-8") if unit_path.exists() else None
        if existing is not None and not existing.startswith(_MANAGED_HEADER):
            return AutostartResult(False, "Autostart was not changed: ciel-chat.service already "
                                   "exists and is not managed by Ciel.")
        if not enabled and existing is None:
            return AutostartResult(True, "Chat login autostart is off.")
        if sys.platform != "linux" or shutil.which("systemctl") is None:
            return AutostartResult(False, "Login autostart needs Linux with systemd user services. "
                                   "You can still start chat with: ciel chat run")
        if not _systemctl("show-environment"):
            return AutostartResult(False, "Your systemd user session is unavailable. Run "
                                   "ciel chat setup from your logged-in desktop to retry. "
                                   "Chat can still be started with: ciel chat run")
        if enabled:
            content = render_unit(project_dir)
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(content, encoding="utf-8")
            unit_path.chmod(0o600)
            if not _systemctl("daemon-reload") or not _systemctl("enable", UNIT_NAME):
                return AutostartResult(False, "The chat service file was saved, but login "
                                       "autostart could not be enabled. Rerun ciel chat setup.")
            return AutostartResult(True, "Chat will start when you log in. It can receive and "
                                   "send messages while this PC is on and connected.")
        if not _systemctl("disable", "--now", UNIT_NAME):
            return AutostartResult(False, "Chat autostart could not be disabled. Rerun "
                                   "ciel chat setup from your logged-in desktop.")
        unit_path.unlink()
        if not _systemctl("daemon-reload"):
            return AutostartResult(False, "Chat autostart was disabled, but systemd could not "
                                   "reload. Rerun ciel chat setup to check it.")
        return AutostartResult(True, "Chat login autostart is off.")
    except (OSError, UnicodeError, ValueError):
        return AutostartResult(False, "Chat autostart could not be changed. Check your user "
                               "service directory and rerun ciel chat setup.")


def main(action: str) -> int:
    """Manage login startup from the chat CLI without initializing voice or providers."""
    if action == "status":
        status = autostart_status()
        print(f"Chat login autostart: {status}.")
        return 0 if status in {"enabled", "not enabled", "off"} else 1
    if action not in {"enable", "disable"}:
        print("Choose enable, disable, or status.")
        return 1
    from dotenv import dotenv_values

    from ciel.chat.setup import _save_configuration
    from ciel.config import get_settings

    if action == "enable" and not get_settings().chat.services:
        print("Choose a chat connection first with: ciel chat setup")
        return 1
    result = set_autostart(action == "enable", Path.cwd())
    print(result.message)
    if not result.success:
        return 1
    env_path = Path(".env")
    try:
        raw = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
        configuration = {key.upper(): value for key, value in raw.items() if value is not None}
        configuration["CHAT_AUTOSTART"] = "true" if action == "enable" else "false"
        _save_configuration(env_path, configuration)
    except (OSError, UnicodeError):
        print("The login service changed, but the .env preference could not be saved. "
              "Rerun ciel chat setup to review it.")
        return 1
    return 0
