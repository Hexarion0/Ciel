"""Keep configured chat available automatically through a managed Linux user service."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ciel.config import ChatConfig

UNIT_NAME = "ciel-chat.service"
_MANAGED_HEADER = "# Managed by Ciel chat setup\n"


@dataclass(frozen=True)
class AutostartResult:
    """Report a verified service operation without implying a Telegram connection."""

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


def _configuration_revision(directory: Path) -> str:
    """Track local file edits without reading, hashing, or publishing configuration values."""
    try:
        metadata = (directory / ".env").stat()
    except FileNotFoundError:
        return "absent"
    return ":".join(f"{value:x}" for value in (
        metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_size,
    ))


def render_unit(project_dir: Path, python: Path | None = None) -> str:
    """Build an audio-independent worker using this interpreter and configuration directory."""
    directory = project_dir.expanduser().resolve()
    if any(ord(character) < 32 or ord(character) == 127 for character in str(directory)):
        raise ValueError("Service paths must not contain control characters")
    # WorkingDirectory takes one literal path, not a shell/ExecStart argument.
    # A trailing slash protects trailing whitespace/backslashes in directory names.
    working_directory = str(directory).replace("%", "%%").rstrip("/") + "/"
    # Resolving a venv interpreter symlink would bypass its installed dependencies.
    interpreter = (python or Path(sys.executable)).expanduser().absolute()
    command = " ".join(_quoted(value, command=True) for value in (
        str(interpreter), "-m", "ciel", "chat", "_worker",
    ))
    return (
        _MANAGED_HEADER
        + f"# Local configuration revision: {_configuration_revision(directory)}\n"
        + "[Unit]\nDescription=Ciel private chat\nStartLimitIntervalSec=0\n\n"
        + "[Service]\nType=simple\n"
        + f"WorkingDirectory={working_directory}\n"
        + f"ExecStart={command}\n"
        + "Environment=PYTHONUNBUFFERED=1\nRestart=on-failure\nRestartSec=10\nTimeoutStopSec=10\n\n"
        + "[Install]\nWantedBy=default.target\n"
    )


def _run_systemctl(*arguments: str) -> subprocess.CompletedProcess[str] | None:
    """Run a bounded service command without exposing potentially private diagnostics."""
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


def _managed_content(path: Path) -> str | None:
    """Read only the exact Ciel-managed unit; never overwrite an unrelated service."""
    if path.is_symlink():
        raise ValueError("The chat service is a symlink")
    content = path.read_text(encoding="utf-8") if path.exists() else None
    if content is not None and not content.startswith(_MANAGED_HEADER):
        raise ValueError("The existing chat service is not managed by Ciel")
    return content


@contextmanager
def _service_lock() -> Iterator[None]:
    """Serialize setup/startup with owner pairing, without waiting behind a pairing prompt."""
    import fcntl

    directory = _unit_path().parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".ciel-chat.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                         0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Chat setup or pairing is already in progress") from None
        yield
    finally:
        os.close(descriptor)


def _available() -> bool:
    return (sys.platform == "linux" and shutil.which("systemctl") is not None
            and _systemctl("show-environment"))


def autostart_status() -> str:
    """Report verified service enablement and process state without contacting Telegram."""
    try:
        if _managed_content(_unit_path()) is None:
            return "not installed"
        if not _available():
            return "automatic startup unavailable"
        enabled = _run_systemctl("is-enabled", UNIT_NAME)
        if enabled is None or enabled.returncode != 0 or enabled.stdout.strip() != "enabled":
            return "automatic startup needs repair"
        return "automatic · running" if _systemctl("is-active", "--quiet", UNIT_NAME) else (
            "automatic · not running; startup will retry"
        )
    except (OSError, UnicodeError, ValueError):
        return "automatic startup unavailable; check the managed user service"


def _write_unit(path: Path, content: str) -> None:
    """Replace the managed unit atomically, so interruption cannot leave a partial service."""
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".ciel-chat-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def set_autostart(
    enabled: bool, project_dir: Path, *, restart: bool = False,
) -> AutostartResult:
    """Reconcile service selection; selected chat always starts now and at every login."""
    try:
        path = _unit_path()
        existing = _managed_content(path)
        if not enabled and existing is None:
            return AutostartResult(True, "No chat services are selected.")
        if not _available():
            return AutostartResult(False, "Automatic chat startup is unavailable. Ciel needs a "
                                   "logged-in Linux desktop with systemd user services. "
                                   "Startup will retry the next time you open Ciel or save setup.")
        with _service_lock():
            existing = _managed_content(path)
            if not enabled:
                if not _systemctl("disable", "--now", UNIT_NAME):
                    return AutostartResult(False, "Chat settings were saved, but the previous "
                                           "chat service could not be stopped.")
                path.unlink(missing_ok=True)
                if not _systemctl("daemon-reload"):
                    return AutostartResult(False, "Chat stopped, but the user service manager "
                                           "could not refresh its saved configuration.")
                return AutostartResult(True, "Chat connections are disabled.")
            content = render_unit(project_dir)
            changed = existing != content
            if changed:
                _write_unit(path, content)
            # Retry reload even if a previous attempt wrote the file but could not load it.
            if not _systemctl("daemon-reload"):
                return AutostartResult(False, "Chat settings were saved, but automatic "
                                       "startup could not refresh. Ciel will retry later.")
            if not _systemctl("enable", UNIT_NAME):
                return AutostartResult(False, "Chat settings were saved, but automatic startup "
                                       "could not be enabled. Ciel will retry later.")
            action = "restart" if restart or (changed and existing is not None) else "start"
            if not _systemctl(action, UNIT_NAME):
                return AutostartResult(False, "Automatic chat is enabled, but its worker could "
                                       "not start. Ciel will retry the next time it opens.")
            if not _systemctl("is-active", "--quiet", UNIT_NAME):
                return AutostartResult(False, "Automatic chat is enabled, but its worker is not "
                                       "running yet. The service will retry; "
                                       "check ciel chat status.")
            return AutostartResult(True, "Chat starts automatically now and when you log in. "
                                   "It works while this PC is on and connected; pair Telegram "
                                   "before sending messages.")
    except (OSError, UnicodeError, ValueError) as error:
        reason = " Pairing is in progress; it will resume chat when finished." if (
            isinstance(error, ValueError) and "in progress" in str(error)
        ) else " Check the Ciel-managed user service and its directory permissions."
        return AutostartResult(False, "Automatic chat startup could not be changed." + reason)


@contextmanager
def pause_for_pairing() -> Iterator[AutostartResult]:
    """Stop the managed poller and exclude automatic restarts until pairing releases its lock."""
    stack = ExitStack()
    try:
        existing = _managed_content(_unit_path())
        if existing is not None and not _available():
            operation = AutostartResult(False, "Ciel could not pause automatic chat for pairing. "
                                        "Retry from your logged-in desktop.")
        else:
            stack.enter_context(_service_lock())
            if existing is not None and not _systemctl("stop", UNIT_NAME):
                operation = AutostartResult(False, "Ciel could not pause automatic chat for "
                                            "pairing. No pairing was changed.")
            else:
                operation = AutostartResult(True, "Automatic chat paused for pairing.")
    except (OSError, UnicodeError, ValueError):
        operation = AutostartResult(False, "Ciel could not pause automatic chat. Another setup "
                                    "may be in progress, or the managed service is unavailable.")
    try:
        yield operation
    finally:
        stack.close()


def reconcile_chat_startup(
    configuration: ChatConfig, *, project_dir: Path, restart: bool = False,
) -> AutostartResult:
    """Honor service selection regardless of obsolete CHAT_AUTOSTART preferences."""
    selected = "telegram" in configuration.services
    token = configuration.telegram_bot_token
    ready = bool(selected and token and token.get_secret_value().strip())
    result = set_autostart(ready, project_dir, restart=restart)
    if selected and not ready:
        message = "Telegram needs a bot token in ciel chat setup before automatic chat can start."
        if not result.success:
            message += " " + result.message
        return AutostartResult(False, message)
    return result
