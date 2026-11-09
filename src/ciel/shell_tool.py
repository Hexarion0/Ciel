"""Shell commands on the owner's PC for the main model's ``exec`` tool.

Commands run as the owner in bash, from the home directory, with no stdin (so
``sudo`` cannot prompt), a timeout that ends the whole process group, and output
bounded and secret-masked before any model sees it. Child processes do not inherit
environment variables that look like credentials.

Approval is decided by the application, never the model: a command that needs it
is stored as pending for its conversation, and only the owner's own reply ("yes")
in the next turn lets the same exact command run, once, within 60 seconds.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ciel.logging import SecretMaskingFilter

APPROVAL_SECONDS = 60.0
_TAIL_CHARS = 2_000
_KILL_GRACE_SECONDS = 1.0

# Read-only commands that run without asking in "safe" mode, when used with
# plain arguments (no pipes, redirects, substitutions, or chaining). Commands whose
# flags can print environments or read files (ps e, date -f) are left out.
SAFE_COMMANDS = frozenset({
    "df", "du", "free", "id", "lsblk", "lscpu", "lsusb", "lspci", "nproc",
    "nvidia-smi", "pwd", "sensors", "uname", "uptime", "w", "who", "whoami", "which",
})
# Safe only with exactly these arguments: nvidia-smi -f writes a file and other
# flags change GPU state; sensors -s writes chip limits.
_SAFE_ARGUMENTS = {
    "nvidia-smi": {(), ("-L",), ("--list-gpus",), ("-q",), ("--query",)},
    "sensors": {(), ("-A",), ("-u",), ("-j",), ("-f",)},
}
# Line breaks and invisible or direction-changing characters can hide part of a
# command in the approval prompt.
_HIDDEN_TEXT = re.compile(r"[\n\r\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")
_SHELL_SYNTAX = re.compile(r"[;&|`$<>(){}\\*?!\n]")
# Obviously destructive or disruptive commands, matched on the raw text and on
# every nested "sh -c" string; asked about when safeguards are on.
_DESTRUCTIVE = re.compile(
    r"\bmkfs(?:\.\w+)?\b|\bwipefs\b|\bfdisk\b|\bparted\b|\bsgdisk\b|\bblkdiscard\b"
    r"|\bdd\b[^\n]*\bof=/dev/|>\s*/dev/(?:sd|nvme|hd|mmcblk|vd)"
    r"|\bshred\b|\btruncate\b|\b(?:shutdown|reboot|poweroff|halt)\b|\binit\s+[06]\b"
    r"|\bsystemctl\s+(?:poweroff|reboot|halt|suspend|hibernate|kexec)\b"
    r"|\bkill\s+(?:-\S+\s+)*-1\b|\bpkill\s+(?:-\S+\s+)*-u\b|\bkillall5\b"
    r"|\bfind\b[^\n]*\s-(?:delete|exec\s+rm)\b"
    r"|\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b|\b(?:ba|z|da)?sh\s+<\(\s*(?:curl|wget)"
    r"|:\(\)\s*\{[^}]*\};\s*:",
    re.I,
)
# Paths whose recursive removal, chmod, or chown is never routine.
_PROTECTED = re.compile(
    r"^(?:/|/\*|~|~/|~/\*|~/\..*|\$\{?HOME\}?(?:/|/\*|/\..*)?|/home(?:/[^/]+)?/?\*?"
    r"|/home/[^/]+/\..*|/(?:etc|usr|boot|var|bin|sbin|lib|lib64|opt|root|srv|dev|proc|sys)"
    r"(?:/.*)?)$",
)
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
_NETWORK = re.compile(r"\b(?:curl|wget|git\s+(?:clone|pull|fetch)|aria2c|scp|rsync|http)\b",
                      re.I)
_APPROVAL = re.compile(
    r"^\s*(?:yes|yeah|yep|y|sure|ok(?:ay)?|do it|run it|go ahead|go for it|confirm(?:ed)?"
    r"|approve(?:d)?)(?:\s+please)?[\s.!]*$",
    re.I,
)
# Speech approval needs a deliberate phrase: "okay", "yeah", and "sure" are common
# in background talk and in transcription mistakes.
_SPOKEN_APPROVAL = re.compile(
    r"^\s*(?:yes|run it|do it|go ahead|confirm(?:ed)?|approve(?:d)?)"
    r"(?:\s+(?:please|run it|do it))?[\s.!]*$",
    re.I,
)
_SECRET_ENV = re.compile(
    r"KEY|TOKEN|SECRET|PASS|CREDENTIAL|COOKIE|AUTH|_PAT$|DSN|PRIVATE|SIGNATURE", re.I,
)
# Paths and sockets, not secrets: GUI, D-Bus, and SSH-agent commands need them.
_SAFE_ENV = {"XAUTHORITY", "SSH_AUTH_SOCK", "GPG_AGENT_INFO", "DBUS_SESSION_BUS_ADDRESS",
             "SESSION_MANAGER", "DESKTOP_SESSION", "PASSWORD_STORE_DIR", "KEYMAP"}
_URL_CREDENTIALS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
# Bounded repeats: unbounded [\w-]* on both sides backtracks for minutes on long
# "password-password-..." text while holding the GIL.
_INLINE_SECRET = re.compile(
    r"\b[\w-]{0,40}?(?:password|passwd|passphrase|api[_-]?key|access[_-]?token|auth[_-]?token"
    r"|bot[_-]?token|client[_-]?secret|secret|_pass\b|_token\b)[\w-]{0,40}\s*[:=]\s*\S+",
    re.I,
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S,
)
_UNPRINTABLE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class CommandResult:
    """What a command did, bounded for a tool message."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    canceled: bool
    truncated: bool
    seconds: float

    def payload(self, command: str) -> dict:
        result = {"command": command, "exit_code": self.exit_code, "stdout": self.stdout,
                  "stderr": self.stderr, "seconds": round(self.seconds, 2)}
        if self.timed_out:
            result["timed_out"] = True
        if self.canceled:
            result["canceled"] = True
        if self.truncated:
            result["truncated"] = True
        return result


def is_approval(text: str, *, spoken: bool = False) -> bool:
    """The owner's own reply approving a pending command."""
    return bool((_SPOKEN_APPROVAL if spoken else _APPROVAL).match(text))


def _words(command: str) -> list[str]:
    try:
        return shlex.split(command, comments=False)
    except ValueError:
        return command.split()


def is_destructive(command: str, depth: int = 0) -> bool:
    """Whether a command could wipe data, disks, or the session, nested shells included."""
    if _DESTRUCTIVE.search(command):
        return True
    words = _words(command)
    for index, word in enumerate(words):
        name = word.rsplit("/", 1)[-1]
        if name in {"rm", "chmod", "chown", "chgrp"}:
            arguments = words[index + 1:]
            recursive = any(re.fullmatch(r"-[a-zA-Z]*[rR][a-zA-Z]*|--recursive", item)
                            for item in arguments)
            if recursive and any(_PROTECTED.match(item.rstrip(";&|")) for item in arguments):
                return True
    # Recurse once into the command string after the first shell or eval:
    # bash -c '...', sh -lc "...", eval '...'. Once per level keeps this linear.
    if depth < 3:
        for index, word in enumerate(words):
            if word.rsplit("/", 1)[-1] in _SHELLS or word == "eval":
                nested = " ".join(item for item in words[index + 1:]
                                  if not item.startswith("-"))
                return bool(nested) and is_destructive(nested, depth + 1)
    return False


def hides_text(command: str) -> bool:
    """Whether the command has line breaks or invisible characters a prompt could hide."""
    return bool(_HIDDEN_TEXT.search(command))


def is_safe_listed(command: str) -> bool:
    """A plain read-only command from the safe list, with arguments it allows."""
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    if not words or words[0] not in SAFE_COMMANDS or _SHELL_SYNTAX.search(command):
        return False
    allowed = _SAFE_ARGUMENTS.get(words[0])
    return allowed is None or tuple(words[1:]) in allowed


def fetches_remote(command: str) -> bool:
    """Whether a command downloads text whose content the model should not trust."""
    return bool(_NETWORK.search(command))


def approval_reason(command: str, *, mode: str, after_web: bool, safeguards: bool) -> str:
    """Why this command needs the owner's approval, or "" to run it now."""
    if mode == "always":
        return "every command is approved first"
    if safeguards and after_web:
        return "web or screen text read recently could have suggested it"
    if safeguards and is_destructive(command):
        return "it could delete data or shut the computer down"
    if mode == "safe" and not is_safe_listed(command):
        return "it isn't on the read-only list"
    return ""


def mask(text: str) -> str:
    """Hide secret-looking values before text reaches a model or a log."""
    text = _PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", SecretMaskingFilter._redact(text))
    text = _URL_CREDENTIALS.sub("://[REDACTED_SECRET]@", text)
    return _INLINE_SECRET.sub("[REDACTED_SECRET]", text)


class _Capture:
    """Read one pipe in a thread, keeping its head and tail within a size limit."""

    def __init__(self, pipe, limit: int) -> None:
        self.limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0
        self._thread = threading.Thread(target=self._read, args=(pipe,), daemon=True)
        self._thread.start()

    def _read(self, pipe) -> None:
        try:
            for chunk in iter(lambda: pipe.read1(65_536), b""):
                room = self.limit - len(self.head)
                if room > 0:
                    self.head.extend(chunk[:room])
                    chunk = chunk[room:]
                if chunk:
                    self.tail.extend(chunk)
                    if len(self.tail) > _TAIL_CHARS:
                        self.dropped += len(self.tail) - _TAIL_CHARS
                        del self.tail[:-_TAIL_CHARS]
        except (OSError, ValueError):
            pass

    def text(self, wait: float) -> tuple[str, bool]:
        # A background child can hold the pipe open after the command exits.
        self._thread.join(wait)
        head = self.head.decode("utf-8", "replace")
        tail = self.tail.decode("utf-8", "replace")
        # Binary output decodes to replacement and control characters that grow
        # several-fold as JSON; keep the text within the configured size.
        head = _UNPRINTABLE.sub("", head)[:self.limit]
        tail = _UNPRINTABLE.sub("", tail)[-_TAIL_CHARS:]
        dropped = self.dropped
        body = head + (f"\n[... {dropped} bytes omitted ...]\n" if dropped else "") + tail
        return mask(body), bool(dropped)


def _child_environment() -> dict[str, str]:
    return {
        name: value for name, value in os.environ.items()
        if name in _SAFE_ENV or name.startswith("XDG_")
        or not (_SECRET_ENV.search(name) or _URL_CREDENTIALS.search(value))
    }


def _end_group(process: subprocess.Popen) -> None:
    """End the command's whole process group, including children that ignore SIGTERM."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # The leader exiting does not end the group: always follow with SIGKILL.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def run_command(
    command: str, *, timeout: float, max_output_chars: int,
    cancel_event: threading.Event | None = None,
) -> CommandResult:
    """Run ``command`` in bash as the owner; never raises for a failing command."""
    shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    started = time.monotonic()
    process = subprocess.Popen(
        [shell, "-c", command], cwd=Path.home(), env=_child_environment(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    out = _Capture(process.stdout, max_output_chars)
    err = _Capture(process.stderr, max_output_chars // 2)
    timed_out = canceled = False
    deadline = started + timeout
    while process.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            canceled = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        try:
            process.wait(0.1)
        except subprocess.TimeoutExpired:
            pass
    if timed_out or canceled:
        _end_group(process)
    stdout, cut_out = out.text(0.5)
    stderr, cut_err = err.text(0.2)
    return CommandResult(
        exit_code=process.returncode, stdout=stdout, stderr=stderr,
        timed_out=timed_out, canceled=canceled, truncated=cut_out or cut_err,
        seconds=time.monotonic() - started,
    )


class Approvals:
    """One pending command per conversation, approved once by the owner's reply."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[str, float]] = {}

    def offer(self, scope: str, command: str) -> None:
        with self._lock:
            self._pending[scope] = (command, time.monotonic() + APPROVAL_SECONDS)

    def approved(self, scope: str, command: str, owner_text: str, *,
                 spoken: bool = False) -> bool:
        """Consume the pending command if the owner just approved this exact one."""
        with self._lock:
            pending = self._pending.get(scope)
            if pending is None or time.monotonic() > pending[1]:
                self._pending.pop(scope, None)
                return False
            if pending[0] != command or not is_approval(owner_text, spoken=spoken):
                return False
            del self._pending[scope]
            return True

    def clear(self, scope: str) -> None:
        with self._lock:
            self._pending.pop(scope, None)


APPROVALS = Approvals()
