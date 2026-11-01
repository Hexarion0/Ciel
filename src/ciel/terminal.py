"""Keyboard messages and local controls alongside the voice listener."""

from __future__ import annotations

import codecs
import getpass
import io
import os
import select
import sys
import threading
from collections.abc import Callable
from typing import TextIO

from ciel.conversation import contains_retired_local_command
from ciel.terminal_output import TerminalOutput

_PROVIDER_COMMANDS = {"/fast", "/strong", "/deep"}
COMMAND_GROUPS = (
    ("Conversation", (
        ("/stop", "Stop the current recording and reply"),
        ("/mute [on|off]", "Toggle or set microphone mute"),
        ("/unmute", "Turn the microphone back on"),
        ("/mode [ambient|wake]", "Choose how CIEL listens"),
    )),
    ("Memory and awareness", (
        ("/memory [status|pause|resume]", "Manage saved-memory recall"),
        ("/observe [on|off|pause|resume|status]", "Control temporary screen hints"),
    )),
    ("Model and reasoning", (
        ("/reasoning [auto|on|off]", "Set reasoning for this session"),
        ("/fast MESSAGE", "Use the fast route for one message"),
        ("/strong MESSAGE", "Use the strong route for one message"),
        ("/deep MESSAGE", "Use the deep route for one message"),
    )),
    ("Session", (
        ("/status", "Show your current session"),
        ("/help COMMAND", "Get help with a command"),
        ("/exit", "Close CIEL; /quit also works"),
    )),
)
_COMMAND_CHOICES = tuple(sorted(
    ((label.split()[0], description) for _, rows in COMMAND_GROUPS for label, description in rows),
    key=lambda row: {"/help": 0, "/status": 1}.get(row[0], 2),
))
_COMMAND_ARGUMENTS = {
    "/mute": ("on", "off"), "/mode": ("ambient", "wake"),
    "/memory": ("status", "pause", "resume"), "/reasoning": ("auto", "on", "off"),
    "/observe": ("on", "off", "pause", "resume", "status"),
    "/help": tuple(name[1:] for name, _ in _COMMAND_CHOICES),
}


def command_suggestions(draft: str) -> list[tuple[str, str]]:
    """Complete only a leading slash command or its supported local arguments."""
    if not draft.startswith("/"):
        return []
    if " " not in draft:
        return [(name, description) for name, description in _COMMAND_CHOICES
                if name.startswith(draft.casefold())]
    command, argument = draft.casefold().split(" ", 1)
    description = dict(_COMMAND_CHOICES).get(command, "")
    return [(f"{command} {value}", description)
            for value in _COMMAND_ARGUMENTS.get(command, ()) if value.startswith(argument)]
_COMMAND_HELP = {
    "/status": "/status — Show the current session, microphone, listening mode, and memory status.",
    "/mute": "/mute [on|off] — Toggle the microphone, or explicitly mute/unmute it.",
    "/unmute": "/unmute — Unmute the microphone; typed messages always remain available.",
    "/mode": "/mode [ambient|wake] — Show or change this session's listening mode.",
    "/memory": "/memory [status|pause|resume] — Inspect or pause/resume saved-memory recall. "
               "Saved facts are kept; recent conversation and learning remain active.",
    "/reasoning": "/reasoning [auto|on|off] — Show or change reasoning for this session.",
    "/observe": "/observe [on|off|pause|resume|status] — Opt in to temporary screen "
                "observation and occasional coding hints. Off at every launch.",
    "/stop": "/stop — Cancel the current recording and reply.",
    "/exit": "/exit — Shut down CIEL. /quit is an alias.",
    "/quit": "/quit — Shut down CIEL. Alias for /exit.",
    "/help": "/help [command] — Show all controls or help for one command.",
    "/fast": "/fast MESSAGE — Send one message using the fast route.",
    "/strong": "/strong MESSAGE — Send one message using the strong route.",
    "/deep": "/deep MESSAGE — Send one message using the deep route.",
}
_COMMAND_USAGE = {
    "/status": "/status", "/mute": "/mute [on|off]", "/unmute": "/unmute",
    "/mode": "/mode [ambient|wake]", "/memory": "/memory [status|pause|resume]",
    "/reasoning": "/reasoning [auto|on|off]",
    "/observe": "/observe [on|off|pause|resume|status]",
    "/stop": "/stop", "/exit": "/exit", "/quit": "/quit",
    "/help": "/help [command]",
}


class TerminalInput:
    """Read complete terminal lines without blocking reply or audio workers."""

    def __init__(
        self,
        submit: Callable[[str], bool],
        stop_reply: Callable[[], None],
        toggle_mute: Callable[[], bool],
        exit_app: Callable[[], None],
        write: Callable[[str], None],
        stream: TextIO | None = None,
        output: TextIO | None = None,
        owner_name: str | None = None,
        *,
        status: Callable[[], str] | None = None,
        set_mute: Callable[[bool], bool] | None = None,
        mode: Callable[[str | None], str] | None = None,
        memory: Callable[[str], str] | None = None,
        reasoning: Callable[[str | None], str] | None = None,
        observe: Callable[[str], str] | None = None,
        microphone_available: Callable[[], bool] | None = None,
        show_help: Callable[[], None] | None = None,
    ) -> None:
        self.stream = sys.stdin if stream is None else stream
        self.output = sys.stdout if output is None else output
        self.owner_name = owner_name or getpass.getuser()
        self.submit = submit
        self.stop_reply = stop_reply
        self.toggle_mute = toggle_mute
        self.exit_app = exit_app
        self.write = write
        self.status = status
        self.set_mute = set_mute
        self.mode = mode
        self.memory = memory
        self.reasoning = reasoning
        self.observe = observe
        self.microphone_available = microphone_available
        self.show_help = show_help
        self._command_lock = threading.RLock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._completion_query = ""
        self._completion_index = 0
        self._completion_dismissed: str | None = None
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""

    def _suggestions(self, draft: str, cursor: int) -> list[tuple[str, str]]:
        if draft != self._completion_query:
            self._completion_query = draft
            self._completion_index = 0
            self._completion_dismissed = None
        if (cursor != len(draft) or self._completion_dismissed == draft
                or self._history_index is not None or not self.output.supports_command_picker()):
            return []
        return command_suggestions(draft)

    def _redraw(self, draft: str, cursor: int) -> None:
        choices = self._suggestions(draft, cursor)
        if choices:
            self._completion_index %= len(choices)
        self.output.set_input(self.owner_name, draft, cursor=cursor,
                              suggestions=choices, selected=self._completion_index)

    def _navigate(self, draft: str, cursor: int, direction: int) -> tuple[str, int]:
        choices = self._suggestions(draft, cursor)
        if choices:
            self._completion_index = (self._completion_index + direction) % len(choices)
        elif self._history:
            if self._history_index is None:
                if direction > 0:
                    return draft, cursor
                self._history_draft = draft
                self._history_index = len(self._history)
            self._history_index = max(0, min(len(self._history), self._history_index + direction))
            draft = (self._history_draft if self._history_index == len(self._history)
                     else self._history[self._history_index])
            cursor = len(draft)
        return draft, cursor

    def handle_line(self, line: str) -> None:
        """Handle controls locally; send ordinary text through the listener queue."""
        text = line.strip()
        if not text:
            return
        if contains_retired_local_command(text):
            with self._command_lock:
                self.write("/local is no longer available. Send your message normally.")
            return
        words = text.split()
        command = words[0].casefold()
        if text.startswith("/") and command not in _PROVIDER_COMMANDS:
            with self._command_lock:
                try:
                    self._handle_command(command, [word.casefold() for word in words[1:]])
                except Exception:
                    self.write(f"Could not run {command}. Try again or use /help.")
        elif not self.submit(text):
            self.write("Message wasn't submitted: the listener is stopping.")

    def _handle_command(self, command: str, arguments: list[str]) -> None:
        """Validate local controls before invoking runtime callbacks."""
        if command not in _COMMAND_USAGE:
            self.write("Unknown command. Use /help for controls.")
            return
        choices = {
            "/mute": {"on", "off"}, "/mode": {"ambient", "wake"},
            "/memory": {"status", "pause", "resume"},
            "/reasoning": {"auto", "on", "off"},
            "/observe": {"on", "off", "pause", "resume", "status"},
        }
        if (
            len(arguments) > 1
            or arguments and command not in {*choices, "/help"}
            or arguments and command in choices and arguments[0] not in choices[command]
        ):
            self.write(f"Usage: {_COMMAND_USAGE[command]}")
            return
        if command in {"/exit", "/quit"}:
            self.exit_app()
            self.write("Exiting CIEL; shutting down the microphone and voice worker.")
            self._closed.set()
        elif command == "/stop":
            self.stop_reply()
            self.write("Stopped the current reply.")
        elif command in {"/mute", "/unmute"}:
            if self.microphone_available is not None and not self.microphone_available():
                self.write("Microphone unavailable; typed messages still work. "
                           "Connect an input and restart CIEL to use voice.")
                return
            if command == "/mute" and not arguments:
                muted = self.toggle_mute()
            elif self.set_mute is None:
                self.write(f"{command} is unavailable in this session.")
                return
            else:
                muted = self.set_mute(command == "/mute" and arguments == ["on"])
            self.write("Microphone muted; typed messages still work." if muted
                       else "Microphone unmuted.")
        elif command == "/help":
            if arguments:
                topic = "/" + arguments[0].lstrip("/")
                self.write(_COMMAND_HELP.get(topic, "Unknown command. Use /help for controls."))
            else:
                if self.show_help is not None:
                    self.show_help()
                else:
                    from ciel.terminal_ui import ConsoleUI

                    output = io.StringIO()
                    ConsoleUI(stream=output).commands(COMMAND_GROUPS)
                    self.write(output.getvalue().rstrip())
        elif command == "/status":
            self.write(self.status() if self.status is not None else
                       "/status is unavailable in this session.")
        elif command == "/mode":
            self.write(self.mode(arguments[0] if arguments else None) if self.mode is not None
                       else "/mode is unavailable in this session.")
        elif command == "/memory":
            self.write(self.memory(arguments[0] if arguments else "status")
                       if self.memory is not None else "/memory is unavailable in this session.")
        elif command == "/reasoning":
            self.write(self.reasoning(arguments[0] if arguments else None)
                       if self.reasoning is not None else
                       "/reasoning is unavailable in this session.")
        elif command == "/observe":
            self.write(self.observe(arguments[0] if arguments else "status")
                       if self.observe is not None else
                       "/observe is unavailable in this session.")

    def start(self) -> None:
        """Start reading stdin; EOF ends keyboard input without stopping voice."""
        self._thread = threading.Thread(target=self._read, name="ciel-keyboard", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        terminal_settings = None
        descriptor = None
        try:
            descriptor = self.stream.fileno()
            interactive = os.isatty(descriptor) and isinstance(self.output, TerminalOutput) and (
                not self.output.chat or self.output.interactive
            )
            if interactive:
                import termios
                import tty

                terminal_settings = termios.tcgetattr(descriptor)
                tty.setcbreak(descriptor, termios.TCSANOW)
            decoder = codecs.getincrementaldecoder(self.stream.encoding or "utf-8")("replace")
            pending = ""
            cursor = 0
            self._show_prompt()
            escape = ""
            while not self._closed.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.1)
                if not ready:
                    if interactive:
                        self.output.refresh_layout()
                    if escape == "\x1b":
                        self._completion_dismissed = pending
                        escape = ""
                        self._redraw(pending, cursor)
                    continue
                data = os.read(descriptor, 4096)
                if not data:
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        self.handle_line(pending)
                    break
                decoded = decoder.decode(data)
                if interactive:
                    for char in decoded:
                        if self._closed.is_set():
                            break
                        if escape:
                            if escape == "\x1b" and char not in {"[", "O"}:
                                self._completion_dismissed = pending
                                escape = ""
                            else:
                                escape += char
                                if char.isalpha() or char == "~" or len(escape) > 8:
                                    key, escape = escape[2:], ""
                                    if key in {"A", "B"}:
                                        pending, cursor = self._navigate(
                                            pending, cursor, -1 if key == "A" else 1,
                                        )
                                    elif key == "D":
                                        cursor = max(0, cursor - 1)
                                    elif key == "C":
                                        cursor = min(len(pending), cursor + 1)
                                    elif key in {"H", "1~", "7~"}:
                                        cursor = 0
                                    elif key in {"F", "4~", "8~"}:
                                        cursor = len(pending)
                                    elif key == "3~":
                                        pending = pending[:cursor] + pending[cursor + 1:]
                                        self._history_index = None
                                    self._redraw(pending, cursor)
                                continue
                        if char == "\x1b":
                            escape = char
                            continue
                        elif char in {"\t", "\r", "\n"}:
                            choices = self._suggestions(pending, cursor)
                            completion = (choices[self._completion_index % len(choices)][0]
                                          if choices else None)
                            if completion is not None and (
                                char == "\t" or completion.casefold() != pending.casefold()
                            ):
                                pending = completion
                                if char == "\t" and (
                                    completion in _COMMAND_ARGUMENTS
                                    or completion in _PROVIDER_COMMANDS
                                ):
                                    pending += " "
                                cursor = len(pending)
                                self._history_index = None
                            elif char != "\t":
                                self.output.submit_input(pending)
                                line, pending, cursor = pending, "", 0
                                if line.strip() and (
                                    not self._history or self._history[-1] != line
                                ):
                                    self._history = [*self._history[-99:], line]
                                self._history_index = None
                                self.handle_line(line)
                        elif char in {"\x7f", "\b"}:
                            if cursor:
                                pending = pending[:cursor - 1] + pending[cursor:]
                                cursor -= 1
                            self._history_index = None
                        elif char == "\x15":  # Ctrl-U: clear draft.
                            pending, cursor = "", 0
                            self._history_index = None
                        elif char == "\x17":  # Ctrl-W: remove last word.
                            before = pending[:cursor].rstrip()
                            before = before.rsplit(" ", 1)[0] if " " in before else ""
                            pending = before + pending[cursor:]
                            cursor = len(before)
                            self._history_index = None
                        elif char == "\x01":  # Ctrl-A: beginning of line.
                            cursor = 0
                        elif char == "\x05":  # Ctrl-E: end of line.
                            cursor = len(pending)
                        elif char == "\x04" and not pending:  # Ctrl-D: keyboard EOF.
                            return
                        elif char.isprintable():
                            pending = pending[:cursor] + char + pending[cursor:]
                            cursor += 1
                            self._history_index = None
                        if not self._closed.is_set():
                            self._redraw(pending, cursor)
                    continue
                pending += decoded
                while "\n" in pending and not self._closed.is_set():
                    line, pending = pending.split("\n", 1)
                    self.handle_line(line)
                    if not self._closed.is_set():
                        self._show_prompt()
        except (OSError, ValueError):
            if not self._closed.is_set():
                self.write("Keyboard input unavailable; voice listening is still active.")
        finally:
            if terminal_settings is not None:
                termios.tcsetattr(descriptor, termios.TCSANOW, terminal_settings)
            if isinstance(self.output, TerminalOutput):
                self.output.hide_input()

    def _show_prompt(self) -> None:
        """Show a typing label in interactive terminals without polluting piped output."""
        if self.stream.isatty() and self.output.isatty():
            if isinstance(self.output, TerminalOutput):
                self.output.set_input(self.owner_name, "")
            else:
                self.output.write(f"[{self.owner_name}]: ")
                self.output.flush()

    def close(self) -> None:
        """Release the reader promptly when the application exits."""
        self._closed.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.3)
