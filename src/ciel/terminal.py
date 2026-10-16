"""Keyboard messages and local controls alongside the voice listener."""

from __future__ import annotations

import codecs
import getpass
import os
import select
import sys
import threading
from collections.abc import Callable
from typing import TextIO

from ciel.terminal_output import TerminalOutput


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
    ) -> None:
        self.stream = sys.stdin if stream is None else stream
        self.output = sys.stdout if output is None else output
        self.owner_name = owner_name or getpass.getuser()
        self.submit = submit
        self.stop_reply = stop_reply
        self.toggle_mute = toggle_mute
        self.exit_app = exit_app
        self.write = write
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def handle_line(self, line: str) -> None:
        """Handle controls locally; send ordinary text through the listener queue."""
        text = line.strip()
        if not text:
            return
        command = text.casefold()
        if command == "/exit":
            self.exit_app()
            self.write("Exiting CIEL; shutting down the microphone and voice worker.")
            self._closed.set()
        elif command == "/stop":
            self.stop_reply()
            self.write("Stopped the current reply.")
        elif command == "/mute":
            muted = self.toggle_mute()
            self.write("Microphone muted; typed messages still work." if muted
                       else "Microphone unmuted.")
        elif command == "/help":
            self.write("Type a message and press Enter. /mute toggles the microphone; "
                       "/stop cancels the reply; /exit quits; /help shows these controls.")
        elif text.startswith("/") and text.split()[0].casefold() not in {
            "/fast", "/strong", "/deep", "/local",
        }:
            self.write("Unknown command. Use /help for controls.")
        elif not self.submit(text):
            self.write("Message wasn't submitted: the listener is stopping.")

    def start(self) -> None:
        """Start reading stdin; EOF ends keyboard input without stopping voice."""
        self._thread = threading.Thread(target=self._read, name="ciel-keyboard", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        terminal_settings = None
        descriptor = None
        try:
            descriptor = self.stream.fileno()
            interactive = os.isatty(descriptor) and isinstance(self.output, TerminalOutput)
            if interactive:
                import termios
                import tty

                terminal_settings = termios.tcgetattr(descriptor)
                tty.setcbreak(descriptor, termios.TCSANOW)
            decoder = codecs.getincrementaldecoder(self.stream.encoding or "utf-8")("replace")
            pending = ""
            self._show_prompt()
            escape = ""
            while not self._closed.is_set():
                ready, _, _ = select.select([descriptor], [], [], 0.1)
                if not ready:
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
                            escape += char
                            if char.isalpha() or char == "~" or len(escape) > 8:
                                escape = ""
                            continue
                        if char == "\x1b":
                            escape = char
                        elif char in {"\r", "\n"}:
                            self.output.submit_input(pending)
                            line, pending = pending, ""
                            self.handle_line(line)
                        elif char in {"\x7f", "\b"}:
                            pending = pending[:-1]
                        elif char == "\x15":  # Ctrl-U: clear draft.
                            pending = ""
                        elif char == "\x17":  # Ctrl-W: remove last word.
                            pending = pending.rstrip().rsplit(" ", 1)[0] if " " in (
                                pending.rstrip()
                            ) else ""
                        elif char == "\x04" and not pending:  # Ctrl-D: keyboard EOF.
                            return
                        elif char.isprintable():
                            pending += char
                        if not self._closed.is_set():
                            self.output.set_input(self.owner_name, pending)
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
