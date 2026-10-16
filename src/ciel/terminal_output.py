"""Coordinate background output with an editable terminal input row."""

from __future__ import annotations

import logging
import shutil
import threading
import unicodedata
from collections.abc import Callable
from typing import TextIO


def _cells(text: str) -> int:
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
               for char in text)


class TerminalOutput:
    """Keep an input draft below logs and a naturally wrapping speech caption.

    Completed output lines become scrollback. Partial speech and keyboard input
    share a redraw region protected by one lock; terminal echo is disabled by
    the keyboard reader, so typed characters never enter a speech caption.
    """

    def __init__(self, stream: TextIO, *, width: int | None = None) -> None:
        self.stream = stream
        self.width = width
        self._lock = threading.RLock()
        self._pending = ""
        self._label = ""
        self._draft = ""
        self._input_active = False
        self._live_rows = 0

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def flush(self) -> None:
        with self._lock:
            self.stream.flush()

    def write(self, text: str) -> int:
        """Write output above the draft, preserving an unfinished speech line."""
        with self._lock:
            if not self._input_active:
                self.stream.write(text)
                self.stream.flush()
                return len(text)
            self._clear_locked()
            combined = self._pending + text
            boundary = combined.rfind("\n")
            if boundary >= 0:
                self.stream.write(combined[:boundary + 1])
                self._pending = combined[boundary + 1:]
            else:
                self._pending = combined
            self._draw_locked()
            return len(text)

    def set_input(self, owner: str, draft: str) -> None:
        """Redraw the owner's input without changing any pending speech."""
        with self._lock:
            self._clear_locked()
            safe_owner = "".join(char for char in owner if char.isprintable())
            self._label = f"[{safe_owner}]: "
            self._draft = "".join(char for char in draft if char.isprintable())
            self._input_active = True
            self._draw_locked()

    def submit_input(self, text: str) -> None:
        """Save the entered line separately and keep the current speech visible."""
        with self._lock:
            self._clear_locked()
            safe_text = "".join(char for char in text if char.isprintable())
            self.stream.write(self._label + safe_text + "\n")
            self._draft = ""
            self._draw_locked()

    def hide_input(self) -> None:
        """Leave scrollback intact and restore an ordinary output stream."""
        with self._lock:
            self._clear_locked()
            if self._pending:
                self.stream.write(self._pending)
            self._pending = ""
            self._input_active = False
            self.stream.flush()

    def _clear_locked(self) -> None:
        if self._live_rows:
            self.stream.write("\r\x1b[2K")
            for _ in range(self._live_rows - 1):
                self.stream.write("\x1b[1A\r\x1b[2K")
        self._live_rows = 0

    def _draw_locked(self) -> None:
        columns = max(2, self.width or shutil.get_terminal_size().columns)
        if self._pending:
            self.stream.write(self._pending + "\n")
            self._live_rows = max(1, (_cells(self._pending) + columns - 1) // columns)
        # Keep the input row short so backspace, redraws, and narrow terminals
        # cannot move the cursor into speech. The complete draft is submitted.
        prompt = self._label + self._draft
        while prompt and _cells(prompt) >= columns:
            prompt = prompt[1:]
        self.stream.write(prompt)
        self._live_rows += 1
        self.stream.flush()


def install_terminal_output(output: TerminalOutput) -> Callable[[], None]:
    """Route console logs through the display and return a restore hook."""
    originals = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is output.stream:
            originals.append((handler, handler.stream))
            handler.setStream(output)

    def restore() -> None:
        for handler, stream in originals:
            handler.setStream(stream)

    return restore
