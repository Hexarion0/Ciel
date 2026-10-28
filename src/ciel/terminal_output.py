"""Coordinate background output with an editable terminal input row."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import unicodedata
from collections.abc import Callable
from typing import TextIO

_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _safe_text(text: str, *, color: bool = False) -> str:
    """Allow printable content and newlines, never cursor or terminal commands."""
    text = _ESCAPE.sub(lambda match: match[0] if color and _SGR.fullmatch(match[0]) else "", text)
    text = text.expandtabs(4)
    return "".join(token for token in re.findall(r"\x1b\[[0-9;]*m|[^\x1b]", text)
                   if color and _SGR.fullmatch(token) or token == "\n"
                   or len(token) == 1 and unicodedata.category(token) not in {"Cc", "Cf", "Cs"})


def _plain_cells(text: str) -> int:
    return _cells(_SGR.sub("", text))


def _split_rows(text: str, width: int) -> list[str]:
    """Wrap display cells explicitly, retaining safe color escapes and blank lines."""
    rows = [""]
    cells = 0
    for token in re.findall(r"\x1b\[[0-9;]*m|[^\x1b]", text):
        if token == "\n":
            rows.append("")
            cells = 0
            continue
        size = 0 if _SGR.fullmatch(token) else _cells(token)
        if size > width:
            token, size = "?", 1
        if size and cells + size > width:
            rows.append("")
            cells = 0
        rows[-1] += token
        cells += size
    return rows


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

    def __init__(
        self, stream: TextIO, *, width: int | None = None, chat: bool = False,
    ) -> None:
        self.stream = stream
        self.width = width
        self.chat = chat
        self.interactive = bool(getattr(stream, "isatty", lambda: False)()) and (
            os.environ.get("TERM", "") not in {"dumb", "unknown"}
        )
        self._color = self.interactive and "NO_COLOR" not in os.environ
        try:
            "❯─·".encode(getattr(stream, "encoding", None) or "utf-8")
            self._unicode = True
        except (LookupError, UnicodeError):
            self._unicode = False
        self._lock = threading.RLock()
        self._pending = ""
        self._label = ""
        self._draft = ""
        self._input_active = False
        self._live_rows = 0
        self._cursor_from_bottom = 0
        self._live_lines: list[str] = []

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def flush(self) -> None:
        with self._lock:
            self.stream.flush()

    def write(self, text: str) -> int:
        """Write output above the draft, preserving an unfinished speech line."""
        return self._write(text, styled=False)

    def write_ui(self, text: str) -> int:
        """Accept trusted presentation colors from ConsoleUI, never cursor escapes."""
        return self._write(text, styled=True)

    def _write(self, text: str, *, styled: bool) -> int:
        with self._lock:
            original_length = len(text)
            if self.chat:
                text = self._text(text, color=styled and self._color)
            if not self._input_active:
                self.stream.write(text)
                self.stream.flush()
                return original_length
            self._clear_locked()
            combined = self._pending + text
            boundary = combined.rfind("\n")
            if boundary >= 0:
                self.stream.write(combined[:boundary + 1])
                self._pending = combined[boundary + 1:]
            else:
                self._pending = combined
            self._draw_locked()
            return original_length

    def set_input(self, owner: str, draft: str) -> None:
        """Redraw the owner's input without changing any pending speech."""
        with self._lock:
            if self.chat and not self.interactive:
                return
            self._clear_locked()
            safe_owner = "".join(char for char in owner if char.isprintable())
            self._label = f"[{safe_owner}]: "
            self._draft = self._text(draft).replace("\n", "")
            self._input_active = True
            self._draw_locked()

    def submit_input(self, text: str) -> None:
        """Save the entered line separately and keep the current speech visible."""
        with self._lock:
            self._clear_locked()
            safe_text = self._text(text).replace("\n", " ")
            if self.chat:
                prefix = "❯ " if self._unicode else "> "
                rows = _split_rows(prefix + safe_text, self._columns())
                self.stream.write("\n".join(rows) + "\n\n")
            else:
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
            if self.chat and self._live_lines:
                physical_width = self._columns() + 1
                row_counts = [max(1, (_plain_cells(line) + physical_width - 1) // physical_width)
                              for line in self._live_lines]
                self._live_rows = sum(row_counts)
                self._cursor_from_bottom = sum(row_counts[-2:])
            if self._cursor_from_bottom:
                self.stream.write(f"\x1b[{self._cursor_from_bottom}B")
            self.stream.write("\r\x1b[2K")
            for _ in range(self._live_rows - 1):
                self.stream.write("\x1b[1A\r\x1b[2K")
        self._live_rows = 0
        self._cursor_from_bottom = 0
        self._live_lines = []

    def _columns(self) -> int:
        """Leave the last cell empty to avoid terminals' delayed auto-wrap."""
        if self.width is not None:
            return max(1, self.width - 1)
        try:
            columns = os.get_terminal_size(self.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size().columns
        return max(1, columns - 1)

    def _text(self, text: str, *, color: bool = False) -> str:
        text = _safe_text(text, color=color)
        if not self._unicode:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(char for char in text if not unicodedata.combining(char))
            text = text.encode("ascii", errors="replace").decode("ascii")
        return text

    def _muted(self, text: str) -> str:
        return f"\x1b[38;5;245m{text}\x1b[0m" if self._color else text

    def _draw_chat_locked(self) -> None:
        columns = self._columns()
        rows = []
        if self._pending:
            rows = _split_rows(self._pending, columns)
            # Keep a bounded live caption on screen; older rows become scrollback.
            try:
                height = os.get_terminal_size(self.fileno()).lines
            except (AttributeError, OSError, ValueError):
                height = shutil.get_terminal_size().lines
            keep = max(1, height - 6)
            if len(rows) > keep:
                self.stream.write("\n".join(rows[:-keep]) + "\n")
                rows = rows[-keep:]
                self._pending = "".join(rows)
            self.stream.write("\n".join(rows) + "\n")
            self._live_rows = len(rows)
        prefix = "❯ " if self._unicode else "> "
        if _cells(prefix) >= columns:
            prefix = ">" if columns > 1 else ""
        draft = self._draft
        while draft and _cells(prefix + draft) > columns:
            draft = draft[1:]
        rule = self._muted(("─" if self._unicode else "-") * min(columns, 100))
        hint = "Enter send  ·  /help  ·  /stop" if self._unicode else "Enter send  /help  /stop"
        hint = _split_rows(hint, columns)[0]
        self.stream.write(rule + "\n" + prefix + draft + "\n" + rule + "\n" + self._muted(hint))
        self._live_lines = [*rows, rule, prefix + draft, rule, hint]
        self._live_rows += 4
        self._cursor_from_bottom = 2
        self.stream.write("\x1b[2A\r")
        position = _cells(prefix + draft)
        if position:
            self.stream.write(f"\x1b[{position}C")
        self.stream.flush()

    def _draw_locked(self) -> None:
        if self.chat:
            if self._input_active and self.interactive:
                self._draw_chat_locked()
            return
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
