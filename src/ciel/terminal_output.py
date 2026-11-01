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


def _join_rows(rows: list[str], soft_breaks: set[int]) -> str:
    """Preserve native terminal wrapping instead of inserting new source newlines."""
    return "".join(row + ("" if index in soft_breaks else "\n")
                   for index, row in enumerate(rows[:-1])) + (rows[-1] if rows else "")


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
        self._draft_cursor = 0
        self._suggestions: list[tuple[str, str]] = []
        self._selected_suggestion = 0
        self._input_active = False
        self._live_rows = 0
        self._cursor_from_bottom = 0
        self._live_lines: list[str] = []
        self._anchor_height = 0
        self._output_row = 0
        self._layout_size: tuple[int, int] | None = None
        self._transcript: list[str] = []
        self._transcript_tail = ""
        self._transcript_revision = 0
        self._transcript_cache: tuple[int, int, list[str]] | None = None
        self._transcript_soft_breaks: set[int] = set()
        self._transcript_top = 0
        self._transcript_changed = False

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
                if self.chat and self.interactive:
                    self._remember_locked(text)
                self.stream.write(text)
                self.stream.flush()
                return original_length
            self._clear_locked()
            combined = self._pending + text
            boundary = combined.rfind("\n")
            if boundary >= 0:
                self._commit_locked(combined[:boundary + 1])
                self._pending = combined[boundary + 1:]
            else:
                self._pending = combined
            self._draw_locked()
            return original_length

    def set_input(
        self, owner: str, draft: str, *, cursor: int | None = None,
        suggestions: list[tuple[str, str]] | tuple = (), selected: int = 0,
    ) -> None:
        """Redraw the owner's input without changing any pending speech."""
        with self._lock:
            if self.chat and not self.interactive:
                return
            self._clear_locked()
            safe_owner = "".join(char for char in owner if char.isprintable())
            self._label = f"[{safe_owner}]: "
            self._draft = self._text(draft).replace("\n", "")
            self._draft_cursor = max(0, min(len(self._draft),
                                           len(draft) if cursor is None else cursor))
            self._suggestions = [(self._text(key), self._text(description))
                                 for key, description in suggestions]
            self._selected_suggestion = selected % len(suggestions) if suggestions else 0
            self._input_active = True
            self._draw_locked()

    def submit_input(self, text: str) -> None:
        """Save the entered line separately and keep the current speech visible."""
        with self._lock:
            self._clear_locked()
            safe_text = self._text(text).replace("\n", " ")
            if self.chat:
                self._commit_locked(self._input_message(safe_text))
            else:
                self.stream.write(self._label + safe_text + "\n")
            self._draft = ""
            self._draft_cursor = 0
            self._suggestions = []
            self._draw_locked()

    def _input_message(self, text: str) -> str:
        prefix = "❯ " if self._unicode else "> "
        rows = _split_rows(prefix + self._text(text).replace("\n", " "), self._columns())
        return "\n".join(rows) + "\n"

    def received_input(self, text: str) -> None:
        """Show accepted speech as a user message without changing the keyboard draft."""
        self.write(self._input_message(text))

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
        if self._anchor_height:
            # Absolute positioning also works after a terminal height change.
            physical_width = self._columns() + 1
            occupied = sum(max(1, (_plain_cells(line) + physical_width - 1) // physical_width)
                           for line in self._live_lines)
            self._output_row = max(1, min(self._height(), self._anchor_height)
                                   - occupied + 1)
            self.stream.write(f"\x1b[{self._output_row};1H\x1b[0J")
            self._anchor_height = 0
            self._live_rows = 0
            self._cursor_from_bottom = 0
            self._live_lines = []
            return
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

    def _commit_locked(self, text: str) -> None:
        """Advance the output position while completed replies enter normal scrollback."""
        if self.chat and self.interactive:
            self._remember_locked(text)
            self._transcript_changed = True
            return
        self.stream.write(text)
        if self._output_row:
            advance = len(_split_rows(text, self._columns() + 1)) - 1
            self._output_row = min(self._height(), self._output_row + advance)

    def seed_transcript(self, text: str) -> None:
        """Retain startup text already printed before this display took ownership."""
        with self._lock:
            self._remember_locked(self._text(text, color=self._color))

    def _remember_locked(self, text: str) -> None:
        lines = (self._transcript_tail + text).split("\n")
        self._transcript.extend(lines[:-1])
        self._transcript_tail = lines[-1]
        self._transcript_revision += 1

    def _transcript_rows(self, columns: int) -> list[str]:
        if self._transcript_cache is None or self._transcript_cache[:2] != (
            self._transcript_revision, columns,
        ):
            lines = [*self._transcript]
            if self._transcript_tail:
                lines.append(self._transcript_tail)
            rows: list[str] = []
            self._transcript_soft_breaks = set()
            for line in lines:
                wrapped = _split_rows(line, columns + 1)
                self._transcript_soft_breaks.update(range(len(rows), len(rows) + len(wrapped) - 1))
                rows.extend(wrapped)
            self._transcript_cache = (self._transcript_revision, columns, rows)
        return self._transcript_cache[2]

    def refresh_layout(self) -> None:
        """Keep an idle composer at the bottom when its terminal is resized."""
        with self._lock:
            if (self.chat and self.interactive and self._input_active
                    and self._layout_size != (self._columns(), self._height())):
                self._clear_locked()
                self._draw_locked()

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

    def _height(self) -> int:
        try:
            return os.get_terminal_size(self.fileno()).lines
        except (AttributeError, OSError, ValueError):
            return shutil.get_terminal_size().lines

    def supports_command_picker(self) -> bool:
        """Offer selection only when the composer has room to display its choices."""
        return self.chat and self.interactive and self._height() >= 10 and self._columns() >= 12

    def _draw_chat_locked(self) -> None:
        columns = self._columns()
        height = self._height()
        menu = []
        if self._suggestions and self.supports_command_picker():
            available = height - 4
            count = min(len(self._suggestions), available if len(self._suggestions) <= available
                        else available - 1)
            start = max(0, min(self._selected_suggestion - count // 2,
                               len(self._suggestions) - count))
            for index in range(start, start + count):
                key, description = self._suggestions[index]
                active = index == self._selected_suggestion
                marker = ("› " if self._unicode else "> ") if active else "  "
                line = marker + key + "  " + description
                line = _split_rows(line, columns)[0]
                if active and self._color:
                    line = "\x1b[38;5;110m" + line + "\x1b[0m"
                menu.append(line)
            if len(self._suggestions) > count:
                menu.append(self._muted(_split_rows(
                    f"  {self._selected_suggestion + 1} / {len(self._suggestions)} commands",
                    columns,
                )[0]))
        rows = []
        if self._pending:
            rows = _split_rows(self._pending, columns)
            # Keep a bounded live caption on screen; older rows become scrollback.
            keep = max(1, height - 6)
            if len(rows) > keep:
                self._commit_locked("\n".join(rows[:-keep]) + "\n")
                rows = rows[-keep:]
                self._pending = "".join(rows)
        prefix = "❯ " if self._unicode else "> "
        if _cells(prefix) >= columns:
            prefix = ">" if columns > 1 else ""
        start = 0
        while start < self._draft_cursor and _cells(
            prefix + self._draft[start:self._draft_cursor],
        ) >= columns:
            start += 1
        draft = self._draft[start:]
        while draft and _cells(prefix + draft) > columns:
            draft = draft[:-1]
        rule = self._muted(("─" if self._unicode else "-") * min(columns, 100))
        hint = ("Enter send  ·  / commands  ·  ↑↓ history" if self._unicode else
                "Enter send  / commands  Up/Down history")
        if menu:
            complete = self._suggestions[self._selected_suggestion][0].casefold() == (
                self._draft.casefold()
            )
            action = "run" if complete else "select"
            index = f"{self._selected_suggestion + 1}/{len(self._suggestions)}"
            hint = (f"↑↓ {index}  ·  Tab fill  ·  Enter {action}  ·  Esc close"
                    if self._unicode else
                    f"Up/Down {index}  Tab fill  Enter {action}  Esc close")
        hint = _split_rows(hint, columns)[0]
        bar = [rule, prefix + draft, rule, hint] if height >= 4 else [prefix + draft]
        visible_rows = rows[-max(1, height - len(bar)):]
        if len(bar) >= height:
            visible_rows = []
        self._live_lines = [*visible_rows, *bar]
        self._live_rows = len(self._live_lines)
        top = max(1, height - self._live_rows + 1)
        transcript = self._transcript_rows(columns)
        viewport = top - 1
        transcript_top = max(0, len(transcript) - viewport)
        if self._transcript_changed and self._layout_size == (columns, height):
            outgoing = transcript[self._transcript_top:transcript_top]
            for offset in range(0, len(outgoing), height):
                chunk = outgoing[offset:offset + height]
                first = self._transcript_top + offset
                soft = {index - first for index in self._transcript_soft_breaks
                        if first <= index < first + len(chunk) - 1}
                self.stream.write("\x1b[1;1H" + _join_rows(chunk, soft))
                self.stream.write(f"\x1b[{height};1H" + "\n" * len(chunk))
        content = transcript[transcript_top:] if viewport else []
        content += [""] * (viewport - len(content))
        # The picker overlays the text area, including a live reply, without
        # changing its viewport. Dismissal restores all underlying content.
        frame = [*content, *self._live_lines]
        covered = set()
        for index, line in enumerate(menu):
            position = height - len(bar) - len(menu) + index
            if position >= 0:
                frame[position] = line
                covered.add(position)
        soft = {index - transcript_top for index in self._transcript_soft_breaks
                if transcript_top <= index < min(len(transcript) - 1,
                                                transcript_top + viewport - 1)
                and index - transcript_top not in covered
                and index - transcript_top + 1 not in covered}
        self.stream.write("\x1b[1;1H\x1b[0J" + _join_rows(frame, soft))
        self._transcript_top = transcript_top
        self._transcript_changed = False
        if len(self._transcript) > 2000:
            removed = self._transcript[:-2000]
            self._transcript = self._transcript[-2000:]
            self._transcript_top = max(0, self._transcript_top - sum(
                len(_split_rows(line, columns + 1)) for line in removed
            ))
            self._transcript_revision += 1
        self._output_row = top
        self._anchor_height = height
        self._layout_size = (columns, height)
        self._cursor_from_bottom = 2 if height >= 4 else 0
        position = _cells(prefix + self._draft[start:self._draft_cursor])
        self.stream.write(f"\x1b[{height - self._cursor_from_bottom};{position + 1}H")
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
        back = _cells(self._draft[self._draft_cursor:])
        if back:
            self.stream.write(f"\x1b[{back}D")
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
