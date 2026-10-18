"""Small celestial console presentation with portable, accessible input fallbacks."""

from __future__ import annotations

import os
import re
import select
import shutil
import sys
import unicodedata
from collections.abc import Sequence
from itertools import zip_longest
from typing import TextIO

_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
_COLORS = {"cyan": "\x1b[38;5;87m", "violet": "\x1b[38;5;183m", "muted": "\x1b[38;5;245m"}
_ASCII_SYMBOLS = str.maketrans({"✦": "*", "◇": "*", "✓": "+", "·": ".", "›": ">",
                               "—": "-", "–": "-", "…": "...", "’": "'", "“": '"', "”": '"'})


def _cells(text: str) -> int:
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap words and long labels using terminal cells rather than codepoint counts."""
    width = max(1, width)
    lines: list[str] = []
    line = ""
    for word in text.split():
        if line and _cells(line + " " + word) <= width:
            line += " " + word
            continue
        if line:
            lines.append(line)
            line = ""
        for char in word:
            if _cells(char) > width:
                char = "?"
            if line and _cells(line + char) > width:
                lines.append(line)
                line = ""
            line += char
    if line or not lines:
        lines.append(line)
    return lines


class ConsoleUI:
    """Render complete console blocks; select with arrows or validated numbered input."""

    def __init__(self, stream: TextIO | None = None, input_stream: TextIO | None = None) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.input_stream = sys.stdin if input_stream is None else input_stream
        self._terminal = bool(getattr(self.stream, "isatty", lambda: False)()) and (
            os.environ.get("TERM", "") not in {"dumb", "unknown"}
        )
        self._color = self._terminal and "NO_COLOR" not in os.environ
        underlying = self.stream
        encoding = getattr(underlying, "encoding", None)
        seen: set[int] = set()
        while not encoding and hasattr(underlying, "stream") and id(underlying) not in seen:
            seen.add(id(underlying))
            underlying = underlying.stream
            encoding = getattr(underlying, "encoding", None)
        try:
            "✦◇✓·›╭─╮│╰╯".encode(encoding or "utf-8")
            self._unicode = True
        except (UnicodeError, LookupError):
            self._unicode = False

    def _text(self, value: str) -> str:
        text = _ESCAPE.sub("", str(value))
        text = "".join(" " if char.isspace() else char for char in text)
        text = "".join(char for char in text
                       if unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
        text = " ".join(text.split())
        if not self._unicode:
            text = unicodedata.normalize("NFKD", text.translate(_ASCII_SYMBOLS))
            text = text.encode("ascii", errors="replace").decode("ascii")
        return text

    def _size(self) -> os.terminal_size:
        try:
            size = os.get_terminal_size(self.stream.fileno())
            if size.columns > 0 and size.lines > 0:
                return size
        except (AttributeError, OSError, ValueError):
            pass
        return shutil.get_terminal_size(fallback=(80, 24))

    def _width(self) -> int:
        return max(1, min(60, self._size().columns))

    def _paint(self, text: str, color: str) -> str:
        return _COLORS[color] + text + "\x1b[0m" if self._color else text

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _message(self, marker: str, text: str, color: str) -> None:
        prefix = self._text(marker) + " "
        width = self._width()
        if _cells(prefix) >= width:
            prefix = ""
        lines = _wrap(self._text(text), width - _cells(prefix))
        self._write("\n".join(self._paint(prefix, color) + line if index == 0 else
                               " " * _cells(prefix) + line
                               for index, line in enumerate(lines)) + "\n")

    def banner(self, subtitle: str, version: str | None = None) -> None:
        """Introduce CIEL with a compact cyan/violet celestial heading."""
        heading = self._text("✦  C I E L" + (f"   v{version}" if version else ""))
        lines = [self._paint(line, "violet") for line in _wrap(heading, self._width())]
        lines.extend(self._paint(line, "muted") for line in
                     _wrap(self._text(subtitle), self._width()))
        lines.append(self._paint(("─" if self._unicode else "-") * self._width(), "cyan"))
        self._write("\n".join(lines) + "\n")

    def step(self, title: str) -> None:
        """Introduce the next setup or startup step."""
        self._message("◇", title, "violet")

    def note(self, text: str) -> None:
        """Show a short status or explanation."""
        self._message("·", text, "muted")

    def finish(self, text: str) -> None:
        """Show the final setup or readiness status."""
        self._message("✓", text, "cyan")

    def summary(self, rows: Sequence[tuple[str, str]]) -> None:
        """Show the selected configuration as a compact panel."""
        self.panel(rows)

    def panel(self, rows: Sequence[tuple[str, str]]) -> None:
        """Write an entire aligned status panel atomically to prevent interleaved logs."""
        width = self._width()
        if width < 10:
            lines = [line for label, value in rows
                     for line in _wrap(self._text(f"{label}: {value}"), width)]
            self._write("\n".join(lines) + "\n")
            return
        left, right, vertical, horizontal = ("╭", "╮", "│", "─") if self._unicode else (
            "+", "+", "|", "-",
        )
        bottom_left, bottom_right = ("╰", "╯") if self._unicode else ("+", "+")
        def border(text: str) -> str:
            return self._paint(text, "cyan")

        lines = [border(left + horizontal * (width - 2) + right)]
        inner = width - 4
        label_width = min(14, max(1, inner // 3))
        value_width = inner - label_width - 2
        for label, value in rows:
            labels = _wrap(self._text(label), label_width)
            values = _wrap(self._text(value), value_width)
            for key, content in zip_longest(labels, values, fillvalue=""):
                padded_key = key + " " * (label_width - _cells(key))
                padded_value = content + " " * (value_width - _cells(content))
                lines.append(border(vertical) + " " + self._paint(padded_key, "violet")
                             + "  " + padded_value + " " + border(vertical))
        lines.append(border(bottom_left + horizontal * (width - 2) + bottom_right))
        self._write("\n".join(lines) + "\n")

    def prompt(self, label: str) -> str:
        """Read one line of text, preserving EOF and interruption for the caller."""
        marker = " › " if self._unicode else " > "
        width = self._width()
        if width <= len(marker):
            marker = ">"
        lines = _wrap(self._text(label), max(1, width - len(marker)))
        gap = "\n" if _cells(lines[-1]) + len(marker) > width else ""
        self._write("\n".join(lines) + gap + marker)
        answer = self.input_stream.readline()
        if answer == "":
            raise EOFError
        if not getattr(self.input_stream, "isatty", lambda: False)():
            self._write("\n")
        return answer.rstrip("\r\n")

    def choose(
        self, title: str, choices: Sequence[tuple[str, str]], default: str | None = None,
    ) -> str:
        """Return a choice value, honoring the selected default on Enter."""
        choices = list(choices)
        values = [value for value, _label in choices]
        if not choices or len(set(values)) != len(values):
            raise ValueError("Choices must have distinct values and cannot be empty")
        if default is not None and default not in values:
            raise ValueError("Default must be one of the choice values")
        self.step(title)
        if self._terminal and getattr(self.input_stream, "isatty", lambda: False)():
            try:
                import termios
                import tty

                fd = self.input_stream.fileno()
                saved = termios.tcgetattr(fd)
            except (ImportError, AttributeError, OSError, ValueError):
                pass
            else:
                try:
                    tty.setcbreak(fd, termios.TCSANOW)
                    self._write("\x1b[?25l")
                    return self._choose_arrows(
                        choices, values.index(default) if default is not None else 0, fd,
                    )
                finally:
                    try:
                        termios.tcsetattr(fd, termios.TCSANOW, saved)
                    finally:
                        self._write("\x1b[?25h")
        for index, (_value, label) in enumerate(choices, 1):
            suffix = " (default)" if _value == default else ""
            self.note(f"{index}. {label}{suffix}")
        while True:
            hint = f" [{values.index(default) + 1}]" if default is not None else ""
            answer = self.prompt(f"Choose 1-{len(choices)}{hint}").strip()
            if not answer and default is not None:
                return default
            if answer.isdecimal() and 1 <= int(answer) <= len(choices):
                return choices[int(answer) - 1][0]
            self.note(f"Enter a number from 1 to {len(choices)}.")

    def _choose_arrows(self, choices: Sequence[tuple[str, str]], selected: int, fd: int) -> str:
        self.note("Up/Down to move, Enter to select")
        drawn = 0
        while True:
            width = self._width()
            prefix = "› " if self._unicode else "> "
            if width <= 2:
                prefix = ""
            lines = []
            selected_line = 0
            for index, (_value, label) in enumerate(choices):
                wrapped = _wrap(self._text(label), max(1, width - len(prefix)))
                if index == selected:
                    selected_line = len(lines)
                for row, text in enumerate(wrapped):
                    marker = prefix if index == selected and row == 0 else " " * len(prefix)
                    lines.append(self._paint(marker + text, "cyan") if index == selected else
                                 marker + text)
            height = max(1, self._size().lines - 8)
            start = max(0, min(selected_line - height // 2, len(lines) - height))
            visible = lines[start:start + height]
            count = max(drawn, len(visible))
            redraw = f"\x1b[{drawn}A\r" if drawn else ""
            redraw += "".join("\r\x1b[2K" + line + "\n" for line in
                              [*visible, *([""] * (count - len(visible)))])
            self._write(redraw)
            drawn = count
            key = self._key_char(fd)
            if key in {"\r", "\n"}:
                return choices[selected][0]
            if not key or key == "\x04":
                raise EOFError
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x1b" and select.select([fd], [], [], 0.05)[0]:
                introducer = self._key_char(fd)
                if introducer in {"[", "O"} and select.select([fd], [], [], 0.05)[0]:
                    direction = self._key_char(fd)
                    if direction in {"A", "B"}:
                        selected = (selected + (1 if direction == "B" else -1)) % len(choices)

    def _key_char(self, fd: int) -> str:
        # A TextIOWrapper may prefetch the complete escape sequence, making select()
        # miss buffered bytes. Real terminals read exactly one byte from the fd.
        return os.read(fd, 1).decode("latin-1") if os.isatty(fd) else self.input_stream.read(1)
