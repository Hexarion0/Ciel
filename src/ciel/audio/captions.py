"""Terminal captions driven by playback progress, without a separate typing timer."""

from __future__ import annotations

import logging
import re
import sys
import threading
from collections.abc import Callable
from typing import TextIO

from ciel.logging import SecretMaskingFilter

_TERMINAL_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])"
)


def _plain_text(text: str) -> str:
    """Keep caption text on one line and remove terminal control sequences."""
    text = SecretMaskingFilter._redact(_TERMINAL_ESCAPE.sub("", text))
    return " ".join(
        "".join(char for char in text if char.isspace() or ord(char) >= 32 and ord(char) != 127)
        .split()
    )


class TerminalCaptions:
    """Append playback-driven captions, letting the terminal wrap naturally.

    Sentences share one reply line and one prefix until ``close``, interruption,
    or a log ends the line. TTY output appends only newly visible characters;
    redirected output appends only completed or interrupted sentence snapshots.
    All output, including caption-aware logs, shares one lock.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        prefix: str = "CIEL: ",
        width: int | None = None,
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.prefix = _plain_text(prefix).rstrip() + " "
        # Retain the width argument for callers; wrapping is handled by the terminal.
        self.width = width
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.RLock()
        self._full_text = ""
        self._visible_text = ""
        self._active = False
        self._line_open = False
        self._emitted_text = ""

    def start(self, full_text: str) -> None:
        """Start a sentence, preserving a previously unfinished visible prefix."""
        with self._lock:
            if self._active:
                self._finish_locked(interrupted=True)
            self._full_text = _plain_text(full_text)
            self._visible_text = ""
            self._active = bool(self._full_text)

    def update(
        self,
        visible_text: str,
        *,
        finished: bool = False,
        interrupted: bool = False,
    ) -> None:
        """Show a valid prefix; ignore mismatches and never rewind visible words."""
        with self._lock:
            if not self._active:
                return
            visible = _plain_text(visible_text)
            if not self._full_text.startswith(visible):
                if interrupted:
                    # A partially revealed secret may not match its final mask.
                    # Save the last safe prefix without leaving a stale caption.
                    self._finish_locked(interrupted=True)
                return
            if len(visible) >= len(self._visible_text):
                self._visible_text = visible
            if finished or interrupted:
                self._finish_locked(interrupted=interrupted)
            elif self.is_tty:
                self._draw_locked()

    def close(self) -> None:
        """Save any visible unfinished speech and leave a clean terminal line."""
        with self._lock:
            if self._active:
                self._finish_locked(interrupted=True)
            self._end_line_locked()

    def write_log(self, message: str) -> None:
        """Separate logs from speech, then restore any active caption prefix."""
        with self._lock:
            self._end_line_locked()
            self.stream.write(message.rstrip("\r\n") + "\n")
            self._emitted_text = ""
            if self.is_tty and self._active:
                self._draw_locked()
            self.stream.flush()

    def _draw_locked(self) -> None:
        if not self._visible_text or self._visible_text == self._emitted_text:
            return
        if not self._emitted_text:
            self.stream.write(" " if self._line_open else self.prefix)
            self._line_open = True
        self.stream.write(self._visible_text[len(self._emitted_text):])
        self._emitted_text = self._visible_text
        self.stream.flush()

    def _end_line_locked(self) -> None:
        if self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    def _finish_locked(self, *, interrupted: bool) -> None:
        if self._visible_text:
            self._draw_locked()
            if interrupted:
                self.stream.write(" [interrupted]")
        if interrupted:
            self._end_line_locked()
        self.stream.flush()
        self._full_text = self._visible_text = self._emitted_text = ""
        self._active = False


class CaptionLoggingHandler(logging.StreamHandler):
    """Preserve formatted logging while sharing the caption renderer's lock."""

    def __init__(self, captions: TerminalCaptions) -> None:
        super().__init__(captions.stream)
        self.captions = captions

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.captions.write_log(self.format(record))
        except Exception:
            self.handleError(record)


def _replace_handler(
    logger: logging.Logger, original: logging.Handler, replacement: logging.Handler,
) -> bool:
    """Keep handler order, including filters that redact before other handlers."""
    with logging._lock:
        if original not in logger.handlers:
            return False
        logger.handlers[logger.handlers.index(original)] = replacement
        return True


def install_caption_logging(
    captions: TerminalCaptions, logger: logging.Logger | None = None,
) -> Callable[[], None]:
    """Replace matching console handlers and return an idempotent restore hook.

    Existing formatter, level, and filters (including secret redaction) are
    copied. Handlers for other streams or files are left in place. Call
    ``captions.close()`` before the returned hook when leaving the voice loop.
    """
    target = logging.getLogger() if logger is None else logger
    replacements: list[tuple[logging.Handler, logging.Handler]] = []
    for original in list(target.handlers):
        if (
            not isinstance(original, logging.StreamHandler)
            or original.stream is not captions.stream
        ):
            continue
        if isinstance(original, CaptionLoggingHandler):
            continue
        replacement = CaptionLoggingHandler(captions)
        replacement.setLevel(original.level)
        replacement.setFormatter(original.formatter)
        for log_filter in original.filters:
            replacement.addFilter(log_filter)
        if _replace_handler(target, original, replacement):
            replacements.append((original, replacement))

    def restore() -> None:
        for original, replacement in replacements:
            if _replace_handler(target, replacement, original):
                replacement.close()

    return restore
