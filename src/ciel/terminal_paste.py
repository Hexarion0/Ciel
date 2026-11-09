"""Pasted text in the terminal draft: one block, never a line-by-line submit.

Terminals that support bracketed paste wrap pasted text in ``ESC[200~`` and
``ESC[201~`` once ``ESC[?2004h`` is sent. Short single-line pastes are inserted
as ordinary text. Longer or multi-line pastes become a placeholder such as
``[Pasted text #1 +42 lines]`` that the owner can type around; Enter expands it.
"""

from __future__ import annotations

import re

ENABLE = "\x1b[?2004h"
DISABLE = "\x1b[?2004l"
START = "200~"  # Escape-sequence keys as the terminal reader collects them.
END_SEQUENCE = "\x1b[201~"
# Single-line pastes up to this length stay inline as ordinary draft text.
INLINE_LIMIT = 200
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Editing keys, which a paste never contains (CR and tab are allowed: terminals
# send pasted line breaks as CR).
_EDITING = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """Unify line endings and drop control characters a paste can carry."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return _CONTROL.sub("", text)


def looks_pasted(chunk: str) -> bool:
    """A line break followed by more text in one read is a paste, not typing.

    Used only when the terminal did not mark the paste itself. Editing keys
    (backspace, Ctrl-U, arrows) mean a burst of typing, never pasted text.
    """
    return not _EDITING.search(chunk) and bool(re.search(r"[\r\n][^\r\n]", chunk))


class Pastes:
    """Placeholders in the current draft and the full text each stands for."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}
        # Short pastes that went in as plain text but start like a command.
        self._inline_commands: list[str] = []
        self._count = 0
        # Set by the reader while one unmarked paste arrives in several reads.
        self._continuing = False

    def _label(self, number: int, text: str) -> str:
        lines = text.count("\n") + 1
        size = f"+{lines} lines" if lines > 1 else f"{len(text)} chars"
        return f"[Pasted text #{number} {size}]"

    def insert(self, draft: str, cursor: int, text: str) -> tuple[str, int]:
        """Insert a paste at the cursor, as text or a placeholder."""
        text = normalize(text)
        # A later read of the same unmarked paste joins it whole, line breaks included.
        if self._extends_last(draft, cursor):
            return self._extend_last(draft, cursor, text)
        if "\n" not in text.strip("\n") and len(text) <= INLINE_LIMIT:
            piece = text.strip("\n")
            if piece.lstrip().startswith("/"):
                self._inline_commands.append(piece.strip())
        else:
            self._count += 1
            piece = self._label(self._count, text)
            self._items[piece] = text
        return draft[:cursor] + piece + draft[cursor:], cursor + len(piece)

    def _last(self) -> str | None:
        return next(reversed(self._items), None)

    def _extends_last(self, draft: str, cursor: int) -> bool:
        last = self._last()
        return last is not None and self._continuing and draft[:cursor].endswith(last)

    def _extend_last(self, draft: str, cursor: int, text: str) -> tuple[str, int]:
        last = self._last()
        assert last is not None
        number = int(re.search(r"#(\d+)", last).group(1))
        combined = self._items.pop(last) + text
        label = self._label(number, combined)
        self._items[label] = combined
        start = cursor - len(last)
        return draft[:start] + label + draft[cursor:], start + len(label)

    def continuing(self, value: bool) -> None:
        """Merge the next unmarked paste chunk into the previous placeholder."""
        self._continuing = value

    def span_at(self, draft: str, cursor: int) -> tuple[int, int] | None:
        """The placeholder that ends at or starts at ``cursor``, if any."""
        for label in self._items:
            for match in re.finditer(re.escape(label), draft):
                if cursor in (match.start(), match.end()):
                    return match.start(), match.end()
        return None

    def remove_before(self, draft: str, cursor: int) -> tuple[str, int] | None:
        """Backspace over a whole placeholder that ends at the cursor."""
        for label in self._items:
            if draft[:cursor].endswith(label):
                start = cursor - len(label)
                return draft[:start] + draft[cursor:], start
        return None

    def remove_after(self, draft: str, cursor: int) -> str | None:
        """Delete a whole placeholder that starts at the cursor."""
        for label in self._items:
            if draft[cursor:].startswith(label):
                return draft[:cursor] + draft[cursor + len(label):]
        return None

    def contains(self, draft: str) -> bool:
        """Whether the draft holds pasted text that must be sent as a message.

        True for a placeholder, and for a short inline paste that starts with
        "/", so a pasted "/stop" is never run as a command.
        """
        stripped = draft.lstrip()
        return (any(label in draft for label in self._items)
                or any(stripped.startswith(text) for text in self._inline_commands))

    def expand(self, draft: str) -> str:
        """The text to send: every placeholder replaced by its pasted content.

        One pass, so pasted text that quotes another placeholder stays as written.
        """
        if not self._items:
            return draft
        labels = sorted(self._items, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(label) for label in labels))
        return pattern.sub(lambda match: self._items[match.group(0)], draft)

    def clear(self) -> None:
        self._items.clear()
        self._inline_commands.clear()
        self._continuing = False
