"""Private local pairing and at-most-once delivery state for messaging."""

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class ChatAlreadyRunning(RuntimeError):
    """Another process owns this chat state file's polling lock."""


@contextmanager
def polling_lock(path: str | Path) -> Iterator[None]:
    """Prevent local pairing and polling processes from consuming each other's updates."""
    import fcntl

    lock_path = Path(f"{Path(path).resolve()}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ChatAlreadyRunning(
                "Ciel chat is already running. Stop it before pairing or starting another copy."
            ) from None
        yield
    finally:
        os.close(descriptor)


class ChatState:
    """Persist owner binding and update claims without storing messages or bot tokens."""

    def __init__(self, path: str | Path) -> None:
        selected = Path(path)
        selected.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(selected, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        self._db = sqlite3.connect(selected)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS telegram_bots ("
            "bot_id INTEGER PRIMARY KEY, owner_id INTEGER, next_offset INTEGER NOT NULL DEFAULT 0,"
            "last_update_at REAL NOT NULL DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS telegram_receipts ("
            "bot_id INTEGER NOT NULL, update_id INTEGER NOT NULL, status TEXT NOT NULL,"
            "PRIMARY KEY (bot_id, update_id));"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(telegram_bots)")}
        if "last_update_at" not in columns:
            self._db.execute(
                "ALTER TABLE telegram_bots ADD COLUMN last_update_at REAL NOT NULL DEFAULT 0",
            )
        self._db.commit()

    def owner(self, bot_id: int) -> int | None:
        """Return the explicitly paired private owner for this exact bot identity."""
        row = self._db.execute(
            "SELECT owner_id FROM telegram_bots WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        return row[0] if row else None

    def bind(self, bot_id: int, owner_id: int) -> None:
        """Bind an owner after local code verification, preserving the consumed offset."""
        with self._db:
            self._db.execute(
                "INSERT INTO telegram_bots (bot_id, owner_id) VALUES (?, ?) "
                "ON CONFLICT(bot_id) DO UPDATE SET owner_id = excluded.owner_id",
                (bot_id, owner_id),
            )

    def unpair(self) -> None:
        """Remove local owner bindings while retaining delivery offsets against replay."""
        with self._db:
            self._db.execute("UPDATE telegram_bots SET owner_id = NULL")

    def bindings(self) -> list[tuple[int, int]]:
        """Return local bot/owner identifiers for a terminal status display."""
        return self._db.execute(
            "SELECT bot_id, owner_id FROM telegram_bots WHERE owner_id IS NOT NULL"
        ).fetchall()

    def offset(self, bot_id: int) -> int:
        """Return the next never-claimed Telegram update ID."""
        row = self._db.execute(
            "SELECT next_offset FROM telegram_bots WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        return row[0] if row else 0

    def recover_idle_offset(self, bot_id: int, now: float) -> bool:
        """Allow Telegram's randomized update IDs after a week without updates.

        Keep recent receipts so resetting a cursor never authorizes a repeated mutation.
        Telegram retains incoming updates for at most 24 hours; freshness is checked
        separately by the transport before processing a request.
        """
        with self._db:
            changed = self._db.execute(
                "UPDATE telegram_bots SET next_offset = 0, last_update_at = ? "
                "WHERE bot_id = ? AND last_update_at <= ?",
                (now, bot_id, now - 7 * 24 * 60 * 60),
            ).rowcount
        return bool(changed)

    def claim(self, bot_id: int, update_id: int, *, now: float | None = None) -> bool:
        """Consume before invoking memory or AI, so a retry cannot repeat a mutation."""
        with self._db:
            if update_id < self.offset(bot_id):
                return False
            self._db.execute(
                "INSERT INTO telegram_bots (bot_id, next_offset, last_update_at) VALUES (?, ?, ?) "
                "ON CONFLICT(bot_id) DO UPDATE SET "
                "next_offset = MAX(next_offset, excluded.next_offset), "
                "last_update_at = excluded.last_update_at",
                (bot_id, update_id + 1, now if now is not None else time.time()),
            )
            inserted = self._db.execute(
                "INSERT OR IGNORE INTO telegram_receipts VALUES (?, ?, 'claimed')",
                (bot_id, update_id),
            ).rowcount
            self._db.execute(
                "DELETE FROM telegram_receipts WHERE bot_id = ? AND rowid NOT IN "
                "(SELECT rowid FROM telegram_receipts WHERE bot_id = ? "
                "ORDER BY rowid DESC LIMIT 1000)", (bot_id, bot_id),
            )
            return bool(inserted)

    def finish(self, bot_id: int, update_id: int, status: str) -> None:
        """Record completion without retaining private request or reply text."""
        with self._db:
            self._db.execute(
                "UPDATE telegram_receipts SET status = ? WHERE bot_id = ? AND update_id = ?",
                (status, bot_id, update_id),
            )

    def close(self) -> None:
        """Close the SQLite handle."""
        self._db.close()
