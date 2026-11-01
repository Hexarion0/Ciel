"""Read-only, owner-admitted conversation catalogue with opaque process-local handles."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from typing import Any

from ciel.conversation import strip_internal_reply_notes
from ciel.logging import SecretMaskingFilter
from ciel.memory.archive_scope import admitted_archive_sessions
from ciel.memory.models import normalize_sources
from ciel.memory.store import MemoryStore


class ConversationCatalog:
    """Recheck ownership on every read; browsing never changes the active conversation."""

    def __init__(self, store: MemoryStore, scope: Callable[[], tuple[str, ...]]) -> None:
        self.store = store
        self.scope = scope
        self._key = secrets.token_bytes(32)

    def handle(self, session_id: str) -> str:
        """Produce a non-reversible handle, valid only after admission."""
        return hmac.new(self._key, session_id.encode(), hashlib.sha256).hexdigest()[:32]

    def _groups(self) -> dict[str, tuple[str, ...]]:
        scope = self.scope()
        sessions = {*scope, *admitted_archive_sessions(self.store, scope, include_private=True)}
        # Archive IDs must have a valid launch suffix. Legacy exact prefixes are admitted.
        sessions = {
            s
            for s in sessions
            if s in scope
            or ":run:" not in s
            or re.fullmatch(r"[a-f0-9]{32}", s.partition(":run:")[2])
        }
        groups = {s: (s,) for s in sessions}
        # Merge only projections explicitly recorded by the shared memory context.
        with self.store._lock:
            rows = (
                self.store._get_connection()
                .execute(
                    "SELECT summary_session_id,member_session_id FROM memory_session_members",
                )
                .fetchall()
            )
        projections: dict[str, set[str]] = {}
        for row in rows:
            projections.setdefault(row[0], set()).add(row[1])
        for members in projections.values():
            if len(members) < 2 or not members <= sessions:
                continue
            desktops = [s for s in members if s.startswith("desktop_session:")]
            if len(desktops) != 1:
                continue
            desktop = desktops[0]
            for member in members:
                groups.pop(member, None)
            groups[desktop] = tuple(sorted(members))
        return groups

    def resolve(self, handle: str) -> tuple[str, ...]:
        """Reject unknown or newly revoked handles without exposing session names."""
        for identity, members in self._groups().items():
            if hmac.compare_digest(self.handle(identity), handle):
                return members
        raise ValueError("This conversation is no longer available.")

    @staticmethod
    def _bounds(before: int | None, limit: int, maximum: int) -> None:
        if (
            before is not None
            and (type(before) is not int or before < 1)
            or type(limit) is not int
            or not 1 <= limit <= maximum
        ):
            raise ValueError("Invalid history page.")

    def list(self, before: int | None = None, limit: int = 30) -> dict[str, Any]:
        """List bounded summaries, newest first, with deterministic user-authored titles."""
        self._bounds(before, limit, 50)
        scope = self.scope()
        entries = []
        for identity, members in self._groups().items():
            marks = ",".join("?" for _ in members)
            with self.store._lock:
                conn = self.store._get_connection()
                last = conn.execute(
                    f"SELECT MAX(id) FROM conversation_turns WHERE session_id IN ({marks})",
                    members,
                ).fetchone()[0]
                first = conn.execute(
                    f"SELECT content FROM conversation_turns WHERE session_id IN ({marks}) "
                    "AND role='user' ORDER BY id LIMIT 1",
                    members,
                ).fetchone()
            if last is None or before is not None and last >= before:
                continue
            title = " ".join(first[0].split())[:80] if first else "Conversation"
            entries.append(
                {
                    "handle": self.handle(identity),
                    "title": SecretMaskingFilter._redact(title),
                    "cursor": last,
                }
            )
        if scope != self.scope():
            raise ValueError("The conversation owner changed.")
        entries.sort(key=lambda item: item["cursor"], reverse=True)
        return {
            "conversations": entries[:limit],
            "next_before": entries[limit - 1]["cursor"] if len(entries) > limit else None,
        }

    def read(self, handle: str, before: int | None = None, limit: int = 50) -> dict[str, Any]:
        """Read one chronological page without model calls or history mutations."""
        self._bounds(before, limit, 100)
        members = self.resolve(handle)
        marks = ",".join("?" for _ in members)
        where = " AND id < ?" if before is not None else ""
        with self.store._lock:
            rows = (
                self.store._get_connection()
                .execute(
                    f"SELECT * FROM conversation_turns WHERE session_id IN ({marks}){where} "
                    "ORDER BY id DESC LIMIT ?",
                    (*members, *((before,) if before is not None else ()), limit + 1),
                )
                .fetchall()
            )
            turns = [self.store._row_to_turn(row) for row in rows[:limit]]
        if members != self.resolve(handle):
            raise ValueError("The conversation owner changed.")
        messages = [
            {
                "id": t.id,
                "role": t.role,
                "text": SecretMaskingFilter._redact(strip_internal_reply_notes(t.content)),
                "sources": normalize_sources(t.sources),
                "time": t.timestamp.timestamp(),
            }
            for t in reversed(turns)
            if t.role in {"user", "assistant"}
        ]
        return {
            "handle": handle,
            "messages": messages,
            "next_before": turns[-1].id if len(rows) > limit else None,
        }
