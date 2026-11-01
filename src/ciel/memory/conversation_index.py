"""Conversation titles, hidden flags, and owner-scoped listing for people and the model.

Hiding a conversation only removes it from the sidebar list: its turns and any saved
memory stay in the database, and Ciel can still find and read it when asked. Titles are
chosen by Ciel (automatically or on request) and can change at any time.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any

from ciel.conversation import strip_internal_reply_notes
from ciel.logging import SecretMaskingFilter
from ciel.memory.archive_scope import admitted_archive_sessions
from ciel.memory.store import MemoryStore

TITLE_PREFIX = "conversation_title:"
HIDDEN_PREFIX = "conversation_hidden:"
MAX_TITLE_CHARS = 60
MAX_READ_CHARS = 2000  # per message, so long pasted turns cannot flood the model


def clean_title(text: str) -> str:
    """One short line, no quotes or trailing punctuation, secrets masked."""
    line = " ".join(str(text).split())
    line = line.strip(" \"'“”‘’`*#").rstrip(" .!?:;,")
    line = SecretMaskingFilter._redact(line)[:MAX_TITLE_CHARS].rstrip()
    if not line:
        raise ValueError("A conversation title needs some words.")
    return line


def conversation_groups(
    store: MemoryStore, scope: tuple[str, ...], *, include_private: bool,
) -> dict[str, tuple[str, ...]]:
    """Admitted conversations keyed by identity; launches merged as recorded in memory."""
    sessions = {*scope, *admitted_archive_sessions(store, scope, include_private=include_private)}
    # Archive IDs must have a valid launch suffix. Legacy exact prefixes are admitted.
    sessions = {
        s for s in sessions
        if s in scope or ":run:" not in s
        or re.fullmatch(r"[a-f0-9]{32}", s.partition(":run:")[2])
    }
    groups = {s: (s,) for s in sessions}
    with store._lock:
        rows = store._get_connection().execute(
            "SELECT summary_session_id,member_session_id FROM memory_session_members",
        ).fetchall()
    projections: dict[str, set[str]] = {}
    for row in rows:
        projections.setdefault(row[0], set()).add(row[1])
    for members in projections.values():
        if len(members) < 2 or not members <= sessions:
            continue
        desktops = [s for s in members if s.startswith("desktop_session:")]
        if len(desktops) != 1:
            continue
        for member in members:
            groups.pop(member, None)
        groups[desktops[0]] = tuple(sorted(members))
    return groups


def conversation_id(identity: str) -> str:
    """A short, stable reference the model can use; resolved only through admission."""
    return "c" + hashlib.sha256(identity.encode()).hexdigest()[:10]


class ConversationIndex:
    """Titles and hidden flags, one settings row per conversation beside its history.

    Separate rows make every write a single atomic upsert, so the titler, the web app,
    the model's rename tool, and the separate chat worker never overwrite each other.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def _entry(self, identity: str) -> dict[str, Any]:
        value = self.store.get_setting(TITLE_PREFIX + identity, None)
        return value if isinstance(value, dict) else {}

    def title(self, identity: str) -> str | None:
        title = self._entry(identity).get("title")
        return title if isinstance(title, str) and title else None

    def titled_at(self, identity: str) -> int:
        """How many owner messages the conversation had when its title was last chosen."""
        turns = self._entry(identity).get("turns", 0)
        return turns if isinstance(turns, int) else 0

    def set_title(self, identity: str, title: str, *, turns: int = 0) -> str:
        cleaned = clean_title(title)
        self.store.set_setting(TITLE_PREFIX + identity, {"title": cleaned, "turns": int(turns)})
        return cleaned

    def hidden(self) -> set[str]:
        with self.store._lock:
            rows = self.store._get_connection().execute(
                "SELECT name,value_json FROM memory_settings WHERE name LIKE ?",
                (HIDDEN_PREFIX + "%",),
            ).fetchall()
        return {row[0][len(HIDDEN_PREFIX):] for row in rows if row[1] == "true"}

    def set_hidden(self, identity: str, hidden: bool) -> None:
        self.store.set_setting(HIDDEN_PREFIX + identity, bool(hidden))

    def user_turns(self, members: tuple[str, ...] | str) -> int:
        """How many messages the owner has sent across a conversation's sessions."""
        members = (members,) if isinstance(members, str) else members
        marks = ",".join("?" for _ in members)
        with self.store._lock:
            row = self.store._get_connection().execute(
                f"SELECT COUNT(*) FROM conversation_turns WHERE session_id IN ({marks}) "
                "AND role='user'", members,
            ).fetchone()
        return int(row[0] or 0)

    @staticmethod
    def group_of(
        groups: dict[str, tuple[str, ...]], session_id: str,
    ) -> tuple[str, tuple[str, ...]]:
        """The conversation identity and sessions a session belongs to (merged launches)."""
        for identity, members in groups.items():
            if session_id == identity or session_id in members:
                return identity, members
        return session_id, (session_id,)

    def _latest(self, groups: dict[str, tuple[str, ...]]) -> dict[str, tuple[int, int]]:
        """Newest turn id and message count per conversation, in one grouped query."""
        owner = {member: identity for identity, members in groups.items() for member in members}
        if not owner:
            return {}
        marks = ",".join("?" for _ in owner)
        with self.store._lock:
            rows = self.store._get_connection().execute(
                f"SELECT session_id, MAX(id), COUNT(*) FROM conversation_turns "
                f"WHERE session_id IN ({marks}) AND role IN ('user','assistant') "
                "GROUP BY session_id", tuple(owner),
            ).fetchall()
        latest: dict[str, tuple[int, int]] = {}
        for session_id, last, count in rows:
            identity = owner[session_id]
            previous_last, previous_count = latest.get(identity, (0, 0))
            latest[identity] = (max(previous_last, last), previous_count + count)
        return latest

    def _details(self, members: tuple[str, ...]) -> tuple[str | None, float]:
        """First owner message (fallback title) and the time of the latest turn."""
        marks = ",".join("?" for _ in members)
        with self.store._lock:
            conn = self.store._get_connection()
            first = conn.execute(
                f"SELECT content FROM conversation_turns WHERE session_id IN ({marks}) "
                "AND role='user' ORDER BY id LIMIT 1", members,
            ).fetchone()
            latest = conn.execute(
                f"SELECT timestamp FROM conversation_turns WHERE session_id IN ({marks}) "
                "ORDER BY id DESC LIMIT 1", members,
            ).fetchone()
        return (first[0] if first else None), (_timestamp(latest[0]) if latest else 0.0)

    def display_title(self, identity: str, first_message: str | None) -> str:
        """Ciel's chosen title, else the first thing the owner said."""
        chosen = self.title(identity)
        if chosen:
            return chosen
        fallback = " ".join(first_message.split())[:80] if first_message else "Conversation"
        return SecretMaskingFilter._redact(fallback)

    def summaries(
        self, scope: tuple[str, ...], *, include_private: bool, current: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Newest-first conversation list for the model, hidden ones included and marked."""
        groups = conversation_groups(self.store, scope, include_private=include_private)
        current_identity = self.group_of(groups, current)[0] if current else ""
        latest = self._latest(groups)
        newest = sorted(latest, key=lambda identity: latest[identity][0], reverse=True)
        hidden, entries = self.hidden(), []
        for identity in newest[:max(1, min(limit, 50))]:
            first, stamp = self._details(groups[identity])
            entries.append({
                "id": conversation_id(identity), "title": self.display_title(identity, first),
                "last_active": datetime.fromtimestamp(stamp).isoformat(timespec="minutes")
                if stamp else "", "messages": latest[identity][1],
                "current": identity == current_identity,
                "hidden_from_list": identity in hidden,
            })
        return entries

    def read(
        self, scope: tuple[str, ...], reference: str, *, include_private: bool,
        before: int | None = None, limit: int = 40,
    ) -> dict[str, Any]:
        """One chronological page of an admitted conversation, as data for the model."""
        groups = conversation_groups(self.store, scope, include_private=include_private)
        match = next(((identity, members) for identity, members in groups.items()
                      if conversation_id(identity) == reference), None)
        if match is None:
            raise ValueError("That conversation is not available.")
        identity, members = match
        marks = ",".join("?" for _ in members)
        where = " AND id < ?" if before is not None else ""
        with self.store._lock:
            rows = self.store._get_connection().execute(
                f"SELECT * FROM conversation_turns WHERE session_id IN ({marks}){where} "
                "AND role IN ('user','assistant') ORDER BY id DESC LIMIT ?",
                (*members, *((before,) if before is not None else ()), limit + 1),
            ).fetchall()
            turns = [self.store._row_to_turn(row) for row in rows[:limit]]
        first, _ = self._details(members)
        return {
            "id": reference, "title": self.display_title(identity, first),
            "messages": [{
                "role": turn.role, "time": turn.timestamp.isoformat(timespec="minutes"),
                "text": SecretMaskingFilter._redact(self.store.redact_forgotten(
                    strip_internal_reply_notes(turn.content)))[:MAX_READ_CHARS],
            } for turn in reversed(turns)],
            "earlier_before": turns[-1].id if len(rows) > limit else None,
        }


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0
