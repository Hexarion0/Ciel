"""SQLite persistence engine for short-term conversation context and long-term memories."""

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raphael.logging import get_logger
from raphael.memory.models import ConversationTurn, MemoryItem, MemoryType, normalize_sources

logger = get_logger("memory.store")


class MemoryStore:
    """Thread-safe SQLite storage for long-term facts, preferences, and session history."""

    def __init__(self, db_path: str | Path = "data/raphael.db") -> None:
        self.db_path_str = str(db_path)
        self._is_in_memory = self.db_path_str == ":memory:"

        if not self._is_in_memory:
            path_obj = Path(self.db_path_str)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._local = threading.local()
        self._memory_connection: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection with WAL mode and row factory."""
        if self._is_in_memory and self._memory_connection is not None:
            return self._memory_connection
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path_str,
                check_same_thread=False,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            if not self._is_in_memory:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._local.conn = conn
            if self._is_in_memory:
                self._memory_connection = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create necessary tables and indices if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'user_explicit',
                        confidence REAL NOT NULL DEFAULT 1.0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL DEFAULT 'default',
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        tokens INTEGER NOT NULL DEFAULT 0,
                        latency REAL NOT NULL DEFAULT 0.0,
                        sources_json TEXT NOT NULL DEFAULT '[]'
                    );
                    """
                )
                columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(conversation_turns)")
                }
                if "sources_json" not in columns:
                    conn.execute(
                        "ALTER TABLE conversation_turns "
                        "ADD COLUMN sources_json TEXT NOT NULL DEFAULT '[]';"
                    )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_keys ("
                    "fact_key TEXT PRIMARY KEY, memory_id INTEGER UNIQUE NOT NULL "
                    "REFERENCES memories(id) ON DELETE CASCADE);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS forgotten_memories ("
                    "fact_key TEXT PRIMARY KEY, patterns_json TEXT NOT NULL);"
                )
                # Index existing structured project anchors without removing any records.
                for row in conn.execute(
                    "SELECT id, metadata_json FROM memories ORDER BY updated_at DESC;"
                ).fetchall():
                    metadata = json.loads(row["metadata_json"])
                    key = metadata.get("fact_key")
                    if key:
                        scope = metadata.get("project") or metadata.get("scope", "user")
                        conn.execute(
                            "INSERT OR IGNORE INTO memory_keys VALUES (?, ?);",
                            (f"{scope}:{key}", row["id"]),
                        )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conv_session "
                    "ON conversation_turns(session_id, timestamp);"
                )
            logger.debug("Memory database initialized at %s", self.db_path_str)

    # -------------------------------------------------------------------------
    # Long-Term Memory Operations
    # -------------------------------------------------------------------------

    def save_memory(self, item: MemoryItem) -> int:
        """Insert or update a memory item. Returns the assigned memory ID."""
        with self._lock:
            conn = self._get_connection()
            now_iso = datetime.now(timezone.utc).isoformat()
            type_val = (
                item.memory_type.value
                if isinstance(item.memory_type, MemoryType)
                else str(item.memory_type)
            )
            meta_str = json.dumps(item.metadata)

            with conn:
                if item.id is not None:
                    # Update existing
                    cursor = conn.execute(
                        """
                        UPDATE memories
                        SET content = ?, memory_type = ?, source = ?, confidence = ?,
                            updated_at = ?, metadata_json = ?
                        WHERE id = ?;
                        """,
                        (
                            item.content,
                            type_val,
                            item.source,
                            item.confidence,
                            now_iso,
                            meta_str,
                            item.id,
                        ),
                    )
                    if cursor.rowcount > 0:
                        return item.id

                # Insert new
                cursor = conn.execute(
                    """
                    INSERT INTO memories (
                        content, memory_type, source, confidence,
                        created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        item.content,
                        type_val,
                        item.source,
                        item.confidence,
                        item.created_at.isoformat() if item.created_at else now_iso,
                        now_iso,
                        meta_str,
                    ),
                )
                item.id = cursor.lastrowid
                return item.id

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        """Retrieve a specific memory by ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("SELECT * FROM memories WHERE id = ?;", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    def get_fact(self, key: str) -> MemoryItem | None:
        """Retrieve the current value of a stable, namespaced fact key."""
        with self._lock:
            row = (
                self._get_connection()
                .execute(
                    "SELECT m.* FROM memories m JOIN memory_keys k ON k.memory_id=m.id "
                    "WHERE k.fact_key=?;",
                    (key,),
                )
                .fetchone()
            )
            return self._row_to_memory(row) if row else None

    def upsert_fact(self, key: str, item: MemoryItem) -> int:
        """Atomically replace a keyed fact, retaining its original record and revision history."""
        scope, fact_key = key.split(":", 1)
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute(
                    "SELECT m.* FROM memories m JOIN memory_keys k ON k.memory_id=m.id "
                    "WHERE k.fact_key=?;",
                    (key,),
                ).fetchone()
                observed = datetime.now(timezone.utc).isoformat()
                metadata = {
                    **item.metadata,
                    "scope": scope,
                    "fact_key": fact_key,
                    "observed_at": observed,
                }
                if scope == "raphael":
                    metadata["project"] = "raphael"
                if row:
                    old = self._row_to_memory(row)
                    revisions = old.metadata.get("revisions", [])
                    if old.content != item.content:
                        revisions = [
                            *revisions,
                            {
                                "content": old.content,
                                "value": old.metadata.get("value"),
                                "recorded_at": old.updated_at.isoformat(),
                            },
                        ][-20:]
                    metadata["revisions"] = revisions
                    memory_id = row["id"]
                    conn.execute(
                        "UPDATE memories SET content=?, memory_type=?, source=?, confidence=?, "
                        "updated_at=?, metadata_json=? WHERE id=?;",
                        (
                            item.content,
                            item.memory_type.value,
                            item.source,
                            item.confidence,
                            observed,
                            json.dumps(metadata),
                            memory_id,
                        ),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO memories (content,memory_type,source,confidence,"
                        "created_at,updated_at,metadata_json) VALUES (?,?,?,?,?,?,?);",
                        (
                            item.content,
                            item.memory_type.value,
                            item.source,
                            item.confidence,
                            observed,
                            observed,
                            json.dumps(metadata),
                        ),
                    )
                    memory_id = cursor.lastrowid
                    conn.execute("INSERT INTO memory_keys VALUES (?, ?);", (key, memory_id))
                forgotten = conn.execute(
                    "SELECT patterns_json FROM forgotten_memories WHERE fact_key=?;", (key,),
                ).fetchone()
                if forgotten:
                    # Re-authorize only this value, keeping unrelated old values private.
                    authorized = {item.content.casefold()}
                    if isinstance(item.metadata.get("value"), str):
                        authorized.add(item.metadata["value"].strip().casefold())
                    patterns = [
                        pattern for pattern in json.loads(forgotten["patterns_json"])
                        if pattern.casefold() not in authorized
                    ]
                    if patterns:
                        conn.execute(
                            "UPDATE forgotten_memories SET patterns_json=? WHERE fact_key=?;",
                            (json.dumps(patterns), key),
                        )
                    else:
                        conn.execute("DELETE FROM forgotten_memories WHERE fact_key=?;", (key,))
                item.id = memory_id
                return memory_id

    def forget_memories(self, items: list[MemoryItem]) -> int:
        """Remove selected durable facts and suppress their values in replayed history."""
        with self._lock:
            conn = self._get_connection()
            deleted = 0
            with conn:
                for item in items:
                    key_row = conn.execute(
                        "SELECT fact_key FROM memory_keys WHERE memory_id=?;",
                        (item.id,),
                    ).fetchone()
                    key = key_row["fact_key"] if key_row else f"record:{item.id}"
                    patterns = [item.content]
                    forgotten = conn.execute(
                        "SELECT patterns_json FROM forgotten_memories WHERE fact_key=?;", (key,),
                    ).fetchone()
                    if forgotten:
                        patterns.extend(json.loads(forgotten["patterns_json"]))
                    value = item.metadata.get("value")
                    if value is None:
                        description = re.match(
                            r"^(?:i (?:have|own|use|prefer)|my \w+ is)\s+"
                            r"(?:(?:a|an|the)\s+)?(.+)$",
                            item.content,
                            re.I,
                        )
                        if description:
                            value = description.group(1)
                    if isinstance(value, str) and len(value.strip()) >= 3:
                        patterns.append(value.strip())
                    # Include replaced values so older summaries cannot restore a prior value.
                    patterns.extend(
                        revision["content"] for revision in item.metadata.get("revisions", [])
                    )
                    patterns.extend(
                        revision["value"]
                        for revision in item.metadata.get("revisions", [])
                        if isinstance(revision.get("value"), str) and len(revision["value"]) >= 3
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO forgotten_memories VALUES (?, ?);",
                        (key, json.dumps(list(dict.fromkeys(patterns)))),
                    )
                    deleted += conn.execute(
                        "DELETE FROM memories WHERE id=?;",
                        (item.id,),
                    ).rowcount
            return deleted

    def redact_forgotten(self, text: str) -> str:
        """Mask forgotten values before sending archived or recent text to a model."""
        with self._lock:
            rows = (
                self._get_connection()
                .execute("SELECT patterns_json FROM forgotten_memories;")
                .fetchall()
            )
        for row in rows:
            for pattern in json.loads(row["patterns_json"]):
                text = re.sub(
                    r"(?<!\w)" + re.escape(pattern) + r"(?!\w)",
                    "[forgotten fact]",
                    text,
                    flags=re.IGNORECASE,
                )
        return text

    def search_memories(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        *,
        exclude_types: tuple[MemoryType, ...] = (),
    ) -> list[MemoryItem]:
        """Search memory content using case-insensitive substring / keyword matching."""
        with self._lock:
            conn = self._get_connection()
            words = [w.strip() for w in query.split() if w.strip()]
            sql = "SELECT * FROM memories WHERE "
            conditions: list[str] = []
            params: list[Any] = []

            # Match any keyword in content
            if words:
                content_clauses = ["content LIKE ?" for _ in words]
                conditions.append(f"({' OR '.join(content_clauses)})")
                params.extend([f"%{w}%" for w in words])

            if memory_type is not None:
                type_val = (
                    memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
                )
                conditions.append("memory_type = ?")
                params.append(type_val)

            if exclude_types:
                placeholders = ", ".join("?" for _ in exclude_types)
                conditions.append(f"memory_type NOT IN ({placeholders})")
                params.extend(memory_type.value for memory_type in exclude_types)

            sql += " AND ".join(conditions) if conditions else "1 = 1"
            sql += " ORDER BY updated_at DESC LIMIT ?;"
            params.append(limit)

            cursor = conn.execute(sql, params)
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """List the most recent memory items."""
        with self._lock:
            conn = self._get_connection()
            if memory_type is not None:
                type_val = (
                    memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
                )
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ? "
                    "ORDER BY updated_at DESC LIMIT ?;",
                    (type_val, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?;",
                    (limit,),
                )
            return [self._row_to_memory(r) for r in cursor.fetchall()]

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory item by ID."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
                return cursor.rowcount > 0

    def delete_by_pattern(self, pattern: str) -> int:
        """Delete memories matching a search pattern (case-insensitive). Returns count deleted."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE content LIKE ?;",
                    (f"%{pattern}%",),
                )
                return cursor.rowcount

    def count_memories(self, memory_type: MemoryType | None = None) -> int:
        """Return total count of memories stored."""
        with self._lock:
            conn = self._get_connection()
            if memory_type is not None:
                type_val = (
                    memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
                )
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?;",
                    (type_val,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM memories;")
            return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # Short-Term Conversation History Operations
    # -------------------------------------------------------------------------

    def save_turn(self, turn: ConversationTurn) -> int:
        """Save a single conversation turn to history."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO conversation_turns (
                        session_id, role, content, timestamp,
                        provider, model, tokens, latency, sources_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        turn.session_id,
                        turn.role,
                        turn.content,
                        turn.timestamp.isoformat()
                        if turn.timestamp
                        else datetime.now(timezone.utc).isoformat(),
                        turn.provider,
                        turn.model,
                        turn.tokens,
                        turn.latency,
                        json.dumps(normalize_sources(turn.sources)),
                    ),
                )
                turn.id = cursor.lastrowid
                return turn.id

    def get_recent_turns(
        self,
        session_id: str = "default",
        limit: int = 10,
    ) -> list[ConversationTurn]:
        """Retrieve the N most recent conversation turns for a session in chronological order."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC;
                """,
                (session_id, limit),
            )
            return [self._row_to_turn(r) for r in cursor.fetchall()]

    def get_session_summary(self, session_id: str) -> MemoryItem | None:
        """Retrieve the latest summary using exact metadata identity."""
        with self._lock:
            row = (
                self._get_connection()
                .execute(
                    "SELECT * FROM memories WHERE memory_type = ? "
                    "AND json_extract(metadata_json, '$.session_id') = ? "
                    "ORDER BY updated_at DESC, id DESC LIMIT 1;",
                    (MemoryType.CONVERSATION.value, session_id),
                )
                .fetchone()
            )
            if not row:
                return None
            summary = self._row_to_memory(row)
            # Older releases stored a count rather than an ID. Infer the cursor
            # once from that count so an upgrade does not repeat old requests.
            if "last_turn_id" not in summary.metadata:
                count = summary.metadata.get("older_turns_count", 0)
                if isinstance(count, int) and count > 0:
                    turn = (
                        self._get_connection()
                        .execute(
                            "SELECT id FROM conversation_turns WHERE session_id = ? "
                            "ORDER BY id ASC LIMIT 1 OFFSET ?;",
                            (session_id, count - 1),
                        )
                        .fetchone()
                    )
                    if turn:
                        summary.metadata["last_turn_id"] = turn["id"]
            return summary

    def get_summary_batch(
        self, session_id: str, after_id: int, keep_recent: int, limit: int
    ) -> list[ConversationTurn]:
        """Read a bounded batch of new turns outside the active context window."""
        with self._lock:
            rows = (
                self._get_connection()
                .execute(
                    "SELECT * FROM conversation_turns WHERE session_id = ? AND id > ? "
                    "AND id NOT IN (SELECT id FROM conversation_turns WHERE session_id = ? "
                    "ORDER BY id DESC LIMIT ?) ORDER BY id ASC LIMIT ?;",
                    (session_id, after_id, session_id, keep_recent, limit),
                )
                .fetchall()
            )
            return [self._row_to_turn(row) for row in rows]

    def clear_session(self, session_id: str) -> int:
        """Atomically clear a session's turns and persisted summaries."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversation_turns WHERE session_id = ?;",
                    (session_id,),
                )
                cleared = cursor.rowcount
                conn.execute(
                    "DELETE FROM memories WHERE memory_type = ? "
                    "AND json_extract(metadata_json, '$.session_id') = ?;",
                    (MemoryType.CONVERSATION.value, session_id),
                )
                return cleared

    # -------------------------------------------------------------------------
    # Helper / Conversion Methods
    # -------------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        meta = {}
        try:
            meta = json.loads(row["metadata_json"])
        except Exception:
            pass

        created_dt = datetime.now(timezone.utc)
        try:
            created_dt = datetime.fromisoformat(row["created_at"])
        except Exception:
            pass

        updated_dt = datetime.now(timezone.utc)
        try:
            updated_dt = datetime.fromisoformat(row["updated_at"])
        except Exception:
            pass

        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"],
            confidence=float(row["confidence"]),
            created_at=created_dt,
            updated_at=updated_dt,
            metadata=meta,
        )

    def _row_to_turn(self, row: sqlite3.Row) -> ConversationTurn:
        sources: Any = []
        try:
            sources = json.loads(row["sources_json"])
        except (TypeError, ValueError):
            pass
        ts = datetime.now(timezone.utc)
        try:
            ts = datetime.fromisoformat(row["timestamp"])
        except Exception:
            pass

        return ConversationTurn(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            timestamp=ts,
            provider=row["provider"],
            model=row["model"],
            tokens=row["tokens"],
            latency=float(row["latency"]),
            sources=normalize_sources(sources),
        )

    def close(self) -> None:
        """Close thread-local database connection if active."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
            if self._is_in_memory:
                self._memory_connection = None
