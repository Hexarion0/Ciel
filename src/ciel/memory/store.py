"""SQLite persistence engine for short-term conversation context and long-term memories."""

import json
import math
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ciel.logging import get_logger
from ciel.memory.models import ConversationTurn, MemoryItem, MemoryType, normalize_sources
from ciel.memory.recall_terms import terms

logger = get_logger("memory.store")

_SESSION_ID_SQL = (
    "CASE WHEN json_valid(metadata_json) "
    "THEN json_extract(metadata_json, '$.session_id') END"
)
_CURRENT_MEMORY_SQL = (
    "NOT EXISTS (SELECT 1 FROM memory_search_documents d "
    "WHERE d.memory_id=memories.id AND d.fact_key IS NOT NULL "
    "AND NOT EXISTS (SELECT 1 FROM memory_keys k "
    "WHERE k.fact_key=d.fact_key AND k.memory_id=d.memory_id))"
)
# Bump whenever recall_terms normalization changes so existing postings are rebuilt once.
_RECALL_INDEX_VERSION = 1
_MAX_RECALL_QUERY_TERMS = 64
_UNGUARDED_FACT_CONTENT = object()


class MemoryConflictError(RuntimeError):
    """A guarded fact update no longer matches the value approved by its caller."""


class MemoryStore:
    """Thread-safe SQLite storage for long-term facts, preferences, and session history."""

    def __init__(self, db_path: str | Path = "data/ciel.db") -> None:
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
            conn.create_function("memory_log", 1, math.log, deterministic=True)
            conn.create_function("memory_casefold", 1, str.casefold, deterministic=True)
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
                conn.execute("BEGIN IMMEDIATE;")
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conv_session "
                    "ON conversation_turns(session_id, timestamp);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conv_session_id "
                    "ON conversation_turns(session_id, id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_updated "
                    "ON memories(updated_at DESC, id DESC);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_type_updated "
                    "ON memories(memory_type, updated_at DESC, id DESC);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_session_summary "
                    f"ON memories(memory_type, ({_SESSION_ID_SQL}), updated_at DESC, id DESC);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_session_state ("
                    "session_id TEXT PRIMARY KEY, generation INTEGER NOT NULL DEFAULT 0);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_session_members ("
                    "summary_session_id TEXT NOT NULL, member_session_id TEXT NOT NULL, "
                    "PRIMARY KEY(summary_session_id, member_session_id));"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_members_archive "
                    "ON memory_session_members(member_session_id, summary_session_id);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_settings ("
                    "name TEXT PRIMARY KEY, value_json TEXT NOT NULL);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_index_state ("
                    "name TEXT PRIMARY KEY, version INTEGER NOT NULL);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_search_documents ("
                    "memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE, "
                    "fact_key TEXT);"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_terms ("
                    "memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE, "
                    "term TEXT NOT NULL, PRIMARY KEY(memory_id, term));"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_terms_lookup "
                    "ON memory_terms(term, memory_id);"
                )
                version = conn.execute(
                    "SELECT version FROM memory_index_state WHERE name='recall_terms';"
                ).fetchone()
                if version is None or version["version"] != _RECALL_INDEX_VERSION:
                    conn.execute("DELETE FROM memory_terms;")
                    conn.execute("DELETE FROM memory_search_documents;")
                    # One upgrade pass preserves records and repairs missing fact-key mappings.
                    for row in conn.execute(
                        "SELECT * FROM memories ORDER BY updated_at DESC, id DESC;"
                    ).fetchall():
                        self._sync_recall_index(conn, self._row_to_memory(row))
                    conn.execute(
                        "INSERT OR REPLACE INTO memory_index_state VALUES ('recall_terms', ?);",
                        (_RECALL_INDEX_VERSION,),
                    )
            logger.debug("Memory database initialized at %s", self.db_path_str)

    # -------------------------------------------------------------------------
    # Long-Term Memory Operations
    # -------------------------------------------------------------------------

    @staticmethod
    def _decode_mapping(value: Any) -> dict[str, Any]:
        """Treat damaged or non-object legacy JSON as absent metadata."""
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _decode_patterns(value: Any) -> list[str]:
        """Ignore malformed tombstones rather than breaking all model context reads."""
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        if not isinstance(decoded, list):
            return []
        return [pattern for pattern in decoded if isinstance(pattern, str) and pattern]

    @staticmethod
    def _structured_key(metadata: dict[str, Any]) -> str | None:
        """Resolve valid legacy and current namespaced keys without stringifying garbage."""
        key = metadata.get("fact_key")
        scope = metadata.get("project") or metadata.get("scope", "user")
        if isinstance(key, str) and key.strip() and isinstance(scope, str) and scope.strip():
            return f"{scope}:{key}"
        return None

    def _sync_recall_index(self, conn: sqlite3.Connection, item: MemoryItem) -> None:
        """Maintain normalized word postings in the same transaction as the memory write."""
        conn.execute("DELETE FROM memory_terms WHERE memory_id=?;", (item.id,))
        conn.execute("DELETE FROM memory_search_documents WHERE memory_id=?;", (item.id,))
        key = self._structured_key(item.metadata)
        if item.memory_type == MemoryType.CONVERSATION:
            conn.execute("DELETE FROM memory_keys WHERE memory_id=?;", (item.id,))
            return
        conn.execute(
            "DELETE FROM memory_keys WHERE memory_id=? AND fact_key IS NOT ?;", (item.id, key),
        )
        if key:
            # An index rebuild must never reinstate an older duplicate of a forgotten fact.
            forgotten = conn.execute(
                "SELECT patterns_json FROM forgotten_memories WHERE fact_key=?;", (key,),
            ).fetchone()
            if forgotten:
                patterns = {
                    pattern.strip().casefold()
                    for pattern in self._decode_patterns(forgotten["patterns_json"])
                }
                identifiers = {item.content.strip().casefold()}
                value = item.metadata.get("value")
                if isinstance(value, str) and value.strip():
                    identifiers.add(value.strip().casefold())
                if identifiers & patterns:
                    # Older initializers could already have restored this mapping.
                    conn.execute(
                        "DELETE FROM memory_keys WHERE fact_key=? AND memory_id=?;",
                        (key, item.id),
                    )
            else:
                conn.execute("INSERT OR IGNORE INTO memory_keys VALUES (?, ?);", (key, item.id))
        conn.execute("INSERT INTO memory_search_documents VALUES (?, ?);", (item.id, key))
        fact_key = item.metadata.get("fact_key", "")
        indexed_terms = terms(item.content + " " + str(fact_key).replace("_", " "))
        conn.executemany(
            "INSERT INTO memory_terms(memory_id, term) VALUES (?, ?);",
            ((item.id, term) for term in indexed_terms),
        )

    def _save_memory_on_connection(self, conn: sqlite3.Connection, item: MemoryItem) -> int:
        """Write a record and its recall postings while the caller owns the transaction."""
        now_iso = datetime.now(timezone.utc).isoformat()
        type_val = (
            item.memory_type.value if isinstance(item.memory_type, MemoryType)
            else str(item.memory_type)
        )
        meta_str = json.dumps(item.metadata)
        if item.id is not None:
            cursor = conn.execute(
                "UPDATE memories SET content=?, memory_type=?, source=?, confidence=?, "
                "updated_at=?, metadata_json=? WHERE id=?;",
                (item.content, type_val, item.source, item.confidence, now_iso, meta_str, item.id),
            )
            if cursor.rowcount:
                self._sync_recall_index(conn, item)
                return item.id
        cursor = conn.execute(
            "INSERT INTO memories(content,memory_type,source,confidence,created_at,updated_at,"
            "metadata_json) VALUES (?,?,?,?,?,?,?);",
            (item.content, type_val, item.source, item.confidence,
             item.created_at.isoformat() if item.created_at else now_iso, now_iso, meta_str),
        )
        item.id = cursor.lastrowid
        self._sync_recall_index(conn, item)
        return item.id

    def save_memory(self, item: MemoryItem) -> int:
        """Insert or update a memory item. Returns the assigned memory ID."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                return self._save_memory_on_connection(conn, item)

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

    def upsert_fact(
        self, key: str, item: MemoryItem, *,
        expected_content: str | None | object = _UNGUARDED_FACT_CONTENT,
    ) -> int:
        """Replace a keyed fact, optionally requiring an unchanged value or absent record.

        Omit ``expected_content`` for an unguarded update. Pass a string to require
        that exact current content, or ``None`` to require that no fact exists.
        """
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
                if expected_content is not _UNGUARDED_FACT_CONTENT:
                    current_content = row["content"] if row else None
                    if current_content != expected_content:
                        raise MemoryConflictError(
                            "The saved fact changed before this update could be applied."
                        )
                observed = datetime.now(timezone.utc).isoformat()
                metadata = {
                    **item.metadata,
                    "scope": scope,
                    "fact_key": fact_key,
                    "observed_at": observed,
                }
                if scope == "ciel":
                    metadata["project"] = "ciel"
                if row:
                    old = self._row_to_memory(row)
                    revisions = old.metadata.get("revisions", [])
                    revisions = (
                        [revision for revision in revisions if isinstance(revision, dict)]
                        if isinstance(revisions, list) else []
                    )
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
                        pattern for pattern in self._decode_patterns(forgotten["patterns_json"])
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
                item.metadata = metadata
                self._sync_recall_index(conn, item)
                return memory_id

    def forget_memories(self, items: list[MemoryItem]) -> int:
        """Remove selected durable facts and suppress their values in replayed history."""
        with self._lock:
            conn = self._get_connection()
            deleted = 0
            with conn:
                conn.execute("BEGIN IMMEDIATE;")
                for item in items:
                    # Callers may have selected this record before a concurrent correction.
                    row = conn.execute("SELECT * FROM memories WHERE id=?;", (item.id,)).fetchone()
                    if row is None:
                        continue
                    item = self._row_to_memory(row)
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
                        patterns.extend(self._decode_patterns(forgotten["patterns_json"]))
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
                    if isinstance(value, str) and value.strip():
                        patterns.append(value.strip())
                    # Include replaced values so older summaries cannot restore a prior value.
                    revisions = item.metadata.get("revisions", [])
                    revisions = (
                        [revision for revision in revisions if isinstance(revision, dict)]
                        if isinstance(revisions, list) else []
                    )
                    patterns.extend(
                        revision["content"] for revision in revisions
                        if isinstance(revision.get("content"), str) and revision["content"]
                    )
                    patterns.extend(
                        revision["value"]
                        for revision in revisions
                        if isinstance(revision.get("value"), str) and revision["value"].strip()
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
            for pattern in self._decode_patterns(row["patterns_json"]):
                text = re.sub(
                    r"(?<!\w)" + re.escape(pattern) + r"(?!\w)",
                    "[forgotten fact]",
                    text,
                    flags=re.IGNORECASE,
                )
        return text

    def ranked_recall_candidates(
        self, query_terms: set[str], limit: int = 200,
    ) -> list[tuple[float, MemoryItem]]:
        """Rank matching indexed durable records across the entire memory collection."""
        selected_terms = sorted(term for term in query_terms if isinstance(term, str) and term)[
            :_MAX_RECALL_QUERY_TERMS
        ]
        if not selected_terms or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in selected_terms)
        with self._lock:
            conn = self._get_connection()
            document_count = conn.execute(
                "SELECT COUNT(*) FROM memory_search_documents d "
                "WHERE d.fact_key IS NULL OR EXISTS (SELECT 1 FROM memory_keys k "
                "WHERE k.fact_key=d.fact_key AND k.memory_id=d.memory_id);"
            ).fetchone()[0]
            rows = conn.execute(
                "WITH matched AS ("
                "SELECT t.memory_id,t.term FROM memory_terms t "
                "JOIN memory_search_documents d ON d.memory_id=t.memory_id "
                f"WHERE t.term IN ({placeholders}) AND (d.fact_key IS NULL OR EXISTS ("
                "SELECT 1 FROM memory_keys k WHERE k.fact_key=d.fact_key "
                "AND k.memory_id=d.memory_id))), "
                "frequencies AS (SELECT term,COUNT(*) AS documents FROM matched GROUP BY term) "
                "SELECT m.*,SUM(memory_log(1.0 + ? / f.documents)) "
                "* COUNT(*) / ? AS recall_score FROM matched t "
                "JOIN frequencies f ON f.term=t.term JOIN memories m ON m.id=t.memory_id "
                "GROUP BY m.id ORDER BY recall_score DESC,m.updated_at DESC,m.id DESC LIMIT ?;",
                (*selected_terms, float(document_count), float(len(selected_terms)), limit),
            ).fetchall()
            return [(float(row["recall_score"]), self._row_to_memory(row)) for row in rows]

    def get_setting(self, name: str, default: Any = None) -> Any:
        """Read one JSON application preference without saving a missing default."""
        with self._lock:
            row = self._get_connection().execute(
                "SELECT value_json FROM memory_settings WHERE name=?;", (name,),
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except (TypeError, ValueError):
            return default

    def set_setting(self, name: str, value: Any) -> None:
        """Persist a JSON preference atomically for all users of this memory database."""
        encoded = json.dumps(value, allow_nan=False)
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    "INSERT INTO memory_settings(name,value_json) VALUES (?,?) "
                    "ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json;",
                    (name, encoded),
                )

    def find_memories_containing(self, text: str, limit: int = 2) -> list[MemoryItem]:
        """Find literal durable matches across all records before applying a bounded limit.

        Targeted forgetting needs to detect ambiguity even when another matching fact
        is older than the newest recall or listing window.
        """
        if not text.strip() or limit <= 0:
            return []
        with self._lock:
            rows = self._get_connection().execute(
                "SELECT * FROM memories WHERE memory_type!=? "
                f"AND {_CURRENT_MEMORY_SQL} AND instr(memory_casefold(content),?)>0 "
                "ORDER BY updated_at DESC,id DESC LIMIT ?;",
                (MemoryType.CONVERSATION.value, text.casefold(), limit),
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

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
            conditions: list[str] = [_CURRENT_MEMORY_SQL]
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
            sql += " ORDER BY updated_at DESC, id DESC LIMIT ?;"
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
                    f"AND {_CURRENT_MEMORY_SQL} ORDER BY updated_at DESC, id DESC LIMIT ?;",
                    (type_val, limit),
                )
            else:
                cursor = conn.execute(
                    f"SELECT * FROM memories WHERE {_CURRENT_MEMORY_SQL} "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?;",
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

    def register_session_projection(self, summary_session_id: str, session_ids: list[str]) -> None:
        """Track summary dependencies so a channel reset invalidates every shared view."""
        if session_ids == [summary_session_id]:
            return
        with self._lock:
            connection = self._get_connection()
            with connection:
                connection.executemany(
                    "INSERT OR IGNORE INTO memory_session_members VALUES (?, ?)",
                    [(summary_session_id, session) for session in session_ids],
                )

    def get_context_turn_count(self, session_ids: list[str]) -> int:
        """Count turns in an explicitly selected owner context."""
        if not session_ids:
            return 0
        placeholders = ",".join("?" for _ in session_ids)
        with self._lock:
            return self._get_connection().execute(
                f"SELECT COUNT(*) FROM conversation_turns WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchone()[0]

    def get_context_turns(
        self, session_ids: list[str], limit: int = 10,
    ) -> list[ConversationTurn]:
        """Read a chronological bounded tail across the exact selected channel archives."""
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        with self._lock:
            rows = self._get_connection().execute(
                "SELECT * FROM (SELECT * FROM conversation_turns "
                f"WHERE session_id IN ({placeholders}) ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                [*session_ids, limit],
            ).fetchall()
            return [self._row_to_turn(row) for row in rows]

    def get_context_summary_batch(
        self, session_ids: list[str], after_id: int, keep_recent: int, limit: int,
    ) -> list[ConversationTurn]:
        """Read the next summary batch across a shared history, keeping its recent tail."""
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        with self._lock:
            rows = self._get_connection().execute(
                f"SELECT * FROM conversation_turns WHERE session_id IN ({placeholders}) "
                "AND id > ? AND id NOT IN (SELECT id FROM conversation_turns "
                f"WHERE session_id IN ({placeholders}) ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC LIMIT ?",
                [*session_ids, after_id, *session_ids, keep_recent, limit],
            ).fetchall()
            return [self._row_to_turn(row) for row in rows]

    def get_session_summary(self, session_id: str) -> MemoryItem | None:
        """Retrieve the latest summary using exact metadata identity."""
        with self._lock:
            return self._get_session_summary_on_connection(self._get_connection(), session_id)

    def _get_session_summary_on_connection(
        self, conn: sqlite3.Connection, session_id: str,
    ) -> MemoryItem | None:
        """Load summary identity and normalize the cursor while the caller holds the lock."""
        row = conn.execute(
            "SELECT * FROM memories WHERE memory_type = ? "
            f"AND ({_SESSION_ID_SQL}) = ? ORDER BY updated_at DESC, id DESC LIMIT 1;",
            (MemoryType.CONVERSATION.value, session_id),
        ).fetchone()
        if not row:
            return None
        summary = self._row_to_memory(row)
        # Older releases stored a count rather than an ID. Infer their starting cursor.
        cursor = summary.metadata.get("last_turn_id")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            cursor = 0
            count = summary.metadata.get("older_turns_count", 0)
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                turn = conn.execute(
                    "SELECT id FROM conversation_turns WHERE session_id = ? "
                    "ORDER BY id ASC LIMIT 1 OFFSET ?;", (session_id, count - 1),
                ).fetchone()
                if turn:
                    cursor = turn["id"]
            summary.metadata["last_turn_id"] = cursor
        count = summary.metadata.get("older_turns_count", 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            summary.metadata["older_turns_count"] = 0
        return summary

    def get_session_generation(self, session_id: str) -> int:
        """Return the shared invalidation generation for a persisted conversation session."""
        with self._lock:
            row = self._get_connection().execute(
                "SELECT generation FROM memory_session_state WHERE session_id=?;", (session_id,),
            ).fetchone()
            return int(row["generation"]) if row else 0

    def save_session_summary(
        self, item: MemoryItem, *, expected_generation: int, expected_last_turn_id: int,
    ) -> bool:
        """Commit only when no clear or competing summary changed the worker's snapshot."""
        session_id = item.metadata.get("session_id")
        last_turn_id = item.metadata.get("last_turn_id")
        if (
            item.memory_type != MemoryType.CONVERSATION
            or not isinstance(session_id, str)
            or not isinstance(last_turn_id, int)
            or isinstance(last_turn_id, bool)
            or last_turn_id <= expected_last_turn_id
        ):
            raise ValueError(
                "A session summary requires a session identity and an advancing cursor"
            )
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute(
                    "SELECT generation FROM memory_session_state WHERE session_id=?;",
                    (session_id,),
                ).fetchone()
                generation = int(row["generation"]) if row else 0
                if generation != expected_generation:
                    return False
                previous = self._get_session_summary_on_connection(conn, session_id)
                current_cursor = previous.metadata.get("last_turn_id", 0) if previous else 0
                if current_cursor != expected_last_turn_id:
                    return False
                if (previous.id if previous else None) != item.id:
                    return False
                self._save_memory_on_connection(conn, item)
                return True

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
        """Clear this archive and invalidate dependent summaries without deleting other channels."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("BEGIN IMMEDIATE;")
                affected = [session_id, *[row[0] for row in conn.execute(
                    "SELECT summary_session_id FROM memory_session_members "
                    "WHERE member_session_id=?", (session_id,),
                )]]
                conn.executemany(
                    "INSERT INTO memory_session_state(session_id,generation) VALUES (?,1) "
                    "ON CONFLICT(session_id) DO UPDATE SET generation=generation+1;",
                    [(identity,) for identity in affected],
                )
                cursor = conn.execute(
                    "DELETE FROM conversation_turns WHERE session_id = ?;",
                    (session_id,),
                )
                cleared = cursor.rowcount
                placeholders = ",".join("?" for _ in affected)
                conn.execute(
                    "DELETE FROM memories WHERE memory_type = ? "
                    f"AND ({_SESSION_ID_SQL}) IN ({placeholders});",
                    [MemoryType.CONVERSATION.value, *affected],
                )
                return cleared

    # -------------------------------------------------------------------------
    # Helper / Conversion Methods
    # -------------------------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryItem:
        meta = self._decode_mapping(row["metadata_json"])

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
