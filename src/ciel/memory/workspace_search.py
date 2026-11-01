"""Rebuildable keyword and optional semantic recall over workspace Markdown.

Markdown is authoritative. This database owns only derived search tables; deleting
or resetting it never changes notes, provenance, or the conversation database.
Embedding clients are supplied explicitly and must enforce their own I/O timeout.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ciel.memory.workspace import MemoryWorkspace

_TOKEN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|\w+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_DATED_NOTE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:-.+)?\.md$")
_SCHEMA_VERSION = "1"
_MAX_DIMENSIONS = 8192
_CACHE_LIMIT = 50000
_SEARCH_INFERENCE_SECONDS = 0.1


class EmbeddingClient(Protocol):
    """Explicit bounded embedding adapter; set ``local_only`` for local inference."""

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Return one finite, nonzero, same-dimension vector per input."""


@dataclass(frozen=True)
class WorkspaceSearchResult:
    """A source-citable excerpt from a live, permitted memory document."""

    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str = "memory"
    origin: str = "owner"
    local_only: bool = False

    def to_dict(self) -> dict:
        """Return the stable tool-result representation."""
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceIndexReport:
    """Counts and an optional degraded indexing reason."""

    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    embedded: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        """Return plain values for CLI and diagnostics consumers."""
        return asdict(self)


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector(value: object, dimensions: int = 0) -> list[float]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= _MAX_DIMENSIONS:
        raise ValueError("Invalid embedding vector size")
    if dimensions and len(value) != dimensions:
        raise ValueError("Embedding dimensions changed; reset or force-rebuild the index")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Embedding vector must contain numbers")
    numbers = [float(x) for x in value]
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Embedding vector contains nonfinite values")
    norm = math.sqrt(sum(x * x for x in numbers))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding vector must have a finite direction")
    return [x / norm for x in numbers]


def _chunks(text: str, tokens: int, overlap: int) -> list[tuple[int, int, str]]:
    """Chunk Unicode text without losing original source-line references."""
    matches = list(_TOKEN.finditer(text))
    if not matches:
        return []
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    result = []
    offset = 0
    while offset < len(matches):
        stop = min(len(matches), offset + tokens)
        start_char = matches[offset].start()
        end_char = matches[stop - 1].end()
        # A pathological single unbroken token must not bypass the character bound.
        for char_start in range(start_char, end_char, 8000):
            char_end = min(end_char, char_start + 8000)
            result.append((
                bisect.bisect_right(line_starts, char_start),
                bisect.bisect_right(line_starts, char_end - 1),
                text[char_start:char_end],
            ))
        if stop == len(matches):
            break
        offset = stop - overlap
    return result


class WorkspaceMemorySearch:
    """Search editable memory files with no implicit models or network access."""

    def __init__(
        self, workspace: MemoryWorkspace, *, db_path: str | Path | None = None,
        embedding_client: EmbeddingClient | None = None, embedding_model: str = "",
        chunk_tokens: int = 400, chunk_overlap: int = 80, candidate_limit: int = 200,
        max_documents: int = 10000, max_chunks: int = 50000,
        max_document_chars: int = 2_000_000, sql_timeout: float = 5.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not 8 <= chunk_tokens <= 2000 or not 0 <= chunk_overlap < chunk_tokens:
            raise ValueError("Invalid memory chunk size or overlap")
        if not 6 <= candidate_limit <= 1000:
            raise ValueError("Candidate limit must be between 6 and 1000")
        if not 1 <= max_documents <= 100000 or not 1 <= max_chunks <= 200000:
            raise ValueError("Invalid memory indexing limits")
        if not 1000 <= max_document_chars <= 10_000_000:
            raise ValueError("Invalid memory document size limit")
        if not math.isfinite(sql_timeout) or not 0 < sql_timeout <= 60:
            raise ValueError("SQL timeout must be between 0 and 60 seconds")
        if not isinstance(embedding_model, str) or len(embedding_model) > 256:
            raise ValueError("Invalid embedding model identity")
        if embedding_client is not None and not embedding_model.strip():
            raise ValueError("An explicit embedding model is required with an embedding client")
        self.workspace = workspace
        self.db_path = (
            Path(db_path).expanduser().absolute()
            if db_path else workspace.root / ".ciel" / "search.sqlite3"
        )
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model.strip()
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.candidate_limit = candidate_limit
        self.max_documents = max_documents
        self.max_chunks = max_chunks
        self.max_document_chars = max_document_chars
        self.sql_timeout = sql_timeout
        self._now = now if callable(now) else lambda: datetime.now(timezone.utc)
        self._lock = threading.RLock()
        self._embedding_lock = threading.Lock()
        self._search_inference: threading.Thread | None = None
        self._semantic_cache: dict[tuple, dict[str, float]] = {}
        self._generation = 0
        self._closed = False
        self._watch_stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self._watch_interval = 0.0
        self._error: str | None = None
        self._embedding_retry_at = 0.0
        self._live: dict[str, object] = {}
        self._last_report = WorkspaceIndexReport()
        self._local_embeddings = bool(
            getattr(embedding_client, "local_only", False)
            or getattr(embedding_client, "is_local", False)
        )
        provider = getattr(embedding_client, "provider_id", "")
        if embedding_client is not None and not provider:
            provider = f"{type(embedding_client).__module__}.{type(embedding_client).__qualname__}"
        self._identity = json.dumps(
            [provider, self.embedding_model, "local" if self._local_embeddings else "remote"],
            separators=(",", ":"),
        )

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("Workspace memory search is closed")
        self._guard_database_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._guard_database_path()
        descriptor = os.open(self.db_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Memory search database must be a regular file")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        conn = sqlite3.connect(self.db_path, timeout=self.sql_timeout)
        conn.row_factory = sqlite3.Row
        deadline = time.monotonic() + self.sql_timeout
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workspace_search_meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS workspace_search_documents (
                    path TEXT PRIMARY KEY, digest TEXT NOT NULL, modified_ns INTEGER NOT NULL,
                    origin TEXT NOT NULL, local_only INTEGER NOT NULL, session_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workspace_search_chunks (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL,
                    path_key TEXT NOT NULL, basename_key TEXT NOT NULL, stem_key TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS workspace_search_chunks_path
                    ON workspace_search_chunks(path);
                CREATE INDEX IF NOT EXISTS workspace_search_chunks_hash
                    ON workspace_search_chunks(text_hash);
                CREATE VIRTUAL TABLE IF NOT EXISTS workspace_search_fts USING fts5(
                    chunk_id UNINDEXED, body, tokenize='unicode61 remove_diacritics 0'
                );
                CREATE TABLE IF NOT EXISTS workspace_search_embeddings (
                    identity TEXT NOT NULL, text_hash TEXT NOT NULL, dimensions INTEGER NOT NULL,
                    vector TEXT NOT NULL, PRIMARY KEY(identity,text_hash)
                );
            """)
            return conn
        except Exception:
            conn.close()
            raise

    def _guard_database_path(self) -> None:
        for component in [*reversed(self.db_path.parents), self.db_path]:
            if component.is_symlink():
                raise ValueError("Memory search database paths cannot contain symlinks")
        for suffix in ("-journal", "-wal", "-shm"):
            if Path(str(self.db_path) + suffix).is_symlink():
                raise ValueError("Memory search database sidecars cannot be symlinks")

    @staticmethod
    def _metadata(conn: sqlite3.Connection) -> dict[str, str]:
        return dict(conn.execute("SELECT key,value FROM workspace_search_meta"))

    def _documents(self, paths: tuple[str, ...] | None = None) -> dict[str, object]:
        documents = (self.workspace.documents() if paths is None else
                     self.workspace.documents(paths=paths))
        if len(documents) > self.max_documents:
            raise ValueError("Memory workspace exceeds the configured document limit")
        result = {}
        total_chars = 0
        for document in documents:
            path = str(document.path).replace("\\", "/")
            parts = Path(path).parts
            if Path(path).is_absolute() or ".." in parts or not (
                path in {"MEMORY.md", "USER.md"}
                or path.startswith("memory/") and path.endswith(".md")
            ):
                raise ValueError("Workspace returned a path outside Markdown memory roots")
            if len(document.text) > self.max_document_chars:
                raise ValueError(f"Memory document exceeds the size limit: {path}")
            total_chars += len(document.text)
            if total_chars > 32_000_000:
                raise ValueError("Memory workspace exceeds the total text size limit")
            result[path] = document
        return result

    def _embeddings(
        self, conn: sqlite3.Connection, chunks: list[tuple], documents: dict,
        metadata: dict[str, str], force: bool, current: Callable[[], bool] | None = None,
    ) -> tuple[list[tuple], int, str | None]:
        if self.embedding_client is None:
            return [], 0, None
        dimensions = 0 if force else int(metadata.get("dimensions", "0"))
        if metadata.get("identity") != self._identity:
            dimensions = 0
        existing = {} if force else {
            row["text_hash"]: row for row in conn.execute(
                "SELECT * FROM workspace_search_embeddings WHERE identity=?", (self._identity,),
            )
        }
        pending = {}
        for chunk in chunks:
            if documents[chunk[1]].local_only and not self._local_embeddings:
                continue
            content_hash = chunk[5]
            cached = existing.get(content_hash)
            if cached is not None and 0 < cached["dimensions"] <= _MAX_DIMENSIONS:
                if not dimensions or cached["dimensions"] == dimensions:
                    dimensions = cached["dimensions"]
                    continue
            pending[content_hash] = chunk[4]
        if time.monotonic() < self._embedding_retry_at:
            return [], dimensions, self._error
        if not pending:
            return [], dimensions, None
        rows = []
        items = list(pending.items())
        try:
            for offset in range(0, len(items), 16):
                if self._closed or (current is not None and not current()):
                    return [], dimensions, None
                batch = items[offset:offset + 16]
                values = self.embedding_client.embed(
                    [text for _, text in batch], model=self.embedding_model,
                )
                if not isinstance(values, (list, tuple)) or len(values) != len(batch):
                    raise ValueError("Embedding response count does not match its inputs")
                for (content_hash, _), value in zip(batch, values):
                    vector = _vector(value, dimensions)
                    dimensions = len(vector)
                    rows.append((self._identity, content_hash, dimensions, json.dumps(vector)))
            self._embedding_retry_at = 0.0
            return rows, dimensions, None
        except Exception as exc:
            # Publish fresh text even when inference fails; never publish a partial vector batch.
            self._embedding_retry_at = time.monotonic() + 30.0
            previous = int(metadata.get("dimensions", "0"))
            if metadata.get("identity") != self._identity:
                previous = 0
            return [], previous, f"Embedding indexing unavailable: {type(exc).__name__}: {exc}"

    def sync(
        self, force: bool = False, *, _embed: bool = True,
        _current: Callable[[], bool] | None = None,
        _paths: tuple[str, ...] | None = None,
    ) -> WorkspaceIndexReport:
        """Publish fresh text atomically, then infer without holding the search lock.

        Explicit synchronization may wait for its configured embedding adapter.
        Concurrent searches reconcile text independently and never queue behind it.
        """
        if _paths == ():
            return self._last_report
        with self._lock:
            documents = self._documents(_paths)
            if _paths is None:
                self._live = documents
            else:
                self._live = {path: doc for path, doc in self._live.items() if path not in _paths}
                self._live.update(documents)
            conn = self._connect()
            try:
                metadata = self._metadata(conn)
                condition = ("" if _paths is None else
                             f" WHERE path IN ({','.join('?' for _ in _paths)})")
                arguments = () if _paths is None else _paths
                previous = {
                    row["path"]: row for row in conn.execute(
                        "SELECT * FROM workspace_search_documents" + condition, arguments,
                    )
                }
                other_documents = conn.execute(
                    "SELECT count(*) FROM workspace_search_documents",
                ).fetchone()[0] - len(previous)
                if other_documents + len(documents) > self.max_documents:
                    raise ValueError("Memory workspace exceeds the configured document limit")
                rebuild = force or metadata.get("version") != _SCHEMA_VERSION or (
                    metadata.get("chunking") != f"{self.chunk_tokens}:{self.chunk_overlap}"
                )
                source_changed = {
                    path for path, doc in documents.items()
                    if path not in previous or any((
                        previous[path]["digest"] != doc.digest,
                        previous[path]["modified_ns"] != doc.modified_ns,
                        previous[path]["origin"] != doc.origin,
                        bool(previous[path]["local_only"]) != bool(doc.local_only),
                        previous[path]["session_id"] != doc.session_id,
                    ))
                }
                changed = set(documents) if rebuild else source_changed
                removed = set(previous) - set(documents)
                chunks = []
                for path in sorted(changed):
                    for ordinal, (start, end, text) in enumerate(_chunks(
                        documents[path].text, self.chunk_tokens, self.chunk_overlap,
                    )):
                        content_hash = _digest(text)
                        chunks.append((
                            _digest(f"{path}:{ordinal}:{start}:{end}:{content_hash}"),
                            path, start, end,
                            text, content_hash, _normalized(path), _normalized(Path(path).name),
                            _normalized(Path(path).stem),
                        ))
                previous_chunks = conn.execute(
                    "SELECT * FROM workspace_search_chunks" + condition +
                    " ORDER BY path,start_line,id", arguments,
                ).fetchall()
                unchanged_chunks = [tuple(row) for row in previous_chunks
                                    if row["path"] not in changed | removed]
                all_chunks = unchanged_chunks + chunks
                other_chunks = conn.execute(
                    "SELECT count(*) FROM workspace_search_chunks",
                ).fetchone()[0] - len(previous_chunks)
                total_chunks = other_chunks + len(all_chunks)
                if total_chunks > self.max_chunks:
                    raise ValueError("Memory workspace exceeds the configured chunk limit")
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    for path in sorted(changed | removed):
                        conn.execute(
                            "DELETE FROM workspace_search_fts WHERE chunk_id IN "
                            "(SELECT id FROM workspace_search_chunks WHERE path=?)", (path,),
                        )
                        conn.execute("DELETE FROM workspace_search_chunks WHERE path=?", (path,))
                        conn.execute("DELETE FROM workspace_search_documents WHERE path=?", (path,))
                    conn.executemany(
                        "INSERT INTO workspace_search_documents VALUES (?,?,?,?,?,?)",
                        [(path, documents[path].digest, documents[path].modified_ns,
                          documents[path].origin, int(documents[path].local_only),
                          documents[path].session_id) for path in sorted(changed)],
                    )
                    conn.executemany(
                        "INSERT INTO workspace_search_chunks VALUES (?,?,?,?,?,?,?,?,?)", chunks,
                    )
                    conn.executemany(
                        "INSERT INTO workspace_search_fts(chunk_id,body) VALUES (?,?)",
                        [(chunk[0], _normalized(chunk[4])) for chunk in chunks],
                    )
                    dimensions = metadata.get("dimensions", "0") if (
                        metadata.get("identity") == self._identity
                    ) else "0"
                    updates = [("identity", self._identity), ("dimensions", dimensions)]
                    # A partial rebuild cannot certify unchanged, unselected chunks.
                    # Leave their old format marker for the next full synchronization.
                    if _paths is None or not other_documents:
                        updates.extend([
                            ("version", _SCHEMA_VERSION),
                            ("chunking", f"{self.chunk_tokens}:{self.chunk_overlap}"),
                        ])
                    conn.executemany(
                        "INSERT OR REPLACE INTO workspace_search_meta VALUES (?,?)",
                        updates,
                    )
                if source_changed or removed or (rebuild and (
                    _paths is None or set(map(tuple, previous_chunks)) != set(all_chunks)
                )):
                    self._generation += 1
                    self._semantic_cache.clear()
                generation = self._generation
                if not self._embedding_retry_at:
                    self._error = None
                self._last_report = WorkspaceIndexReport(
                    len(changed), len(documents) - len(changed), len(removed), total_chunks,
                    error=self._error,
                )
                report = self._last_report
            except (sqlite3.Error, ValueError) as exc:
                self._error = f"Memory indexing failed: {type(exc).__name__}: {exc}"
                self._last_report = WorkspaceIndexReport(error=self._error)
                return self._last_report
            finally:
                conn.close()

        if not _embed or self.embedding_client is None or not self._embedding_lock.acquire(False):
            return report
        try:
            with self._lock:
                conn = self._connect()
            try:
                embeddings, dimensions, error = self._embeddings(
                    conn, all_chunks, documents, metadata, force,
                    lambda: generation == self._generation and (
                        _current is None or _current()
                    ),
                )
                with self._lock:
                    if self._closed or generation != self._generation or (
                        _current is not None and not _current()
                    ):
                        return report
                    live = self._documents(_paths)
                    if set(live) != set(documents) or any(
                        (live[path].digest, live[path].origin, live[path].local_only)
                        != (doc.digest, doc.origin, doc.local_only)
                        for path, doc in documents.items()
                    ):
                        return report
                    deadline = time.monotonic() + self.sql_timeout
                    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                    with conn:
                        if force and not error and _paths is None:
                            conn.execute(
                                "DELETE FROM workspace_search_embeddings WHERE identity=?",
                                (self._identity,),
                            )
                        conn.executemany(
                            "INSERT OR REPLACE INTO workspace_search_embeddings VALUES (?,?,?,?)",
                            embeddings,
                        )
                        conn.execute(
                            "INSERT OR REPLACE INTO workspace_search_meta VALUES ('dimensions',?)",
                            (str(dimensions),),
                        )
                        excess = conn.execute(
                            "SELECT count(*) FROM workspace_search_embeddings",
                        ).fetchone()[0] - _CACHE_LIMIT
                        if excess > 0:
                            conn.execute(
                                "DELETE FROM workspace_search_embeddings WHERE rowid IN "
                                "(SELECT rowid FROM workspace_search_embeddings ORDER BY CASE "
                                "WHEN identity != ? THEN 0 WHEN text_hash NOT IN "
                                "(SELECT text_hash FROM workspace_search_chunks) THEN 1 "
                                "ELSE 2 END,rowid LIMIT ?)", (self._identity, excess),
                            )
                    self._error = error
                    self._last_report = WorkspaceIndexReport(
                        report.indexed, report.unchanged, report.removed, report.chunks,
                        len(embeddings), error,
                    )
                    return self._last_report
            except (sqlite3.Error, ValueError) as exc:
                with self._lock:
                    self._error = (
                        f"Memory embedding publication failed: {type(exc).__name__}: {exc}"
                    )
                    self._last_report = WorkspaceIndexReport(
                        report.indexed, report.unchanged, report.removed, report.chunks,
                        error=self._error,
                    )
                    return self._last_report
            finally:
                conn.close()
        finally:
            self._embedding_lock.release()

    def _rows(
        self, conn: sqlite3.Connection, local_only: bool,
        allowed_paths: tuple[str, ...] | None = None,
        allowed_origins: tuple[str, ...] | None = None,
    ) -> list[sqlite3.Row]:
        conditions = ["(? OR d.local_only=0)"]
        arguments: list[object] = [int(local_only)]
        for column, values in (("d.path", allowed_paths), ("d.origin", allowed_origins)):
            if values is not None:
                conditions.append(f"{column} IN ({','.join('?' for _ in values)})")
                arguments.extend(values)
        arguments.append(self.max_chunks)
        return conn.execute(
            "SELECT c.*,d.digest,d.origin,d.local_only FROM workspace_search_chunks c "
            "JOIN workspace_search_documents d ON c.path=d.path "
            f"WHERE {' AND '.join(conditions)} ORDER BY c.path,c.start_line,c.id LIMIT ?",
            arguments,
        ).fetchall()

    @staticmethod
    def _permitted(
        row: sqlite3.Row, live: dict, local_only: bool,
        allowed_paths: tuple[str, ...] | None, allowed_origins: tuple[str, ...] | None,
    ) -> bool:
        document = live.get(row["path"])
        return document is not None and (
            row["digest"] == document.digest
            and (local_only or not document.local_only)
            and (allowed_paths is None or row["path"] in allowed_paths)
            and (allowed_origins is None or document.origin in allowed_origins)
        )

    def _lexical(self, conn: sqlite3.Connection, query: str, allowed: set[str]) -> dict[str, float]:
        terms = list(dict.fromkeys(_WORD.findall(_normalized(query))))
        if not terms:
            return {}
        if len(terms) > 128:
            raise ValueError("Memory search query has too many terms")
        conn.execute("CREATE TEMP TABLE eligible_chunks (id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO eligible_chunks VALUES (?)", [(key,) for key in allowed])
        selected = []
        for separator in (" AND ", " OR "):
            expression = separator.join('"' + term.replace('"', '""') + '"' for term in terms)
            # Scope before LIMIT; private rows must not crowd out eligible public hits.
            selected = conn.execute(
                "SELECT f.chunk_id,bm25(workspace_search_fts) AS rank "
                "FROM workspace_search_fts f JOIN workspace_search_chunks c ON c.id=f.chunk_id "
                "JOIN eligible_chunks e ON e.id=f.chunk_id "
                "WHERE workspace_search_fts MATCH ? "
                "ORDER BY rank,c.path,c.start_line LIMIT ?",
                (expression, self.candidate_limit),
            ).fetchall()
            if selected:
                break
        if not selected and re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", query):
            # unicode61 treats an unspaced CJK sentence as one token. A bounded
            # substring fallback permits natural short queries without a model.
            selected = conn.execute(
                "SELECT f.chunk_id,-1.0 AS rank FROM workspace_search_fts f "
                "JOIN workspace_search_chunks c ON c.id=f.chunk_id "
                "JOIN eligible_chunks e ON e.id=f.chunk_id "
                "WHERE instr(f.body,?)>0 "
                "ORDER BY c.path,c.start_line LIMIT ?",
                (_normalized(query), self.candidate_limit),
            ).fetchall()
        scores = {row["chunk_id"]: max(0.0, -row["rank"]) for row in selected
                  if row["chunk_id"] in allowed}
        maximum = max(scores.values(), default=0.0)
        return {key: value / maximum if maximum else 1.0 for key, value in scores.items()}

    def _semantic(
        self, conn: sqlite3.Connection, query: str, rows: list[sqlite3.Row], local_only: bool,
    ) -> dict[str, float]:
        if self.embedding_client is None or (local_only and not self._local_embeddings):
            return {}
        if time.monotonic() < self._embedding_retry_at:
            return {}
        metadata = self._metadata(conn)
        dimensions = int(metadata.get("dimensions", "0"))
        if dimensions <= 0 or metadata.get("identity") != self._identity:
            return {}
        cached = {row["text_hash"]: row for row in conn.execute(
            "SELECT * FROM workspace_search_embeddings WHERE identity=?", (self._identity,),
        )}
        eligible = [row for row in rows if row["text_hash"] in cached
                    and (not row["local_only"] or self._local_embeddings)]
        if not eligible:
            return {}
        try:
            values = self.embedding_client.embed([query], model=self.embedding_model)
            if not isinstance(values, (list, tuple)) or len(values) != 1:
                raise ValueError("Embedding query response count is invalid")
            query_vector = _vector(values[0], dimensions)
            scores = {}
            deadline = time.monotonic() + self.sql_timeout
            for row in eligible:
                if time.monotonic() >= deadline:
                    self._error = "Semantic scoring deadline reached; recall is partial"
                    break
                vector = _vector(json.loads(cached[row["text_hash"]]["vector"]), dimensions)
                scores[row["id"]] = max(0.0, sum(a * b for a, b in zip(query_vector, vector)))
            return scores
        except Exception as exc:
            self._error = f"Semantic memory unavailable: {type(exc).__name__}: {exc}"
            self._embedding_retry_at = time.monotonic() + 30.0
            return {}

    def _bounded_semantic(
        self, query: str, local_only: bool, allowed_paths: tuple[str, ...] | None,
        allowed_origins: tuple[str, ...] | None,
    ) -> dict[str, float]:
        """Give optional inference a small budget and allow only one pending worker.

        A slow adapter may finish in the background; a subsequent identical query
        can use its cached scores while the current query keeps fresh keyword access.
        """
        if self.embedding_client is None or (local_only and not self._local_embeddings):
            return {}
        with self._lock:
            key = (query, local_only, allowed_paths, allowed_origins, self._generation)
            if key in self._semantic_cache:
                return self._semantic_cache[key]
            if self._embedding_lock.locked() or (
                self._search_inference is not None and self._search_inference.is_alive()
            ):
                return {}
            finished = threading.Event()
            output: dict[str, float] = {}

            def infer() -> None:
                try:
                    self.sync(_paths=allowed_paths)
                    if not self._embedding_lock.acquire(False):
                        return
                    try:
                        with self._lock:
                            if self._closed or key[-1] != self._generation:
                                return
                            conn = self._connect()
                        try:
                            with self._lock:
                                rows = [row for row in self._rows(
                                    conn, local_only, allowed_paths, allowed_origins,
                                ) if self._permitted(
                                    row, self._live, local_only, allowed_paths, allowed_origins,
                                )]
                            scores = self._semantic(conn, query, rows, local_only)
                        finally:
                            conn.close()
                        with self._lock:
                            if self._closed or key[-1] != self._generation:
                                return
                            output.update(scores)
                            if scores:
                                if len(self._semantic_cache) >= 32:
                                    self._semantic_cache.pop(next(iter(self._semantic_cache)))
                                self._semantic_cache[key] = scores
                    finally:
                        self._embedding_lock.release()
                except Exception as exc:
                    with self._lock:
                        if not self._closed:
                            self._error = (
                                f"Semantic memory unavailable: {type(exc).__name__}: {exc}"
                            )
                finally:
                    finished.set()

            self._search_inference = threading.Thread(
                target=infer, name="ciel-workspace-memory-inference", daemon=True,
            )
            self._search_inference.start()
        if finished.wait(min(_SEARCH_INFERENCE_SECONDS, self.sql_timeout)):
            return dict(output)
        return {}

    def _decay(self, path: str) -> float:
        filename = Path(path).name
        if not path.startswith("memory/") or not _DATED_NOTE.fullmatch(filename):
            return 1.0
        try:
            observed = date.fromisoformat(filename[:10])
        except ValueError:
            return 1.0
        age = max(0, (self._now().date() - observed).days)
        return 0.5 ** (age / 30.0)

    def search(
        self, query: str, *, max_results: int = 6, min_score: float = 0.35,
        local_only: bool = False, allowed_paths: tuple[str, ...] | None = None,
        allowed_origins: tuple[str, ...] | None = None,
    ) -> list[WorkspaceSearchResult]:
        """Retrieve fresh excerpts, optionally restricted to exact paths and origins.

        ``None`` allows every path/origin; an empty tuple allows none. Restrictions
        apply before candidate selection, scoring, diversification, and result limits.
        Optional inference gets a 100 ms budget; keyword recall remains available
        while one pending inference worker finishes in the background.
        """
        if not isinstance(query, str) or len(query) > 4096:
            raise ValueError("Memory search query must be at most 4096 characters")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not (
            1 <= max_results <= 100
        ):
            raise ValueError("Memory search result limit must be between 1 and 100")
        if not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not (
            0 <= min_score <= 1
        ):
            raise ValueError("Memory search minimum score must be between 0 and 1")
        for values in (allowed_paths, allowed_origins):
            if values is not None and (
                not isinstance(values, tuple) or len(values) > 1000
                or any(not isinstance(value, str) or not value or len(value) > 4096
                       for value in values)
            ):
                raise ValueError("Memory search scopes must be bounded tuples of nonempty strings")
        query = query.strip()
        if not query or not _WORD.search(query):
            return []
        if len(set(_WORD.findall(_normalized(query)))) > 128:
            raise ValueError("Memory search query has too many terms")
        if allowed_paths == () or allowed_origins == ():
            return []
        self.sync(_embed=False, _paths=allowed_paths)
        semantic = self._bounded_semantic(query, local_only, allowed_paths, allowed_origins)
        with self._lock:
            conn = self._connect()
            try:
                live = self._documents(allowed_paths)
                rows = [row for row in self._rows(conn, local_only, allowed_paths, allowed_origins)
                        if self._permitted(
                            row, live, local_only, allowed_paths, allowed_origins,
                        )]
                by_id = {row["id"]: row for row in rows}
                lexical = self._lexical(conn, query, set(by_id))
                path_query = _normalized(query).replace("\\", "/")
                candidates = {}
                for row in rows:
                    exact = 3 if row["path_key"] == path_query else (
                        2 if row["basename_key"] == path_query else (
                            1 if row["stem_key"] == path_query else 0
                        )
                    )
                    path_match = exact or path_query in row["path_key"]
                    text_score = lexical.get(row["id"], 0.0)
                    vector_score = semantic.get(row["id"], 0.0)
                    if not text_score and not vector_score and not path_match:
                        continue
                    relevance = 0.7 * vector_score + 0.3 * text_score if semantic else text_score
                    if path_match:
                        relevance = max(relevance, 1.0 if exact else 0.4)
                    # Lexical matches remain eligible even below the hybrid minimum.
                    if relevance < min_score and not text_score:
                        continue
                    score = relevance * self._decay(row["path"])
                    candidates[row["id"]] = (exact, score, row)
                ranked = sorted(
                    candidates.values(),
                    key=lambda item: (-item[0], -item[1], item[2]["path"], item[2]["start_line"]),
                )
                # Never let content withdrawn during inference consume a result slot.
                live = self._documents(allowed_paths)
                ranked = [item for item in ranked if self._permitted(
                    item[2], live, local_only, allowed_paths, allowed_origins,
                )][:self.candidate_limit * 2]
                if semantic and lexical:
                    ranked = self._diverse(ranked, max_results)
                return [WorkspaceSearchResult(
                    row["path"], row["start_line"], row["end_line"], score,
                    row["text"][:700], origin=live[row["path"]].origin,
                    local_only=bool(live[row["path"]].local_only),
                ) for _, score, row in ranked[:max_results]]
            finally:
                conn.close()

    @staticmethod
    def _diverse(ranked: list[tuple], limit: int) -> list[tuple]:
        selected = []
        tokens = {row["id"]: set(_WORD.findall(_normalized(row["text"])))
                  for _, _, row in ranked}
        while ranked and len(selected) < limit:
            best_tier = ranked[0][0]

            def utility(item: tuple) -> tuple:
                words = tokens[item[2]["id"]]
                redundancy = max((
                    len(words & tokens[old[2]["id"]])
                    / max(1, len(words | tokens[old[2]["id"]])) for old in selected
                ), default=0.0)
                return item[0], 0.7 * item[1] - 0.3 * redundancy

            winner = max((item for item in ranked if item[0] == best_tier), key=utility)
            selected.append(winner)
            ranked.remove(winner)
        return selected

    def get(
        self, path: str, *, from_line: int = 1, lines: int = 200, max_chars: int = 8000,
        local_only: bool = False,
    ):
        """Delegate exact excerpt reads to the workspace's safe source reader."""
        if self._closed:
            raise RuntimeError("Workspace memory search is closed")
        return self.workspace.get(
            path, from_line=from_line, lines=lines, max_chars=max_chars, local_only=local_only,
        )

    def status(self) -> dict:
        """Inspect derived-index state without invoking an embedding provider."""
        with self._lock:
            conn = self._connect()
            try:
                metadata = self._metadata(conn)
                return {
                    "database": str(self.db_path),
                    "documents": conn.execute(
                        "SELECT count(*) FROM workspace_search_documents",
                    ).fetchone()[0],
                    "chunks": conn.execute(
                        "SELECT count(*) FROM workspace_search_chunks",
                    ).fetchone()[0],
                    "mode": "hybrid" if self.embedding_client is not None else "keyword",
                    "model": self.embedding_model,
                    "dimensions": int(metadata.get("dimensions", "0")),
                    "watching": bool(self._watcher and self._watcher.is_alive()),
                    "watch_interval_seconds": self._watch_interval,
                    "error": self._error,
                }
            finally:
                conn.close()

    def reset_index(self) -> None:
        """Clear only derived workspace-search data; leave every source intact."""
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    for table in (
                        "workspace_search_fts", "workspace_search_chunks",
                        "workspace_search_documents", "workspace_search_embeddings",
                        "workspace_search_meta",
                    ):
                        conn.execute(f"DELETE FROM {table}")
                self._live = {}
                self._generation += 1
                self._semantic_cache.clear()
                self._error = None
                self._embedding_retry_at = 0.0
            finally:
                conn.close()

    def start_watching(
        self, interval_seconds: float = 1.5, *, current: Callable[[], bool] | None = None,
    ) -> bool:
        """Start one optional poller; ``current`` can stop a retired runtime's watcher.

        Polling never starts implicitly. The explicit embedding client's timeout
        also bounds inference performed by the watcher. Search always reconciles
        sources independently, including when watching is stopped or degraded.
        """
        if not isinstance(interval_seconds, (int, float)) or not math.isfinite(
            interval_seconds,
        ) or not 0.05 <= interval_seconds <= 300:
            raise ValueError("Memory watch interval must be between 0.05 and 300 seconds")
        if current is not None and not callable(current):
            raise ValueError("Memory watcher current must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("Workspace memory search is closed")
            if self._watcher is not None and self._watcher.is_alive():
                return False
            self._watch_stop.clear()
            self._watch_interval = float(interval_seconds)
            self._watcher = threading.Thread(
                target=self._watch, args=(float(interval_seconds), current),
                name="ciel-workspace-memory-watch", daemon=True,
            )
            self._watcher.start()
            return True

    def _watch(self, interval: float, current: Callable[[], bool] | None) -> None:
        delay = interval
        while not self._watch_stop.wait(delay):
            try:
                if current is not None and not current():
                    return
                if self._watch_stop.is_set():
                    return
                report = self.sync(_current=lambda: not self._watch_stop.is_set() and (
                    current is None or current()
                ))
                delay = min(30.0, max(interval, delay * 2)) if report.error else interval
            except Exception as exc:
                with self._lock:
                    if self._closed:
                        return
                    self._error = f"Memory watcher unavailable: {type(exc).__name__}: {exc}"
                delay = min(30.0, max(interval, delay * 2))

    def stop_watching(self) -> None:
        """Wake and join the optional daemon without holding the indexing lock."""
        self._watch_stop.set()
        watcher = self._watcher
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=max(1.0, self.sql_timeout))

    def close(self) -> None:
        """Stop the optional watcher and release this index's runtime ownership."""
        with self._lock:
            self._closed = True
            self._generation += 1
            self._semantic_cache.clear()
        self.stop_watching()
