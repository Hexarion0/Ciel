"""Optional CPU-only Ollama recall, indexed explicitly and disabled without a model.

Only a loopback /api/embed endpoint is supported. Model installation and selection are
separate user operations; this module never downloads a model or enables semantic recall.
"""

import hashlib
import ipaddress
import json
import math
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import numpy as np

from ciel.logging import get_logger
from ciel.memory.models import MemoryItem
from ciel.memory.store import MemoryStore

logger = get_logger("memory.semantic")

_MAX_DIMENSIONS = 8192
_MAX_QUERY_CHARS = 2000
_QUERY_CACHE_SIZE = 64
_FAILURE_BACKOFF_SECONDS = 30.0
_CANONICAL_SQL = (
    "(d.fact_key IS NULL OR EXISTS (SELECT 1 FROM memory_keys k "
    "WHERE k.fact_key=d.fact_key AND k.memory_id=d.memory_id))"
)


@dataclass(frozen=True)
class SemanticIndexReport:
    """Counts from an explicit indexing attempt, including recoverable provider failure."""

    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    error: str | None = None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _model_identity(model: str) -> str:
    """Accept Ollama's explicit :latest spelling while rejecting a different model."""
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def _validated_vector(value: object, dimensions: int | None = None) -> np.ndarray:
    """Reject nonnumeric, nonfinite, zero, inconsistent, or excessive embedding vectors."""
    if (
        not isinstance(value, list) or not 1 <= len(value) <= _MAX_DIMENSIONS
        or any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in value)
    ):
        raise ValueError("Invalid embedding vector")
    vector = np.asarray(value, dtype=np.float64)
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError("Embedding dimensions changed")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding contains nonfinite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has no finite direction")
    return vector / norm


class LocalSemanticRecall:
    """Use previously indexed local embeddings to supplement protected lexical anchors."""

    def __init__(
        self, store: MemoryStore, *, model: str = "",
        base_url: str = "http://127.0.0.1:11434", timeout: float = 2.0,
        min_similarity: float = 0.35,
    ) -> None:
        parsed = urlsplit(base_url)
        try:
            loopback = parsed.hostname == "localhost" or ipaddress.ip_address(
                parsed.hostname or "",
            ).is_loopback
            parsed.port
        except ValueError:
            loopback = False
        if (
            parsed.scheme != "http" or not loopback or parsed.username is not None
            or parsed.password is not None or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
        ):
            raise ValueError("Semantic recall requires an HTTP loopback Ollama endpoint")
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("Embedding timeout must be between 0 and 60 seconds")
        if not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
            raise ValueError("Minimum embedding similarity must be between 0 and 1")
        if not isinstance(model, str) or len(model) > 256 or any(char.isspace() for char in model):
            raise ValueError("Invalid local embedding model name")
        self.store = store
        self.model = _model_identity(model) if model else ""
        self.endpoint = base_url.rstrip("/") + "/api/embed"
        self.timeout = timeout
        self.min_similarity = min_similarity
        self._lock = threading.Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._unavailable_until = 0.0

    def _embed(self, texts: list[str], dimensions: int | None = None) -> list[np.ndarray]:
        """Request CPU embeddings without proxies, redirect following, or implicit truncation."""
        with httpx.Client(
            timeout=self.timeout, trust_env=False, follow_redirects=False,
        ) as client:
            response = client.post(self.endpoint, json={
                "model": self.model, "input": texts, "truncate": False,
                "options": {"num_gpu": 0}, "keep_alive": "5m",
            })
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise ValueError("Embedding response has no model identity")
        if _model_identity(payload["model"]) != self.model:
            raise ValueError("Embedding response used a different model")
        values = payload.get("embeddings")
        if not isinstance(values, list) or len(values) != len(texts):
            raise ValueError("Embedding response count does not match its inputs")
        vectors: list[np.ndarray] = []
        for value in values:
            vector = _validated_vector(value, dimensions)
            dimensions = len(vector)
            vectors.append(vector)
        return vectors

    def _indexed_rows(self) -> list[sqlite3.Row]:
        """Read only canonical durable records that still exist in the current store."""
        with self.store._lock:
            conn = self.store._get_connection()
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_vectors';"
            ).fetchone() is None:
                return []
            return conn.execute(
                "SELECT m.*,v.content_hash,v.dimensions,v.vector_json FROM memory_vectors v "
                "JOIN memories m ON m.id=v.memory_id "
                "JOIN memory_search_documents d ON d.memory_id=m.id "
                f"WHERE v.model=? AND {_CANONICAL_SQL} ORDER BY m.id;", (self.model,),
            ).fetchall()

    def reindex(self, batch_size: int = 16) -> SemanticIndexReport:
        """Explicitly update persisted embeddings, preserving valid unchanged vectors."""
        if not self.model:
            return SemanticIndexReport(error="No local embedding model is configured")
        if not 1 <= batch_size <= 64:
            raise ValueError("Embedding batch size must be between 1 and 64")
        with self._lock:
            return self._reindex(batch_size)

    def _reindex(self, batch_size: int) -> SemanticIndexReport:
        with self.store._lock:
            conn = self.store._get_connection()
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_vectors ("
                    "memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE, "
                    "model TEXT NOT NULL, content_hash TEXT NOT NULL, dimensions INTEGER NOT NULL, "
                    "vector_json TEXT NOT NULL, PRIMARY KEY(memory_id,model));"
                )
            records = conn.execute(
                "SELECT m.* FROM memories m JOIN memory_search_documents d ON d.memory_id=m.id "
                f"WHERE {_CANONICAL_SQL} ORDER BY m.id;"
            ).fetchall()
            existing = {
                row["memory_id"]: row for row in conn.execute(
                    "SELECT * FROM memory_vectors WHERE model=?;", (self.model,),
                )
            }
        current_hashes = {row["id"]: _content_hash(row["content"]) for row in records}
        stale_ids = [
            memory_id for memory_id, row in existing.items()
            if current_hashes.get(memory_id) != row["content_hash"]
        ]
        with self.store._lock:
            conn = self.store._get_connection()
            with conn:
                conn.executemany(
                    "DELETE FROM memory_vectors WHERE memory_id=? AND model=?;",
                    ((memory_id, self.model) for memory_id in stale_ids),
                )
        pending: list[sqlite3.Row] = []
        unchanged = 0
        dimensions: int | None = None
        for row in records:
            old = existing.get(row["id"])
            if old is not None and old["content_hash"] == current_hashes[row["id"]]:
                try:
                    vector = _validated_vector(json.loads(old["vector_json"]), old["dimensions"])
                    if dimensions is not None and len(vector) != dimensions:
                        raise ValueError("Stored embedding dimensions disagree")
                    dimensions = len(vector)
                    unchanged += 1
                    continue
                except (TypeError, ValueError):
                    pass
            pending.append(row)
        indexed = failed = 0
        error = None
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            try:
                vectors = self._embed([row["content"] for row in batch], dimensions)
                dimensions = len(vectors[0])
            except (httpx.HTTPError, TypeError, ValueError, OverflowError) as exc:
                failed = len(pending) - offset
                error = f"Local embedding request failed: {type(exc).__name__}"
                self._unavailable_until = time.monotonic() + _FAILURE_BACKOFF_SECONDS
                break
            with self.store._lock:
                conn = self.store._get_connection()
                with conn:
                    conn.execute("BEGIN IMMEDIATE;")
                    for row, vector in zip(batch, vectors):
                        # Recheck after inference: a forgotten or corrected fact cannot return.
                        live = conn.execute(
                            "SELECT m.content FROM memories m JOIN memory_search_documents d "
                            f"ON d.memory_id=m.id WHERE m.id=? AND {_CANONICAL_SQL};", (row["id"],),
                        ).fetchone()
                        if (
                            live is None
                            or _content_hash(live["content"]) != current_hashes[row["id"]]
                        ):
                            continue
                        conn.execute(
                            "INSERT OR REPLACE INTO memory_vectors VALUES (?,?,?,?,?);",
                            (row["id"], self.model, current_hashes[row["id"]], len(vector),
                             json.dumps(vector.tolist(), allow_nan=False)),
                        )
                        indexed += 1
        if error is None:
            self._unavailable_until = 0.0
        self._query_cache.clear()
        return SemanticIndexReport(indexed, unchanged, len(stale_ids), failed, error)

    def rank(self, query: str, limit: int = 4) -> list[tuple[float, MemoryItem]]:
        """Query fresh indexed vectors; unavailable or malformed local responses return no hits."""
        if (
            not self.model or not query.strip() or limit <= 0
            or self.store.get_setting("recall_enabled", True) is False
            or time.monotonic() < self._unavailable_until
        ):
            return []
        with self._lock:
            try:
                documents: list[tuple[sqlite3.Row, np.ndarray]] = []
                dimensions: int | None = None
                for row in self._indexed_rows():
                    if row["content_hash"] != _content_hash(row["content"]):
                        continue
                    try:
                        vector = _validated_vector(
                            json.loads(row["vector_json"]), row["dimensions"],
                        )
                        if dimensions is not None and len(vector) != dimensions:
                            continue
                    except (TypeError, ValueError):
                        continue
                    dimensions = len(vector)
                    documents.append((row, vector))
                if not documents:
                    return []
                text = query.strip()[:_MAX_QUERY_CHARS]
                cache_key = _content_hash(text)
                query_vector = self._query_cache.get(cache_key)
                if query_vector is None or len(query_vector) != dimensions:
                    query_vector = self._embed([text], dimensions)[0]
                    self._query_cache[cache_key] = query_vector
                    if len(self._query_cache) > _QUERY_CACHE_SIZE:
                        self._query_cache.popitem(last=False)
                self._query_cache.move_to_end(cache_key)
                # A deletion or correction may have happened while embedding the query.
                live_hashes = {
                    row["id"]: _content_hash(row["content"]) for row in self._indexed_rows()
                }
                if self.store.get_setting("recall_enabled", True) is False:
                    return []
                ranked = [
                    (float(np.dot(query_vector, vector)), self.store._row_to_memory(row))
                    for row, vector in documents
                    if live_hashes.get(row["id"]) == row["content_hash"]
                ]
                ranked.sort(key=lambda pair: (pair[0], pair[1].id or 0), reverse=True)
                return [pair for pair in ranked[:limit] if pair[0] >= self.min_similarity]
            except (httpx.HTTPError, sqlite3.Error, TypeError, ValueError, OverflowError) as exc:
                logger.debug("Local semantic recall unavailable: %s", type(exc).__name__)
                self._unavailable_until = time.monotonic() + _FAILURE_BACKOFF_SECONDS
                return []

    def hybrid_rank(
        self, query: str, lexical: list[tuple[float, MemoryItem]], *, limit: int = 4,
    ) -> list[tuple[float, MemoryItem]]:
        """Fuse independent rankings, retaining the first lexical result as an exact-word anchor."""
        if limit <= 0 or self.store.get_setting("recall_enabled", True) is False:
            return []
        semantic = self.rank(query, limit=max(12, limit * 3))
        if self.store.get_setting("recall_enabled", True) is False:
            return []
        # Lexical candidates were selected before a potentially slow embedding request.
        # Do not return their old text if a correction, forgetting, or purge happened meanwhile.
        fresh_lexical: list[tuple[float, MemoryItem]] = []
        with self.store._lock:
            conn = self.store._get_connection()
            for score, item in lexical:
                live = conn.execute(
                    "SELECT m.content FROM memories m JOIN memory_search_documents d "
                    f"ON d.memory_id=m.id WHERE m.id=? AND {_CANONICAL_SQL};", (item.id,),
                ).fetchone()
                if live is not None and live["content"] == item.content:
                    fresh_lexical.append((score, item))
        lexical = fresh_lexical
        if not semantic:
            return lexical[:limit]
        candidates: dict[int, tuple[float, MemoryItem]] = {}
        for ranking in (lexical, semantic):
            for position, (_, item) in enumerate(ranking, start=1):
                if item.id is None:
                    continue
                previous = candidates.get(item.id, (0.0, item))[0]
                candidates[item.id] = (previous + 1.0 / (60 + position), item)
        if lexical and lexical[0][1].id in candidates:
            anchor_id = lexical[0][1].id
            _, anchor = candidates[anchor_id]
            candidates[anchor_id] = (1.0 + max(score for score, _ in candidates.values()), anchor)
        return sorted(candidates.values(), key=lambda pair: pair[0], reverse=True)[:limit]
