"""Explicit project journals for decisions, constraints, and unfinished work."""

import hashlib
import re
import unicodedata

from ciel.memory.models import MemoryItem, MemoryType
from ciel.memory.store import MemoryStore

_ENTRY = re.compile(
    r"project\s+([^:\n]{1,60}?)\s+(decision|constraint|next step|abandoned|status|goal)"
    r"\s*:\s*(.{2,600})", re.I,
)


def _scope(name: str) -> str:
    """Keep project identities stable across case, spacing, and Unicode equivalents."""
    normalized = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    return "project/" + hashlib.sha256(normalized.encode()).hexdigest()[:24]


def parse_project_entry(text: str) -> tuple[str, MemoryItem] | None:
    """Recognize a labeled user declaration without inferring project decisions from dialogue."""
    match = _ENTRY.fullmatch(text.strip().rstrip(".!"))
    if match is None or "?" in text:
        return None
    name, label, value = (part.strip() for part in match.groups())
    if not name or any(ord(char) < 32 for char in name + value):
        return None
    category = label.casefold().replace(" ", "_")
    fact_key = category
    if category in {"decision", "constraint", "abandoned"}:
        normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        fact_key += "_" + hashlib.sha256(normalized.encode()).hexdigest()[:20]
    item = MemoryItem(
        content=f"Project {name} {label.casefold()}: {value}.", memory_type=MemoryType.PROJECT,
        metadata={"project_name": name, "entry_kind": category, "value": value},
    )
    return f"{_scope(name)}:{fact_key}", item


def list_project_memories(store: MemoryStore, name: str, limit: int = 50) -> list[MemoryItem]:
    """Inspect only the current durable journal entries for one exact project identity."""
    if limit <= 0:
        return []
    with store._lock:
        rows = store._get_connection().execute(
            "SELECT m.* FROM memories m JOIN memory_keys k ON k.memory_id=m.id "
            "WHERE k.fact_key LIKE ? ORDER BY m.updated_at DESC,m.id DESC LIMIT ?;",
            (_scope(name) + ":%", min(limit, 500)),
        ).fetchall()
        return [store._row_to_memory(row) for row in rows]
