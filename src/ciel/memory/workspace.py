"""Owner-editable Markdown memory with private provenance and atomic local writes.

Markdown is the content source. The hidden registry only identifies managed facts,
records provenance, and makes append operations idempotent; it never grants authority
from frontmatter or other editable prose.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_EXCERPT_CHARS = 32000
MAX_EXCERPT_LINES = 1000
_REGISTRY = ".ciel-memory.json"
_DOCUMENT_CACHE_ENTRIES = 4096
_DOCUMENT_CACHE_BYTES = 16 * 1024 * 1024
_REGISTRY_CACHE_BYTES = 256 * 1024
_MARKER = re.compile(
    r"(?m)^<!-- ciel-memory:([a-f0-9]{32}) -->\n(.*?)\n<!-- /ciel-memory:\1 -->[ \t]*(?:\n|$)",
    re.S,
)


class MemoryWorkspaceError(ValueError):
    """A path, document, or provenance record cannot be used safely."""


class MemoryWorkspaceConflict(MemoryWorkspaceError):
    """The owner changed a curated document before a guarded write."""


@dataclass(frozen=True)
class MemoryDocument:
    """An exact Markdown document with independently recorded provenance."""

    path: str
    text: str
    modified_ns: int
    digest: str
    origin: str = "owner"
    local_only: bool = False
    session_id: str = ""


@dataclass(frozen=True)
class MemoryExcerpt:
    """A bounded, line-addressed excerpt of one workspace document."""

    path: str
    text: str
    from_line: int
    to_line: int
    truncated: bool
    next_line: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return values suitable for an application-owned memory-read tool."""
        return asdict(self)


class MemoryWorkspace:
    """Read and write one private Markdown workspace across threads and processes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute()
        self._mutex = threading.RLock()
        self._depth = 0
        self._lock_descriptor: int | None = None
        self._document_cache: OrderedDict[str, tuple[tuple[int, ...], MemoryDocument, int]] = (
            OrderedDict()
        )
        self._document_cache_bytes = 0
        self._registry_cache: tuple[tuple[int, ...], dict[str, Any]] | None = None
        self._ensure_root()

    def _ensure_root(self) -> None:
        for component in [*reversed(self.root.parents), self.root]:
            if component.is_symlink():
                raise MemoryWorkspaceError("Memory workspace paths cannot contain symlinks")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir():
            raise MemoryWorkspaceError("Memory workspace must be a directory")
        self.root.chmod(0o700)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize cooperating writers, including writers in other processes."""
        import fcntl

        with self._mutex:
            self._ensure_root()
            if self._depth == 0:
                lock_path = self._internal_path(".ciel-memory.lock")
                descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                try:
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise MemoryWorkspaceError("Memory workspace lock must be a regular file")
                    os.fchmod(descriptor, 0o600)
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except BaseException:
                    os.close(descriptor)
                    raise
                self._lock_descriptor = descriptor
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._lock_descriptor is not None:
                    fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
                    os.close(self._lock_descriptor)
                    self._lock_descriptor = None

    @staticmethod
    def _parts(path: str) -> tuple[str, ...]:
        if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
            raise MemoryWorkspaceError("A relative Markdown path is required")
        parts = tuple(path.split("/"))
        if PurePosixPath(path).is_absolute() or any(part in {"", ".", ".."} for part in parts):
            raise MemoryWorkspaceError("Memory paths cannot leave the workspace")
        return parts

    def _checked_path(self, parts: tuple[str, ...], *, create_parents: bool = False) -> Path:
        candidate = self.root
        for index, part in enumerate(parts):
            candidate = candidate / part
            if candidate.is_symlink():
                raise MemoryWorkspaceError("Memory paths cannot contain symlinks")
            if index < len(parts) - 1:
                if create_parents:
                    candidate.mkdir(exist_ok=True, mode=0o700)
                if candidate.exists() and not candidate.is_dir():
                    raise MemoryWorkspaceError("Memory path parent must be a directory")
            elif candidate.exists() and not candidate.is_file():
                raise MemoryWorkspaceError("Memory document must be a regular file")
        return candidate

    def _path(self, path: str, *, create_parents: bool = False) -> Path:
        parts = self._parts(path)
        if not (
            path in {"MEMORY.md", "USER.md"}
            or len(parts) >= 2 and parts[0] == "memory"
            and path.endswith(".md") and not any(part.startswith(".") for part in parts)
        ):
            raise MemoryWorkspaceError("Only MEMORY.md, USER.md, and memory/**/*.md are readable")
        return self._checked_path(parts, create_parents=create_parents)

    def _internal_path(self, path: str, *, create_parents: bool = False) -> Path:
        return self._checked_path(self._parts(path), create_parents=create_parents)

    @staticmethod
    def _file_identity(status: os.stat_result) -> tuple[int, ...]:
        """Include replacement, in-place edit, permission, and ownership changes."""
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns,
                status.st_ctime_ns, status.st_mode, status.st_uid, status.st_gid)

    @staticmethod
    def _read(path: Path) -> tuple[str, os.stat_result]:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_size > MAX_DOCUMENT_BYTES:
                raise MemoryWorkspaceError(
                    "Memory document is not regular or exceeds the size limit"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                raw = source.read(MAX_DOCUMENT_BYTES + 1)
            if len(raw) > MAX_DOCUMENT_BYTES:
                raise MemoryWorkspaceError("Memory document exceeds the size limit")
            if MemoryWorkspace._file_identity(status) != MemoryWorkspace._file_identity(
                os.fstat(descriptor),
            ) or MemoryWorkspace._file_identity(status) != MemoryWorkspace._file_identity(
                path.stat(follow_symlinks=False),
            ):
                raise MemoryWorkspaceError("Memory document changed while it was being read")
            try:
                return raw.decode("utf-8"), status
            except UnicodeDecodeError as error:
                raise MemoryWorkspaceError("Memory documents must be UTF-8") from error
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, text: str) -> None:
        raw = text.encode("utf-8")
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise MemoryWorkspaceError("Memory document exceeds the size limit")
        descriptor, temporary = tempfile.mkstemp(prefix=".memory-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as target:
                os.fchmod(target.fileno(), 0o600)
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
            if path.is_symlink():
                raise MemoryWorkspaceError("Memory paths cannot contain symlinks")
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _registry(self) -> dict[str, Any]:
        path = self._internal_path(_REGISTRY)
        if not path.exists():
            self._registry_cache = None
            return {"version": 1, "files": {}, "facts": {}, "entries": {}}
        identity = self._file_identity(path.stat(follow_symlinks=False))
        if self._registry_cache is not None and self._registry_cache[0] == identity:
            return deepcopy(self._registry_cache[1])
        self._registry_cache = None
        text, status = self._read(path)
        try:
            state = json.loads(text)
        except ValueError as error:
            raise MemoryWorkspaceError("Memory provenance registry is malformed") from error
        if not isinstance(state, dict) or state.get("version") != 1:
            raise MemoryWorkspaceError("Memory provenance registry has an unsupported format")
        for key in ("files", "facts", "entries"):
            if not isinstance(state.get(key), dict):
                raise MemoryWorkspaceError("Memory provenance registry is malformed")
        if status.st_size <= _REGISTRY_CACHE_BYTES:
            self._registry_cache = (self._file_identity(status), deepcopy(state))
        return state

    def _save_registry(self, state: dict[str, Any]) -> None:
        self._atomic_write(self._internal_path(_REGISTRY), json.dumps(state, ensure_ascii=False))
        self._registry_cache = None

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _provenance(state: dict[str, Any], path: str) -> dict[str, Any]:
        value = state["files"].get(path, {})
        return value if isinstance(value, dict) else {}

    def _document(self, path: str, state: dict[str, Any]) -> MemoryDocument:
        metadata = self._provenance(state, path)
        try:
            target = self._path(path)
            status = target.stat(follow_symlinks=False)
            identity = self._file_identity(status)
            cached = self._document_cache.get(path)
            if cached is not None and cached[0] == identity:
                text, digest = cached[1].text, cached[1].digest
            else:
                self._discard_document(path)
                text, status = self._read(target)
                identity = self._file_identity(status)
                if identity != self._file_identity(target.stat(follow_symlinks=False)):
                    raise MemoryWorkspaceError("Memory document changed while it was being read")
                digest = self._digest(text)
            document = MemoryDocument(
                path, text, status.st_mtime_ns, digest,
                str(metadata.get("origin", "owner")),
                path.startswith("memory/private/") or bool(metadata.get("local_only", False)),
                str(metadata.get("session_id", "")),
            )
            self._cache_document(identity, document)
            return document
        except (OSError, MemoryWorkspaceError):
            self._discard_document(path)
            raise

    def _discard_document(self, path: str) -> None:
        previous = self._document_cache.pop(path, None)
        if previous is not None:
            self._document_cache_bytes -= previous[2]

    def _cache_document(self, identity: tuple[int, ...], document: MemoryDocument) -> None:
        self._discard_document(document.path)
        size = sys.getsizeof(document.text) + sys.getsizeof(document.path) + 256
        if size > _DOCUMENT_CACHE_BYTES:
            return
        while self._document_cache and (
            len(self._document_cache) >= _DOCUMENT_CACHE_ENTRIES
            or self._document_cache_bytes + size > _DOCUMENT_CACHE_BYTES
        ):
            self._discard_document(next(iter(self._document_cache)))
        self._document_cache[document.path] = identity, document, size
        self._document_cache_bytes += size

    def _document_paths(self) -> list[str]:
        paths = [name for name in ("MEMORY.md", "USER.md") if self._path(name).exists()]
        memory = self.root / "memory"
        if memory.is_symlink():
            raise MemoryWorkspaceError("Memory paths cannot contain symlinks")
        if memory.exists():
            if not memory.is_dir():
                raise MemoryWorkspaceError("memory must be a directory")
            for directory, names, files in os.walk(memory, followlinks=False):
                for name in [*names, *files]:
                    if (Path(directory) / name).is_symlink():
                        raise MemoryWorkspaceError("Memory paths cannot contain symlinks")
                names[:] = sorted(name for name in names if not name.startswith("."))
                paths.extend(
                    (Path(directory) / name).relative_to(self.root).as_posix()
                    for name in sorted(files) if name.endswith(".md") and not name.startswith(".")
                )
        return paths

    def documents(self, *, paths: tuple[str, ...] | None = None) -> list[MemoryDocument]:
        """Read fresh Markdown, optionally restricted to exact paths without archive traversal."""
        with self.locked():
            state = self._registry()
            selected = self._document_paths() if paths is None else [
                path for path in dict.fromkeys(paths) if self._path(path).exists()
            ]
            live = set(selected)
            checked = set(self._document_cache) if paths is None else set(paths)
            for path in checked - live:
                self._discard_document(path)
            return [self._document(path, state) for path in selected]

    def provenance(self, path: str) -> dict[str, Any]:
        """Read application-owned source classification without interpreting note prose."""
        with self.locked():
            self._path(path)
            metadata = dict(self._provenance(self._registry(), path))
            origin = metadata.get("origin", "owner")
            origins = metadata.get("origins", [origin])
            if not isinstance(origins, list) or not all(isinstance(item, str) for item in origins):
                origins = ["untrusted"]
            metadata["origin"] = origin
            metadata["trusted"] = bool(origins) and set(origins) <= {"owner", "agent"}
            return metadata

    def has_provenance(self) -> bool:
        """Distinguish a new workspace import from deleted notes in an existing workspace."""
        with self.locked():
            return self._internal_path(_REGISTRY).exists()

    def signature(self) -> tuple[Any, ...]:
        """Identify managed fact sources and provenance without scanning the archive.

        Arbitrary notes are indexed separately by ``documents()``. Only paths
        recorded in the independent fact registry can change structured facts.
        """
        with self.locked():
            paths = [*self._fact_paths(self._registry()), _REGISTRY]
            result = []
            for relative in paths:
                path = self._internal_path(relative)
                if path.exists():
                    status = path.stat(follow_symlinks=False)
                    result.append((relative, *self._file_identity(status)))
            return tuple(result)

    def _fact_paths(self, state: dict[str, Any]) -> list[str]:
        paths = set()
        for record in state["facts"].values():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise MemoryWorkspaceError("Managed memory metadata is malformed")
            paths.add(record["path"])
        selected = sorted(paths)
        for path in selected:
            self._path(path)
        return selected

    def get(
        self, path: str, *, from_line: int = 1, lines: int = 200,
        max_chars: int = 8000, local_only: bool = False,
    ) -> MemoryExcerpt:
        """Read a bounded line range without allowing traversal, symlinks, or private leaks."""
        if from_line < 1 or lines < 1 or max_chars < 1:
            raise MemoryWorkspaceError("Line numbers and excerpt limits must be positive")
        with self.locked():
            document = self._document(path, self._registry())
            if document.local_only and not local_only:
                raise MemoryWorkspaceError("This memory document requires local-only context")
            all_lines = document.text.splitlines()
            selected = all_lines[from_line - 1:from_line - 1 + min(lines, MAX_EXCERPT_LINES)]
            budget = min(max_chars, MAX_EXCERPT_CHARS)
            output: list[str] = []
            size = 0
            clipped = False
            for line in selected:
                available = budget - size - bool(output)
                if available <= 0:
                    break
                output.append(line[:available])
                size += len(output[-1]) + (1 if len(output) > 1 else 0)
                if len(line) > available:
                    clipped = True
                    break
            to_line = from_line + len(output) - 1
            truncated = clipped or to_line < len(all_lines)
            return MemoryExcerpt(path, "\n".join(output), from_line, to_line, truncated,
                                 to_line + 1 if truncated and to_line < len(all_lines) else None)

    def _record_provenance(
        self, state: dict[str, Any], path: str, *, origin: str,
        session_id: str, local_only: bool,
    ) -> None:
        previous = self._provenance(state, path)
        previous_origin = previous.get("origin")
        origins = previous.get("origins", [previous_origin] if previous_origin else [])
        if not isinstance(origins, list) or not all(isinstance(item, str) for item in origins):
            origins = ["untrusted"]
        state["files"][path] = {
            "origin": origin if previous_origin in {None, origin} else "mixed",
            "origins": sorted(set([*origins, origin])),
            "local_only": local_only or bool(previous.get("local_only", False)),
            "session_id": (session_id
                           if previous.get("session_id", session_id) == session_id else ""),
        }

    def _append(
        self, path: str, text: str, *, entry_id: str, origin: str,
        session_id: str, local_only: bool, internal: bool = False,
        expected_digest: str | None = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise MemoryWorkspaceError("A nonempty memory entry is required")
        with self.locked():
            state = self._registry()
            target = (self._internal_path if internal else self._path)(path, create_parents=True)
            old = self._read(target)[0] if target.exists() else ""
            if expected_digest is not None and self._digest(old) != expected_digest:
                raise MemoryWorkspaceConflict("The curated memory changed before this write")
            identity = self._digest(path + "\0" + entry_id) if entry_id else ""
            if identity and identity in state["entries"]:
                return path
            new_text = old.rstrip() + ("\n\n" if old.strip() else "") + text.strip() + "\n"
            self._atomic_write(target, new_text)
            self._record_provenance(state, path, origin=origin, session_id=session_id,
                                    local_only=local_only)
            if identity:
                state["entries"][identity] = {"path": path, "origin": origin,
                                             "session_id": session_id, "local_only": local_only}
            self._save_registry(state)
            return path

    def append_daily(
        self, text: str, *, now: datetime | None = None, entry_id: str = "",
        origin: str = "agent", session_id: str = "", local_only: bool = False,
    ) -> str:
        """Append a dated observation once, without making it a confirmed structured fact."""
        date = (now or datetime.now().astimezone()).date().isoformat()
        prefix = "memory/private" if local_only else "memory"
        return self._append(f"{prefix}/{date}.md", text, entry_id=entry_id, origin=origin,
                            session_id=session_id, local_only=local_only)

    def append_curated(
        self, text: str, *, entry_id: str = "", origin: str = "owner", session_id: str = "",
        local_only: bool = False, expected_digest: str | None = None,
    ) -> str:
        """Append to the real curated layer, optionally guarding an owner-edit preimage."""
        path = "memory/private/MEMORY.md" if local_only else "MEMORY.md"
        return self._append(path, text, entry_id=entry_id, origin=origin, session_id=session_id,
                            local_only=local_only, expected_digest=expected_digest)

    def append_dream(self, text: str, *, entry_id: str = "") -> str:
        """Append a review-log entry excluded from automatic memory retrieval."""
        return self._append("DREAMS.md", text, entry_id=entry_id, origin="agent",
                            session_id="", local_only=False, internal=True)

    def save_preimage(self, text: str, *, entry_id: str = "") -> str:
        """Save an immutable private preimage excluded from memory candidate documents."""
        name = self._digest(entry_id or uuid4().hex)
        path = f".memory-state/preimages/{name}.md"
        with self.locked():
            target = self._internal_path(path, create_parents=True)
            if not target.exists():
                self._atomic_write(target, text)
            return path

    def replace_curated(
        self, text: str, *, expected_digest: str, origin: str = "agent", session_id: str = "",
        local_only: bool = False,
    ) -> str:
        """Replace reviewed prose while guarding the preimage and managed current facts."""
        path = "memory/private/MEMORY.md" if local_only else "MEMORY.md"
        with self.locked():
            state = self._registry()
            target = self._path(path, create_parents=True)
            old = self._read(target)[0] if target.exists() else ""
            if self._digest(old) != expected_digest:
                raise MemoryWorkspaceConflict("The curated memory changed before this write")
            old_blocks = {match.group(1): match.group(0).strip() for match in _MARKER.finditer(old)}
            new_blocks = {match.group(1): match.group(0).strip()
                          for match in _MARKER.finditer(text)}
            if origin != "owner" and any(new_blocks.get(key) != value
                                         for key, value in old_blocks.items()):
                raise MemoryWorkspaceConflict("Consolidation cannot replace managed current facts")
            self.save_preimage(old)
            self._atomic_write(target, text)
            self._record_provenance(state, path, origin=origin, session_id=session_id,
                                    local_only=local_only)
            self._save_registry(state)
            return path

    def bootstrap(
        self, *, max_chars: int = 8000, profile_max_chars: int = 4000,
        include_daily: bool = False, now: datetime | None = None, local_only: bool = False,
    ) -> list[MemoryExcerpt]:
        """Read bounded curated/profile context and optionally today's and yesterday's notes."""
        if max_chars <= 0:
            return []
        with self.locked():
            paths = ["USER.md", "MEMORY.md"]
            if local_only:
                paths.append("memory/private/MEMORY.md")
            if include_daily:
                today = (now or datetime.now().astimezone()).date()
                for day in (today, today - timedelta(days=1)):
                    paths.append(f"memory/{day.isoformat()}.md")
                    if local_only:
                        paths.append(f"memory/private/{day.isoformat()}.md")
            remaining = max_chars
            excerpts = []
            state = self._registry()
            for path in paths:
                if remaining <= 0 or not self._path(path).exists():
                    continue
                if self._document(path, state).local_only and not local_only:
                    continue
                budget = min(remaining, profile_max_chars) if path == "USER.md" else remaining
                if budget <= 0:
                    continue
                excerpt = self.get(path, lines=MAX_EXCERPT_LINES, max_chars=budget,
                                   local_only=local_only)
                if excerpt.text:
                    excerpts.append(excerpt)
                    remaining -= len(excerpt.text)
            return excerpts

    def put_fact(self, uid: str, record: dict[str, Any], text: str) -> dict[str, Any]:
        """Replace one managed block, keeping content authoritative in the curated file."""
        if not re.fullmatch(r"[a-f0-9]{32}", uid):
            raise MemoryWorkspaceError("Managed memory identity is invalid")
        with self.locked():
            state = self._registry()
            old_record = state["facts"].get(uid, {})
            private = bool(record.get("metadata", {}).get("local_only", False))
            path = "memory/private/MEMORY.md" if private else "MEMORY.md"
            previous_path = old_record.get("path")
            if previous_path and previous_path != path:
                previous_target = self._path(previous_path)
                if previous_target.exists():
                    previous_text = self._read(previous_target)[0]
                    previous_blocks = [match for match in _MARKER.finditer(previous_text)
                                       if match.group(1) == uid]
                    if len(previous_blocks) > 1:
                        raise MemoryWorkspaceError(
                            "A managed memory block has duplicate identities"
                        )
                    if previous_blocks:
                        match = previous_blocks[0]
                        self._atomic_write(previous_target, previous_text[:match.start()]
                                           + previous_text[match.end():])
            target = self._path(path, create_parents=True)
            old = self._read(target)[0] if target.exists() else ""
            matches = [match for match in _MARKER.finditer(old) if match.group(1) == uid]
            if len(matches) > 1:
                raise MemoryWorkspaceError("A managed memory block has duplicate identities")
            block = f"<!-- ciel-memory:{uid} -->\n{text.strip()}\n<!-- /ciel-memory:{uid} -->\n"
            if matches:
                match = matches[0]
                new = old[:match.start()] + block + old[match.end():]
            else:
                new = old.rstrip() + ("\n\n" if old.strip() else "") + block
            self._atomic_write(target, new)
            stored = {**record, "path": path, "digest": self._digest(text.strip())}
            stored.pop("content", None)
            value = stored.get("metadata", {}).get("value")
            if isinstance(value, str) and value and text.count(value) == 1:
                stored["value_template"] = text.split(value, 1)
            state["facts"][uid] = stored
            self._record_provenance(state, path, origin="owner", session_id="", local_only=private)
            self._save_registry(state)
            return {**stored, "content": text.strip()}

    def remove_fact(self, uid: str) -> None:
        """Delete the authoritative block while leaving unrelated owner prose intact."""
        with self.locked():
            state = self._registry()
            record = state["facts"].pop(uid, None)
            if not isinstance(record, dict):
                return
            target = self._path(record["path"])
            if target.exists():
                old = self._read(target)[0]
                matches = [match for match in _MARKER.finditer(old) if match.group(1) == uid]
                if len(matches) > 1:
                    raise MemoryWorkspaceError("A managed memory block has duplicate identities")
                if matches:
                    match = matches[0]
                    self._atomic_write(target, old[:match.start()] + old[match.end():])
            self._save_registry(state)

    def forget_patterns(self, patterns: list[str]) -> None:
        """Remove known forgotten wording from notes and private consolidation artifacts."""
        selected = [pattern for pattern in patterns if isinstance(pattern, str) and pattern]
        if not selected:
            return

        def redact(text: str) -> str:
            for pattern in selected:
                text = re.sub(r"(?<!\w)" + re.escape(pattern) + r"(?!\w)",
                              "[forgotten fact]", text, flags=re.I)
            return text

        def redact_values(value: Any) -> Any:
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, list):
                return [redact_values(item) for item in value]
            if isinstance(value, dict):
                return {key: redact_values(item) for key, item in value.items()}
            return value

        with self.locked():
            state = self._registry()
            paths = self._document_paths()
            dreams = self._internal_path("DREAMS.md")
            if dreams.exists():
                paths.append("DREAMS.md")
            preimages = self.root / ".memory-state" / "preimages"
            if preimages.is_symlink() or preimages.parent.is_symlink():
                raise MemoryWorkspaceError("Memory paths cannot contain symlinks")
            if preimages.exists():
                paths.extend(path.relative_to(self.root).as_posix()
                             for path in preimages.iterdir() if path.suffix == ".md")
            for relative in paths:
                path = self._internal_path(relative)
                text = self._read(path)[0]
                cleaned = redact(text)
                if cleaned != text:
                    self._atomic_write(path, cleaned)
            state["facts"] = redact_values(state["facts"])
            self._save_registry(state)

    def facts(self) -> dict[str, dict[str, Any]]:
        """Rebuild managed records from current prose and invalidate stale structured values."""
        with self.locked():
            state = self._registry()
            documents = {path: self._document(path, state) for path in self._fact_paths(state)
                         if self._path(path).exists()}
            blocks: dict[tuple[str, str], list[str]] = {}
            for path, document in documents.items():
                for match in _MARKER.finditer(document.text):
                    blocks.setdefault((path, match.group(1)), []).append(match.group(2).strip())
            result = {}
            changed = False
            for uid, stored in state["facts"].items():
                if not isinstance(stored, dict):
                    raise MemoryWorkspaceError("Managed memory metadata is malformed")
                matching = blocks.get((stored.get("path"), uid), [])
                if len(matching) > 1:
                    raise MemoryWorkspaceError("A managed memory block has duplicate identities")
                if not matching:
                    continue
                text = matching[0]
                if not text:
                    continue
                record = {**stored, "metadata": dict(stored.get("metadata", {}))}
                if documents[stored["path"]].local_only:
                    record["metadata"]["local_only"] = True
                digest = self._digest(text)
                if digest != stored.get("digest"):
                    record["metadata"].pop("value", None)
                    template = stored.get("value_template")
                    if (isinstance(template, list) and len(template) == 2
                            and all(isinstance(part, str) for part in template)):
                        prefix, suffix = template
                        if text.startswith(prefix) and (not suffix or text.endswith(suffix)):
                            value = text[len(prefix):len(text) - len(suffix) if suffix else None]
                            if value.strip():
                                record["metadata"]["value"] = value
                    record["source"] = "user_edited"
                    record["updated_at"] = datetime.fromtimestamp(
                        documents[stored["path"]].modified_ns / 1e9, timezone.utc,
                    ).isoformat()
                    record["metadata"]["observed_at"] = record["updated_at"]
                    record["digest"] = digest
                    state["facts"][uid] = record
                    changed = True
                result[uid] = {**record, "content": text}
            if changed:
                self._save_registry(state)
            return result
