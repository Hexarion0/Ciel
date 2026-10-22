"""Private runtime-only image cache shared by the owner's voice and chat processes."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ciel.screen.capture import (
    MAX_CAPTURE_BYTES,
    MAX_IMAGE_BYTES,
    MAX_MONITORS,
    Capture,
    Cursor,
    Monitor,
    ScreenCaptureError,
    ScreenImage,
    decode_image,
)

CAPTURE_TTL_SECONDS = 15 * 60
MAX_CAPTURES = 2
MAX_CROPS = 8
_SAFE_FILE = re.compile(r"[0-9a-f]{32}\.png\Z")
_SAFE_SCOPE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_METADATA_BYTES = 256 * 1024


class ScreenCacheError(ScreenCaptureError):
    """A temporary capture expired or its private cache could not be accessed."""


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise ScreenCacheError("Ciel's temporary screen directory must be private to your user.")


def _runtime_root(root: Path | None) -> Path:
    if root is not None:
        result = Path(root)
        if not result.is_absolute():
            raise ScreenCacheError("The temporary screen directory must be an absolute path.")
        _private_directory(result)
        return result
    configured = os.environ.get("XDG_RUNTIME_DIR")
    runtime = Path(configured) if configured else Path(f"/run/user/{os.getuid()}")
    try:
        info = runtime.lstat()
    except OSError as exc:
        raise ScreenCacheError("The logged-in user's runtime directory is unavailable.") from exc
    if (not runtime.is_absolute() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise ScreenCacheError("The logged-in user's runtime directory is not private.")
    result = runtime / "ciel-screen"
    _private_directory(result)
    return result


def _process_identity(pid: int) -> str | None:
    """Linux PID plus process start ticks disambiguates a recycled PID after a crash."""
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
        fields = value[value.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return f"{pid}:{fields[19]}"
    except (OSError, IndexError, ValueError):
        return None


def _read_private(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > limit):
            raise ScreenCacheError("A temporary screen cache file is invalid.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ScreenCacheError("A temporary screen cache file is too large.")
        return data
    finally:
        os.close(descriptor)


def _write_private(path: Path, data: bytes) -> None:
    temporary = path.parent / f".{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(descriptor)
        os.replace(temporary, path)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _scope_lock(directory: Path) -> Iterator[None]:
    descriptor = os.open(
        directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600,
    )
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise ScreenCacheError("Ciel's temporary screen lock is not private.")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _empty_state(*, lifecycle_id: str | None = None) -> dict[str, Any]:
    return {"version": 1, "generation": uuid.uuid4().hex, "lifecycle_id": lifecycle_id,
            "leases": {}, "captures": []}


def _read_state(directory: Path) -> dict[str, Any]:
    try:
        state = json.loads(_read_private(directory / "state.json", _MAX_METADATA_BYTES))
        if (not isinstance(state, dict) or state.get("version") != 1
                or not isinstance(state.get("generation"), str)
                or not isinstance(state.get("leases"), dict)
                or not isinstance(state.get("captures"), list)):
            raise ValueError("invalid state")
        return state
    except FileNotFoundError:
        return _empty_state()
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ScreenCacheError("The temporary screen cache metadata is invalid.") from exc


def _save_state(directory: Path, state: dict[str, Any]) -> None:
    value = json.dumps(state, allow_nan=False, separators=(",", ":")).encode()
    if len(value) > _MAX_METADATA_BYTES:
        raise ScreenCacheError("The temporary screen cache metadata is too large.")
    _write_private(directory / "state.json", value)


def _live_leases(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    leases = {}
    for token, value in state["leases"].items():
        if not isinstance(value, dict) or not isinstance(value.get("pid"), int):
            continue
        identity = _process_identity(value["pid"])
        if identity is not None and identity == value.get("identity"):
            leases[token] = value
    return leases


def _remove_unreferenced(directory: Path, state: dict[str, Any]) -> None:
    keep = {
        value.get("file")
        for capture in state["captures"] if isinstance(capture, dict)
        for value in capture.get("images", []) if isinstance(value, dict)
    }
    for path in directory.iterdir():
        if ((_SAFE_FILE.fullmatch(path.name) or path.name.endswith(".tmp"))
                and path.name not in keep):
            path.unlink(missing_ok=True)


def clear_runtime_caches(root: Path | None = None) -> None:
    """Invalidate all same-user captures, including work held by another frontend."""
    directory = _runtime_root(root)
    for scope in directory.iterdir():
        if not _SAFE_SCOPE.fullmatch(scope.name):
            continue
        _private_directory(scope)
        with _scope_lock(scope):
            state = _empty_state()
            _save_state(scope, state)
            _remove_unreferenced(scope, state)


class CaptureCache:
    """Two captures, at most fifteen minutes old, shared only by live same-owner handles.

    The first live handle starts a generation. Other frontends attach to that generation.
    Closing/crashing the last handle ends it; a later process never restores its images.
    An explicit lifecycle_id may additionally identify a separately managed capture service.
    """

    def __init__(self, scope: str, root: Path | None = None,
                 *, lifecycle_id: str | None = None, sweep_interval: float = 15.0) -> None:
        if not scope or len(scope) > 16_384:
            raise ScreenCacheError("A valid owner scope is required for screen capture.")
        self.root = _runtime_root(root)
        self.directory = self.root / hashlib.sha256(scope.encode()).hexdigest()
        _private_directory(self.directory)
        self._mutex = threading.RLock()
        self._token = uuid.uuid4().hex
        self._closed = False
        self._stop = threading.Event()
        identity = _process_identity(os.getpid())
        if identity is None:
            raise ScreenCacheError("The screen cache could not identify its running process.")
        with _scope_lock(self.directory):
            state = _read_state(self.directory)
            state["leases"] = _live_leases(state)
            if (not state["leases"] or (
                    lifecycle_id is not None and state.get("lifecycle_id") != lifecycle_id)):
                state = _empty_state(lifecycle_id=lifecycle_id)
            self._generation = state["generation"]
            state["leases"][self._token] = {"pid": os.getpid(), "identity": identity}
            self._expire(state)
            _save_state(self.directory, state)
            _remove_unreferenced(self.directory, state)
        self._thread = threading.Thread(
            target=self._sweep, args=(max(0.05, min(sweep_interval, 30.0)),),
            name="ciel-screen-expiry", daemon=True,
        )
        self._thread.start()

    def _expire(self, state: dict[str, Any]) -> None:
        now = time.time()
        state["captures"] = [
            entry for entry in state["captures"]
            if isinstance(entry, dict) and isinstance(entry.get("captured_at"), (int, float))
            and math.isfinite(entry["captured_at"])
            and 0 <= now - entry["captured_at"] < CAPTURE_TTL_SECONDS
        ][-MAX_CAPTURES:]

    @contextlib.contextmanager
    def _state(self) -> Iterator[dict[str, Any]]:
        with self._mutex, _scope_lock(self.directory):
            if self._closed:
                raise ScreenCacheError("The screen capture service has closed.")
            state = _read_state(self.directory)
            if state["generation"] != self._generation or self._token not in state["leases"]:
                raise ScreenCacheError(
                    "The screen capture service was reset; request a fresh image.",
                )
            self._expire(state)
            # Publish expiry before image reads, including when reading corrupt data raises.
            _save_state(self.directory, state)
            _remove_unreferenced(self.directory, state)
            yield state
            _save_state(self.directory, state)
            _remove_unreferenced(self.directory, state)

    def _sweep(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                with self._state():
                    pass
            except (OSError, ScreenCacheError):
                return

    def _store_image(self, image: ScreenImage) -> dict[str, Any]:
        decoded = decode_image(image.data)
        if (decoded.size != (image.width, image.height) or image.mime_type != "image/png"
                or not image.name or len(image.name) > 512
                or not all(math.isfinite(value) for value in (image.origin_x, image.origin_y))
                or (image.logical_width is not None and (
                    not math.isfinite(image.logical_width) or image.logical_width <= 0))
                or (image.logical_height is not None and (
                    not math.isfinite(image.logical_height) or image.logical_height <= 0))):
            raise ScreenCacheError("The captured image metadata is invalid.")
        filename = f"{uuid.uuid4().hex}.png"
        _write_private(self.directory / filename, image.data)
        value = asdict(replace(image, data=b""))
        del value["data"]
        value["file"] = filename
        value["bytes"] = len(image.data)
        return value

    def add(self, capture: Capture) -> None:
        """Store one complete fresh observation, evicting the oldest parent and its crops."""
        if (not capture.id or len(capture.id) > 256 or not math.isfinite(capture.captured_at)
                or not 0 <= time.time() - capture.captured_at < CAPTURE_TTL_SECONDS
                or not 0 < len(capture.monitors) <= MAX_MONITORS
                or not 0 < len(capture.images) <= MAX_MONITORS + 1 + MAX_CROPS
                or sum(len(image.data) for image in capture.images) > MAX_CAPTURE_BYTES
                or len({image.name for image in capture.images}) != len(capture.images)):
            raise ScreenCacheError("The capture is expired or exceeded the cache limits.")
        with self._state() as state:
            images = []
            try:
                images = [self._store_image(image) for image in capture.images]
                value = {"id": capture.id, "captured_at": capture.captured_at,
                         "monitors": [asdict(monitor) for monitor in capture.monitors],
                         "cursor": asdict(capture.cursor), "images": images}
                state["captures"] = [
                    entry for entry in state["captures"] if entry.get("id") != capture.id
                ] + [value]
                state["captures"] = sorted(
                    state["captures"], key=lambda entry: entry["captured_at"],
                )[-MAX_CAPTURES:]
            except BaseException:
                _remove_unreferenced(self.directory, state)
                raise

    def _load_image(self, value: dict[str, Any]) -> ScreenImage:
        filename = value.get("file")
        if not isinstance(filename, str) or not _SAFE_FILE.fullmatch(filename):
            raise ScreenCacheError("A temporary screen image reference is invalid.")
        try:
            data = _read_private(self.directory / filename, MAX_IMAGE_BYTES)
            values = {key: item for key, item in value.items() if key not in {"file", "bytes"}}
            image = ScreenImage(data=data, **values)
            decoded = decode_image(data)
            if decoded.size != (image.width, image.height):
                raise ValueError("invalid dimensions")
            return image
        except (OSError, TypeError, ValueError) as exc:
            raise ScreenCacheError("That temporary screen image is no longer available.") from exc

    def recent(self) -> list[Capture]:
        """Read currently valid captures, oldest first, enforcing expiry before access."""
        with self._state() as state:
            captures = []
            try:
                for value in state["captures"]:
                    captures.append(Capture(
                        id=value["id"], captured_at=value["captured_at"],
                        monitors=tuple(Monitor(**monitor) for monitor in value["monitors"]),
                        cursor=Cursor(**value["cursor"]),
                        images=tuple(self._load_image(image) for image in value["images"]),
                    ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ScreenCacheError("The temporary screen cache metadata is invalid.") from exc
            return captures

    def get_image(self, capture_id: str, name: str = "desktop") -> ScreenImage:
        """Read an available parent/crop without extending the parent's lifetime."""
        for capture in self.recent():
            if capture.id == capture_id:
                return capture.image(name)
        raise ScreenCacheError("That screenshot has expired; it cannot be viewed again.")

    def contains(self, capture_id: str) -> bool:
        """Check liveness at a provider/delivery boundary without exposing pixels."""
        with self._state() as state:
            return any(value["id"] == capture_id for value in state["captures"])

    def add_crop(self, capture_id: str, image: ScreenImage) -> ScreenImage:
        """Attach a bounded derived image to the parent's original expiry time."""
        with self._state() as state:
            for capture in state["captures"]:
                if capture["id"] != capture_id:
                    continue
                images = capture["images"]
                base_names = {"desktop", *(monitor["name"] for monitor in capture["monitors"])}
                if image.name in base_names:
                    raise ScreenCacheError("A crop cannot replace an original screen image.")
                others = [entry for entry in images if entry["name"] != image.name]
                if (len([entry for entry in others if entry["name"] not in base_names]) >= MAX_CROPS
                        or sum(entry.get("bytes", 0) for entry in others)
                        + len(image.data) > MAX_CAPTURE_BYTES):
                    raise ScreenCacheError("The derived screen images exceeded the cache limits.")
                capture["images"] = [*others, self._store_image(image)]
                return image
        raise ScreenCacheError("That screenshot has expired; its crop cannot be saved.")

    def clear(self) -> None:
        """Delete this owner's images; caller must first cancel in-flight capture work."""
        with self._state() as state:
            state["captures"] = []

    def close(self) -> None:
        """Release a frontend lease, deleting pixels when the shared service ends."""
        self._stop.set()
        if hasattr(self, "_thread") and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        with self._mutex, _scope_lock(self.directory):
            if self._closed:
                return
            self._closed = True
            state = _read_state(self.directory)
            if state["generation"] != self._generation:
                return
            state["leases"].pop(self._token, None)
            state["leases"] = _live_leases(state)
            if not state["leases"]:
                state = _empty_state()
            _save_state(self.directory, state)
            _remove_unreferenced(self.directory, state)

    def __enter__(self) -> CaptureCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
