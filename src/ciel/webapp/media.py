"""In-memory temporary media with atomic per-device and owner byte budgets."""

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class _Media:
    data: bytes
    owner_key: str
    device_id: str
    request_id: str
    revision: int
    content_type: str
    expires: float


class MediaStore:
    """Opaque handles never name files or grant authority outside their exact scope."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        item_limit: int = 10 * 1024**2,
        device_limit: int = 32 * 1024**2,
        owner_limit: int = 64 * 1024**2,
    ) -> None:
        self.clock = clock
        self.item_limit, self.device_limit, self.owner_limit = (
            item_limit,
            device_limit,
            owner_limit,
        )
        self._items: dict[str, _Media] = {}
        self._lock = threading.RLock()

    def _expire(self) -> None:
        now = self.clock()
        self._items = {h: m for h, m in self._items.items() if m.expires > now}

    def put(
        self,
        data: bytes,
        *,
        owner_key: str,
        device_id: str,
        request_id: str,
        revision: int,
        content_type: str,
        ttl: float,
    ) -> str:
        """Reserve a bounded blob under a trusted, server-assigned scope."""
        with self._lock:
            self._expire()
            size = len(data)
            device_bytes = sum(
                len(m.data)
                for m in self._items.values()
                if m.device_id == device_id and m.owner_key == owner_key
            )
            owner_bytes = sum(len(m.data) for m in self._items.values() if m.owner_key == owner_key)
            if (
                not 0 < size <= self.item_limit
                or device_bytes + size > self.device_limit
                or owner_bytes + size > self.owner_limit
                or not 0 < ttl <= 900
            ):
                raise ValueError("Temporary media limit reached. Remove an attachment and retry.")
            handle = secrets.token_urlsafe(24)
            self._items[handle] = _Media(
                bytes(data),
                owner_key,
                device_id,
                request_id,
                revision,
                content_type,
                self.clock() + ttl,
            )
            return handle

    def get(
        self, handle: str, *, owner_key: str, device_id: str, request_id: str, revision: int
    ) -> bytes:
        """Release bytes only to the exact currently admitted request scope."""
        with self._lock:
            self._expire()
            media = self._items.get(handle)
            if media is None or (
                media.owner_key,
                media.device_id,
                media.request_id,
                media.revision,
            ) != (
                owner_key,
                device_id,
                request_id,
                revision,
            ):
                raise ValueError("This attachment expired. Please attach it again.")
            return media.data

    def invalidate(self, **filters) -> None:
        """Discard media on cancellation, selection changes, expiry, or revocation."""
        with self._lock:
            self._items = {
                h: m
                for h, m in self._items.items()
                if not all(getattr(m, key) == value for key, value in filters.items())
            }

    def discard(self, handle: str, **scope) -> None:
        """Remove only an attachment whose complete scope is already admitted."""
        with self._lock:
            self.get(handle, **scope)
            self._items.pop(handle, None)
