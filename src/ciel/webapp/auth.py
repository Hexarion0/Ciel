"""Local device pairing; bearer tokens are retained only as hashes, in memory and on disk."""

import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

PAIRED_DEVICE_LIFETIME = 30 * 24 * 3600
"""Paired devices stay connected until this long passes without use."""
BROWSER_SESSION_LIFETIME = 12 * 3600
"""Unpaired open home-network sessions stay in memory only, for this long."""
_RENEWAL_SAVE_INTERVAL = 24 * 3600
_RETRY_AFTER_FAILED_SAVE = 300
_DIGEST = re.compile(r"[0-9a-f]{64}")
_DEVICE_ID = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class DeviceSession:
    """A connected owner device and its CSRF secret."""

    device_id: str
    owner_key: str
    expires_at: float
    generation: int = 0
    csrf: str = field(default="", repr=False)
    token: str = field(default="", repr=False)
    paired: bool = False


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class DeviceAuth:
    """Single-use codes expire after five minutes; paired devices after 30 idle days."""

    def __init__(self, owner_key: str, *, clock: Callable[[], float] = time.time) -> None:
        self.owner_key, self.clock = owner_key, clock
        self._lock = threading.RLock()
        self._code = ""
        self._code_expiry = 0.0
        self._devices: dict[str, DeviceSession] = {}
        self._saved_expiry: dict[str, float] = {}
        self._attempts: deque[tuple[float, str]] = deque(maxlen=500)
        self._store: Path | None = None
        self._retry_save_at = 0.0

    def attach_store(self, path: Path) -> None:
        """Restore paired devices from a private file and save later pairing changes there."""
        with self._lock:
            try:
                if path.parent.exists():
                    path.parent.chmod(0o700)
                if path.exists():
                    path.chmod(0o600)
            except OSError as error:
                logger.warning("Not using paired-device file %s; it cannot be made "
                               "owner-only: %s", path, error)
                return
            self._store = path
            now = self.clock()
            for digest, record in self._read_store(path).items():
                # Never trust a saved expiry beyond one idle window from now.
                expires_at = min(record["expires_at"], now + PAIRED_DEVICE_LIFETIME)
                if expires_at > now:
                    self._devices[digest] = DeviceSession(
                        record["device_id"], self.owner_key, expires_at,
                        csrf=record["csrf"], paired=True,
                    )
                    self._saved_expiry[digest] = expires_at

    def issue_code(self) -> str:
        """Issue a short-lived code for display only on the trusted PC."""
        with self._lock:
            self._code = f"{secrets.randbelow(100000000):08d}"
            self._code_expiry = self.clock() + 300
            return self._code

    def pair(self, code: str, *, source: str = "local") -> DeviceSession:
        """Exchange one code for an unguessable device token, subject to rate limits."""
        with self._lock:
            now = self.clock()
            while self._attempts and self._attempts[0][0] <= now - 60:
                self._attempts.popleft()
            if len(self._attempts) >= 50 or sum(s == source for _, s in self._attempts) >= 5:
                raise ValueError("Too many pairing attempts. Wait one minute.")
            if (
                not self._code
                or now >= self._code_expiry
                or not secrets.compare_digest(self._code, code)
            ):
                self._attempts.append((now, source))
                raise ValueError("The pairing code is incorrect or expired.")
            self._code = ""
            device = self.create_session(paired=True)
            self._save()
            return device

    def create_session(self, *, paired: bool = False) -> DeviceSession:
        """Allocate a browser identity; only paired devices are kept across restarts."""
        with self._lock:
            token = secrets.token_urlsafe(32)
            lifetime = PAIRED_DEVICE_LIFETIME if paired else BROWSER_SESSION_LIFETIME
            device = DeviceSession(
                secrets.token_hex(16), self.owner_key, self.clock() + lifetime,
                csrf=secrets.token_urlsafe(32), paired=paired,
            )
            self._devices[_digest(token)] = device
            return replace(device, token=token)

    def authenticate(self, cookie: str) -> DeviceSession | None:
        """Authenticate by token digest; each use restarts a paired device's idle window."""
        if not isinstance(cookie, str) or len(cookie) > 128:
            return None
        with self._lock:
            digest = _digest(cookie)
            device = self._devices.get(digest)
            now = self.clock()
            if device is None:
                return None
            if device.expires_at <= now:
                del self._devices[digest]
                return None
            if device.paired:
                device = replace(device, expires_at=now + PAIRED_DEVICE_LIFETIME)
                self._devices[digest] = device
                renewal_due = (device.expires_at - self._saved_expiry.get(digest, 0.0)
                               >= _RENEWAL_SAVE_INTERVAL)
                if renewal_due and now >= self._retry_save_at:
                    self._save()
            return device

    def revoke(self, device_id: str) -> None:
        """Immediately revoke a device; in-flight guards recheck authentication."""
        with self._lock:
            self._devices = {h: d for h, d in self._devices.items() if d.device_id != device_id}
            self._save()

    def revoke_all(self) -> list[str]:
        """Disconnect every paired device and return their identities."""
        with self._lock:
            removed = [d.device_id for d in self._devices.values() if d.paired]
            self._devices = {h: d for h, d in self._devices.items() if not d.paired}
            self._save()
            return removed

    def clear(self) -> None:
        """Forget sessions in memory on shutdown; saved pairings stay for the next start."""
        with self._lock:
            # Detach first: a request racing shutdown must not save the emptied set.
            self._store = None
            self._devices.clear()
            self._saved_expiry.clear()
            self._code = ""

    def _read_store(self, path: Path) -> dict[str, dict]:
        """Read valid saved records; a missing or damaged file means no saved devices."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            logger.warning("Ignoring unreadable paired-device file %s: %s", path, error)
            return {}
        records = data.get("devices") if isinstance(data, dict) else None
        if not isinstance(records, list):
            logger.warning("Ignoring paired-device file %s with an unknown format", path)
            return {}
        valid: dict[str, dict] = {}
        for record in records:
            if (
                isinstance(record, dict)
                and isinstance(record.get("token_sha256"), str)
                and _DIGEST.fullmatch(record["token_sha256"])
                and isinstance(record.get("device_id"), str)
                and _DEVICE_ID.fullmatch(record["device_id"])
                and isinstance(record.get("csrf"), str)
                and 0 < len(record["csrf"]) <= 128
                and isinstance(record.get("expires_at"), int | float)
                and not isinstance(record["expires_at"], bool)
                and math.isfinite(record["expires_at"])
            ):
                valid[record["token_sha256"]] = record
        return valid

    def _save(self) -> None:
        """Atomically write paired devices to an owner-only file. Caller holds the lock."""
        if self._store is None:
            return
        now = self.clock()
        self._devices = {h: d for h, d in self._devices.items() if d.expires_at > now}
        paired = {h: d for h, d in self._devices.items() if d.paired}
        document = {"version": 1, "devices": [
            {"token_sha256": h, "device_id": d.device_id, "csrf": d.csrf,
             "expires_at": d.expires_at}
            for h, d in paired.items()
        ]}
        directory = self._store.parent
        temporary = directory / f".{self._store.name}.{secrets.token_hex(4)}.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
                handle.flush()
                os.fsync(handle.fileno())  # never leave an empty file after power loss
            os.replace(temporary, self._store)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            self._retry_save_at = now + _RETRY_AFTER_FAILED_SAVE
            logger.warning("Could not save paired devices to %s: %s", self._store, error)
            return
        self._retry_save_at = 0.0
        self._saved_expiry = {h: d.expires_at for h, d in paired.items()}
