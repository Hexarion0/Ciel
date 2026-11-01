"""Atomically save a group of host preferences without dropping unrelated secrets."""

import os
import tempfile
import threading
from pathlib import Path

from dotenv import set_key

_ENV_LOCK = threading.RLock()


def save_preferences(path: Path, changes: dict[str, str]) -> None:
    """Publish one complete private configuration snapshot, preserving existing comments."""
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ENV_LOCK:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".ciel-config-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(path.read_text() if path.is_file() else "")
        try:
            temporary.chmod(0o600)
            for name, value in changes.items():
                set_key(temporary, name, value)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
