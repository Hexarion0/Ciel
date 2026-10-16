"""Explicit Kaggle pilot storage policy; committed full-state resume stays mandatory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

GIB = 1024**3
MIN_FREE = 3 * GIB


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    sync_directory(path.parent)


def enabled() -> bool:
    return os.environ.get("CIEL_KAGGLE_PILOT") == "1"


def require_space(path: Path, write_bytes: int = 0) -> None:
    """Reserve the complete new file AND the safety floor, without deleting old state."""
    free = shutil.disk_usage(path).free
    required = MIN_FREE + write_bytes
    if free < required:
        raise RuntimeError(
            f"Kaggle pilot storage stop: {free / GIB:.2f} GiB free; need "
            f"{required / GIB:.2f} GiB including the 3 GiB reserve. "
            "The previous committed full optimizer checkpoint is preserved."
        )


def state_bytes(value) -> int:
    """Conservative tensor payload estimate, plus metadata/serialization headroom."""

    def size(item):
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(size(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(size(v) for v in item)
        return 0

    return size(value) + 64 * 1024**2


def prune_committed_periodic(output: Path, current: Path) -> None:
    """Called only AFTER replacement, fsync, checksum and latest-pointer commit."""
    native = output / "checkpoints/native"
    if not current.resolve().is_relative_to(native.resolve()):
        raise ValueError("Periodic checkpoint must stay in the native run directory")
    for old in native.glob("*/checkpoint_*.pth"):
        if old.resolve() != current.resolve():
            old.unlink()
