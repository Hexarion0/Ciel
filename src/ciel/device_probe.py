"""Isolate native audio discovery so an unavailable sound service cannot hang tools."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ciel.platform.base import AudioDeviceInfo


def list_devices(*, timeout: float = 5.0) -> list[AudioDeviceInfo]:
    """List metadata in a bounded child process, without opening capture/playback streams."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ciel.device_probe"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("Audio device discovery timed out.") from None
    if result.returncode:
        raise RuntimeError("Audio device discovery failed.")

    from ciel.platform.base import AudioDeviceInfo

    try:
        devices = json.loads(result.stdout)
        return [AudioDeviceInfo(**device) for device in devices]
    except (ValueError, TypeError):
        raise RuntimeError("Audio device discovery returned invalid metadata.") from None


def _main() -> int:
    """Run native discovery only in the explicitly launched worker process."""
    try:
        from ciel.platform import get_audio_backend

        devices = get_audio_backend().list_devices()
        print(json.dumps([asdict(device) for device in devices]))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
