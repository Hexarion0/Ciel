"""Compatibility entry point for the packaged, isolated Chatterbox Turbo worker."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "src/ciel/audio/turbo_worker.py"),
        run_name="__main__",
    )
