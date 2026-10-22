"""Invalidate existing temporary images when application ownership or history changes."""

import os
from pathlib import Path


def invalidate_screen_images() -> None:
    """Clear only an existing runtime cache; history reset never starts capture services."""
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    if not (runtime / "ciel-screen").exists():
        return
    from ciel.logging import get_logger
    from ciel.screen.cache import ScreenCacheError, clear_runtime_caches

    try:
        clear_runtime_caches()
    except (OSError, ScreenCacheError):
        # Revocation/reset is already committed. An unusable cache cannot undo it.
        get_logger("screen").warning(
            "Temporary screen cleanup failed; old captures remain invalidated by "
            "owner and history checks. Check runtime-directory permissions.",
        )
