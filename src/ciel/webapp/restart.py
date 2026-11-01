"""Replace a cleanly stopped web process after an explicit browser restart action."""

import os
import sys


def restart_web() -> None:
    """Keep the same interpreter and launch options, without opening another PC tab."""
    arguments = [
        value for value in sys.argv[1:] if value not in {"--open-browser", "--no-open-browser"}
    ]
    os.execv(sys.executable, [sys.executable, "-m", "ciel", *arguments, "--no-open-browser"])
