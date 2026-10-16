"""Private experiment paths, fixed evaluation text, and atomic metadata writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_IDS = ("3mZ2ZbciDPM", "iP0AY_BnlHE", "cv0ovU6kb-s", "lSaNBdJbhds")
XTTS_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
INTRO_END = {"3mZ2ZbciDPM": 95.0, "lSaNBdJbhds": 118.0}
EVALUATION_TEXTS = {
    "short": "Yes. I'm here.",
    "conversational": "You changed the configuration again, didn't you? "
                      "Show me what you changed and I'll take a look.",
    "warm": "It's alright. Show me what happened, and we'll figure it out together.",
    "amused": "You broke it again? Somehow, I'm not even surprised anymore.",
    "technical": "The service is running correctly, but the environment variable is missing.",
    "soft": "You're still awake? That's alright. I'll stay here for a little while.",
    "long": "Everything looks steady so far. Let's check the remaining settings together, "
            "then we can leave the system alone for a while. "
            "You don't need to keep watching it; I'll tell you if something needs attention. "
            "And before you change that configuration again, let me finish checking this one.",
}


def digest(path: Path) -> str:
    """Hash audio/assets without loading a large recording into memory."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path: Path, value: dict) -> None:
    """Persist progress atomically; never expose credentials in a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    """Reject absolute/traversing paths in imported local review files."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Audio path must remain inside its dataset")
    return path
