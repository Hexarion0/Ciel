"""Fixed material and provenance shared by all shootout engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STYLE_REVISION = "7b3c5e94fb0ef7d2c778f772dd8b671269bf5009"
STYLE_MODEL_REVISION = "3aa7ba7f8f275ec13dce21682a61494c35089e2a"
XTTS_MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
SENTENCES = {
    "short": "Yes. I'm here.",
    "conversational": (
        "You changed the configuration again, didn't you? "
        "Show me what you changed and I'll take a look."
    ),
    "warm": "It's alright. Show me what happened, and we'll figure it out together.",
    "amused": "You broke it again? Somehow, I'm not even surprised anymore.",
    "technical": (
        "The service is running correctly, but the environment variable is missing. "
        "Fix that first, then restart the process."
    ),
    "soft": "You're still awake? That's alright. I'll stay here for a little while.",
    "long": (
        "Everything looks steady so far. Let's check the remaining settings together, "
        "then we can leave the system alone for a while. "
        "You don't need to keep watching it; I'll tell you if something needs attention. "
        "And before you change that configuration again, let me finish checking this one."
    ),
}
CONFIGURATIONS = {
    **{f"style-{device}-{steps}": {
        "engine": "styletts2", "device": device, "precision": "fp32", "steps": steps,
    } for device in ("cpu", "cuda") for steps in (1, 3, 5)},
    "xtts-cuda-fp16": {"engine": "xtts", "device": "cuda", "precision": "fp16 AMP"},
    "xtts-cuda-fp32": {"engine": "xtts", "device": "cuda", "precision": "fp32"},
    "xtts-cpu-fp32": {"engine": "xtts", "device": "cpu", "precision": "fp32"},
    "chatterbox-baseline": {
        "engine": "chatterbox", "device": "cuda", "precision": "existing unchanged",
    },
    "neutts-air-cpu-q8": {"engine": "neutts", "device": "cpu", "precision": "Q8_0 / FP32 ONNX"},
}


def evaluation_python(configuration: dict) -> Path:
    """Keep incompatible optional-engine dependencies in their own environment."""
    environment = "neutts" if configuration["engine"] == "neutts" else "shootout"
    return ROOT / f"data/voice/envs/{environment}/bin/python"


def digest(path: Path) -> str:
    """Hash private audio/model files without exposing their contents."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path: Path, value: dict) -> None:
    """Replace report metadata atomically; audio and references are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
