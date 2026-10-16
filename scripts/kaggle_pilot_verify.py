"""Check the downloaded full resume state on CPU without initializing CUDA."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def verify(directory: Path) -> dict:
    import torch

    workspace = directory / "outputs/ciel-xtts-training"
    pointer = json.loads((workspace / "checkpoints/latest/checkpoint.json").read_text())
    checkpoint = (workspace / pointer["path"]).resolve()
    if not checkpoint.is_relative_to(workspace.resolve()):
        raise ValueError("Resume pointer leaves its workspace")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != pointer["sha256"]:
        raise ValueError("Downloaded full resume checkpoint checksum differs")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    required = {"model", "optimizer", "scheduler", "scaler", "config", "step", "epoch",
                "ciel_resume"}
    if not required <= set(state) or any(state[k] is None for k in required):
        raise ValueError("Incomplete native resume state")
    if not state["optimizer"].get("state") or not state["scaler"]:
        raise ValueError("Optimizer moments or AMP scaler missing")
    if not state["ciel_resume"]["optimizer_boundary"]:
        raise ValueError("Checkpoint was not committed at an optimizer boundary")
    identity = json.loads((workspace / "config/identity.json").read_text())
    if state["ciel_resume"]["identity"] != identity:
        raise ValueError("Saved native state and workspace recipe identity differ")
    full_files = list((workspace / "checkpoints/native").glob("*/checkpoint_*.pth"))
    if len(full_files) != 1 or full_files[0].resolve() != checkpoint:
        raise ValueError("Expected exactly one latest full optimizer checkpoint")
    steps = sorted({float(v["step"]) for v in state["optimizer"]["state"].values()
                    if "step" in v})
    if not steps or min(steps) < 1 or state["epoch"] != 2 or state["step"] != 738:
        raise ValueError("Missing trained optimizer updates or unexpected pilot epoch/step")
    if torch.cuda.is_initialized():
        raise RuntimeError("Local verification unexpectedly initialized CUDA")
    result = {"status": "verified", "sha256": digest.hexdigest(),
              "bytes": checkpoint.stat().st_size, "epoch": state["epoch"],
              "step": state["step"], "epoch_complete": state["ciel_resume"]["epoch_complete"],
              "optimizer_parameter_states": len(state["optimizer"]["state"]),
              "optimizer_step_values": steps, "scheduler_present": True, "scaler_present": True,
              "precision_policy": identity.get("precision_policy"),
              "prompt_mask_policy": identity.get("prompt_mask_policy"),
              "latest_full_checkpoint_count": 1, "cuda_initialized": False,
              "production_changed": False}
    (directory / "local-full-resume-check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve()), indent=2))
