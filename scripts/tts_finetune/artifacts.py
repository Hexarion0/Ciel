"""Private, checksummed cloud exports and experimental-only local registration."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from .common import EVALUATION_TEXTS, ROOT, digest, dump, safe_path
from .xtts import inference_checkpoint


def unpack(source: Path, target: Path) -> None:
    """Unpack a new private workspace; reject traversal, links and duplicate members."""
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    with zipfile.ZipFile(source) as archive:
        names = set()
        for info in archive.infolist():
            path = safe_path(target, info.filename)
            if (path in names or "\\" in info.filename
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Duplicate, ambiguous path or symlink in archive")
            names.add(path)
        target.mkdir(parents=True)
        archive.extractall(target)


def archive_directory(source: Path, output: Path) -> Path:
    """Write once and checksum every included file; no symlinks or external recordings."""
    if output.exists():
        raise FileExistsError(output)
    files = [p for p in sorted(source.rglob("*")) if p.is_file()]
    if any(p.is_symlink() or not p.resolve().is_relative_to(source.resolve()) for p in files):
        raise ValueError("Artifacts cannot contain external links")
    checksums = {str(p.relative_to(source)): digest(p) for p in files}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(source)))
        archive.writestr("checksums.json", json.dumps(checksums, indent=2) + "\n")
    dump(output.with_suffix(".sha256.json"), {"file": output.name, "sha256": digest(output)})
    return output


def export_cloud(workspace: Path, models: Path, dataset: Path, output: Path,
                 recovery: bool = False) -> Path:
    """Compact inference export, or explicitly separate full optimizer recovery export."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in ("config", "logs", "samples"):
        source = workspace / name
        if source.is_dir():
            shutil.copytree(source, output / name)
    for name in ("training-summary.json", "epoch-metrics.json", "stop-reason.json"):
        if (workspace / name).exists():
            shutil.copy2(workspace / name, output / name)
    best = json.loads((workspace / "checkpoints/best/checkpoint.json").read_text())
    selected = safe_path(workspace, best["path"])
    if digest(selected) != best["sha256"]:
        raise ValueError("Best checkpoint failed its committed checksum")
    if recovery:
        if best.get("full_optimizer_resume") is False:
            raise ValueError(
                "Kaggle pilot best is inference-only. Preserve/download the original workspace "
                "and its latest full checkpoint for strict resume; do not use best for resume."
            )
        # Latest and best only; do not duplicate every multi-GiB periodic optimizer state.
        seen = set()
        for kind in ("latest", "best"):
            pointer = json.loads((workspace / f"checkpoints/{kind}/checkpoint.json").read_text())
            source = safe_path(workspace, pointer["path"])
            if digest(source) != pointer["sha256"]:
                raise ValueError(f"{kind} checkpoint failed checksum")
            destination = safe_path(output, pointer["path"])
            if pointer["path"] not in seen:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                seen.add(pointer["path"])
            dump(output / f"checkpoints/{kind}/checkpoint.json", pointer)
        (output / "README.txt").write_text(
            "Private full-state recovery: unpack into a NEW persistent workspace.\n"
            "Use the same transfer dataset/code and pinned base assets.\n"
            "train-xtts VERIFIED PRETRAINED THIS_WORKSPACE --epochs 2 --resume\n"
            "Use a larger reviewed TOTAL --epochs only after listening.\n"
            "Interrupted epochs replay; optimizer/scheduler/scaler and global step persist.\n")
    else:
        final = output / "final"
        final.mkdir()
        inference_checkpoint(selected, final / "model.pth")
        for name in ("config.json", "vocab.json", "speakers_xtts.pth", "LICENSE.txt",
                     "training-inventory.json"):
            shutil.copy2(models / name, final / name)
        shutil.copy2(dataset / "evaluation/reference.wav", final / "reference.wav")
        dump(final / "inventory.json", {"checkpoint": best, "files": {
            p.name: digest(p) for p in final.iterdir() if p.is_file()}})
        dump(output / "experiment.json", {"schema": 1, "engine": "xtts",
             "status": "experimental_awaiting_blind_listening", "production_enabled": False,
             "checkpoint": "final/model.pth", "model_directory": "final",
             "reference": "final/reference.wav", "texts": EVALUATION_TEXTS,
             "selected_by": "best validation loss; NOT a listening winner"})
        shutil.copy2(ROOT / "docs/tts-cloud-preflight.md", output / "LOCAL-INTEGRATION.md")
        guide = ROOT / "docs/tts-kaggle-pilot.md"
        if guide.is_file():
            shutil.copy2(guide, output / "KAGGLE-PILOT.md")
        (output / "README.txt").write_text(
            "Experimental XTTS only. Do not change Ciel production configuration.\n"
            "Import with: .venv/bin/python scripts/ciel_tts_finetune.py import-xtts "
            "ciel-xtts-final.zip data/voice/experiments/ciel-xtts-pilot\n"
            "Then use evaluate xtts OUTPUT --experiment "
            "data/voice/experiments/ciel-xtts-pilot/experiment.json\n"
            "See docs/tts-cloud-preflight.md for exact benchmark and blind comparison commands.\n"
            "Speaker conditioning is recomputed once and cached against the checkpoint hash.\n"
            "Keep LICENSE.txt: CPML personal non-commercial evaluation.\n")
    return archive_directory(output, output.with_suffix(".zip"))


def import_experiment(archive: Path, destination: Path) -> Path:
    """Register a private experimental inference bundle without touching runtime settings."""
    unpack(archive, destination)
    checksums = json.loads((destination / "checksums.json").read_text())
    actual = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    if actual != set(checksums) | {"checksums.json"}:
        raise ValueError("Archive checksum inventory is incomplete")
    for name, expected in checksums.items():
        if digest(safe_path(destination, name)) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    manifest = destination / "experiment.json"
    value = json.loads(manifest.read_text())
    if value.get("engine") != "xtts" or value.get("production_enabled") is not False:
        raise ValueError("Expected an experimental XTTS inference export")
    for key in ("checkpoint", "reference", "model_directory"):
        if not safe_path(destination, value[key]).exists():
            raise ValueError(f"Missing experimental inference asset: {key}")
    return manifest
