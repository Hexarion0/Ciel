"""Feed real downloaded pilot WAVs to Ciel's existing rating-gated blind page."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.tts_finetune.common import EVALUATION_TEXTS, digest, dump, safe_path


def comparison(imported: Path, output: Path, include_progression: bool = False) -> Path:
    """Compare the selected checkpoint, or all saved epoch audio plus base XTTS.

    The progression option does not require or imply retained model weights for every
    epoch: native listening manifests preserve the original full checkpoint hashes.
    """
    from scripts.tts_finetune import evaluate

    final = imported / "final"
    reference_hash = digest(final / "reference.wav")
    baseline = imported / "samples/baseline/chatterbox"
    base = json.loads((baseline / "manifest.json").read_text())
    if base["reference_sha256"] != reference_hash or base["texts"] != EVALUATION_TEXTS:
        raise ValueError("Chatterbox reference/texts differ from the trained export")
    best = json.loads((final / "inventory.json").read_text())["checkpoint"]
    candidates = []
    for manifest in (imported / "samples").glob("step-*/manifest.json"):
        data = json.loads(manifest.read_text())
        if data["reference_sha256"] != reference_hash:
            raise ValueError("XTTS listening reference mismatch")
        if (data.get("checkpoint") or {}).get("sha256") == best["sha256"]:
            candidates.append(manifest)
    # The epoch monitor links samples to the latest FULL state. Compare native step/epoch
    # when best is compact; those samples were synthesized from that exact native epoch state.
    if not candidates:
        for manifest in (imported / "samples").glob("step-*/manifest.json"):
            data = json.loads(manifest.read_text())
            pointer = data.get("checkpoint") or {}
            if (pointer.get("step"), pointer.get("epoch")) == (
                best["step"],
                best["epoch"],
            ):
                candidates.append(manifest)
    if len(candidates) != 1:
        raise ValueError(
            "Cannot uniquely bind best trained checkpoint to real listening WAVs"
        )
    trained = json.loads(candidates[0].read_text())
    if trained["reference_sha256"] != reference_hash:
        raise ValueError("XTTS reference mismatch")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    sources = [
        (
            "Chatterbox Turbo · accepted baseline",
            baseline,
            [
                {
                    "name": name,
                    "text": EVALUATION_TEXTS[name],
                    "audio": row["file"],
                    "sha256": row["sha256"],
                }
                for name, row in base["audio"].items()
            ],
            None,
        ),
        (
            f"XTTS · Kaggle trained epoch {best['epoch']}",
            candidates[0].parent,
            trained["rows"],
            best["sha256"],
        ),
    ]
    if include_progression:
        initial = imported / "samples/baseline/xtts-zero-shot"
        zero_shot = json.loads((initial / "manifest.json").read_text())
        if (zero_shot["reference_sha256"] != reference_hash or zero_shot["step"] != 0
                or zero_shot.get("checkpoint") is not None):
            raise ValueError("Invalid native XTTS zero-shot baseline provenance")
        sources = [sources[0], ("XTTS · zero-shot baseline", initial, zero_shot["rows"], None)]
        epochs = []
        for manifest in (imported / "samples").glob("step-*/manifest.json"):
            value = json.loads(manifest.read_text())
            checkpoint = value.get("checkpoint") or {}
            if (value["reference_sha256"] != reference_hash
                    or checkpoint.get("epoch", 0) < 1 or checkpoint.get("step", 0) < 1
                    or checkpoint.get("step") != value["step"]
                    or checkpoint.get("epoch_complete") is not True
                    or len(checkpoint.get("sha256", "")) != 64):
                raise ValueError("Invalid trained epoch listening provenance")
            if value.get("inference") != zero_shot.get("inference"):
                raise ValueError("XTTS baseline and epoch listening protocols differ")
            epochs.append((checkpoint["epoch"], checkpoint["step"], manifest.parent, value))
        if len({(epoch, step) for epoch, step, _, _ in epochs}) != len(epochs):
            raise ValueError("Duplicate trained epoch listening manifests")
        for epoch, _, directory, value in sorted(epochs):
            sources.append((f"XTTS · Kaggle trained epoch {epoch}", directory, value["rows"],
                            value["checkpoint"]["sha256"]))
    for index, (label, source, rows, checkpoint) in enumerate(sources):
        directory = output / f"source-{index}"
        directory.mkdir(exist_ok=True)
        checked = []
        if (len(rows) != len(EVALUATION_TEXTS)
                or {r["name"] for r in rows} != set(EVALUATION_TEXTS)):
            raise ValueError("Missing/duplicate listening texts")
        for row in rows:
            if row["text"] != EVALUATION_TEXTS[row["name"]]:
                raise ValueError("Listening transcript differs")
            audio = safe_path(source, row["audio"])
            if digest(audio) != row["sha256"]:
                raise ValueError("Listening WAV checksum differs")
            target = directory / f"{row['name']}.wav"
            shutil.copy2(audio, target)
            checked.append(
                {
                    "sentence": row["name"],
                    "text": row["text"],
                    "iteration": 1,
                    "audio": target.name,
                }
            )
        result = directory / "result.json"
        dump(
            result,
            {
                "status": "complete_awaiting_listening",
                "label": label,
                "reference_sha256": reference_hash,
                "texts": EVALUATION_TEXTS,
                "checkpoint_sha256": checkpoint,
                "rows": checked,
                "protocol": "Downloaded native epoch WAVs; listening only; no runtime claims",
            },
        )
        results.append(result)
    # Existing page expects a repository reference. Supply an isolated fixture root only.
    fixture = output / "reference-root"
    reference = fixture / "data/voice/references/ciel/reference-1.wav"
    reference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final / "reference.wav", reference)
    previous = evaluate.ROOT
    try:
        evaluate.ROOT = fixture
        return evaluate.listening(output / "page", results)
    finally:
        evaluate.ROOT = previous


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imported", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-progression", action="store_true")
    args = parser.parse_args()
    print(comparison(args.imported.resolve(), args.output.resolve(), args.include_progression))
