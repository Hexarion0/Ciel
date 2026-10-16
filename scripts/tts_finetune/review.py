"""Local listening review and leakage-resistant exports of explicitly verified speech."""

from __future__ import annotations

import html
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .common import ROOT, digest, dump, safe_path
from .corpus import normalized, write_metadata

CHECKS = ("transcript_verified", "same_speaker_verified", "audio_quality_verified")


def load_dataset(directory: Path) -> dict:
    """Bind review decisions to one immutable candidate manifest and its audio hashes."""
    manifest = json.loads((directory / "dataset.json").read_text())
    manifest["manifest_sha256"] = digest(directory / "dataset.json")
    return manifest


def import_reviews(directory: Path, path: Path) -> Path:
    """Validate the whole import before replacing any existing local review decisions."""
    manifest = load_dataset(directory)
    incoming = json.loads(path.read_text())
    if incoming.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Review belongs to a different candidate manifest")
    clips = {row["id"]: row for row in manifest["clips"]}
    destination = directory / "reviews.json"
    previous = json.loads(destination.read_text()) if destination.exists() else {"rows": {}}
    if previous.get("manifest_sha256", manifest["manifest_sha256"]) != manifest["manifest_sha256"]:
        raise ValueError("Existing reviews belong to a different manifest")
    validated = {}
    for identifier, row in incoming["rows"].items():
        if identifier not in clips or row.get("audio_sha256") != clips[identifier]["audio_sha256"]:
            raise ValueError(f"Unknown/changed audio in review: {identifier}")
        if row.get("decision") not in {"pending", "approve", "reject"}:
            raise ValueError(f"Invalid review decision: {identifier}")
        if any(not isinstance(row.get(key), bool) for key in CHECKS):
            raise ValueError(f"Verification checks must be explicit booleans: {identifier}")
        text = row.get("transcript", "")
        if not isinstance(text, str) or not isinstance(row.get("notes", ""), str):
            raise ValueError(f"Transcript/notes invalid: {identifier}")
        text = text.strip()
        if row["decision"] == "approve" and not text:
            raise ValueError(f"Transcript/notes invalid: {identifier}")
        if any(character in text for character in ("|", "\n", "\r", "\x00")):
            raise ValueError(f"Transcript must be one plain line without pipes: {identifier}")
        if row["decision"] == "approve" and not all(row[key] for key in CHECKS):
            raise ValueError(f"Approval requires all three listening checks: {identifier}")
        audio = safe_path(directory, clips[identifier]["audio"])
        if digest(audio) != row["audio_sha256"]:
            raise ValueError(f"Audio changed since review: {identifier}")
        validated[identifier] = {**row, "transcript": text}
    previous["rows"].update(validated)
    previous.update({"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
                     "recorded_at": incoming.get("recorded_at"),
                     "verification": "Explicit listening decisions; not inferred from ASR"})
    dump(destination, previous)
    return destination


def verified(directory: Path, manifest: dict) -> list[dict]:
    """Pending/rejected clips and unchecked transcripts cannot silently enter training."""
    reviews = directory / "reviews.json"
    if not reviews.exists():
        return []
    value = json.loads(reviews.read_text())
    if value["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Reviews no longer match the candidate manifest")
    result = []
    for row in manifest["clips"]:
        review = value["rows"].get(row["id"], {})
        if review.get("decision") != "approve":
            continue
        if not all(review.get(key) is True for key in CHECKS):
            raise ValueError(f"Incomplete approval: {row['id']}")
        if review.get("audio_sha256") != digest(safe_path(directory, row["audio"])):
            raise ValueError(f"Approved audio changed: {row['id']}")
        text = review["transcript"].strip()
        if not text or any(c in text for c in "|\n\r\x00"):
            raise ValueError(f"Invalid approved transcript: {row['id']}")
        result.append({**row, "transcript": text, "review": review})
    return result


def split(rows: list[dict], seed: int = 73) -> tuple[list[dict], list[dict], list[dict]]:
    """Hold out time blocks; purge near-boundary clips and duplicate validation text."""
    groups = sorted({(row["source_id"], int(row["start"] // 120)) for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two independent recording/time blocks for validation")
    random.Random(seed).shuffle(groups)
    target = sum(row["prepared_seconds"] for row in rows) * 0.1
    selected, seconds = set(), 0.0
    for group in groups[:-1]:
        selected.add(group)
        seconds += sum(row["prepared_seconds"] for row in rows
                       if (row["source_id"], int(row["start"] // 120)) == group)
        if seconds >= target:
            break
    validation = [row for row in rows
                  if (row["source_id"], int(row["start"] // 120)) in selected]
    texts = {normalized(row["transcript"]) for row in validation}
    train, purged = [], []
    for row in rows:
        if row in validation:
            continue
        near = any(row["source_id"] == other["source_id"]
                   and row["start"] < other["end"] + 12
                   and row["end"] > other["start"] - 12 for other in validation)
        if near or normalized(row["transcript"]) in texts:
            purged.append(row)
        else:
            train.append(row)
    if not train or not validation:
        raise ValueError("Insufficient independent verified clips after leakage purge")
    return train, validation, purged


def export(directory: Path, output: Path, minimum_minutes: float = 30) -> Path:
    """Export only reviewed clips, without changing candidates or active voice references."""
    manifest = load_dataset(directory)
    rows = verified(directory, manifest)
    seconds = sum(row["prepared_seconds"] for row in rows)
    if seconds < minimum_minutes * 60:
        raise ValueError(f"Only {seconds / 60:.2f} verified minutes; need {minimum_minutes:g}. "
                         "Review more material rather than filling training with unverified clips.")
    train, validation, purged = split(rows)
    if sum(row["prepared_seconds"] for row in train + validation) < minimum_minutes * 60:
        raise ValueError("Leakage purge puts the dataset below the minimum verified duration")
    if output.exists():
        raise FileExistsError("Use a fresh export directory; verified datasets are immutable")
    output.mkdir(parents=True)
    (output / "wavs").mkdir()
    exported = []
    for partition, values in (("train", train), ("validation", validation)):
        converted = []
        for row in values:
            destination = output / "wavs" / f"{row['id']}.wav"
            shutil.copy2(safe_path(directory, row["audio"]), destination)
            item = {**row, "audio": str(destination.relative_to(output)), "split": partition}
            converted.append(item)
            exported.append(item)
        write_metadata(output / f"metadata_{partition}.csv", converted)
    dump(output / "provenance.json", {
        "schema_version": 1, "status": "human_verified_training_export",
        "manifest_sha256": manifest["manifest_sha256"], "sources": manifest["sources"],
        "permission": manifest["permission"], "clips": exported,
        "split": {"seed": 73, "block_seconds": 120, "boundary_purge_seconds": 12,
                  "validation_fraction_target": 0.1, "duplicate_text_across_splits": False,
                  "train_clips": len(train), "validation_clips": len(validation),
                  "train_seconds": sum(r["prepared_seconds"] for r in train),
                  "validation_seconds": sum(r["prepared_seconds"] for r in validation),
                  "purged_ids": [row["id"] for row in purged]},
    })
    baseline = ROOT / "data/voice/references/ciel/reference-1.wav"
    if baseline.is_file():
        if digest(baseline) != manifest["baseline_sha256"]:
            raise ValueError("The evaluation reference changed; preserve the original baseline")
        (output / "evaluation").mkdir()
        shutil.copy2(baseline, output / "evaluation/reference.wav")
    return output / "provenance.json"


def render(directory: Path) -> Path:
    """Create a paginated local listening workbench; no bulk auto-approval control."""
    manifest = load_dataset(directory)
    baseline = ROOT / "data/voice/references/ciel/reference-1.wav"
    if digest(baseline) != manifest["baseline_sha256"]:
        raise ValueError("The baseline no longer matches this experiment")
    destination = directory / "baseline.wav"
    if destination.exists() and digest(destination) != manifest["baseline_sha256"]:
        raise ValueError("Refusing to replace a different existing baseline")
    if not destination.exists():
        shutil.copy2(baseline, destination)
    checks_path = directory / "asr-checks.json"
    checks = json.loads(checks_path.read_text())["rows"] if checks_path.exists() else {}
    review_path = directory / "reviews.json"
    reviews = json.loads(review_path.read_text())["rows"] if review_path.exists() else {}
    clips = []
    for row in manifest["clips"]:
        check = checks.get(row["id"], {})
        if check and check.get("audio_sha256") != row["audio_sha256"]:
            raise ValueError("ASR check belongs to different audio")
        clips.append({**row, "asr": check.get("text", "Not cross-checked yet"),
                      "flags": row["flags"] + check.get("flags", []),
                      "proposal": check.get("text") or row["transcript"]})
    counts = Counter(row["source_id"] for row in clips)
    settings = {"manifest_sha256": manifest["manifest_sha256"], "clips": clips,
                "reviews": reviews, "sources": manifest["sources"], "counts": dict(counts)}
    template = Path(__file__).with_name("review.html").read_text()
    output = template.replace("__SETTINGS__", json.dumps(settings).replace("<", "\\u003c"))
    output = output.replace("__COUNT__", str(len(clips))).replace(
        "__MINUTES__", html.escape(f"{manifest['candidate_seconds'] / 60:.2f}"))
    path = directory / "review.html"
    path.write_text(output, encoding="utf-8")
    return path
