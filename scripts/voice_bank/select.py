"""Build a provisional shortlist; never finalize references without human audition."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .analyze import dump, style_hint
from .audio import sha256


def shortlist(directory: Path, baseline: Path) -> Path:
    """Apply identity/noise screening; retain uncertain style labels as review hints."""
    report = json.loads((directory / "screening.json").read_text())
    identity = json.loads((directory / "speaker-screening.json").read_text())
    vectors = {key: np.asarray(value) for key, value in identity["vectors"].items()}
    unique = {row["id"]: row for row in report["pool"]}
    # Candidate pool anchor is a new-source reference, never the previous baseline.
    acceptable = [row for row in unique.values()
                  if identity["within_clip_cosines"][row["id"]] >= 0.7
                  and row["metrics"]["voiced_fraction_proxy"] >= 0.3]
    if not acceptable:
        raise ValueError("No stable voiced reference passed screening; human review required")
    anchor = max(acceptable, key=lambda row: row["score"])
    target = vectors[anchor["id"]]
    baseline_vector = np.asarray(identity["baseline_vector"])
    for row in report["pool"]:
        row["speaker_cosine_to_anchor"] = float(vectors[row["id"]] @ target)
        row["speaker_cosine_to_baseline"] = float(vectors[row["id"]] @ baseline_vector)
        row["within_clip_cosine"] = identity["within_clip_cosines"][row["id"]]
        row["identity_status"] = (
            "consistent proxy; audition pending" if row["speaker_cosine_to_anchor"] >= 0.75
            and row["within_clip_cosine"] >= 0.7 else "flagged; excluded from automatic shortlist"
        )
    result = {
        "status": "awaiting_user_audition", "permission": report["permission"],
        "sources": report["sources"], "candidates": [], "recommendation": {},
        "speaker_screening": {
            "anchor": anchor["id"], "encoder_sha256": identity["encoder_sha256"],
            "cuda_initialized": identity["cuda_initialized"],
            "interpretation": identity["interpretation"],
            "minimum_anchor_cosine_for_shortlist": 0.75,
            "minimum_within_clip_cosine": 0.7,
            "thresholds_calibrated": False,
        },
        "human_acceptance": None,
        "notes": (
            "All choices are provisional waveform/embedding/context shortlists. "
            "Source music, reverb, actor identity, emotional delivery, and superiority "
            "to baseline have not been established by listening. Runtime unchanged."
        ),
    }
    selected = []
    for style in ("neutral", "warm", "amused", "soft"):
        rows = [row for row in {row["id"]: row for row in report["pool"]}.values()
                if row["identity_status"].startswith("consistent")]
        if style != "neutral":
            rows = [row for row in rows if style_hint(row["caption"], style)]
        rows.sort(key=lambda row: (
            row["score"] + min(style_hint(row["caption"], style), 2) * 0.1
            + row["speaker_cosine_to_anchor"] * 0.15
        ), reverse=True)
        count = 0
        for row in rows:
            if any(row["source_id"] == previous["source_id"]
                   and abs(row["start"] - previous["start"]) < 15 for previous in selected):
                continue
            count += 1
            selected.append(row)
            item = dict(row)
            item["intended_style"] = style
            item["id"] = f"candidate-{style}-{count:02}"
            item["screening_id"] = row["id"]
            path = directory / f"{item['id']}.wav"
            if path.exists():
                raise FileExistsError(path)
            shutil.copy2(directory / row["reference"], path)
            item["reference"] = path.name
            result["candidates"].append(item)
            if count == 2:
                break
        result["recommendation"][style] = {
            "candidate": f"candidate-{style}-01" if count else None,
            "fallback": None if count else "neutral",
            "status": "provisional; listen before accepting" if count else "missing clean example",
        }
    # Keep a fourth-source comparator even if it did not enter the first eight.
    # This is a neutral audition slot, not an assertion of neutral delivery.
    for source in report["sources"]:
        if any(row["source_id"] == source["id"] for row in selected):
            continue
        options = [row for row in report["pool"] if row["source_id"] == source["id"]
                   and row["identity_status"].startswith("consistent")]
        if options:
            row = max(options, key=lambda row: row["score"])
            selected.append(row)
            count = sum(item["intended_style"] == "neutral" for item in result["candidates"]) + 1
            item = {**row, "id": f"candidate-neutral-{count:02}", "intended_style": "neutral",
                    "screening_id": row["id"], "notes": "Additional source comparator; audition."}
            path = directory / f"{item['id']}.wav"
            if path.exists():
                raise FileExistsError(path)
            shutil.copy2(directory / row["reference"], path)
            item["reference"] = path.name
            result["candidates"].append(item)
    for style in ("firm", "focused"):
        result["recommendation"][style] = {
            "candidate": None, "fallback": "neutral",
            "status": "No listening-confirmed controlled/analytical delivery; do not force a file.",
        }
    for source in result["sources"]:
        rows = [row for row in report["pool"] if row["source_id"] == source["id"]]
        source["speaker_anchor_cosine_range"] = [
            min(row["speaker_cosine_to_anchor"] for row in rows),
            max(row["speaker_cosine_to_anchor"] for row in rows),
        ]
        source["shortlisted_candidates"] = [row["id"] for row in result["candidates"]
                                            if row["source_id"] == source["id"]]
        source["review_status"] = (
            "Contains shortlisted material; listening acceptance pending"
            if source["shortlisted_candidates"] else
            "Not selected by automated shortlist; not proof of a different speaker"
        )
    baseline_copy = directory / "baseline" / "reference-1.wav"
    baseline_copy.parent.mkdir(exist_ok=True)
    if not baseline_copy.exists():
        shutil.copy2(baseline, baseline_copy)
    result["candidates"].insert(0, {
        "id": "baseline-reference-1", "intended_style": "neutral", "source_id": "baseline",
        "reference": str(baseline_copy.relative_to(directory)),
        "unprocessed": str(baseline_copy.relative_to(directory)),
        "reference_sha256": sha256(baseline), "source_url": None,
        "source_timestamp": "Existing baseline", "style_status": "Existing runtime reference",
        "notes": "Preserved comparator; not mixed into the new speaker identity bank.",
    })
    dump(directory / "screening-with-identity.json", report)
    dump(directory / "manifest.json", result)
    return directory / "manifest.json"
