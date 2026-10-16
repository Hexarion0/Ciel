"""Blind progression comparisons use verified native WAVs without loading a model."""

from __future__ import annotations

import json

import pytest

from scripts.kaggle_pilot_compare import comparison
from scripts.tts_finetune.common import EVALUATION_TEXTS, digest, dump


@pytest.fixture
def imported(tmp_path):
    root = tmp_path / "imported"
    final = root / "final"
    final.mkdir(parents=True)
    (final / "reference.wav").write_bytes(b"shared reference fixture")
    reference = digest(final / "reference.wav")
    for name, epoch, step in (("baseline/xtts-zero-shot", 0, 0),
                              ("step-00000369", 1, 369), ("step-00000738", 2, 738)):
        directory = root / "samples" / name
        directory.mkdir(parents=True)
        rows = []
        for key, text in EVALUATION_TEXTS.items():
            audio = directory / f"{key}.wav"
            audio.write_bytes(f"fixture {epoch} {key}".encode())
            rows.append({"name": key, "text": text, "audio": audio.name,
                         "sha256": digest(audio)})
        checkpoint = {"sha256": str(epoch) * 64, "epoch": epoch, "step": step,
                      "epoch_complete": True} if epoch else None
        dump(directory / "manifest.json", {"reference_sha256": reference, "step": step,
             "rows": rows, "inference": "native test_run; fixed seed", "checkpoint": checkpoint})
    dump(final / "inventory.json", {"checkpoint": {"sha256": "2" * 64, "epoch": 2, "step": 738}})
    directory = root / "samples/baseline/chatterbox"
    directory.mkdir(parents=True)
    audio = {}
    for key in EVALUATION_TEXTS:
        path = directory / f"{key}.wav"
        path.write_bytes(f"fixture chatterbox {key}".encode())
        audio[key] = {"file": path.name, "sha256": digest(path)}
    dump(directory / "manifest.json", {"reference_sha256": reference,
         "texts": EVALUATION_TEXTS, "audio": audio})
    return root


def test_existing_selected_checkpoint_comparison_remains_two_samples(imported, tmp_path):
    page = comparison(imported, tmp_path / "selected")
    rounds = json.loads((page.parent / "assignments.json").read_text())["rounds"]
    assert len(rounds) == 7 and all(len(row["samples"]) == 2 for row in rounds)


def test_progression_has_base_both_epochs_and_chatterbox(imported, tmp_path):
    page = comparison(imported, tmp_path / "progression", include_progression=True)
    rounds = json.loads((page.parent / "assignments.json").read_text())["rounds"]
    assert len(rounds) == 7 and all(len(row["samples"]) == 4 for row in rounds)
    expected = {"Chatterbox Turbo · accepted baseline", "XTTS · zero-shot baseline",
                "XTTS · Kaggle trained epoch 1", "XTTS · Kaggle trained epoch 2"}
    assert all({sample["identity"] for sample in row["samples"]} == expected for row in rounds)
    assert all({s["label"] for s in row["samples"]} == set("ABCD") for row in rounds)
    assert page.read_text().count('data-label="D"') == 7
    before = (page.parent / "assignments.json").read_bytes()
    comparison(imported, tmp_path / "progression", include_progression=True)
    assert (page.parent / "assignments.json").read_bytes() == before


@pytest.mark.parametrize("change", ["audio", "reference", "text", "protocol", "duplicate"])
def test_progression_rejects_changed_audio_or_provenance(imported, tmp_path, change):
    path = imported / "samples/step-00000369/manifest.json"
    value = json.loads(path.read_text())
    if change == "audio":
        (path.parent / "short.wav").write_bytes(b"changed audio")
    elif change == "reference":
        value["reference_sha256"] = "different speaker"
    elif change == "text":
        value["rows"][0]["text"] = "Different words."
    elif change == "protocol":
        value["inference"] = "different conditioning"
    else:
        value["rows"].append(value["rows"][0])
    dump(path, value)
    with pytest.raises(ValueError):
        comparison(imported, tmp_path / "invalid", include_progression=True)
