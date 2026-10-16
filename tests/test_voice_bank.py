"""Reference preparation regressions without models, YouTube, or audio devices."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.voice_bank.analyze import permission
from scripts.voice_bank.audio import SOURCE_IDS, prepare, sha256, vtt_words, write_new_audio
from scripts.voice_bank.extract import run as extract
from scripts.voice_bank.page import render
from scripts.voice_bank.preview import cached_state


def test_permission_is_required_before_source_processing(tmp_path):
    path = tmp_path / "permission.json"
    path.write_text(json.dumps({"status": "unconfirmed", "sources": []}))
    with pytest.raises(ValueError, match="not been confirmed"):
        permission(tmp_path)
    path.write_text(json.dumps({"status": "user_confirmed", "sources": [
        {"id": sid} for sid in SOURCE_IDS
    ]}))
    assert permission(tmp_path)["status"] == "user_confirmed"


def test_permission_for_one_source_does_not_authorize_other_sources(tmp_path):
    (tmp_path / "permission.json").write_text(json.dumps({
        "status": "user_confirmed", "sources": [{"id": SOURCE_IDS[0]}],
    }))
    with pytest.raises(ValueError):
        permission(tmp_path)


def test_inline_caption_timestamps_do_not_duplicate_rolling_text(tmp_path):
    path = tmp_path / "source.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n"
        "Hello<00:00:01.500><c> there.</c>\n\n"
        "00:00:03.000 --> 00:00:04.000\nHello there.\n\n"
        "00:00:04.000 --> 00:00:06.000\n"
        "Next<00:00:04.500><c> sentence.</c>\n"
    )
    assert vtt_words(path) == [
        {"start": 1.0, "text": "Hello"}, {"start": 1.5, "text": "there."},
        {"start": 4.0, "text": "Next"}, {"start": 4.5, "text": "sentence."},
    ]


def test_preprocessing_preserves_pitch_and_bounds_gain():
    rate = 48000
    tone = 0.03 * np.sin(2 * np.pi * 180 * np.arange(rate * 2) / rate)
    raw = np.concatenate([np.zeros(rate // 2), tone, np.zeros(rate // 2)])
    prepared, operations = prepare(np.column_stack([raw, raw]), rate)
    frequencies = np.fft.rfftfreq(len(prepared), 1 / 24000)
    peak = frequencies[np.argmax(np.abs(np.fft.rfft(prepared)))]
    assert abs(peak - 180) < 1
    assert 2 <= len(prepared) / 24000 <= 2.25
    assert operations["linear_gain"] <= 2
    assert not operations["denoising"] and not operations["pitch_or_formant_change"]
    assert not operations["compression"] and not operations["separation"]
    assert np.isfinite(prepared).all() and np.max(np.abs(prepared)) <= 0.85


def test_candidate_files_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "reference.wav"
    wave = np.ones(1000, dtype=np.float32) * 0.1
    write_new_audio(path, wave, 24000)
    digest = sha256(path)
    with pytest.raises(FileExistsError):
        write_new_audio(path, wave * 2, 24000)
    assert sha256(path) == digest


def test_state_cache_is_keyed_by_reference_model_and_numerical_mode(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference audio fingerprint")
    model = SimpleNamespace(get_state_for_audio_prompt=MagicMock(return_value={"state": True}))
    exported = []

    def export(_state, path):
        path.write_bytes(b"cached state")
        exported.append(path)

    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(export_model_state=export))
    monkeypatch.setattr("scripts.voice_bank.preview.importlib.metadata.version", lambda _n: "3.3.0")
    _state, fp32, encoded = cached_state(
        model, reference, tmp_path / "cache", language="english_2026-09", quantize=False,
    )
    assert encoded
    _state, reused, encoded = cached_state(
        model, reference, tmp_path / "cache", language="english_2026-09", quantize=False,
    )
    assert not encoded and reused == fp32
    assert model.get_state_for_audio_prompt.call_args.args[0] == fp32
    _state, int8, encoded = cached_state(
        model, reference, tmp_path / "cache", language="english_2026-09", quantize=True,
    )
    assert encoded and int8 != fp32 and len(exported) == 2


def test_cached_state_provenance_mismatch_is_rejected(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wave")
    model = SimpleNamespace(get_state_for_audio_prompt=lambda *_a, **_k: {"state": True})
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(
        export_model_state=lambda _s, path: path.write_bytes(b"state"),
    ))
    monkeypatch.setattr("scripts.voice_bank.preview.importlib.metadata.version", lambda _n: "3.3.0")
    _s, path, _e = cached_state(
        model, reference, tmp_path / "cache", language="english_2026-09", quantize=False,
    )
    path.with_suffix(".json").write_text("{}")
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        cached_state(
            model, reference, tmp_path / "cache", language="english_2026-09", quantize=False,
        )


def test_page_escapes_source_metadata_and_keeps_missing_audio_clear(tmp_path):
    manifest = {
        "sources": [{"id": "source", "title": "<script>alert(1)</script>",
                     "duration_seconds": 20, "review_status": "pending"}],
        "candidates": [{"id": "candidate-neutral-01", "source_id": "source",
                        "intended_style": "neutral", "source_timestamp": "00:00:10",
                        "reference": "missing.wav", "unprocessed": "missing-raw.wav"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    content = render(tmp_path).read_text()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "Not generated yet." in content
    assert "Reveal A/B after listening" in content
    assert 'data-candidate="candidate-neutral-01" disabled' in content


def test_browser_blind_gate_and_note_export():
    if not shutil.which("node"):
        pytest.skip("Node is needed for the audition-page interaction check")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["node", "tests/frontend/voice_bank.cjs"], cwd=root, capture_output=True,
        text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_manual_extract_preserves_sources_and_cannot_replace_a_candidate(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "permission.json").write_text(json.dumps({
        "status": "user_confirmed", "sources": [{"id": sid} for sid in SOURCE_IDS],
    }))
    original = source / f"source-{SOURCE_IDS[0]}.webm"
    original.write_bytes(b"immutable compressed archive")
    rate = 24000
    tone = 0.02 * np.sin(2 * np.pi * 190 * np.arange(rate * 7) / rate)
    write_new_audio(source / f"source-{SOURCE_IDS[0]}.decoded.wav", tone, rate)
    directory = tmp_path / "candidates"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "candidates": [], "sources": [{"id": SOURCE_IDS[0], "shortlisted_candidates": []}],
    }))
    before = sha256(original)
    path = extract(directory, source, identifier="candidate-warm-03", source_id=SOURCE_IDS[0],
                   start=1, end=6, style="warm")
    assert sha256(original) == before
    candidate_hash = sha256(path)
    with pytest.raises(FileExistsError):
        extract(directory, source, identifier="candidate-warm-03", source_id=SOURCE_IDS[0],
                start=1, end=6, style="warm")
    assert sha256(path) == candidate_hash
    row = json.loads((directory / "manifest.json").read_text())["candidates"][0]
    assert row["source_url"].endswith(SOURCE_IDS[0])
    assert row["intended_style"] == "warm" and row["identity_status"] == "CPU recheck pending"
