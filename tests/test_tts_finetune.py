"""Dataset gates and experiment safety without downloaded models or GPU inference."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.tts_finetune import cloud, review, xtts
from scripts.tts_finetune.common import EVALUATION_TEXTS, digest, dump, safe_path
from scripts.tts_finetune.corpus import prepare_audio, utterances
from scripts.tts_finetune.evaluate import evaluate, listening
from scripts.tts_finetune.training import Monitor


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    baseline = tmp_path / "data/voice/references/ciel/reference-1.wav"
    baseline.parent.mkdir(parents=True)
    sf.write(baseline, np.zeros(100), 24000)
    dataset = tmp_path / "dataset"
    (dataset / "clips").mkdir(parents=True)
    rows = []
    for index in range(20):
        audio = dataset / f"clips/{index}.wav"
        sf.write(audio, np.full(3 * 22050, 0.01), 22050)
        rows.append({"id": str(index), "source_id": "speaker-recording",
                     "source_url": "https://example.test/recording",
                     "start": index * 120, "end": index * 120 + 3,
                     "audio": f"clips/{index}.wav", "audio_sha256": digest(audio),
                     "caption": f"A unique utterance letter {chr(97 + index)}",
                     "raw": f"clips/{index}.wav",
                     "transcript": f"A unique utterance letter {chr(97 + index)}",
                     "prepared_seconds": 3,
                     "flags": []})
    dump(dataset / "dataset.json", {"clips": rows, "sources": [], "permission": {},
         "candidate_seconds": 60, "baseline_sha256": digest(baseline)})
    return dataset


def decisions(dataset: Path) -> dict:
    manifest = review.load_dataset(dataset)
    return {"manifest_sha256": manifest["manifest_sha256"], "rows": {
        row["id"]: {"decision": "approve", "transcript": row["transcript"],
                    "audio_sha256": row["audio_sha256"], "notes": "",
                    **dict.fromkeys(review.CHECKS, True)} for row in manifest["clips"]}}


def test_asr_and_captions_never_count_as_verified(corpus):
    assert review.verified(corpus, review.load_dataset(corpus)) == []
    with pytest.raises(ValueError, match="0.00 verified minutes"):
        review.export(corpus, corpus.parent / "export")


@pytest.mark.parametrize("change", ["manifest", "missing_check", "hash", "transcript"])
def test_review_import_rejects_incomplete_or_changed_data(corpus, change):
    value = decisions(corpus)
    if change == "manifest":
        value["manifest_sha256"] = "other"
    elif change == "missing_check":
        value["rows"]["0"]["transcript_verified"] = False
    elif change == "hash":
        value["rows"]["0"]["audio_sha256"] = "other"
    else:
        value["rows"]["0"]["transcript"] = "unverified|column"
    dump(corpus / "incoming.json", value)
    with pytest.raises(ValueError):
        review.import_reviews(corpus, corpus / "incoming.json")
    assert not (corpus / "reviews.json").exists()


def test_approved_export_uses_exact_transcripts_and_fixed_split(corpus):
    value = decisions(corpus)
    value["rows"]["0"]["transcript"] = "A human corrected this sentence."
    dump(corpus / "incoming.json", value)
    review.import_reviews(corpus, corpus / "incoming.json")
    output = corpus.parent / "export"
    review.export(corpus, output, minimum_minutes=0)
    manifest = xtts.validate_export(output, minimum_minutes=0)
    assert manifest["split"]["train_clips"] == 18
    assert manifest["split"]["validation_clips"] == 2
    assert next(r for r in manifest["clips"] if r["id"] == "0")["transcript"] == (
        "A human corrected this sentence.")
    assert (output / "evaluation/reference.wav").is_file()
    assert (output / "metadata_train.csv").read_text().startswith(
        "audio_file|text|speaker_name|emotion_name\n")
    with pytest.raises(ValueError, match="30 verified minutes"):
        xtts.validate_export(output)
    path = output / "metadata_train.csv"
    path.write_text(path.read_text().replace("unique", "edited"))
    with pytest.raises(ValueError, match="CSV no longer matches"):
        xtts.validate_export(output, minimum_minutes=0)


def test_rejected_clip_does_not_require_a_transcript(corpus):
    value = decisions(corpus)
    value["rows"]["0"].update(decision="reject", transcript="", transcript_verified=False)
    dump(corpus / "incoming.json", value)
    review.import_reviews(corpus, corpus / "incoming.json")
    assert len(review.verified(corpus, review.load_dataset(corpus))) == 19


def test_validation_purges_neighbors_and_duplicate_text():
    rows = [{"id": str(i), "source_id": "one", "start": i * 120,
             "end": i * 120 + 3, "prepared_seconds": 3,
             "transcript": f"Utterance {chr(97 + i)}"} for i in range(20)]
    original_train, validation, _ = review.split(rows)
    held = next(r for r in validation if r["start"] > 0)
    rows += [{**held, "id": "neighbor", "start": held["start"] - 4,
              "end": held["start"] - 1, "transcript": "Adjacent speech"},
             {**original_train[0], "id": "duplicate",
              "transcript": held["transcript"]}]
    train, validation, purged = review.split(rows)
    assert not ({r["id"] for r in train} & {r["id"] for r in validation})
    assert not ({r["transcript"] for r in train} & {r["transcript"] for r in validation})
    for row in train:
        assert not any(row["source_id"] == other["source_id"]
                       and row["start"] < other["end"] + 12
                       and row["end"] > other["start"] - 12 for other in validation)
    assert purged


def test_path_escape_and_audio_tampering_are_rejected(corpus):
    with pytest.raises(ValueError):
        safe_path(corpus, "../outside.wav")
    with pytest.raises(ValueError):
        safe_path(corpus, "/tmp/outside.wav")
    dump(corpus / "incoming.json", decisions(corpus))
    (corpus / "clips/0.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Audio changed"):
        review.import_reviews(corpus, corpus / "incoming.json")


def test_six_gb_training_rejected_without_loading_torch(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k:
                        "NVIDIA GeForce GTX 1660 SUPER, 6144, 5300")
    with pytest.raises(RuntimeError, match="Native GPT AdamW training rejected"):
        xtts.gpu_preflight()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "NVIDIA T4, 15360, 14000")
    assert xtts.gpu_preflight()["free_mib"] == 14000


def test_review_page_does_not_auto_approve(corpus):
    output = review.render(corpus)
    text = output.read_text()
    assert "Captions and independent ASR are suggestions" in text
    assert "canApprove()" in text
    assert "byId('transcript_verified').checked=false" in text
    assert "bulk" not in text


def test_audio_processing_preserves_pitch_and_nonoverlapping_windows():
    rate = 48000
    wave = 0.05 * np.sin(2 * np.pi * 210 * np.arange(rate * 4) / rate)
    prepared, operations = prepare_audio(np.column_stack([wave, wave]), rate)
    peak = np.argmax(np.abs(np.fft.rfft(prepared))) * 22050 / len(prepared)
    assert abs(peak - 210) < 1
    assert abs(operations["linear_gain_db"]) <= 3
    assert operations["denoising"] is False
    words = [{"start": i * 0.5, "text": f"word{i}."} for i in range(70)]
    clips = utterances(words, 36, 0)
    assert len(clips) > 2
    assert all(a["end"] <= b["start"] for a, b in zip(clips, clips[1:]))
    assert len({w for row in clips for w in row["caption"].split()}) == sum(
        len(row["caption"].split()) for row in clips)


def test_caption_boundary_never_extends_into_next_word():
    words = [{"start": i * 0.5, "text": f"word{i}."} for i in range(70)]
    clips = utterances(words, 36, 0)
    for row in clips:
        last = int(row["caption"].split()[-1].removeprefix("word").removesuffix("."))
        if last + 1 < len(words):
            assert row["end"] <= words[last + 1]["start"] - 0.099


def test_caption_markup_removal_preserves_spoken_words():
    from scripts.tts_finetune.corpus import caption_text

    assert caption_text("God, how do I phrase this? >> Okay, um I think about you a lot.") == (
        "God, how do I phrase this? Okay, um I think about you a lot.")
    assert caption_text("[Music] Everything is steady.") == "Everything is steady."
    assert caption_text("The value is greater than five.") == "The value is greater than five."


def test_cloud_bundle_contains_no_runtime_or_private_files(tmp_path):
    import zipfile

    output = cloud.bundle(tmp_path / "bundle")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "scripts/tts_finetune/xtts.py" in names
    assert not any(name.endswith((".env", ".wav", ".pth")) for name in names)
    document = json.loads(cloud.notebook(tmp_path / "cloud.ipynb").read_text())
    for item in document["cells"]:
        if item["cell_type"] == "code":
            compile(''.join(item["source"]), "notebook", "exec")


def test_monitor_stops_nonfinite_loss_or_human_request(tmp_path):
    monitor = Monitor(tmp_path)
    trainer = SimpleNamespace(keep_avg_train=SimpleNamespace(avg_values={"avg_loss": float('nan')}))
    with pytest.raises(RuntimeError, match="Non-finite"):
        monitor.step(trainer)
    trainer.keep_avg_train.avg_values = {"avg_loss": 1.0}
    trainer.train_loader = [None, None]
    saved = []
    trainer.save_checkpoint = lambda: saved.append(True)
    (tmp_path / "STOP").touch()
    with pytest.raises(RuntimeError, match="Human listening"):
        monitor.step(trainer)
    assert saved == [True]


def test_no_fine_tuned_audio_is_invented_for_empty_page(tmp_path, monkeypatch):
    from scripts.tts_finetune import evaluate as evaluation

    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    baseline = tmp_path / "data/voice/references/ciel/reference-1.wav"
    baseline.parent.mkdir(parents=True)
    sf.write(baseline, np.zeros(100), 24000)
    text = listening(tmp_path / "listen", []).read_text()
    assert "No fine-tuned checkpoint is available" in text
    assert "ratings_before_reveal" in text
    assert "if(!ready(round,choice))return" in text
    assert len(EVALUATION_TEXTS) == 7


def test_bench_requires_warm_repeats_and_real_sustained_window(tmp_path):
    with pytest.raises(ValueError, match="3 warm iterations"):
        evaluate("xtts", tmp_path / "bench", iterations=1)
    with pytest.raises(ValueError, match="300–600"):
        evaluate("xtts", tmp_path / "bench", sustained_seconds=30)
    with pytest.raises(ValueError, match="inference-xtts first"):
        evaluate("xtts", tmp_path / "bench", checkpoint=tmp_path / "full-training.pth")


def test_inference_export_preserves_model_and_rejects_untrained_checkpoint(tmp_path, monkeypatch):
    model = {"xtts.gpt.weight": "unchanged tensor", "dvae.weight": "frozen tensor"}
    state = {"model": model, "epoch": 0, "step": 0, "optimizer": "large moment tensors"}
    written = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        load=lambda path, **kwargs: state,
        save=lambda value, path: (written.append(value), Path(path).write_bytes(b"model"))))
    source = tmp_path / "full.pth"
    source.write_bytes(b"full checkpoint")
    with pytest.raises(ValueError, match="No trained epoch"):
        xtts.inference_checkpoint(source, tmp_path / "inference.pth")
    state.update(epoch=2, step=500)
    output = xtts.inference_checkpoint(source, tmp_path / "inference.pth")
    assert written[0]["model"] is model
    assert "optimizer" not in written[0]
    assert json.loads(output.with_suffix('.json').read_text())["weights_modified"] is False


def test_checkpoint_inference_and_conditioning_cache_are_separate(tmp_path, monkeypatch):
    from scripts.tts_shootout import providers

    class Tensor:
        dtype = "float32"

        def detach(self):
            return self

        cpu = detach
        contiguous = detach

    class Model:
        calls = []
        encodings = 0

        @staticmethod
        def init_from_config(config):
            return Model()

        def load_checkpoint(self, config, **kwargs):
            self.calls.append(kwargs)

        def to(self, device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([Tensor()])

        def get_conditioning_latents(self, **kwargs):
            Model.encodings += 1
            return Tensor(), Tensor()

    directory = tmp_path / "data/voice/models/xtts-v2"
    directory.mkdir(parents=True)
    (directory / "inventory.json").write_text('{}')
    acceptance = tmp_path / "data/voice/shootout-research/xtts-license-acceptance.json"
    dump(acceptance, {"accepted": True})
    reference, checkpoint = tmp_path / "reference.wav", tmp_path / "checkpoint.pth"
    reference.write_bytes(b"reference")
    checkpoint.write_bytes(b"adapted checkpoint")
    monkeypatch.setattr(providers, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        isfinite=lambda tensor: SimpleNamespace(all=lambda: True)))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(
        save_file=lambda tensors, path: Path(path).write_bytes(b"cached"),
        load_file=lambda path, **kwargs: {"gpt_cond_latent": Tensor(),
                                          "speaker_embedding": Tensor()}))
    monkeypatch.setitem(sys.modules, "TTS.tts.configs.xtts_config", SimpleNamespace(
        XttsConfig=lambda: SimpleNamespace(load_json=lambda path: None)))
    monkeypatch.setitem(sys.modules, "TTS.tts.models.xtts", SimpleNamespace(Xtts=Model))
    options = {"device": "cpu", "precision": "fp32"}
    zero = providers.XTTSProvider(options, reference, tmp_path / "cache")
    fine = providers.XTTSProvider({**options, "checkpoint": checkpoint}, reference,
                                 tmp_path / "cache")
    cached = providers.XTTSProvider({**options, "checkpoint": checkpoint}, reference,
                                    tmp_path / "cache")
    assert zero.state_path != fine.state_path == cached.state_path
    assert Model.encodings == 2
    assert cached.conditioning_encoded is False
    assert Model.calls[0]["checkpoint_dir"] == str(directory)
    assert Model.calls[1]["checkpoint_path"] == str(checkpoint)
    assert Model.calls[1]["vocab_path"] == str(directory / "vocab.json")
