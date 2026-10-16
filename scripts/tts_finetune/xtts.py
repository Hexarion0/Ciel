"""Pinned, native coqui-tts GPT adaptation; deliberately separate from Ciel runtime."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .common import EVALUATION_TEXTS, XTTS_REVISION, digest, dump, safe_path

FILES = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth", "dvae.pth",
         "mel_stats.pth", "LICENSE.txt")


def assets(output: Path) -> Path:
    """Download the real v2 training dependencies at the zero-shot model's revision."""
    from huggingface_hub import snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    snapshot_download("coqui/XTTS-v2", revision=XTTS_REVISION, allow_patterns=list(FILES),
                      local_dir=output, cache_dir=output / ".download-cache")
    dump(output / "training-inventory.json", {
        "repository": "coqui/XTTS-v2", "revision": XTTS_REVISION,
        "files": {name: digest(output / name) for name in FILES},
    })
    return output / "training-inventory.json"


def inference_checkpoint(source: Path, output: Path) -> Path:
    """Keep native XTTS model state unchanged; omit Adam/scheduler state from audition copies."""
    import torch

    if output.exists():
        raise FileExistsError("Preserve the earlier inference export")
    # Use a file path, not load_fsspec's file object: mmap avoids materializing
    # several GiB of optimizer tensors when only the model is needed.
    checkpoint = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("epoch", 0) < 1 or checkpoint.get("step", 0) < 1:
        raise ValueError("No trained epoch/step in checkpoint; cannot label it fine-tuned")
    if not any(key.startswith("xtts.gpt.") for key in checkpoint["model"]):
        raise ValueError("Expected a native GPTTrainer checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint["model"], "epoch": checkpoint["epoch"],
                "step": checkpoint["step"]}, str(output))
    dump(output.with_suffix(".json"), {
        "source_sha256": digest(source), "inference_sha256": digest(output),
        "epoch": checkpoint["epoch"], "step": checkpoint["step"],
        "weights_modified": False, "optimizer_omitted": True,
        "native_format": "Xtts.load_checkpoint reads the model mapping and filters extra keys",
    })
    return output


def gpu_preflight() -> dict:
    """Reject insufficient physical VRAM before loading a model or initializing CUDA."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(selected) != 1 or selected[0] in ("", "-1"):
        raise RuntimeError("Expose exactly one cloud GPU with CUDA_VISIBLE_DEVICES")
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free", f"--id={selected[0]}",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip()
    name, total, free = (part.strip() for part in output.split(","))
    result = {"name": name, "total_mib": float(total), "free_mib": float(free)}
    if result["total_mib"] < 15 * 1024 or result["free_mib"] < 12 * 1024:
        raise RuntimeError(
            f"Native GPT AdamW training rejected on {name}: {total} MiB total, {free} MiB free. "
            "At least 15 GiB total and 12 GiB free VRAM are required; "
            "use 16 GB or larger (24 GB preferred). "
            "No OOM retries or swap workaround."
        )
    return result


def cuda_device_preflight(torch, physical: dict) -> dict:
    """Check CUDA's usable capacity separately from nvidia-smi's physical framebuffer."""
    actual = torch.cuda.get_device_properties(0)
    free, _ = torch.cuda.mem_get_info(0)
    total_mib = actual.total_memory / 1024**2
    reserved_mib = physical["total_mib"] - total_mib
    result = {"name": actual.name, "cuda_total_mib": total_mib,
              "cuda_free_mib": free / 1024**2, "physical": physical,
              "physical_minus_cuda_mib": reserved_mib}
    # Kaggle T4 reports 15360 MiB physically; CUDA excludes driver-reserved memory.
    # Bound this allowance to 512 MiB, only for the explicitly requested T4 pilot.
    pilot_t4 = (os.environ.get("CIEL_KAGGLE_PILOT") == "1"
                and actual.name == physical["name"] and "T4" in actual.name
                and physical["total_mib"] >= 15 * 1024
                and 0 <= reserved_mib <= 512 and total_mib >= 14.5 * 1024)
    print("CUDA memory preflight: " + json.dumps(result), flush=True)
    if (actual.total_memory < 15 * 1024**3 and not pilot_t4) or free < 12 * 1024**3:
        raise RuntimeError(
            "CUDA capacity/free memory rejected: " + json.dumps(result)
            + "; require 15 GiB total (T4 pilot: physical 15 GiB with at most "
            "512 MiB driver reservation), and 12 GiB CUDA-free. No OOM retries."
        )
    result["bounded_t4_driver_reservation_allowance"] = pilot_t4
    return result


def validate_export(dataset: Path, minimum_minutes: float = 30) -> dict:
    """Recheck audio and explicit human verification after a local/cloud transfer."""
    value = json.loads((dataset / "provenance.json").read_text())
    if value.get("status") != "human_verified_training_export":
        raise ValueError("Training requires a verified export, not candidate/ASR metadata")
    duration = 0.0
    partitions = {"train": [], "validation": []}
    ids, paths, texts = set(), set(), {"train": set(), "validation": set()}
    for row in value["clips"]:
        if row["id"] in ids or row["audio"] in paths or row["split"] not in partitions:
            raise ValueError("Duplicate clip ID/path or invalid split in metadata")
        ids.add(row["id"])
        paths.add(row["audio"])
        text = row["transcript"]
        if not text.strip() or any(char in text for char in "|\n\r\x00"):
            raise ValueError("Empty or invalid transcript in metadata")
        texts[row["split"]].add(" ".join(text.casefold().split()))
        review = row["review"]
        if review.get("decision") != "approve" or not all(
            review.get(key) is True for key in
            ("transcript_verified", "same_speaker_verified", "audio_quality_verified")
        ):
            raise ValueError("Unverified clip in export")
        audio = safe_path(dataset, row["audio"])
        if digest(audio) != row["audio_sha256"]:
            raise ValueError(f"Training audio changed: {row['id']}")
        if not 3 <= row["prepared_seconds"] <= 11.6:
            raise ValueError("Unsupported training utterance duration")
        import soundfile as sf

        info = sf.info(audio)
        if (info.samplerate != 22050 or info.channels != 1 or info.frames > 255995
                or abs(info.duration - row["prepared_seconds"]) > 1 / 22050):
            raise ValueError(f"Audio format/duration differs from provenance: {row['id']}")
        partitions[row["split"]].append(row)
        duration += row["prepared_seconds"]
    if texts["train"] & texts["validation"]:
        raise ValueError("Duplicate transcript leaks across training and validation splits")
    if duration < minimum_minutes * 60 or not all(partitions.values()):
        raise ValueError("Need 30 verified minutes with explicit train/validation partitions")
    # Reproduce metadata exactly, including transcripts; a modified CSV is not verified.
    import csv

    for partition, rows in partitions.items():
        with (dataset / f"metadata_{partition}.csv").open(newline="") as handle:
            actual = list(csv.DictReader(handle, delimiter="|"))
        expected = [{"audio_file": row["audio"], "text": row["transcript"],
                     "speaker_name": "ciel", "emotion_name": "neutral"} for row in rows]
        if actual != expected:
            raise ValueError(f"{partition} CSV no longer matches verified provenance")
    return value


def configuration(dataset: Path, models: Path, output: Path, epochs: int):
    """Use GPTArgs/GPTTrainerConfig from the pinned supported recipe, not a new trainer."""
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    inventory = json.loads((models / "training-inventory.json").read_text())
    if inventory["revision"] != XTTS_REVISION:
        raise ValueError("Training model differs from the pinned zero-shot model")
    for name in FILES:
        if digest(models / name) != inventory["files"][name]:
            raise ValueError(f"Model asset changed: {name}")
    data = BaseDatasetConfig(
        formatter="coqui", dataset_name="ciel_verified", path=str(dataset),
        meta_file_train=str(dataset / "metadata_train.csv"),
        meta_file_val=str(dataset / "metadata_validation.csv"), language="en",
    )
    args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=str(models / "mel_stats.pth"), dvae_checkpoint=str(models / "dvae.pth"),
        xtts_checkpoint=str(models / "model.pth"), tokenizer_file=str(models / "vocab.json"),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    config = GPTTrainerConfig(
        # Native start_with_eval consumes epoch 0 without a training pass.
        epochs=epochs + 1, output_path=str(output), run_name="CIEL_XTTS_EXPERIMENT",
        project_name="ciel_speaker_adaptation", dashboard_logger="tensorboard",
        model_args=args, audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050,
                                             output_sample_rate=24000),
        datasets=[data], batch_size=1, eval_batch_size=1, batch_group_size=0,
        num_loader_workers=2, num_eval_loader_workers=0, training_seed=73,
        mixed_precision=True, precision="fp16", run_eval=True,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 0.01},
        lr=5e-6, lr_scheduler="MultiStepLR", scheduler_after_epoch=False,
        lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5},
        # Callbacks save at optimizer boundaries; native batch-based saves can capture
        # incomplete accumulated gradients, which the native checkpoint does not store.
        print_step=25, plot_step=25, save_step=1000000000, save_n_checkpoints=5,
        # Native interrupt saves can occur halfway through gradient accumulation.
        # Keep the last committed optimizer-boundary checkpoint instead.
        save_on_interrupt=False, save_checkpoints=True,
        target_loss="loss_mel_ce", save_all_best=False,
        test_sentences=[{"text": text, "speaker_wav": [str(dataset / "evaluation/reference.wav")],
                         "language": "en"} for text in EVALUATION_TEXTS.values()],
    )
    return config



def train(dataset: Path, models: Path, output: Path, epochs: int = 2,
          restore: Path | None = None, resume: bool = False) -> Path:
    """Cloud-only native GPT adaptation; --restore is a full-state continuation."""
    from .training import train as run

    return run(dataset, models, output, epochs, restore, resume)
