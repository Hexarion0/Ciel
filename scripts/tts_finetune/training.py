"""Cloud orchestration around the pinned native GPTTrainer and Trainer APIs.

No training loop or optimizer replacement. Strict restore intentionally aborts rather
than using Trainer 0.3.3's silent partial-model/failed-optimizer restore fallback.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from importlib.metadata import distributions
from pathlib import Path

from . import kaggle_storage
from .common import EVALUATION_TEXTS, XTTS_REVISION, digest, dump, safe_path
from .xtts import configuration, cuda_device_preflight, gpu_preflight, validate_export

ACCUMULATION = 252


def dataset_report(dataset: Path, models: Path) -> dict:
    """Validate every verified WAV/CSV and native tokenizer before constructing a model."""
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

    value = validate_export(dataset)
    tokenizer = VoiceBpeTokenizer(vocab_file=str(models / "vocab.json"))
    normalized, lengths = [], []
    for row in value["clips"]:
        tokens = tokenizer.encode(row["transcript"], lang="en")
        if not tokens or 0 in tokens or 1 in tokens or len(tokens) > 200:
            raise ValueError(f"Native tokenizer rejected verified transcript: {row['id']}")
        lengths.append(len(tokens))
        normalized.append(" ".join(row["transcript"].casefold().split()))
    durations = [r["prepared_seconds"] for r in value["clips"]]
    result = {"total_clips": len(durations), "total_seconds": sum(durations),
              "shortest_seconds": min(durations), "longest_seconds": max(durations),
              "average_seconds": sum(durations) / len(durations), "sample_rate": 22050,
              "tokenizer_validation": "PASS: all native token IDs and lengths checked",
              "max_text_tokens": max(lengths), "missing_file_count": 0,
              "duplicate_transcript_count": len(normalized) - len(set(normalized)),
              "provenance_sha256": digest(dataset / "provenance.json"),
              "reference_sha256": digest(dataset / "evaluation/reference.wav")}
    for split in ("train", "validation"):
        rows = [r for r in value["clips"] if r["split"] == split]
        result[f"{split}_clips"] = len(rows)
        result[f"{split}_seconds"] = sum(r["prepared_seconds"] for r in rows)
        result[f"{split}_minutes"] = result[f"{split}_seconds"] / 60
    print(json.dumps(result, indent=2), flush=True)
    return result


def recipe_identity(config: dict) -> dict:
    """Only locations and the reviewed total epoch limit may change on continuation."""
    # Native config uses infinity as an unlimited-length sentinel. Preserve its identity
    # explicitly in strict JSON; this is unrelated to the finite-loss training safeguards.
    value = json.loads(json.dumps(config), parse_constant=lambda token: {
        "native_nonfinite_float": token})
    for name in ("epochs", "output_path"):
        value.pop(name, None)
    for row in value["datasets"]:
        for name in ("path", "meta_file_train", "meta_file_val"):
            row[name] = Path(row[name]).name
    for name in ("mel_norm_file", "dvae_checkpoint", "xtts_checkpoint", "tokenizer_file"):
        value["model_args"][name] = Path(value["model_args"][name]).name
    for row in value["test_sentences"]:
        row["speaker_wav"] = [Path(p).name for p in row["speaker_wav"]]
    return value


def restore_full_state(trainer, source: Path, identity: dict) -> dict:
    """Restore native single-optimizer state, fail closed, never silently warm start.

    Native Trainer replays an interrupted epoch; an epoch-complete checkpoint advances
    to the next epoch. This is full optimizer resume, not bitwise dataloader replay.
    Checkpoints only commit at accumulation boundaries, with no pending gradients.
    """
    import torch

    state = torch.load(str(source), map_location="cpu", weights_only=True, mmap=True)
    required = ("model", "optimizer", "scheduler", "scaler", "config", "step", "epoch",
                "ciel_resume")
    if any(key not in state or state[key] is None for key in required):
        raise ValueError("Full resume requires model/optimizer/scheduler/scaler/config/step/epoch")
    meta = state["ciel_resume"]
    if meta["identity"] != identity or not meta.get("optimizer_boundary"):
        raise ValueError("Resume dataset, dependencies, recipe or checkpoint boundary differs")
    if state["step"] < 1 or state["epoch"] < 1 or not state["optimizer"].get("state"):
        raise ValueError("Resume checkpoint has no trained optimizer state")
    # Same components as Trainer.restore_model, with strict errors instead of partial loads.
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    if not trainer.use_amp_scaler or not state["scaler"]:
        raise ValueError("Resume requires the original supported FP16 AMP scaler state")
    trainer.scaler.load_state_dict(state["scaler"])
    trainer.total_steps_done = state["step"]
    trainer.epochs_done = state["epoch"] + int(meta["epoch_complete"])
    trainer.best_loss = meta["best_loss"]
    trainer.continue_run = True
    return {"source_sha256": digest(source), "step": state["step"], "epoch": state["epoch"],
            "next_epoch": trainer.epochs_done, "epoch_complete": meta["epoch_complete"],
            "semantics": "full optimizer/scheduler/scaler resume; interrupted epoch replays"}


class Monitor:
    """Native callbacks: bounded checkpoint intervals and conservative loss safeguards."""

    def __init__(self, output: Path):
        self.output, self.started = output, time.monotonic()
        metrics = output / "epoch-metrics.json"
        self.rows = json.loads(metrics.read_text())["rows"] if metrics.exists() else []
        self.reason = None
        self.batch = 0
        self.epoch_complete = False

    def stop(self, reason: str) -> None:
        self.reason = reason
        dump(self.output / "stop-reason.json", {"reason": reason, "rows": self.rows})
        raise RuntimeError(f"Experiment stopped: {reason}")

    def start_epoch(self, trainer) -> None:
        self.batch = 0
        self.epoch_complete = False

    def finite_forward_loss(self, trainer, losses: dict) -> None:
        invalid = {key: str(float(value)) for key, value in losses.items()
                   if "loss" in key and not math.isfinite(float(value))}
        if invalid:
            dump(self.output / "nonfinite-forward-loss.json", {
                "epoch": trainer.epochs_done, "step": trainer.total_steps_done,
                "batch_in_epoch": self.batch + 1, "invalid_losses": invalid,
                "stage": "before backward and optimizer update",
                "last_committed_checkpoint_preserved": (
                    self.output / "checkpoints/latest/checkpoint.json").exists()})
            self.stop("Non-finite forward loss before backward; retain last committed checkpoint")

    def retained_index(self) -> None:
        pointers = {}
        for path in (self.output / "checkpoints").glob("*/checkpoint.json"):
            pointer = json.loads(path.read_text())
            pointers[str(path.relative_to(self.output))] = {
                **pointer, "retained_on_disk": safe_path(self.output, pointer["path"]).is_file()}
        dump(self.output / "checkpoints/retained-files.json", {"pointers": pointers})

    def step(self, trainer) -> None:
        if kaggle_storage.enabled():
            kaggle_storage.require_space(self.output)
        self.batch += 1
        losses = trainer.keep_avg_train.avg_values
        if any(not math.isfinite(float(v)) for k, v in losses.items() if "loss" in k):
            self.stop("Non-finite training loss; retain the last finite checkpoint")
        boundary = self.batch % ACCUMULATION == 0 or self.batch == len(trainer.train_loader)
        if boundary:
            trainer.save_checkpoint()
            self.retained_index()
            if (self.output / "STOP").exists():
                self.stop("Human listening stop request (STOP file; optimizer boundary)")

    def epoch(self, trainer) -> None:
        import torch

        train, validation = (dict(trainer.keep_avg_train.avg_values),
                             dict(trainer.keep_avg_eval.avg_values))
        invalid = any(not math.isfinite(float(v)) for group in (train, validation)
                      for k, v in group.items() if "loss" in k)
        def clean(values):
            return {k: float(v) if math.isfinite(float(v)) else None for k, v in values.items()}

        row = {"epoch": trainer.epochs_done, "step": trainer.total_steps_done,
               "baseline_only": trainer.start_with_eval,
               "train": clean(train), "validation": clean(validation),
               "elapsed_seconds_this_session": time.monotonic() - self.started,
               "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        self.rows.append(row)
        dump(self.output / "epoch-metrics.json", {"rows": self.rows})
        if invalid:
            self.stop("Non-finite validation/epoch loss; retain last finite checkpoint")
        self.epoch_complete = True
        if not trainer.start_with_eval:
            trainer.save_checkpoint()
            self.retained_index()
        useful = [r for r in self.rows if not r["baseline_only"]]
        if len(useful) >= 3:
            recent = useful[-3:]
            key = "avg_loss_mel_ce"
            if all(key in r["validation"] for r in recent):
                a, b, c = [r["validation"][key] for r in recent]
                if c > b > a and c > a * 1.05:
                    self.stop("Validation deteriorated over three checks (greater than 5%)")


def logger_class():
    """Use native dashboard extension points for atomic saves and standalone test WAVs."""
    from trainer.logging.tensorboard_logger import TensorboardLogger

    class ListeningLogger(TensorboardLogger):
        def __init__(self, output: Path, monitor: Monitor, identity: dict):
            super().__init__(output / "logs", "CIEL_XTTS_EXPERIMENT")
            self.output, self.monitor, self.identity = output, monitor, identity
            self.trainer = None

        def save_model(self, state: dict, path: str) -> None:
            import torch

            trainer = self.trainer
            best = dict(trainer.best_loss)
            if "best_model" in Path(path).name:
                best = state.get("model_loss", best)
            # Best is saved after eval/test; periodic saves are optimizer-boundary callbacks.
            complete = self.monitor.epoch_complete or "best_model" in Path(path).name
            state["ciel_resume"] = {"identity": self.identity, "optimizer_boundary": True,
                                    "epoch_complete": complete, "best_loss": best}
            pilot = kaggle_storage.enabled()
            if pilot:
                if "best_model" in Path(path).name:
                    state = {key: state[key] for key in ("model", "epoch", "step")}
                elif (not state.get("optimizer", {}).get("state")
                      or not state.get("scaler") or state.get("scheduler") is None):
                    raise RuntimeError(
                        "Refusing to commit a non-resumable pilot checkpoint: "
                        "trained optimizer, scheduler and AMP scaler state are mandatory")
                kaggle_storage.require_space(self.output, kaggle_storage.state_bytes(state))
                # Epoch-end saves reuse the last optimizer step's filename. Never replace
                # the file behind a committed pointer before its successor pointer commits.
                if Path(path).exists():
                    source = Path(path)
                    path = str(source.with_name(f"{source.stem}-commit-{time.time_ns()}.pth"))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(str(path) + ".tmp")
            try:
                with temporary.open("xb") as handle:
                    torch.save(state, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            temporary.replace(path)
            if pilot:
                kaggle_storage.sync_directory(Path(path).parent)
            pointer = {"path": str(Path(path).relative_to(self.output)),
                       "sha256": digest(Path(path)),
                       "epoch": state["epoch"], "step": state["step"], "epoch_complete": complete}
            kind = "best" if "best_model" in Path(path).name else "latest"
            if pilot:
                pointer["full_optimizer_resume"] = kind == "latest"
            dump(self.output / "checkpoints" / kind / "checkpoint.json", pointer)
            if pilot:
                kaggle_storage.sync_pointer(
                    self.output / "checkpoints" / kind / "checkpoint.json")
                if kind == "best":
                    for old in Path(path).parent.glob("best_model*.pth"):
                        if old != Path(path):
                            old.unlink()
            if kind == "latest":
                dump(self.output / "checkpoints" / f"step-{state['step']:08d}" / "checkpoint.json",
                     pointer)
                if pilot:
                    kaggle_storage.prune_committed_periodic(self.output, Path(path))
            # Connect the epoch's listening outputs to the exact native saved weights.
            samples = self.output / "samples" / f"step-{state['step']:08d}" / "manifest.json"
            if samples.exists() and complete:
                value = json.loads(samples.read_text())
                value["checkpoint"] = pointer
                dump(samples, value)

        def test_audios(self, step: int, audios: dict, sample_rate: int) -> None:
            import numpy as np
            import soundfile as sf
            import torch

            label = "baseline/xtts-zero-shot" if step == 0 else f"step-{step:08d}"
            directory = self.output / "samples" / label
            if directory.exists():
                raise FileExistsError(f"Listening samples already exist: {directory}")
            directory.mkdir(parents=True)
            rows = []
            for index, (name, text) in enumerate(EVALUATION_TEXTS.items()):
                audio = audios[f"{index}-audio"]
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError(f"Invalid validation synthesis: {name}")
                sf.write(directory / f"{name}.wav", audio, sample_rate, subtype="PCM_16")
                rows.append({"name": name, "text": text, "audio": f"{name}.wav",
                             "sha256": digest(directory / f"{name}.wav")})
            dump(directory / "manifest.json", {"step": step, "rows": rows,
                 "reference_sha256": self.identity["reference_sha256"],
                 "engine": "XTTS base" if step == 0 else "XTTS experimental checkpoint",
                 "inference": "native GPTTrainer.test_run; gpt_cond_len=3; seed=73",
                 "checkpoint": None})
            super().test_audios(step, audios, sample_rate)

    return ListeningLogger


def train(dataset: Path, models: Path, output: Path, epochs: int = 2,
          restore: Path | None = None, resume: bool = False) -> Path:
    """Run the native pretrained GPT adaptation on a sufficiently large cloud GPU only."""
    if not 1 <= epochs <= 10:
        raise ValueError("Choose 1–10 TOTAL training epochs; begin with a two-epoch pilot")
    gpu = gpu_preflight()  # Must precede Torch import or model initialization.
    if restore and resume:
        raise ValueError("Choose --resume for this workspace OR --restore for a full checkpoint")
    if output.exists() and not (restore or resume):
        raise FileExistsError("Preserve this experiment; use --resume or a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    pilot = kaggle_storage.enabled()
    if pilot:
        kaggle_storage.require_space(output)
    elif shutil.disk_usage(output).free < 60 * 1024**3:
        raise RuntimeError("Keep 60 GiB free on persistent storage for native Adam checkpoints")
    if (output / "STOP").exists() or (output / "stop-reason.json").exists():
        raise ValueError("Review STOP/stop-reason.json before archiving them to permit resume")
    if resume:
        pointer = json.loads((output / "checkpoints/latest/checkpoint.json").read_text())
        restore = safe_path(output, pointer["path"])
        if digest(restore) != pointer["sha256"]:
            raise ValueError("Latest checkpoint failed its committed checksum")
        if pointer.get("epoch_complete") and pointer["epoch"] >= epochs:
            raise ValueError("Total epoch limit already reached; audition/export before extending")

    import random

    import numpy as np
    import torch
    from trainer import Trainer, TrainerArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this pinned training environment")
    cuda_gpu = cuda_device_preflight(torch, gpu)
    dump(output / "config/cuda-memory-preflight.json", cuda_gpu)
    config = configuration(dataset, models, output, epochs)
    if pilot:
        config.save_n_checkpoints = 1
        dump(output / "config/kaggle-storage-policy.json", {
            "mode": "kaggle_pilot", "minimum_free_gib": 3,
            "latest_full_optimizer_checkpoints": 1, "best_validation": "inference only",
            "replacement": "old full state retained until new state and pointer commit",
            "resume": "strict model/Adam/scheduler/scaler identity checks unchanged",
            "session_storage": "Kaggle working disk is ephemeral; download persisted outputs",
            "new_submission": "requires explicitly attached prior output for cross-session resume",
        })
    sanity = dataset_report(dataset, models)
    dump(output / "config/dataset-sanity.json", sanity)
    frozen = "".join(sorted(f"{d.metadata['Name']}=={d.version}\n" for d in distributions()))
    dependencies = output / "config/dependencies.txt"
    dependencies.parent.mkdir(parents=True, exist_ok=True)
    dependencies.write_text(frozen)
    identity = {"provenance_sha256": sanity["provenance_sha256"],
                "reference_sha256": sanity["reference_sha256"], "base_revision": XTTS_REVISION,
                "recipe": recipe_identity(config.to_dict()), "dependencies": frozen,
                "grad_accum_steps": ACCUMULATION}
    stable_perceiver = os.environ.get("CIEL_XTTS_FP32_PERCEIVER") == "1"
    if stable_perceiver:
        from .precision import POLICY

        identity["precision_policy"] = POLICY.copy()
        dump(output / "config/precision-policy.json", POLICY)
    mask_fix = os.environ.get("CIEL_XTTS_MASK_UNITS_FIX") == "1"
    if mask_fix:
        from .prompt_mask import POLICY as MASK_POLICY

        identity["prompt_mask_policy"] = MASK_POLICY.copy()
        dump(output / "config/prompt-mask-policy.json", MASK_POLICY)
    prior_identity = output / "config/identity.json"
    if prior_identity.exists() and json.loads(prior_identity.read_text()) != identity:
        raise ValueError("Existing workspace has a different dataset/dependency/recipe identity")
    config.save_json(str(output / "config/requested-config.json"))
    dump(output / "config/identity.json", identity)
    from .cloud import validate_baseline

    baseline = dataset / "baseline/chatterbox"
    validate_baseline(baseline, dataset)
    if not (output / "samples/baseline/chatterbox").exists():
        shutil.copytree(baseline, output / "samples/baseline/chatterbox")
    model = GPTTrainer.init_from_config(config)
    model.requires_grad_(False)
    model.xtts.gpt.requires_grad_(True)
    if stable_perceiver:
        from .precision import install_fp32_perceiver

        install_fp32_perceiver(model)
    if mask_fix:
        from .prompt_mask import install_prompt_mask_fix

        install_prompt_mask_fix(model)
    train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)
    checked = XTTSDataset(config, train_samples + eval_samples, model.xtts.tokenizer,
                         config.audio.sample_rate, is_eval=True)
    if len(checked) != len(train_samples) + len(eval_samples):
        raise ValueError("Native loader rejected verified data; abort instead of dropping clips")
    monitor = Monitor(output)
    logger = logger_class()(output, monitor, identity)

    class StrictTrainer(Trainer):
        def _compute_loss(self, *args, **kwargs):
            outputs, losses = super()._compute_loss(*args, **kwargs)
            if losses:
                monitor.finite_forward_loss(self, losses)
            return outputs, losses

        def save_best_model(self):
            if not pilot:
                return super().save_best_model()
            if self.start_with_eval:
                return  # Select the best TRAINED model, not baseline epoch 0.
            loss = self._pick_target_avg_loss(self.keep_avg_eval)
            if loss is None or not math.isfinite(float(loss)):
                return  # The epoch monitor records the non-finite loss and stops.
            if loss >= self.best_loss["eval_loss"]:
                return
            from trainer.io import save_model

            best = {"train_loss": self._pick_target_avg_loss(self.keep_avg_train),
                    "eval_loss": loss}
            save_model(
                self.config, self._get_model(),
                str(output / "checkpoints/best-inference/best_model_inference.pth"),
                current_step=self.total_steps_done, epoch=self.epochs_done,
                model_loss=best, save_func=self.dashboard_logger.save_model,
            )
            self.best_loss = best

        def restore_model(self):
            if resume and Path(self.args.restore_path).resolve() != restore.resolve():
                raise ValueError("Native continuation did not select the checksum-committed latest")
            self.ciel_restored = restore_full_state(self, Path(self.args.restore_path), identity)

        def _restore_best_loss(self):
            # Native 0.3.3 reads the latest checkpoint's loss, not necessarily best loss.
            # restore_full_state has already restored the committed best-so-far loss.
            if not restore:
                super()._restore_best_loss()

        def test_run(self):
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all())
            random.seed(73)
            np.random.seed(73)
            torch.manual_seed(73)
            try:
                return super().test_run()
            finally:
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])

    report = {"status": "initializing", "gpu": gpu, "dataset": sanity,
              "total_training_epochs_requested": epochs, "effective_batch": ACCUMULATION,
              "model_revision": XTTS_REVISION, "production_changed": False,
              "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
              "gpt_trainable_parameters": sum(p.numel() for p in model.parameters()
                                              if p.requires_grad),
              "listening_acceptance": None, "sessions": []}
    if pilot:
        report["storage_mode"] = "kaggle_pilot_one_latest_full_plus_best_inference"
    previous = output / "training-summary.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        report["sessions"] = old.get("sessions", [])
    if resume:
        # Native continuation reuses its checkpoint directory and retention window.
        # Rebase only paths/total epochs for this runtime before native config loading.
        # The original full checkpoint retains the original config for the audit trail.
        config.save_json(str(restore.parent / "config.json"))
    try:
        trainer = StrictTrainer(
            TrainerArgs(restore_path=str(restore) if restore else "",
                        continue_path=str(restore.parent) if resume else "",
                        start_with_eval=not bool(restore), grad_accum_steps=ACCUMULATION),
            config, output_path=str(output / "checkpoints/native"), model=model,
            train_samples=train_samples, eval_samples=eval_samples, dashboard_logger=logger,
            callbacks={"on_epoch_start": monitor.start_epoch, "on_epoch_end": monitor.epoch,
                       "on_train_step_end": monitor.step}, parse_command_line_args=False)
        logger.trainer = trainer
        if trainer.epochs_done >= config.epochs:
            raise ValueError("Total epoch limit already reached; audition before extending")
        report["resume"] = getattr(trainer, "ciel_restored", None)
        report["status"] = "running"
        dump(previous, report)
        diagnostic_mode = os.environ.get("CIEL_XTTS_FORWARD_DIAGNOSTIC")
        if diagnostic_mode:
            from .forward_diagnostic import inspect_all_training_clips, inspect_first_batch

            diagnostic = (inspect_all_training_clips(trainer) if diagnostic_mode == "all_training"
                          else inspect_first_batch(trainer))
            dump(output / "forward-diagnostic.json", diagnostic)
            report["status"] = "forward_diagnostic_complete_no_updates"
            return previous
        trainer.fit()
        report["status"] = "pilot_complete_awaiting_listening"
    except BaseException as error:
        report["status"] = "stopped" if monitor.reason else "error_or_interrupted"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["epochs"] = monitor.rows
        report["stop_reason"] = monitor.reason
        report["sessions"].append({"elapsed_seconds": time.monotonic() - monitor.started,
                                    "resume": str(restore) if restore else None,
                                    "status": report["status"]})
        report["elapsed_seconds_all_sessions"] = sum(s["elapsed_seconds"]
                                                    for s in report["sessions"])
        report["checkpoints"] = {}
        for kind in ("latest", "best"):
            pointer = output / f"checkpoints/{kind}/checkpoint.json"
            if pointer.exists():
                report["checkpoints"][kind] = json.loads(pointer.read_text())
        dump(previous, report)
        logger.finish()
    return previous
