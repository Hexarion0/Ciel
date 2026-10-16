"""Cloud safety/resume/export regression checks, without real models or CUDA."""

from __future__ import annotations

import ast
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tts_finetune import artifacts, cloud, training, xtts
from scripts.tts_finetune.common import EVALUATION_TEXTS, digest, dump


@pytest.fixture(params=["local", "cloud", "kaggle"])
def archive_unpacker(request, tmp_path, monkeypatch):
    """Exercise the real import helper and the helpers emitted into cloud notebooks."""
    if request.param == "local":
        return artifacts.unpack
    if request.param == "cloud":
        notebook = cloud.notebook(tmp_path / "cloud.ipynb")
        function_name = "extract_new"
    else:
        from scripts import kaggle_pilot_notebook

        root = tmp_path / "checkout"
        original = root / "data/voice/finetuning/20261014-01/cloud/xtts_ciel.ipynb"
        original.parent.mkdir(parents=True)
        original.write_text('{"cells": []}')
        dump(original.parent / "transfer-final/transfer-manifest.json", {"files": {}})
        for relative in ("scripts/ciel_tts_finetune.py", "docs/tts-cloud-preflight.md"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic export fixture")
        monkeypatch.setattr(kaggle_pilot_notebook, "ROOT", root)
        notebook = kaggle_pilot_notebook.build(tmp_path / "kaggle", "test-owner")
        function_name = "extract_checked"
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                namespace = {"Path": Path, "zipfile": zipfile, "space": lambda *_args: None}
                exec(compile(ast.Module(body=[node], type_ignores=[]), "helper", "exec"), namespace)
                return namespace[function_name]
    raise AssertionError(f"Missing generated archive helper: {function_name}")


@pytest.mark.parametrize("memory,allowed", [(15359, False), (15360, True), (24576, True)])
def test_exact_fifteen_gib_gpu_guard(monkeypatch, memory, allowed):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(xtts.subprocess, "check_output",
                        lambda *a, **k: f"Cloud GPU, {memory}, 14000")
    if allowed:
        assert xtts.gpu_preflight()["total_mib"] == memory
    else:
        with pytest.raises(RuntimeError, match="15 GiB"):
            xtts.gpu_preflight()


def test_gpu_guard_respects_visible_device_and_rejects_ambiguous_setup(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    calls = []
    monkeypatch.setattr(xtts.subprocess, "check_output",
                        lambda args, **k: calls.append(args) or "GPU, 24000, 22000")
    xtts.gpu_preflight()
    assert "--id=2" in calls[0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="exactly one"):
        xtts.gpu_preflight()


def test_periodic_save_only_after_native_optimizer_boundary(tmp_path):
    monitor = training.Monitor(tmp_path)
    saves = []
    trainer = SimpleNamespace(keep_avg_train=SimpleNamespace(avg_values={"loss": 1.0}),
                              train_loader=[None] * 369,
                              save_checkpoint=lambda: saves.append(monitor.batch))
    for _ in range(369):
        monitor.step(trainer)
    assert saves == [252, 369]


def test_stop_waits_for_optimizer_boundary(tmp_path):
    monitor = training.Monitor(tmp_path)
    saves = []
    trainer = SimpleNamespace(keep_avg_train=SimpleNamespace(avg_values={"loss": 1.0}),
                              train_loader=[None] * 369,
                              save_checkpoint=lambda: saves.append(monitor.batch))
    (tmp_path / "STOP").touch()
    for _ in range(251):
        monitor.step(trainer)
    assert not saves
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        monitor.step(trainer)
    assert saves == [252]


def test_nonfinite_validation_stops_without_losing_json_report(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)))
    monitor = training.Monitor(tmp_path)
    trainer = SimpleNamespace(keep_avg_train=SimpleNamespace(avg_values={"avg_loss": 1.0}),
                              keep_avg_eval=SimpleNamespace(avg_values={"avg_loss": float("nan")}),
                              epochs_done=1, total_steps_done=369, start_with_eval=False)
    with pytest.raises(RuntimeError, match="Non-finite validation"):
        monitor.epoch(trainer)
    assert json.loads((tmp_path / "epoch-metrics.json").read_text())["rows"][0][
        "validation"]["avg_loss"] is None
    assert (tmp_path / "stop-reason.json").is_file()


def test_full_resume_loads_all_components_and_advances_completed_epoch(tmp_path, monkeypatch):
    identity = {"dataset": "verified", "recipe": "fixed"}
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"native full checkpoint")
    loaded = {}

    class Component:
        def __init__(self, name):
            self.name = name

        def load_state_dict(self, value, **kwargs):
            loaded[self.name] = (value, kwargs)

    state = {"model": {"weight": 3}, "optimizer": {"state": {1: "moments"}},
             "scheduler": {"last_epoch": 20}, "scaler": {"scale": 65536},
             "config": {}, "step": 738, "epoch": 2, "ciel_resume": {
                 "identity": identity, "optimizer_boundary": True, "epoch_complete": True,
                 "best_loss": {"train_loss": 2.0, "eval_loss": 2.2}}}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: state))
    trainer = SimpleNamespace(**{name: Component(name) for name in
                                 ("model", "optimizer", "scheduler", "scaler")},
                              use_amp_scaler=True)
    result = training.restore_full_state(trainer, source, identity)
    assert set(loaded) == {"model", "optimizer", "scheduler", "scaler"}
    assert loaded["model"][1] == {"strict": True}
    assert trainer.total_steps_done == 738 and trainer.epochs_done == 3
    assert trainer.best_loss["eval_loss"] == 2.2
    assert result["epoch_complete"] is True
    state["ciel_resume"]["epoch_complete"] = False
    training.restore_full_state(trainer, source, identity)
    assert trainer.epochs_done == 2
    with pytest.raises(ValueError, match="differs"):
        training.restore_full_state(trainer, source, {"dataset": "other"})
    del state["optimizer"]
    with pytest.raises(ValueError, match="Full resume"):
        training.restore_full_state(trainer, source, identity)


def test_restore_does_not_ignore_incompatible_optimizer(tmp_path, monkeypatch):
    class Broken:
        def load_state_dict(self, *args, **kwargs):
            raise RuntimeError("incompatible optimizer")

    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"checkpoint")
    state = {"model": {}, "optimizer": {"state": {1: {}}}, "scheduler": {}, "scaler": {},
             "config": {}, "step": 252, "epoch": 1,
             "ciel_resume": {"identity": {}, "optimizer_boundary": True}}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: state))
    trainer = SimpleNamespace(model=SimpleNamespace(load_state_dict=lambda *a, **k: None),
                              optimizer=Broken())
    with pytest.raises(RuntimeError, match="incompatible optimizer"):
        training.restore_full_state(trainer, source, {})


@pytest.mark.parametrize("member", ["../escape", "/tmp/escape", "a\\b"])
def test_untrusted_archive_paths_rejected(tmp_path, member, archive_unpacker):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "unsafe")
    with pytest.raises(ValueError):
        archive_unpacker(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("alias", ["./model.pth", "nested/../model.pth", "model.pth/."])
def test_archive_rejects_paths_that_alias_an_existing_member(tmp_path, alias, archive_unpacker):
    archive = tmp_path / "aliases.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("model.pth", "first checkpoint")
        handle.writestr(alias, "replacement checkpoint")
    with pytest.raises(ValueError):
        archive_unpacker(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_export_includes_inference_assets_but_not_full_optimizer(tmp_path, monkeypatch):
    workspace, models, dataset = (tmp_path / n for n in ("workspace", "models", "dataset"))
    checkpoint = workspace / "checkpoints/native/run/best_model_369.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"native model + large Adam states")
    pointer = {"path": str(checkpoint.relative_to(workspace)), "sha256": digest(checkpoint),
               "step": 369, "epoch": 1}
    for kind in ("best", "latest"):
        dump(workspace / f"checkpoints/{kind}/checkpoint.json", pointer)
    dump(workspace / "training-summary.json", {"status": "awaiting_listening"})
    for name in ("config.json", "vocab.json", "speakers_xtts.pth", "LICENSE.txt",
                 "training-inventory.json"):
        models.mkdir(exist_ok=True)
        (models / name).write_text("pinned asset")
    (dataset / "evaluation").mkdir(parents=True)
    (dataset / "evaluation/reference.wav").write_bytes(b"unchanged reference")
    (workspace / "samples/step-00000369").mkdir(parents=True)
    (workspace / "samples/step-00000369/short.wav").write_bytes(b"real audio fixture")

    def inference(source, output):
        output.write_bytes(b"model only")
        dump(output.with_suffix(".json"), {"optimizer_omitted": True})

    monkeypatch.setattr(artifacts, "inference_checkpoint", inference)
    archive = artifacts.export_cloud(workspace, models, dataset, tmp_path / "export")
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert "final/model.pth" in names and "final/vocab.json" in names
        assert "samples/step-00000369/short.wav" in names
        assert not any("best_model_" in n for n in names)
    manifest = artifacts.import_experiment(archive, tmp_path / "registered")
    assert json.loads(manifest.read_text())["production_enabled"] is False
    recovery = artifacts.export_cloud(workspace, models, dataset, tmp_path / "recovery", True)
    with zipfile.ZipFile(recovery) as handle:
        assert len([n for n in handle.namelist() if n.endswith(".pth")]) == 1
    with pytest.raises(FileExistsError):
        artifacts.import_experiment(archive, tmp_path / "registered")


def test_notebook_compiles_and_contains_guards_resume_and_export(tmp_path):
    notebook = cloud.notebook(tmp_path / "xtts_ciel.ipynb")
    document = json.loads(notebook.read_text())
    assert len({c["id"] for c in document["cells"]}) == len(document["cells"])
    text = "\n".join("".join(c["source"]) for c in document["cells"])
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
    for token in ("3.12.14", "xtts_cloud.lock.txt", "--resume", "15 GiB", "STOP",
                  "CIEL_TRAINING_STORAGE", "cloud-check", "export-cloud", "--recovery"):
        assert token in text
    assert len(EVALUATION_TEXTS) == 7
