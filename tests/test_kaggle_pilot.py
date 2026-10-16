"""Regression checks for pilot disk, atomic replacement and compact-best semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tts_finetune import kaggle_storage, training, xtts


@pytest.mark.parametrize("pilot,name,total_mib,free_mib,allowed", [
    (True, "Tesla T4", 15095, 14000, True),
    (True, "Tesla T4", 14848, 13000, True),
    (True, "Tesla T4", 14847, 13000, False),
    (True, "Tesla T4", 15095, 12000, False),
    (False, "Tesla T4", 15095, 14000, False),
    (True, "Other GPU", 15095, 14000, False),
    (True, "GTX 1660 SUPER", 6144, 6000, False),
])
def test_cuda_t4_driver_reservation_is_bounded(monkeypatch, pilot, name, total_mib,
                                               free_mib, allowed):
    if pilot:
        monkeypatch.setenv("CIEL_KAGGLE_PILOT", "1")
    else:
        monkeypatch.delenv("CIEL_KAGGLE_PILOT", raising=False)
    mib = 1024**2
    cuda = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(name=name, total_memory=total_mib*mib),
        mem_get_info=lambda index: (free_mib*mib, total_mib*mib))
    physical = {"name": name, "total_mib": 15360, "free_mib": 14912}
    if allowed:
        assert xtts.cuda_device_preflight(SimpleNamespace(cuda=cuda), physical)[
            "bounded_t4_driver_reservation_allowance"] is True
    else:
        with pytest.raises(RuntimeError, match="CUDA capacity/free memory rejected"):
            xtts.cuda_device_preflight(SimpleNamespace(cuda=cuda), physical)


def test_disk_floor_and_full_replacement_reservation(tmp_path, monkeypatch):
    free = 3 * kaggle_storage.GIB
    monkeypatch.setattr(
        kaggle_storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=free)
    )
    kaggle_storage.require_space(tmp_path)
    with pytest.raises(RuntimeError, match="previous committed full"):
        kaggle_storage.require_space(tmp_path, 1)
    free -= 1
    with pytest.raises(RuntimeError, match="3 GiB reserve"):
        kaggle_storage.require_space(tmp_path)


def test_native_infinite_config_limits_have_strict_json_resume_identity():
    config = {"epochs": 3, "output_path": "/output", "max_audio_len": float("inf"),
              "max_text_len": float("inf"), "datasets": [], "test_sentences": [],
              "model_args": {k: "/assets/" + k for k in
                  ("mel_norm_file", "dvae_checkpoint", "xtts_checkpoint", "tokenizer_file")}}
    identity = training.recipe_identity(config)
    encoded = json.dumps(identity, allow_nan=False)
    assert json.loads(encoded)["max_audio_len"] == {"native_nonfinite_float": "Infinity"}
    assert config["max_audio_len"] == float("inf")
    config["epochs"] = 6
    config["output_path"] = "/another/output"
    assert training.recipe_identity(config) == identity


def test_low_disk_stops_before_another_optimizer_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("CIEL_KAGGLE_PILOT", "1")
    monkeypatch.setattr(
        kaggle_storage.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=2 * kaggle_storage.GIB),
    )
    monitor = training.Monitor(tmp_path)
    with pytest.raises(RuntimeError, match="storage stop"):
        monitor.step(SimpleNamespace())
    assert monitor.batch == 0


def test_nonfinite_forward_stops_before_backward_and_records_exact_loss(tmp_path):
    monitor = training.Monitor(tmp_path)
    trainer = SimpleNamespace(epochs_done=1, total_steps_done=0)
    with pytest.raises(RuntimeError, match="before backward"):
        monitor.finite_forward_loss(trainer, {"loss_text_ce": float("nan"),
                                             "loss_mel_ce": float("inf")})
    record = json.loads((tmp_path / "nonfinite-forward-loss.json").read_text())
    assert record["step"] == 0 and record["batch_in_epoch"] == 1
    assert record["invalid_losses"] == {"loss_text_ce": "nan", "loss_mel_ce": "inf"}
    assert record["stage"] == "before backward and optimizer update"


def test_same_step_commit_failure_preserves_previous_full_pointer(tmp_path, logger, monkeypatch):
    checkpoint = tmp_path / "checkpoints/native/run/checkpoint_369.pth"
    logger.save_model(full_state(369), str(checkpoint))
    pointer = tmp_path / "checkpoints/latest/checkpoint.json"
    previous = pointer.read_bytes(), checkpoint.read_bytes()
    original_dump = training.dump

    def fail_commit(path, value):
        if path == pointer:
            raise OSError("simulated pointer commit failure")
        original_dump(path, value)

    monkeypatch.setattr(training, "dump", fail_commit)
    changed = full_state(369)
    changed["model"]["xtts.gpt.weight"] = [3, 4]
    with pytest.raises(OSError, match="pointer commit"):
        logger.save_model(changed, str(checkpoint))
    assert (pointer.read_bytes(), checkpoint.read_bytes()) == previous


def test_pilot_will_not_commit_an_empty_optimizer_as_resumable(tmp_path, logger):
    path = tmp_path / "checkpoints/native/run/checkpoint_252.pth"
    state = full_state(252)
    state["optimizer"]["state"] = {}
    with pytest.raises(RuntimeError, match="non-resumable pilot checkpoint"):
        logger.save_model(state, str(path))
    assert not path.exists()
    assert not (tmp_path / "checkpoints/latest/checkpoint.json").exists()


@pytest.fixture
def logger(tmp_path, monkeypatch):
    monkeypatch.setenv("CIEL_KAGGLE_PILOT", "1")
    monkeypatch.setattr(
        kaggle_storage.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=100 * kaggle_storage.GIB),
    )

    class Dashboard:
        def __init__(self, *args):
            pass

    module = SimpleNamespace(TensorboardLogger=Dashboard)
    monkeypatch.setitem(sys.modules, "trainer.logging.tensorboard_logger", module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            save=lambda state, handle: handle.write(json.dumps(state).encode())
        ),
    )
    monitor = training.Monitor(tmp_path)
    value = training.logger_class()(tmp_path, monitor, {"recipe": "test"})
    value.trainer = SimpleNamespace(best_loss={"eval_loss": 1.0})
    return value


def full_state(step):
    return {
        "model": {"xtts.gpt.weight": [1, 2]},
        "optimizer": {"state": {"moments": 3}},
        "scheduler": {"last_epoch": step},
        "scaler": {"scale": 2},
        "config": {},
        "epoch": 1,
        "step": step,
    }


def test_failed_replacement_preserves_full_state_and_pointer(
    tmp_path, monkeypatch, logger
):
    first = tmp_path / "checkpoints/native/run/checkpoint_252.pth"
    logger.save_model(full_state(252), str(first))
    pointer = tmp_path / "checkpoints/latest/checkpoint.json"
    before = pointer.read_bytes(), first.read_bytes()

    def fail(state, handle):
        handle.write(b"partial write")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(sys.modules["torch"], "save", fail)
    second = first.with_name("checkpoint_369.pth")
    with pytest.raises(OSError, match="simulated"):
        logger.save_model(full_state(369), str(second))
    assert (pointer.read_bytes(), first.read_bytes()) == before
    assert not second.exists() and not Path(str(second) + ".tmp").exists()


def test_prune_happens_after_new_pointer_commit(tmp_path, logger, monkeypatch):
    first = tmp_path / "checkpoints/native/run/checkpoint_252.pth"
    second = first.with_name("checkpoint_369.pth")
    logger.save_model(full_state(252), str(first))
    original_prune = kaggle_storage.prune_committed_periodic

    def observe(output, current):
        pointer = json.loads(
            (output / "checkpoints/latest/checkpoint.json").read_text()
        )
        assert first.exists() and second.exists()
        assert (output / pointer["path"]) == second
        assert pointer["full_optimizer_resume"] is True
        original_prune(output, current)

    monkeypatch.setattr(kaggle_storage, "prune_committed_periodic", observe)
    logger.save_model(full_state(369), str(second))
    assert second.exists() and not first.exists()
    restored = json.loads(second.read_text())
    assert set(("optimizer", "scheduler", "scaler", "ciel_resume")) <= set(restored)


def test_compact_best_does_not_replace_latest_full_resume(tmp_path, logger):
    latest = tmp_path / "checkpoints/native/run/checkpoint_369.pth"
    logger.save_model(full_state(369), str(latest))
    pointer = tmp_path / "checkpoints/latest/checkpoint.json"
    before = pointer.read_bytes()
    best = tmp_path / "checkpoints/best-inference/best_model_inference.pth"
    logger.save_model(full_state(369), str(best))
    assert set(json.loads(best.read_text())) == {"model", "epoch", "step"}
    assert pointer.read_bytes() == before and latest.exists()
    assert (
        json.loads((tmp_path / "checkpoints/best/checkpoint.json").read_text())[
            "full_optimizer_resume"
        ]
        is False
    )
