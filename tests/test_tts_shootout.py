"""Shootout correctness without loading models or initializing a GPU."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.tts_shootout.common import (
    CONFIGURATIONS,
    SENTENCES,
    digest,
    dump,
    evaluation_python,
)
from scripts.tts_shootout.page import render
from scripts.tts_shootout.providers import cache_key
from scripts.tts_shootout.telemetry import Sampler
from scripts.tts_shootout.worker import measure


class DummySampler:
    metrics = {"rss_mib_peak": 42}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def test_stream_measurement_keeps_every_chunk_and_preserves_audio(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.tts_shootout.worker.Sampler", DummySampler)
    chunks = [np.ones(1000, dtype=np.float32) * level for level in (0.1, 0.2, 0.3)]
    provider = SimpleNamespace(stream=lambda _text: ((chunk, 24000) for chunk in chunks))
    destination = tmp_path / "sample.wav"
    result = measure(provider, "test", destination)
    audio, rate = sf.read(destination)
    assert rate == 24000 and np.allclose(audio, np.concatenate(chunks))
    assert result["chunks"] == 3 and result["audio_seconds"] == 0.125
    assert 0 <= result["first_audio_seconds"] <= result["synthesis_seconds"]
    assert result["resources"]["rss_mib_peak"] == 42
    before = digest(destination)
    with pytest.raises(FileExistsError):
        measure(provider, "test", destination)
    assert digest(destination) == before


@pytest.mark.parametrize("waves", [[], [np.array([np.nan])], [np.array([10.0])]])
def test_failed_inference_cannot_create_a_listening_sample(tmp_path, monkeypatch, waves):
    monkeypatch.setattr("scripts.tts_shootout.worker.Sampler", DummySampler)
    provider = SimpleNamespace(stream=lambda _text: ((wave, 24000) for wave in waves))
    path = tmp_path / "invalid.wav"
    with pytest.raises(ValueError):
        measure(provider, "test", path)
    assert not path.exists()


def test_conditioning_cache_does_not_mix_recordings_models_or_devices(tmp_path):
    reference, inventory = tmp_path / "ref.wav", tmp_path / "model.json"
    reference.write_bytes(b"original reference")
    dump(inventory, {"revision": "one"})
    first = cache_key(reference, {"device": "cpu"}, inventory)
    assert cache_key(reference, {"device": "cuda"}, inventory) != first
    dump(inventory, {"revision": "two"})
    assert cache_key(reference, {"device": "cpu"}, inventory) != first
    reference.write_bytes(b"different reference")
    assert cache_key(reference, {"device": "cpu"}, inventory) != first


def test_blind_assignments_are_persisted_and_names_are_hidden(tmp_path):
    dump(tmp_path / "manifest.json", {
        "run_id": "test", "configurations": CONFIGURATIONS, "rounds": [],
    })
    for identifier in ("style-cpu-5", "xtts-cuda-fp32", "chatterbox-baseline"):
        rows = []
        for name in SENTENCES:
            audio = tmp_path / identifier / f"{name}.wav"
            audio.parent.mkdir(exist_ok=True)
            sf.write(audio, np.ones(2400) * 0.1, 24000)
            rows.append({"sentence": name, "iteration": 1, "audio": str(audio)})
        dump(tmp_path / f"{identifier}.json", {"status": "ok", "rows": rows})
    html = render(tmp_path).read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["rounds"]) == 7
    assignments = json.dumps(manifest["rounds"], sort_keys=True)
    visible = html.split("<script>")[0]
    for engine in ("StyleTTS2", "XTTS-v2", "Chatterbox"):
        assert engine not in visible
    for round_ in manifest["rounds"]:
        assert [sample["label"] for sample in round_["samples"]] == list("ABC")
        for sample in round_["samples"]:
            assert digest(tmp_path / sample["audio"]) == sample["audio_sha256"]
    render(tmp_path)
    assert json.dumps(json.loads((tmp_path / "manifest.json").read_text())["rounds"],
                      sort_keys=True) == assignments


def test_incomplete_or_failed_engine_is_not_substituted_in_main_round(tmp_path):
    dump(tmp_path / "manifest.json", {
        "run_id": "test", "configurations": CONFIGURATIONS, "rounds": [],
    })
    dump(tmp_path / "xtts-cuda-fp32.json", {"status": "error", "error": "failed", "rows": []})
    page = render(tmp_path).read_text()
    assert "still being generated" in page
    assert not json.loads((tmp_path / "manifest.json").read_text())["rounds"]


def test_different_speaker_reference_cannot_enter_an_apples_to_apples_round(tmp_path):
    dump(tmp_path / "manifest.json", {
        "run_id": "test", "configurations": CONFIGURATIONS, "rounds": [],
        "reference_sha256": "same-reference-required",
    })
    dump(tmp_path / "xtts-cuda-fp32.json", {
        "status": "ok", "reference_sha256": "different-reference", "rows": [],
    })
    with pytest.raises(ValueError, match="different reference"):
        render(tmp_path)


def test_gpu_telemetry_keeps_partial_metrics_and_attributes_only_test_vram(monkeypatch):
    def process(pid, rss):
        return SimpleNamespace(pid=pid, memory_info=lambda: SimpleNamespace(rss=rss),
                               cpu_times=lambda: SimpleNamespace(user=1, system=1))

    parent, child = process(123, 8 * 1024**2), process(124, 4 * 1024**2)
    parent.children = lambda recursive: [child]
    monkeypatch.setattr("scripts.tts_shootout.telemetry.psutil.Process", lambda: parent)

    def unsupported(_handle):
        raise RuntimeError("Fan unavailable")

    api = SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _index: "GPU", NVML_TEMPERATURE_GPU=0,
        NVML_VALUE_NOT_AVAILABLE=2**64 - 1,
        nvmlDeviceGetUtilizationRates=lambda _h: SimpleNamespace(gpu=7),
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(used=1024**3),
        nvmlDeviceGetPowerUsage=lambda _h: 20000,
        nvmlDeviceGetTemperature=lambda _h, _sensor: 50,
        nvmlDeviceGetFanSpeed=unsupported,
        nvmlDeviceGetComputeRunningProcesses=lambda _h: [
            SimpleNamespace(pid=123, usedGpuMemory=256 * 1024**2),
            SimpleNamespace(pid=999, usedGpuMemory=128 * 1024**2),
        ],
    )
    monkeypatch.setitem(sys.modules, "pynvml", api)
    with Sampler(interval=1) as sampler:
        snapshot = sampler.snapshot()
    assert snapshot["engine_vram_mib"] == 256
    assert snapshot["rss_mib"] == 12 and snapshot["gpu_power_w"] == 20
    assert sampler.metrics["gpu_fan_percent_peak"] is None
    assert sampler.metrics["gpu_temp_c_peak"] == 50


def test_shootout_browser_choice_before_reveal_and_feedback():
    if not shutil.which("node"):
        pytest.skip("Node is needed for frontend behavior checks")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["node", "tests/frontend/tts_shootout.cjs"], cwd=root,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_optional_engine_uses_separate_environment():
    assert "envs/neutts/" in str(evaluation_python(CONFIGURATIONS["neutts-air-cpu-q8"]))
    assert "envs/shootout/" in str(evaluation_python(CONFIGURATIONS["style-cpu-5"]))


def test_neutts_native_cpu_conditioning_is_reused_and_closed(tmp_path, monkeypatch):
    from scripts.tts_shootout.providers import NeuTTSProvider

    calls = []
    codes = SimpleNamespace(numel=lambda: 10)
    codes.detach = codes.cpu = codes.contiguous = lambda: codes
    inventory = tmp_path / "data/voice/models/neutts-air/inventory.json"
    dump(inventory, {"assets": {
        "neuphonic/neutts-air-q8-gguf": {"directory": str(tmp_path)},
        "neuphonic/neucodec-onnx-decoder": {"directory": str(tmp_path)},
    }})
    transcript = tmp_path / "data/voice/references/ciel/reference-1.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("The actual reference transcript.")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"original speaker")

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["backbone_device"] == kwargs["codec_device"] == "cpu"
            self.backbone = SimpleNamespace(close=lambda: calls.append("closed"),
                                            n_threads=6, n_threads_batch=12)
            self.codec = SimpleNamespace(session=SimpleNamespace(
                get_providers=lambda: ["CPUExecutionProvider"],
            ))
            self.watermarker = object()

        def infer_stream(self, text, actual_codes, actual_transcript):
            assert actual_codes is codes
            assert actual_transcript == transcript.read_text()
            yield np.ones(100) * 0.1

    def encode(path, _metadata, _receipt):
        assert path == reference
        calls.append("encoded")
        return codes

    monkeypatch.setattr("scripts.tts_shootout.providers.ROOT", tmp_path)
    monkeypatch.setattr("scripts.tts_shootout.neutts_reference.encode_reference_cpu", encode)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        isfinite=lambda _codes: SimpleNamespace(all=lambda: True),
    ))
    monkeypatch.setitem(sys.modules, "neutts", SimpleNamespace(NeuTTS=Model))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(
        load_file=lambda _path: {"codes": codes},
        save_file=lambda _value, path: Path(path).write_bytes(b"cached codes"),
    ))
    for _ in range(2):
        provider = NeuTTSProvider(
            CONFIGURATIONS["neutts-air-cpu-q8"], reference, tmp_path / "cache",
        )
        assert len(list(provider.stream("First sentence."))) == 1
        assert len(list(provider.stream("Second sentence."))) == 1
        provider.close()
    assert calls == ["encoded", "closed", "closed"]
