"""CPU-only Nano experiment and fair audition checks without downloaded models."""

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from scripts import chatterbox_nano_audition as nano


@pytest.fixture
def native(tmp_path, monkeypatch):
    # The real adapter changes its isolated process environment before importing
    # Torch. Record those keys here so mocked, in-process tests restore them.
    for name in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "")
    directory, cache = tmp_path / "model", tmp_path / "cache"
    directory.mkdir()
    (directory / "inventory.json").write_text('{"revision":"test"}')
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"unchanged reference")
    state = SimpleNamespace(to=Mock(), save=Mock(side_effect=lambda p: p.write_bytes(b"state")))
    state.to.return_value = state
    module = SimpleNamespace(parameters=lambda: [SimpleNamespace(
        device=SimpleNamespace(type="cpu"))])
    wave = Mock()
    wave.detach.return_value.cpu.return_value.numpy.return_value = np.ones(2400) * .1
    model = SimpleNamespace(t3=module, s3gen=module, ve=module, conds=state, sr=24000,
                            prepare_conditionals=Mock(), generate=Mock(return_value=wave))
    factory = Mock(return_value=model)
    loader = Mock(return_value=state)
    torch = SimpleNamespace(__version__="test", cuda=SimpleNamespace(
        is_initialized=Mock(return_value=False)), set_num_threads=Mock(),
        inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "chatterbox", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", SimpleNamespace(
        ChatterboxTurboTTS=SimpleNamespace(from_local=factory),
        Conditionals=SimpleNamespace(load=loader)))
    monkeypatch.setattr(nano, "version", lambda _name: "test")
    return SimpleNamespace(directory=directory, cache=cache, reference=reference, model=model,
                           factory=factory, loader=loader, torch=torch)


def test_native_api_cpu_model_and_conditioning_reused(native, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    provider = nano.NanoProvider(native.directory, native.reference, native.cache)
    native.factory.assert_called_once_with(str(native.directory), device="cpu", nano=True)
    native.model.prepare_conditionals.assert_called_once_with(
        str(native.reference), exaggeration=0.0)
    assert provider.conditioning_encoded
    for text in ("One.", "Two."):
        [(wave, rate)] = provider.stream(text)
        assert wave.size == 2400 and rate == 24000
    native.model.prepare_conditionals.assert_called_once()
    assert provider.conditioning_loads == 1
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == ""
    provider.close()
    assert provider.model is None
    cached = nano.NanoProvider(native.directory, native.reference, native.cache)
    assert not cached.conditioning_encoded
    native.model.prepare_conditionals.assert_called_once()
    native.loader.assert_called_once_with(Path(cached.state_path), map_location="cpu")


@pytest.mark.parametrize("changed", ["reference", "model"])
def test_cache_invalidates_when_voice_or_model_changes(native, changed):
    first = nano.NanoProvider(native.directory, native.reference, native.cache)
    path = native.reference if changed == "reference" else native.directory / "inventory.json"
    path.write_bytes(b"different")
    second = nano.NanoProvider(native.directory, native.reference, native.cache)
    assert first.state_path != second.state_path
    assert native.model.prepare_conditionals.call_count == 2


def test_cuda_initialized_process_is_rejected_before_model_loading(native):
    native.torch.cuda.is_initialized.return_value = True
    with pytest.raises(RuntimeError, match="fresh CPU-only"):
        nano.NanoProvider(native.directory, native.reference, native.cache)
    native.factory.assert_not_called()


def test_cuda_parameters_are_rejected(native):
    native.model.t3 = SimpleNamespace(parameters=lambda: [SimpleNamespace(
        device=SimpleNamespace(type="cuda"))])
    with pytest.raises(RuntimeError, match="every model parameter on CPU"):
        nano.NanoProvider(native.directory, native.reference, native.cache)
    native.model.prepare_conditionals.assert_not_called()


def test_synthesis_errors_propagate_without_fallback(native):
    provider = nano.NanoProvider(native.directory, native.reference, native.cache)
    native.model.generate.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        list(provider.stream("Test."))


def test_missing_model_download_and_invalid_benchmark_configuration(tmp_path):
    with pytest.raises(FileNotFoundError, match="download first"):
        nano.inventory(tmp_path)
    for iterations, threads in ((1, 4), (3, 0), (3, 13)):
        with pytest.raises(ValueError, match="three warm"):
            nano.run(tmp_path / "output", iterations, threads)


def test_level_matching_preserves_raw_audio_and_creates_complete_blind_rounds(
    tmp_path, monkeypatch,
):
    from scripts.tts_finetune import evaluate

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    reference = tmp_path / "data/voice/references/ciel/reference-1.wav"
    reference.parent.mkdir(parents=True)
    sf.write(reference, np.sin(np.arange(24000)) * .1, 24000)
    monkeypatch.setattr(nano, "BASELINE", baseline)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "pyloudnorm", SimpleNamespace(Meter=lambda rate:
        SimpleNamespace(integrated_loudness=lambda wave: 20 * np.log10(
            np.sqrt(np.mean(wave**2))))))
    source = tmp_path / "nano"
    source.mkdir()
    value = {"status": "complete_awaiting_listening", "label": "Nano CPU",
             "reference_sha256": nano.digest(reference), "texts": nano.EVALUATION_TEXTS,
             "rows": []}
    manifest = {"texts": nano.EVALUATION_TEXTS, "reference_sha256": nano.digest(reference),
                "audio": {}}
    original_hashes = {}
    for name, text in nano.EVALUATION_TEXTS.items():
        for directory, amplitude in ((source, .4), (baseline, .1)):
            path = directory / f"{name}.wav"
            sf.write(path, np.sin(np.arange(24000)) * amplitude, 24000, subtype="FLOAT")
            original_hashes[path] = nano.digest(path)
        value["rows"].append({"sentence": name, "text": text, "iteration": 1,
                              "audio": str(source / f"{name}.wav")})
        manifest["audio"][name] = {"file": f"{name}.wav",
                                   "sha256": nano.digest(baseline / f"{name}.wav")}
    nano.dump(baseline / "manifest.json", manifest)
    nano.dump(source / "result.json", value)
    output = tmp_path / "comparison"
    html = nano.page(source / "result.json", output).read_text()
    assert "Nano CPU" not in html.split("<script>")[0]
    rounds = json.loads((output / "listening/assignments.json").read_text())["rounds"]
    assert len(rounds) == 7 and all(len(r["samples"]) == 2 for r in rounds)
    for name in nano.EVALUATION_TEXTS:
        quiet, _ = sf.read(output / "matched/turbo" / f"{name}.wav")
        loud, _ = sf.read(output / "matched/nano" / f"{name}.wav")
        assert np.allclose(quiet, loud, atol=1e-6)
        assert np.max(np.abs(loud)) < .11
    assert all(nano.digest(path) == checksum for path, checksum in original_hashes.items())
    with pytest.raises(FileExistsError):
        nano.page(source / "result.json", output)
