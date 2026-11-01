"""Fixed-voice workers should retain conditioning, not reference-only neural networks."""

import weakref
from types import SimpleNamespace

import pytest

from ciel.audio.turbo_worker import release_reference_encoders


class Tensor:
    device = "cuda:0"
    dtype = "float32"

    def __init__(self, size=10, **kwargs):
        self.size = size
        self.device = kwargs.get("device", self.device)
        self.dtype = kwargs.get("dtype", self.dtype)

    def numel(self):
        return self.size

    def element_size(self):
        return 4


class Module:
    def __init__(self):
        self.weights = [Tensor()]

    def parameters(self):
        return iter(self.weights)

    def buffers(self):
        return iter([])

    def register_parameter(self, name, value):
        self.weights = [value]
        setattr(self, name, value)


def fake_model():
    torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module, Parameter=lambda value, **_: value),
        empty=lambda size, **kwargs: Tensor(size, **kwargs),
    )
    conditioning = object()
    model = SimpleNamespace(
        ve=Module(), s3gen=SimpleNamespace(tokenizer=Module(), speaker_encoder=Module()),
        conds=conditioning,
    )
    return model, torch


def test_release_frees_all_three_networks_but_retains_device_and_conditioning():
    model, torch = fake_model()
    conditioning = model.conds
    old = [weakref.ref(module) for module in (
        model.ve, model.s3gen.tokenizer, model.s3gen.speaker_encoder)]
    old_weight = weakref.ref(next(model.s3gen.tokenizer.parameters()))
    result = release_reference_encoders(model, torch)
    assert result["reference_encoder_bytes_released"] == 120
    assert model.conds is conditioning
    assert all(reference() is None for reference in old)
    assert old_weight() is None, "The device anchor retained an original reference tensor"
    assert model.ve is None and model.s3gen.speaker_encoder is None
    anchor = next(model.s3gen.tokenizer.parameters())
    assert anchor.device == "cuda:0" and anchor.dtype == "float32"
    assert anchor.numel() == 0


def test_reencoding_after_release_is_explicitly_rejected():
    model, torch = fake_model()
    release_reference_encoders(model, torch)
    with pytest.raises(RuntimeError, match="prepared"):
        model.s3gen.tokenizer.forward(None)


def test_unprepared_reference_never_releases_encoders():
    model, torch = fake_model()
    original = model.s3gen.tokenizer
    model.conds = None
    with pytest.raises(ValueError, match="conditioning"):
        release_reference_encoders(model, torch)
    assert model.s3gen.tokenizer is original
    assert model.ve is not None


def test_unsupported_encoder_layout_fails_before_any_mutation():
    model, torch = fake_model()
    original = model.s3gen.tokenizer
    model.s3gen.speaker_encoder = None
    with pytest.raises((ValueError, AttributeError)):
        release_reference_encoders(model, torch)
    assert model.s3gen.tokenizer is original
    assert model.ve is not None
