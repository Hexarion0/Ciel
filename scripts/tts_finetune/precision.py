"""A recorded FP32 autocast region for the proven XTTS Perceiver FP16 overflow."""

from __future__ import annotations

POLICY = {"conditioning_perceiver": "fp32_autocast_disabled_v1",
          "remaining_training": "native_fp16_amp", "parameter_dtype": "fp32",
          "optimizer_scheduler_scaler": "unchanged"}


def install_fp32_perceiver(model) -> dict:
    """Retain every native parameter and state key; change this forward's arithmetic only."""
    import torch

    perceiver = model.xtts.gpt.conditioning_perceiver
    if getattr(perceiver, "_ciel_fp32_forward", False):
        return POLICY.copy()
    original = perceiver.forward

    def cast(value):
        if isinstance(value, torch.Tensor):
            return value.float() if value.is_floating_point() else value
        if isinstance(value, tuple):
            return tuple(cast(v) for v in value)
        if isinstance(value, list):
            return [cast(v) for v in value]
        if isinstance(value, dict):
            return {k: cast(v) for k, v in value.items()}
        return value

    def forward(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return original(*cast(args), **cast(kwargs))

    perceiver.forward = forward
    perceiver._ciel_fp32_forward = True
    return POLICY.copy()
