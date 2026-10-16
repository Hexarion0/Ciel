"""Compare forward paths on Kaggle without backward calls or optimizer updates."""

from __future__ import annotations

import copy
import math


def inspect_first_batch(trainer) -> dict:
    import torch

    model = trainer._get_model()
    loader = trainer.get_train_dataloader(trainer.training_assets, trainer.train_samples,
                                          verbose=True)
    batch = trainer.format_batch(next(iter(loader)))
    versions = {name: p._version for name, p in model.named_parameters()}
    result = {"optimizer_updates": 0, "backward_calls": 0, "variants": [],
              "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "finite": bool(torch.isfinite(value).all())}
                  for name, value in batch.items() if isinstance(value, torch.Tensor)}}
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()

    for training, amp, masking, stable in [
            (True, True, True, False), (True, False, True, False),
            (True, True, False, False), (False, True, True, False),
            (False, False, True, False), (True, True, True, True),
            (False, True, True, True)]:
        if stable:
            from .precision import install_fp32_perceiver

            install_fp32_perceiver(model)
        model.eval()
        model.xtts.gpt.train(training)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        copied = copy.deepcopy(batch)
        if not masking:
            copied["cond_idxs"] = None
        invalid = []

        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from tensors(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from tensors(item)

        def hook(name):
            def check(module, inputs, outputs):
                if len(invalid) < 8:
                    for tensor in tensors(outputs):
                        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                            invalid.append({"module": name, "dtype": str(tensor.dtype),
                                            "shape": list(tensor.shape)})
                            break
            return check

        handles = [module.register_forward_hook(hook(name))
                   for name, module in model.xtts.gpt.named_modules()]
        row = {"gpt_training": training, "fp16_autocast": amp, "conditioning_mask": masking,
               "fp32_perceiver": stable}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, losses = model.train_step(copied, None)
            row["losses"] = {
                name: float(value) if math.isfinite(float(value)) else str(float(value))
                for name, value in losses.items()}
            row["all_losses_finite"] = all(math.isfinite(float(v)) for v in losses.values())
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            for handle in handles:
                handle.remove()
        row["first_nonfinite_module_outputs"] = invalid
        result["variants"].append(row)
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise RuntimeError("Diagnostic unexpectedly changed model parameters")
    if trainer.optimizer.state or trainer.total_steps_done != 0:
        raise RuntimeError("Diagnostic unexpectedly allocated optimizer state or advanced a step")
    result["parameters_unchanged"] = True
    return result


def inspect_all_training_clips(trainer) -> dict:
    """Visit every verified train clip with deterministic prompts; never optimize."""
    import torch

    model = trainer._get_model()
    versions = {name: p._version for name, p in model.named_parameters()}
    model.eval()
    model.xtts.gpt.train()
    loader = trainer.get_eval_dataloader(trainer.training_assets, trainer.train_samples,
                                         verbose=True)
    rows = []
    for index, raw in enumerate(loader):
        batch = trainer.format_batch(raw)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _, losses = model.train_step(batch, None)
        values = {name: float(value) for name, value in losses.items()}
        if any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Non-finite diagnostic training forward at clip {index}: {values}")
        rows.append({"clip": index, "losses": values})
        if index % 50 == 0:
            print(f"Forward-only validation: {index + 1}/{len(loader)} finite", flush=True)
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise RuntimeError("Diagnostic changed parameters")
    if trainer.optimizer.state or trainer.total_steps_done != 0:
        raise RuntimeError("Diagnostic unexpectedly advanced optimizer state")
    return {"mode": "all_verified_training_clips_forward_only", "clips": len(rows),
            "all_losses_finite": True, "optimizer_updates": 0, "backward_calls": 0,
            "parameters_unchanged": True, "rows": rows}
