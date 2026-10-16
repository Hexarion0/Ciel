"""Correct native 0.27.5 prompt-mask units without disabling masking or changing weights."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import MethodType

UPSTREAM_SHA256 = "7f2d3269741334f8164a2b71496e110e02f7a08d846339b9ef1ee655b0194c6a"
POLICY = {"patch": "native_gpt_mel_mask_units_v1", "upstream_forward_sha256": UPSTREAM_SHA256,
          "conditioning_masking": True, "mel_target_stride_samples": 1024,
          "perceiver_stride_samples": 256, "parameter_state_keys": "unchanged"}


def corrected_forward(source: str, namespace: dict):
    """Change only the two mel-target bounds, after native conditioning-mask calculation."""
    tree = ast.parse(textwrap.dedent(source))
    changed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {"cond_start", "cond_end"}:
            continue
        if not (isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name) and node.value.value.id == "cond_idxs"):
            raise ValueError("Unexpected upstream mask-bound expression")
        original = node.value
        node.value = ast.IfExp(
            test=ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                               attr="use_perceiver_resampler", ctx=ast.Load()),
            body=ast.BinOp(left=ast.BinOp(
                left=original, op=ast.Mult(), right=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="perceiver_cond_length_compression", ctx=ast.Load())),
                op=ast.FloorDiv(), right=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="code_stride_len", ctx=ast.Load())),
            orelse=original)
        changed.append(target.id)
    if sorted(changed) != ["cond_end", "cond_start"]:
        raise ValueError("Expected exactly the two native mel-mask bounds")
    ast.fix_missing_locations(tree)
    scope = dict(namespace)
    exec(compile(tree, "<ciel-native-gpt-mask-unit-fix>", "exec"), scope)
    return scope["forward"]


def install_prompt_mask_fix(model) -> dict:
    gpt = model.xtts.gpt
    if getattr(gpt, "_ciel_mask_units_fixed", False):
        return POLICY.copy()
    original = gpt.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Native GPT.forward differs from reviewed 0.27.5 source; refusing patch")
    gpt.forward = MethodType(corrected_forward(source, original.__globals__), gpt)
    gpt._ciel_mask_units_fixed = True
    return POLICY.copy()
