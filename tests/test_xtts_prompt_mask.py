"""Regression for an all-ignored mel target caused by mismatched prompt-mask units."""

from types import SimpleNamespace

import pytest

from scripts.tts_finetune.prompt_mask import corrected_forward

SOURCE = """
def forward(self, cond_idxs):
    idx = 0
    cond_start = cond_idxs[idx, 0]
    cond_end = cond_idxs[idx, 1]
    return cond_start, cond_end
"""


class Indices:
    def __getitem__(self, index):
        return (0, 33075 // 256)[index[1]]


def test_prompt_loss_bounds_use_audio_code_stride_and_leave_valid_targets():
    fixed = corrected_forward(SOURCE, {})
    model = SimpleNamespace(use_perceiver_resampler=True,
                            perceiver_cond_length_compression=256, code_stride_len=1024)
    original_bounds = (0, 33075 // 256)
    repaired = fixed(model, Indices())
    assert repaired == (0, 33075 // 1024)
    target_count = (66150 + 1023) // 1024 + 5
    assert original_bounds[1] >= target_count  # Original mask removes EVERY mel loss target.
    assert repaired[1] < target_count  # Audio suffix and stop tokens remain supervised.


def test_non_perceiver_bounds_remain_unchanged():
    model = SimpleNamespace(use_perceiver_resampler=False)
    assert corrected_forward(SOURCE, {})(model, Indices()) == (0, 33075 // 256)


def test_unexpected_upstream_source_aborts_instead_of_partial_patching():
    with pytest.raises(ValueError, match="exactly the two"):
        corrected_forward(SOURCE.replace("cond_end =", "other ="), {})
