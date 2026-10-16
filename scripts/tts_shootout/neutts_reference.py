"""One-time native NeuCodec reference encoding with pinned, local checkpoint loading."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from .common import digest, dump


def encode_reference_cpu(reference: Path, inventory: dict, receipt: Path):
    """Preserve native encoding; adapt checkpoint boot to Hub 1.x's removed keywords."""
    import inspect

    import librosa
    import neucodec.model
    import torch
    from neucodec import NeuCodec

    # NeuCodec 0.0.6's from_pretrained passes removed proxies/resume_download
    # keywords to Hub 1.x (required by NeuTTS's Transformers 5.1). Load its pinned
    # checkpoint locally using exactly its native constructor/filter/load logic.
    directory = Path(inventory["assets"]["neuphonic/neucodec"]["directory"])
    codec = NeuCodec(24000, 480)
    weights = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    weights = {key: value for key, value in weights.items()
               if not any(ignored in key for ignored in ("fc_post_s", "SemanticDecoder"))}
    status = codec.load_state_dict(weights, strict=False)
    codec.eval().to("cpu")
    wave, _ = librosa.load(reference, sr=16000, mono=True)
    with torch.inference_mode():
        codes = codec.encode_code(torch.from_numpy(wave).float()[None, None, :])
        codes = codes.squeeze(0).squeeze(0).detach().cpu().contiguous()
    dump(receipt, {
        "source_sha256": digest(Path(inspect.getfile(neucodec.model))),
        "reference_sha256": digest(reference), "device": "cpu", "sample_rate": 16000,
        "boot_adaptation": "local checkpoint; native constructor/filter/load logic",
        "encoding_inference_modified": False, "missing_keys": status.missing_keys,
        "unexpected_keys": status.unexpected_keys, "codes": codes.numel(),
    })
    del weights, codec
    gc.collect()
    return codes


def main():
    """Prepare the comparison's reference codes once, before timed cache-hit startup."""
    import argparse

    from safetensors.torch import save_file

    from .common import ROOT
    from .providers import cache_key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "data/voice/references/ciel/reference-1.wav")
    args = parser.parse_args()
    inventory = ROOT / "data/voice/models/neutts-air/inventory.json"
    cache = ROOT / "data/voice/states/shootout"
    cache.mkdir(parents=True, exist_ok=True)
    key = cache_key(args.reference, {"implementation": "neutts-1.4.1", "device": "cpu"}, inventory)
    state = cache / f"neutts-{key}.safetensors"
    if state.is_file():
        print("Retaining existing encoded reference:", state)
        return
    started = time.perf_counter()
    import torch

    torch.set_num_threads(4)
    codes = encode_reference_cpu(args.reference, json.loads(inventory.read_text()),
                                 state.with_suffix(".json"))
    save_file({"codes": codes}, str(state))
    print("Reference preparation:", round(time.perf_counter() - started, 2), "seconds", state)


if __name__ == "__main__":
    main()
