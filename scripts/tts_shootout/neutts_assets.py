"""Provision pinned optional CPU cloning assets, outside Ciel's production dependencies."""

from __future__ import annotations

import os
from pathlib import Path

from .common import ROOT, digest, dump

MODELS = {
    "neuphonic/neutts-air-q8-gguf": "855f828ba33d1885443a7a4971785e318414dfd2",
    "neuphonic/neucodec": "a940021324df13a07f44185f0a17c6dac3698179",
    "neuphonic/neucodec-onnx-decoder": "ac01424f74ea25b245132252168e2191b3d859fa",
    "facebook/w2v-bert-2.0": "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
}


def prepare() -> Path:
    """Retain native Hub caches with verified main refs for the supported upstream loader."""
    os.environ["HF_HOME"] = str(ROOT / "data/voice/cache/neutts-hf")
    from huggingface_hub import HfApi, snapshot_download

    destination = ROOT / "data/voice/models/neutts-air/inventory.json"
    assets = {}
    for repo, revision in MODELS.items():
        if HfApi().model_info(repo).sha != revision:
            raise ValueError(f"{repo} main changed; inspect before updating pinned assets")
        patterns = ["*.gguf", "*.json", "*.bin", "*.onnx", "meta.yaml", "README.md"]
        if repo.startswith("facebook/"):
            patterns.append("model.safetensors")
        path = Path(snapshot_download(
            repo, revision="main",
            allow_patterns=patterns,
        ))
        if path.name != revision:
            raise ValueError(f"Unexpected snapshot revision for {repo}")
        assets[repo] = {"revision": revision, "directory": str(path),
                        "sha256": {file.name: digest(file) for file in path.iterdir()
                                   if file.is_file()}}
    dump(destination, {"assets": assets, "implementation": "neutts==1.4.1",
                       "codec": "neucodec==0.0.6", "backbone_precision": "Q8_0",
                       "device": "CPU only", "license": "Apache-2.0"})
    return destination


if __name__ == "__main__":
    print(prepare())
