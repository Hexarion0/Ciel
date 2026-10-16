"""Provision pinned evaluation assets, without downloading any new speaker references."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

from .common import ROOT, STYLE_MODEL_REVISION, STYLE_REVISION, XTTS_MODEL_REVISION, digest, dump


def prepare(accept_xtts_license: bool = False):
    """Fetch model/runtime inputs once; explicit XTTS acceptance is required."""
    import nltk
    from huggingface_hub import snapshot_download

    research = ROOT / "data/voice/shootout-research"
    accepted = research / "xtts-license-acceptance.json"
    if accept_xtts_license and not accepted.exists():
        dump(accepted, {"accepted": True, "scope": "non-commercial personal Ciel evaluation",
                        "license": "https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt",
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "method": "explicit --accept-xtts-license command"})
    if not accepted.exists() or not json.loads(accepted.read_text()).get("accepted"):
        raise ValueError("Review XTTS's CPML license, then pass --accept-xtts-license if accepted")
    vendor = ROOT / "data/voice/vendor/styletts2"
    if not vendor.exists():
        subprocess.run(["git", "clone", "--no-checkout",
                        "https://github.com/NeuralVox/StyleTTS2.git", str(vendor)], check=True)
        subprocess.run(["git", "checkout", "--detach", STYLE_REVISION], cwd=vendor, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if revision != STYLE_REVISION:
        raise ValueError("Existing evaluation vendor differs; preserve it before replacing")
    resources = ROOT / "data/voice/cache/nltk"
    for name in ("punkt", "punkt_tab"):
        if not (resources / "tokenizers" / name).is_dir():
            if not nltk.download(name, download_dir=str(resources), quiet=True):
                raise RuntimeError(f"NLTK download failed: {name}")
    repos = [
        ("yl4579/StyleTTS2-LibriTTS", "styletts2-libritts", STYLE_MODEL_REVISION,
         ["Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth"]),
        ("coqui/XTTS-v2", "xtts-v2", XTTS_MODEL_REVISION,
         ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth", "LICENSE.txt"]),
    ]
    for repo, folder, pinned, files in repos:
        destination = ROOT / "data/voice/models" / folder
        inventory = destination / "inventory.json"
        if inventory.exists():
            previous = json.loads(inventory.read_text())
            if previous["revision"] != pinned:
                raise ValueError(f"Existing {folder} revision differs; preserve it")
        if not all((destination / filename).is_file() for filename in files):
            snapshot_download(repo_id=repo, revision=pinned, allow_patterns=files,
                              local_dir=destination, max_workers=2)
        dump(inventory, {"repo": repo, "revision": pinned, "files": files})
        dump(destination / "integrity.json", {
            filename: digest(destination / filename) for filename in files
        })
        print("Verified pinned assets:", repo, pinned, flush=True)
    requirements = ROOT / "scripts/voice_requirements/shootout.txt"
    freeze = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", str(ROOT / "data/voice/envs/shootout/bin/python")],
        text=True,
    )
    research.mkdir(parents=True, exist_ok=True)
    (research / "installed-requirements.txt").write_text(freeze)
    dump(research / "environment-provenance.json", {
        "requirements_sha256": digest(requirements),
        "installed_requirements_sha256": digest(research / "installed-requirements.txt"),
        "style_implementation_revision": revision,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-xtts-license", action="store_true",
                        help="Confirm that you reviewed and accept CPML for non-commercial use")
    args = parser.parse_args()
    prepare(args.accept_xtts_license)


if __name__ == "__main__":
    main()
