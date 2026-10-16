"""Package only experiment code and a private, explicitly verified dataset for cloud runs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .common import EVALUATION_TEXTS, ROOT, digest, dump, safe_path
from .xtts import validate_export


def validate_baseline(directory: Path, dataset: Path) -> dict:
    """Require all seven exact texts, the same reference and intact baseline audio."""
    value = json.loads((directory / "manifest.json").read_text())
    if (value["texts"] != EVALUATION_TEXTS or value["reference_sha256"] !=
            digest(dataset / "evaluation/reference.wav")):
        raise ValueError("Baseline text/reference mismatch")
    if set(value["audio"]) != set(EVALUATION_TEXTS):
        raise ValueError("Baseline does not cover all seven test lines")
    for row in value["audio"].values():
        if digest(safe_path(directory, row["file"])) != row["sha256"]:
            raise ValueError("Baseline audio failed checksum")
    return value


def prepare_baseline(output: Path) -> Path:
    """Reuse existing accepted Chatterbox samples; synthesize only changed/missing text."""
    import shutil

    if output.exists():
        raise FileExistsError(output)
    previous = ROOT / "data/voice/shootouts/20261014-01"
    manifest = json.loads((previous / "manifest.json").read_text())
    result = json.loads((previous / "chatterbox-baseline.json").read_text())
    reference = ROOT / "data/voice/references/ciel/reference-1.wav"
    if result["status"] != "ok" or result["reference_sha256"] != digest(reference):
        raise ValueError("Existing Chatterbox comparison does not match Ciel's baseline")
    output.mkdir(parents=True)
    value = {"engine": "existing unchanged Chatterbox Turbo", "texts": EVALUATION_TEXTS,
             "reference_sha256": digest(reference), "audio": {}}
    provider = None
    try:
        for name, text in EVALUATION_TEXTS.items():
            target = output / f"{name}.wav"
            reuse = manifest["sentences"].get(name) == text
            old = next((r for r in result["rows"] if r["sentence"] == name
                        and r["iteration"] == 1), None)
            if reuse and old:
                source = Path(old["audio"])
                shutil.copy2(source, target)
                provenance = {"reused_from": str(source.relative_to(ROOT)),
                              "original_sha256": digest(source)}
            else:
                from scripts.tts_shootout.providers import ChatterboxProvider
                from scripts.tts_shootout.worker import measure

                if provider is None:
                    provider = ChatterboxProvider({}, reference, output / "cache")
                provenance = {"new_synthesis": measure(provider, text, target),
                              "reason": "Exact evaluation text changed since zero-shot shootout"}
            value["audio"][name] = {"file": target.name, "sha256": digest(target),
                                     "provenance": provenance}
            dump(output / "manifest.json", value)
    finally:
        if provider is not None:
            provider.close()
    return output / "manifest.json"


def bundle(output: Path, dataset: Path | None = None, baseline: Path | None = None) -> Path:
    """Never include .env, credentials, recordings, model weights, or unrelated code."""
    if dataset:
        validate_export(dataset)
        if baseline is None:
            raise ValueError("Cloud transfer requires the prepared Chatterbox baseline")
        validate_baseline(baseline, dataset)
    output.mkdir(parents=True, exist_ok=True)
    code = output / "ciel-finetune-code.zip"
    paths = [ROOT / "scripts/ciel_tts_finetune.py",
             ROOT / "scripts/voice_requirements/xtts_finetune.txt",
             ROOT / "scripts/voice_requirements/xtts_cloud.lock.txt",
             ROOT / "docs/tts-cloud-preflight.md"]
    paths += sorted((ROOT / "scripts/tts_finetune").glob("*.py"))
    paths += sorted((ROOT / "scripts/tts_finetune").glob("*.html"))
    with zipfile.ZipFile(code, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(ROOT)))
    if dataset:
        with zipfile.ZipFile(output / "ciel-verified-dataset.zip", "x",
                             compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(dataset.rglob("*")):
                if path.is_file():
                    if path.is_symlink() or not path.resolve().is_relative_to(dataset.resolve()):
                        raise ValueError("Dataset bundle cannot contain external links")
                    archive.write(path, str(path.relative_to(dataset)))
            for path in sorted(baseline.rglob("*")):
                if path.is_file():
                    safe_path(baseline, str(path.relative_to(baseline)))
                    if path.is_symlink():
                        raise ValueError("Baseline links are not accepted")
                    target = Path("baseline/chatterbox") / path.relative_to(baseline)
                    archive.write(path, str(target))
    dump(output / "transfer-manifest.json", {"files": {
        path.name: digest(path) for path in output.glob("*.zip")}})
    return code


def notebook(output: Path) -> Path:
    """Self-contained private Colab or Linux cloud notebook; no automatic activation."""
    cells = []

    def cell(kind: str, source: str) -> None:
        value = {"cell_type": kind, "id": f"ciel-{len(cells):02d}", "metadata": {},
                 "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    transfer = output.parent / "transfer-final/transfer-manifest.json"
    hashes = json.loads(transfer.read_text())["files"] if transfer.exists() else {}
    cell("markdown", """# Ciel XTTS: final cloud preflight / bounded adaptation
Use a **private Linux x86_64 NVIDIA runtime: 16 GB acceptable, 24 GB preferred**.
Training aborts below **15 GiB total VRAM**; it also requires 12 GiB free before loading.
The selected CUDA device is checked again. No batch shrinking, optimizer replacement or swap.
Keep at least **60 GiB free persistent storage** (including your Drive quota), plus space for
downloads/archives. The two ZIPs contain verified private speech and the Chatterbox baseline;
do not publish them. Existing personal non-commercial XTTS CPML acceptance applies.

This notebook installs its own Python 3.12.14 environment and pinned dependency closure.
It does not need your local repository, .env, HF token, Chatterbox installation or source videos.
Base XTTS assets download from the pinned public checkpoint revision. Internet is required.
It prepares no paid instance and never changes production Ciel. Run cells sequentially.
""")
    cell("code", """import hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile, zipfile
Path = pathlib.Path
# Colab: explicitly authorize your own private Drive for persistent checkpoints.
# Other clouds: set CIEL_TRAINING_STORAGE to a real persistent volume BEFORE this cell.
storage = os.environ.get('CIEL_TRAINING_STORAGE')
if storage:
    STORAGE = Path(storage).expanduser().resolve()
else:
    try:
        from google.colab import drive, files
    except ImportError:
        raise RuntimeError('Set CIEL_TRAINING_STORAGE to a persistent volume, then rerun.')
    drive.mount('/content/drive')
    STORAGE = Path('/content/drive/MyDrive/ciel-private-training')
STORAGE.mkdir(parents=True, exist_ok=True)
WORKSPACE = STORAGE / 'ciel-xtts-training'
ROOT = Path(tempfile.mkdtemp(prefix='ciel-cloud-'))
print('Persistent training output:', WORKSPACE)
print('Temporary code/data/environment:', ROOT)
""")
    cell("code", f"""# Colab: choose ONLY the two ZIPs in cloud/transfer-final.
# Other clouds: place those two files in this notebook's current directory.
EXPECTED = {hashes!r}
if 'google.colab' in sys.modules:
    from google.colab import files
    uploaded = files.upload()
    del uploaded  # Avoid retaining the dataset ZIP as a large Python bytes object.
def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()
def extract_new(source, target):
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(source) as archive:
        names = set()
        for item in archive.infolist():
            path = (target / item.filename).resolve()
            if (Path(item.filename).is_absolute() or not path.is_relative_to(target.resolve())
                or '\\\\' in item.filename or path in names
                or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe or duplicate archive path')
            names.add(path)
        target.mkdir(parents=True)
        archive.extractall(target)
for name in ('ciel-finetune-code.zip', 'ciel-verified-dataset.zip'):
    if not EXPECTED or sha256(name) != EXPECTED.get(name):
        raise ValueError('ZIP checksum mismatch; regenerate notebook after bundle: ' + name)
extract_new('ciel-finetune-code.zip', ROOT / 'code')
extract_new('ciel-verified-dataset.zip', ROOT / 'verified')
CODE = ROOT / 'code'
DATASET = ROOT / 'verified'
MODELS = ROOT / 'pretrained'
""")
    cell("code", """# Bootstrap a pinned interpreter, independent of the notebook kernel.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv==0.12.19'], check=True)
UV = [sys.executable, '-m', 'uv']
subprocess.run(UV + ['python', 'install', '3.12.14'], check=True)
subprocess.run(UV + ['venv', '--python', '3.12.14', str(ROOT / '.training')], check=True)
PY = str(ROOT / '.training/bin/python')
LOCK = str(CODE / 'scripts/voice_requirements/xtts_cloud.lock.txt')
subprocess.run(UV + ['pip', 'install', '--python', PY,
    '--index-url', 'https://download.pytorch.org/whl/cu124',
    'torch==2.6.0+cu124', 'torchaudio==2.6.0+cu124'], check=True)
subprocess.run(UV + ['pip', 'install', '--python', PY, '-r', LOCK], check=True)
subprocess.run(UV + ['pip', 'check', '--python', PY], check=True)
CLI = str(CODE / 'scripts/ciel_tts_finetune.py')
def run(*args):
    subprocess.run([PY, CLI, *map(str, args)], cwd=CODE, check=True)
# The guard rejects a 6 GB card BEFORE importing Torch or loading pretrained weights.
subprocess.run([PY, '-c',
    'from scripts.tts_finetune.xtts import gpu_preflight; print(gpu_preflight())'],
    cwd=CODE, check=True)
run('xtts-assets', MODELS)
run('cloud-check', DATASET, MODELS, '--baseline', DATASET / 'baseline/chatterbox')
""")
    cell("markdown", """## Bounded pilot, full-state continuation and listening stops
The native `coqui-tts==0.27.5` GPTTrainer uses the pretrained v2 GPT, frozen DVAE/vocoder/speaker
encoder, AdamW 5e-6, batch 1 / accumulation 252, and native FP16 AMP with FP32 Adam states.
Epoch 0 only evaluates/generates the pretrained XTTS baseline. **Two training epochs initially**;
all seven fixed test sentences generate WAVs at baseline and every epoch with seed 73.
This small pilot has few optimizer updates; neither success nor failure follows from loss alone.

Full checkpoints commit after every optimizer update (252 batches and the epoch's final batch),
then again after validation/listening. Keep the latest five periodic checkpoints plus the best
validation checkpoint. `checkpoints/latest`, `best`, and `step-*` contain checksummed pointers
to the native files, avoiding duplicate multi-GiB tensors. Logs retain train/eval losses;
`samples/step-*/manifest.json` identifies the exact checkpoint for each WAV.

On reconnect, rerun setup/upload cells using the **same persistent STORAGE**. The training cell
detects the latest committed checkpoint and restores model, AdamW, scheduler, AMP scaler,
global step, epoch, best-so-far loss and config identity. Missing/incompatible state aborts.
An interrupted epoch replays from its beginning; this is not bitwise dataloader/RNG replay.
An epoch-complete checkpoint resumes the next epoch. `TOTAL_EPOCHS` is a total, not extra epochs.

Three consecutively worse validation checks with >5% overall worsening stop training;
non-finite losses stop immediately. For degraded voice/pronunciation or memorization, create
`WORKSPACE/STOP` from a second notebook cell or the storage browser. It stops at the next optimizer
boundary, saving first. Native mid-accumulation interrupt saves are disabled.
Do not kill the VM to request a clean stop. The last committed full state
remains on persistent storage even if the VM disappears. Listen before extending toward 5/10 total
epochs; do not blindly continue because train loss falls. After review, archive STOP and
stop-reason.json before permitting a continuation.
""")
    cell("code", """# Copy Chatterbox baseline and generate base XTTS before updates.
TOTAL_EPOCHS = 2
resume = (WORKSPACE / 'checkpoints/latest/checkpoint.json').exists()
args = ['train-xtts', DATASET, MODELS, WORKSPACE, '--epochs', TOTAL_EPOCHS]
if resume:
    args.append('--resume')
# A rerun after the requested total is reached aborts; audition/export instead.
run(*args)
""")
    cell("code", """# Run after any completed epoch, including a listening/loss stop.
from IPython.display import Audio, display
summary = json.loads((WORKSPACE / 'training-summary.json').read_text())
print(json.dumps(summary, indent=2))
print('Samples:', list((WORKSPACE / 'samples').glob('step-*')))
for directory in sorted((WORKSPACE / 'samples').glob('step-*')):
    print(directory.name, 'conversational (checkpoint manifest in same folder)')
    display(Audio(filename=str(directory / 'conversational.wav')))
print('Use the exported local blind page before making quality decisions.')
""")
    cell("markdown", """## Private export / recovery
The final ZIP contains ONE best-validation inference model (no optimizer), config/vocab/speaker
assets, unchanged reference, license, exact dependencies, losses/summary, all baseline/checkpoint
WAVs and manifests. Best loss is a candidate to audition, never a quality acceptance.
Download it privately and follow `docs/tts-cloud-preflight.md` in the local Ciel repository.
No training base model, DVAE source checkpoint, caches, duplicate Adam checkpoints or raw dataset
are bundled into the inference archive. Frozen components within the native model mapping remain
unchanged for compatible loading.

Persistent storage is the primary disconnect recovery. An optional recovery ZIP below contains
latest + best full states (deduplicated), config, samples and logs, rather than every periodic
checkpoint. Download this BEFORE deleting a rented instance/volume. On a different cloud, unpack
it into a new persistent workspace and rerun the same setup with `--resume`. Never use the compact
inference model for training resume. Keep dataset/code ZIPs and this notebook separately.
""")
    cell("code", """# Increment EXPORT_TAG for each export; earlier exports are preserved.
EXPORT_TAG = 'pilot-01'
destination = STORAGE / ('ciel-xtts-final-' + EXPORT_TAG)
run('export-cloud', WORKSPACE, MODELS, DATASET, destination)
final_zip = destination.with_suffix('.zip')
print('Private final archive:', final_zip, 'SHA256:', sha256(final_zip))
if 'google.colab' in sys.modules:
    files.download(str(final_zip))
""")
    cell("code", """# Optional portable full-state recovery; persistent checkpoints already exist.
MAKE_RECOVERY_ZIP = False  # Set True before deleting the instance or persistent volume.
if MAKE_RECOVERY_ZIP:
    recovery = STORAGE / ('ciel-xtts-recovery-' + EXPORT_TAG)
    run('export-cloud', WORKSPACE, MODELS, DATASET, recovery, '--recovery')
    print('Private recovery archive:', recovery.with_suffix('.zip'))
    if 'google.colab' in sys.modules:
        files.download(str(recovery.with_suffix('.zip')))
""")
    document = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return output
