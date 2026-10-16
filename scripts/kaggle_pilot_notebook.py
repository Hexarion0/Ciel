"""Build an unattended private Kaggle variant of the existing reviewed notebook."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_REF = "hexarion0/ciel-xtts-training"
PILOT_DIRECTORY = ROOT / "data/voice/finetuning/20261014-01/cloud/kaggle-pilot"


def build(destination: Path, owner: str | None = None) -> Path:
    """Build from the archived private inputs for an explicitly selected notebook owner."""
    owner = owner or os.environ.get("KAGGLE_USERNAME")
    if not owner or not owner.strip():
        raise ValueError("Provide a notebook owner or set KAGGLE_USERNAME")
    original = ROOT / "data/voice/finetuning/20261014-01/cloud/xtts_ciel.ipynb"
    original_document = json.loads(original.read_text())
    expected = json.loads(
        (original.parent / "transfer-final/transfer-manifest.json").read_text()
    )["files"]
    inventories = {}
    for name, expected_hash in expected.items():
        archive_path = original.parent / "transfer-final" / name
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Local sealed transfer archive changed: " + name)
        with zipfile.ZipFile(archive_path) as archive:
            inventories[name.removesuffix(".zip")] = {
                i.filename: hashlib.sha256(archive.read(i)).hexdigest()
                for i in archive.infolist()
                if not i.is_dir()
            }
    payload = io.BytesIO()
    files = [
        ROOT / "scripts/ciel_tts_finetune.py",
        ROOT / "docs/tts-cloud-preflight.md",
    ]
    guide = ROOT / "docs/tts-kaggle-pilot.md"
    if guide.is_file():
        files.append(guide)
    files += list((ROOT / "scripts/tts_finetune").glob("*.py"))
    files += list((ROOT / "scripts/tts_finetune").glob("*.html"))
    files += list((ROOT / "scripts/voice_requirements").glob("xtts*"))
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
    blob = payload.getvalue()
    encoded = base64.b64encode(blob).decode("ascii")
    cells = []

    def cell(kind, source):
        if kind == "code":
            source = source.replace("subprocess.run(", "checked(")
        value = {
            "cell_type": kind,
            "id": f"kaggle-pilot-{len(cells):02}",
            "metadata": {},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True),
        }
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell(
        "markdown",
        """
        # Ciel XTTS · private Kaggle two-epoch pilot
        Adapted from the existing reviewed `xtts_ciel.ipynb`; no manual cell edits/uploads.
        Dataset: `hexarion0/ciel-xtts-training`. T4 GPU, internet, private notebook.
        One latest full Adam/scheduler/scaler checkpoint, compact best-trained inference model,
        all listening WAVs and metrics. Atomic replacements retain the prior full state until
        commit; 3 GiB disk floor and pre-write capacity reservations. No optimizer substitution.
        Setup lives in `/tmp`; retained outputs live in `/kaggle/working`.
        Kaggle session disk is ephemeral: successful output persistence/download is required
        for recovery after the session ends. A new push does not implicitly resume prior outputs.
        Interrupted epochs replay; strict full-state restore remains enabled. Existing STOP and
        failure markers require review. Existing personal non-commercial XTTS license applies.
        Production Ciel TTS is unchanged. Two epochs alone cannot establish voice quality.
        T4 memory checks require 15 GiB physical capacity and 12 GiB CUDA-free; a bounded
        512 MiB CUDA driver reservation is allowed only for this explicit T4 pilot.
        The conditioning Perceiver runs in FP32 to prevent measured FP16 overflow;
        native FP16 AMP, AdamW and full optimizer/scheduler/scaler resume are retained.
        A source-checked native fix converts prompt loss-mask bounds to audio-code units;
        conditioning masking remains enabled. All 369 training clips passed forward validation.
    """,
    )
    cell(
        "code",
        """
        import base64, hashlib, io, json, os, pathlib, shutil, signal, subprocess, sys
        import time, zipfile
        Path = pathlib.Path
        os.environ.update(MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='0', CIEL_KAGGLE_PILOT='1',
                          CIEL_XTTS_FP32_PERCEIVER='1',
                          CIEL_XTTS_MASK_UNITS_FIX='1',
                          PYTHONUNBUFFERED='1', PIP_NO_CACHE_DIR='1',
                          UV_CACHE_DIR='/tmp/ciel-xtts-uv-cache',
                          UV_PYTHON_INSTALL_DIR='/tmp/ciel-xtts-python',
                          XDG_CACHE_HOME='/tmp/ciel-xtts-cache')
        INPUT = Path('/kaggle/input/datasets/hexarion0/ciel-xtts-training')
        if not INPUT.is_dir():
            raise FileNotFoundError('Expected attached Kaggle dataset mount: ' + str(INPUT))
        STORAGE = Path('/kaggle/working')
        WORKSPACE = STORAGE / 'ciel-xtts-training'
        ROOT = Path('/tmp/ciel-xtts-kaggle-pilot')
        ROOT.mkdir(parents=True, exist_ok=True)
        CODE, DATASET, MODELS = ROOT / 'code', ROOT / 'verified', ROOT / 'pretrained'
        def sha256(path):
            value = hashlib.sha256()
            with open(path, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    value.update(chunk)
            return value.hexdigest()
        def space(write_bytes=0):
            for location in (STORAGE, ROOT):
                free = shutil.disk_usage(location).free
                if free < 3 * 1024**3 + write_bytes:
                    raise RuntimeError('Storage safety stop at ' + str(location) + ': '
                        + str(round(free / 1024**3, 2)) + ' GiB free; 3 GiB reserve required')
        space()
        def checked(command, check=True, **kwargs):
            # Monitor setup/download/export as well as training. Never delete committed state.
            process = subprocess.Popen(command, start_new_session=True, **kwargs)
            try:
                while process.poll() is None:
                    space()
                    time.sleep(1)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            if check and process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            return process
        def extract_checked(source, target):
            with zipfile.ZipFile(source) as archive:
                names = set()
                for item in archive.infolist():
                    path = (target / item.filename).resolve()
                    if (Path(item.filename).is_absolute() or '\\\\' in item.filename
                        or not path.is_relative_to(target.resolve()) or path in names
                        or (item.external_attr >> 16) & 0o170000 == 0o120000):
                        raise ValueError('Unsafe archive member')
                    names.add(path)
                if target.exists():
                    actual = {str(p.relative_to(target)) for p in target.rglob('*')
                              if p.is_file() and '__pycache__' not in p.parts}
                    wanted = {i.filename for i in archive.infolist() if not i.is_dir()}
                    if actual != wanted:
                        raise ValueError('Existing extraction has a different file inventory')
                    for item in archive.infolist():
                        if not item.is_dir():
                            if sha256(target / item.filename) != hashlib.sha256(
                                archive.read(item)).hexdigest():
                                raise ValueError('Existing extraction changed: ' + item.filename)
                    return
                space(sum(i.file_size for i in archive.infolist()))
                staging = target.with_name(target.name + '.extracting')
                if staging.exists():
                    raise RuntimeError('Incomplete extraction requires review: ' + str(staging))
                staging.mkdir(parents=True)
                archive.extractall(staging)
                staging.replace(target)
        print('Private dataset mount:', INPUT, 'Output:', WORKSPACE, flush=True)
    """,
    )
    cell(
        "code",
        f"""
EXPECTED = {expected!r}
EXPANDED_INVENTORIES = {inventories!r}
for name, expected_hash in EXPECTED.items():
    matches = list(INPUT.rglob(name))
    expanded = INPUT / name.removesuffix('.zip')
    if expanded.is_dir():
        inventory = EXPANDED_INVENTORIES[name.removesuffix('.zip')]
        actual = {{str(p.relative_to(expanded)) for p in expanded.rglob('*') if p.is_file()}}
        if actual != set(inventory):
            raise ValueError('Expanded dataset file inventory differs: ' + name)
        for relative, expected_member_hash in inventory.items():
            path = (expanded / relative).resolve()
            if not path.is_relative_to(expanded.resolve()) or sha256(path) != expected_member_hash:
                raise ValueError('Expanded dataset member checksum mismatch: ' + relative)
        if name == 'ciel-verified-dataset.zip':
            DATASET = expanded
    else:
        if len(matches) != 1 or sha256(matches[0]) != expected_hash:
            raise ValueError('Dataset archive missing, ambiguous or checksum mismatch: ' + name)
        if name == 'ciel-verified-dataset.zip':
            extract_checked(matches[0], DATASET)
# The attached original code ZIP was verified above. This version embeds the reviewed pilot patch.
PATCHED_CODE_SHA256 = {hashlib.sha256(blob).hexdigest()!r}
patched_code = base64.b64decode({encoded!r})
if hashlib.sha256(patched_code).hexdigest() != PATCHED_CODE_SHA256:
    raise ValueError('Embedded pilot code checksum mismatch')
extract_checked(io.BytesIO(patched_code), CODE)
del patched_code
(STORAGE / 'pilot-source.json').write_text(json.dumps({{
    'original_notebook_sha256': {hashlib.sha256(original.read_bytes()).hexdigest()!r},
    'patched_code_sha256': PATCHED_CODE_SHA256, 'dataset_archives': EXPECTED,
    'dataset': {DATASET_REF!r}, 'production_changed': False}}, indent=2))
""",
    )
    cell(
        "code",
        """
        # The GPU guard runs before downloading/installing model dependencies.
        subprocess.run([sys.executable, '-c',
            'from scripts.tts_finetune.xtts import gpu_preflight; print(gpu_preflight())'],
            cwd=CODE, check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                        'uv==0.12.19'], check=True)
        UV = [sys.executable, '-m', 'uv']
        subprocess.run(UV + ['python', 'install', '3.12.14'], check=True)
        VENV = ROOT / '.training'
        PY = str(VENV / 'bin/python')
        if VENV.exists():
            if not Path(PY).is_file():
                raise RuntimeError('Incomplete venv; refusing destructive recreation')
            found = subprocess.check_output([PY, '-c',
                'import platform; print(platform.python_version())'], text=True).strip()
            if found != '3.12.14':
                raise RuntimeError('Refusing to replace existing venv Python ' + found)
            print('Reusing Python', found, 'venv:', VENV, flush=True)
        else:
            subprocess.run(UV + ['venv', '--python', '3.12.14', str(VENV)], check=True)
        LOCK = CODE / 'scripts/voice_requirements/xtts_cloud.lock.txt'
        setup_identity = {'lock_sha256': sha256(LOCK), 'python': '3.12.14',
                          'torch': '2.6.0+cu124', 'torchaudio': '2.6.0+cu124'}
        marker = VENV / 'ciel-setup.json'
        if marker.exists() and json.loads(marker.read_text()) != setup_identity:
            raise ValueError('Pinned environment identity changed; review before reusing')
        if not marker.exists():
            space()
            subprocess.run(UV + ['pip', 'install', '--no-cache', '--python', PY,
                '--index-url', 'https://download.pytorch.org/whl/cu124',
                'torch==2.6.0+cu124', 'torchaudio==2.6.0+cu124'], check=True)
            subprocess.run(UV + ['pip', 'install', '--no-cache', '--python', PY,
                                 '-r', str(LOCK)], check=True)
        subprocess.run(UV + ['pip', 'check', '--python', PY], check=True)
        # Verify actual installed pins even when the marker exists.
        subprocess.run([PY, '-c',
            'import torch,torchaudio; assert torch.__version__ == "2.6.0+cu124"; '
            'assert torchaudio.__version__ == "2.6.0+cu124"'], check=True)
        marker.write_text(json.dumps(setup_identity, indent=2))
        CLI = str(CODE / 'scripts/ciel_tts_finetune.py')
        def run(*args):
            space()
            subprocess.run([PY, CLI, *map(str, args)], cwd=CODE, check=True)
        run('xtts-assets', MODELS)
        run('cloud-check', DATASET, MODELS, '--baseline', DATASET / 'baseline/chatterbox')
    """,
    )
    cell(
        "code",
        """
        TOTAL_EPOCHS = 2
        def guarded_train():
            args = ['train-xtts', DATASET, MODELS, WORKSPACE, '--epochs', TOTAL_EPOCHS]
            if (WORKSPACE / 'checkpoints/latest/checkpoint.json').exists():
                args.append('--resume')
            elif WORKSPACE.exists():
                raise RuntimeError('Existing workspace without full state; inspect before retry')
            log = STORAGE / 'pilot-training.log'
            with log.open('a', buffering=1) as handle:
                process = subprocess.Popen([PY, CLI, *map(str, args)], cwd=CODE,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                position = log.stat().st_size
                try:
                    while process.poll() is None:
                        space()
                        with log.open() as reader:
                            reader.seek(position)
                            chunk = reader.read()
                            position = reader.tell()
                        if chunk:
                            print(chunk, end='', flush=True)
                        time.sleep(2)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
                finally:
                    with log.open() as reader:
                        reader.seek(position)
                        print(reader.read(), end='', flush=True)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, args)
        try:
            guarded_train()
        except BaseException as error:
            (STORAGE / 'pilot-failure.json').write_text(json.dumps({
                'error_type': type(error).__name__, 'error': str(error),
                'latest_checkpoint_preserved': (
                    WORKSPACE / 'checkpoints/latest/checkpoint.json').exists(),
                'automatic_retry': False}, indent=2))
            raise
    """,
    )
    cell(
        "code",
        """
        summary = json.loads((WORKSPACE / 'training-summary.json').read_text())
        if summary['status'] != 'pilot_complete_awaiting_listening':
            raise RuntimeError('Training did not complete; preserve outputs for diagnosis')
        completed = [r for r in summary['epochs'] if not r['baseline_only']]
        if len({r['epoch'] for r in completed}) != TOTAL_EPOCHS:
            raise RuntimeError('Expected exactly two completed training epochs')
        # Reserve enough space for both the inference directory and its uncompressed ZIP.
        best = json.loads((WORKSPACE / 'checkpoints/best/checkpoint.json').read_text())
        best_path = (WORKSPACE / best['path']).resolve()
        if not best_path.is_relative_to(WORKSPACE.resolve()) or sha256(best_path) != best['sha256']:
            raise ValueError('Best inference checkpoint failed its committed checksum')
        other = sum(p.stat().st_size for folder in ('config', 'logs', 'samples')
                    for p in (WORKSPACE / folder).rglob('*') if p.is_file())
        assets = sum((MODELS / n).stat().st_size for n in
                     ('config.json', 'vocab.json', 'speakers_xtts.pth', 'LICENSE.txt'))
        space(2 * (best_path.stat().st_size + other + assets) + 128 * 1024**2)
        destination = STORAGE / 'ciel-xtts-final-kaggle-pilot'
        run('export-cloud', WORKSPACE, MODELS, DATASET, destination)
        final_zip = destination.with_suffix('.zip')
        full = list((WORKSPACE / 'checkpoints/native').glob('*/checkpoint_*.pth'))
        if len(full) != 1:
            raise RuntimeError('Pilot must retain exactly one latest full optimizer checkpoint')
        result = {'status': 'complete_awaiting_local_import_and_blind_listening',
                  'epochs': TOTAL_EPOCHS, 'archive': final_zip.name,
                  'sha256': sha256(final_zip),
                  'latest_full_checkpoint': str(full[0].relative_to(STORAGE)),
                  'full_state_resume_preserved': True, 'production_changed': False}
        (STORAGE / 'pilot-complete.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    """,
    )
    document = {**original_document, "cells": cells}
    document["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    destination.mkdir(parents=True, exist_ok=True)
    notebook = destination / "xtts_ciel_kaggle.ipynb"
    notebook.write_text(json.dumps(document, indent=2) + "\n")
    metadata = {
        "id": f"{owner}/ciel-xtts-kaggle-pilot",
        "title": "Ciel XTTS Kaggle Pilot",
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_REF],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return notebook


if __name__ == "__main__":
    print(build(PILOT_DIRECTORY))
