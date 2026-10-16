"""Run a private forward-only Kaggle diagnosis; never call backward or optimizer.step."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_pilot import auth_environment
from scripts.kaggle_pilot_notebook import PILOT_DIRECTORY, ROOT, build


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require a deliberate account and keep generated diagnostic data private."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='01')
    parser.add_argument('--all-training-forwards', action='store_true')
    parser.add_argument('--owner', default=os.environ.get('KAGGLE_USERNAME'))
    parser.add_argument('--kaggle', default=shutil.which('kaggle') or str(
        ROOT.parent / '.venv-kaggle/bin/kaggle'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[0-9]{2}', args.tag):
        parser.error('--tag must be two digits')
    if not args.owner or not re.fullmatch(r'[A-Za-z0-9_-]+', args.owner):
        parser.error('provide --owner or set KAGGLE_USERNAME to your notebook account')
    if args.output is None:
        args.output = PILOT_DIRECTORY / 'runs' / ('forward-diagnostic-' + args.tag)
    return args


def main() -> None:
    """Build and submit only the explicitly requested private diagnostic notebook."""
    args = parse_args()
    folder = PILOT_DIRECTORY / ('diagnostics-' + args.tag)
    notebook = build(folder, args.owner)
    document = json.loads(notebook.read_text())
    diagnostic_setup = (
        "os.environ['CIEL_XTTS_FORWARD_DIAGNOSTIC'] = 'all_training'\n"
        "os.environ['CIEL_XTTS_FP32_PERCEIVER'] = '1'\n"
        "os.environ['CIEL_XTTS_MASK_UNITS_FIX'] = '1'\n" if args.all_training_forwards else
        "os.environ.pop('CIEL_XTTS_FP32_PERCEIVER', None)\n"
        "os.environ['CIEL_XTTS_FORWARD_DIAGNOSTIC'] = '1'\n")
    document["cells"][-2]["source"].insert(0, diagnostic_setup)
    document["cells"][-1]["source"] = [
        "print((WORKSPACE / 'forward-diagnostic.json').read_text())\n"]
    notebook.write_text(json.dumps(document, indent=2) + "\n")
    metadata_path = folder / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ref = f"{args.owner}/ciel-xtts-kaggle-diagnostics"
    metadata.update(id=ref, title="Ciel XTTS Kaggle Diagnostics")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "submission-attempted.json"
    already_submitted = marker.exists()
    env = auth_environment()
    cli = args.kaggle

    def command(*args, check=True):
        result = subprocess.run([cli, *args], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        with (output / "cli.log").open("a") as handle:
            handle.write("\nkaggle " + " ".join(args) + "\n" + result.stdout)
        if check:
            result.check_returncode()
        return result.stdout

    if not already_submitted:
        marker.write_text(json.dumps({"kernel": ref, "mode": "forward_only_no_optimizer"}))
        print(command("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"),
              flush=True)
    while True:
        status = command("kernels", "status", ref)
        print(status, end="", flush=True)
        match = re.search(r'has status "([^"]+)"', status)
        if not match:
            raise RuntimeError("Unknown diagnostic status; inspect cli.log")
        current = match[1].split('.')[-1].lower()
        if current in {'complete', 'error', 'failed', 'cancelled', 'canceled'}:
            (output / "kaggle-execution.log").write_text(command("kernels", "logs", ref))
            print(command("kernels", "output", ref, "-p", str(output / "outputs")), flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
