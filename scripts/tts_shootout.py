"""Prepare a blind quality shootout; preserve production settings and every existing engine."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(directory: Path, reference: Path):
    from scripts.tts_shootout.common import CONFIGURATIONS, ROOT, SENTENCES, digest, dump

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use the existing run or a new directory; do not replace an audition")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    copied = directory / "reference.wav"
    if copied.exists():
        raise FileExistsError(copied)
    shutil.copy2(reference, copied)
    dump(manifest_path, {
        "run_id": directory.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_measurements_and_listening", "listening_acceptance": None,
        "reference": str(reference.resolve()), "reference_sha256": digest(reference),
        "reference_copy": "reference.wav", "sentences": SENTENCES,
        "configurations": CONFIGURATIONS, "rounds": [],
        "production_env_sha256": digest(ROOT / ".env") if (ROOT / ".env").exists() else None,
        "pocket_quality": "User rejected robotic output; preserved but not reevaluated.",
        "limits": "Engine inference only, no STT/LLM contention. Whole-device GPU telemetry "
                  "is not TTS-specific power. Quality winners require listening. "
                  "Chatterbox inference and conditioning defaults remain unchanged.",
    })
    print(manifest_path)


def run(directory: Path, configurations: list[str], iterations: int, retry_failed: bool = False):
    from scripts.tts_shootout.common import CONFIGURATIONS, ROOT, digest, dump, evaluation_python

    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    reference = directory / manifest["reference_copy"]
    if digest(reference) != manifest["reference_sha256"]:
        raise ValueError("Reference changed; comparison would not use identical audio")
    for identifier in configurations:
        config = CONFIGURATIONS[identifier]
        python = evaluation_python(config)
        result_path = directory / f"{identifier}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["status"] == "ok" or not retry_failed:
                print("Retaining existing result:", identifier, result["status"], flush=True)
                continue
            archive_root = directory / "failed-attempts"
            archive = archive_root / f"{identifier}-{time.time_ns()}"
            archive.mkdir(parents=True)
            for path in (result_path, directory / f"{identifier}.log",
                         directory / "audio" / identifier):
                if path.exists():
                    shutil.move(str(path), archive / path.name)
        log = directory / f"{identifier}.log"
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "src"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "NLTK_DATA": str(ROOT / "data/voice/cache/nltk"),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(ROOT / "data/voice/cache/neutts-hf") if config["engine"] == "neutts"
            else str(Path.home() / ".cache/huggingface"),
        })
        if config["device"] == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        print("Testing", identifier, "with", iterations, "warm iterations", flush=True)
        with log.open("x") as output:
            started = time.perf_counter()
            environment["CIEL_SHOOTOUT_STARTED"] = str(started)
            command = [str(python), "-m", "scripts.tts_shootout.worker",
                       "--configuration", identifier, "--reference", str(reference),
                       "--cache", str(ROOT / "data/voice/states/shootout"),
                       "--output", str(result_path), "--iterations", str(iterations)]
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                process.wait(timeout=2700)
                if process.returncode != 0 or not result_path.exists():
                    dump(result_path, {"configuration_id": identifier, "configuration": config,
                                       "status": "error", "rows": [], "error":
                                       f"Worker exited {process.returncode}; inspect {log.name}"})
            except subprocess.TimeoutExpired:
                terminate_group(process)
                dump(result_path, {"configuration_id": identifier, "configuration": config,
                                   "status": "error", "rows": [],
                                   "error": "Worker exceeded 45 minutes"})
            except BaseException:
                terminate_group(process)
                raise
        result = json.loads(result_path.read_text())
        print(identifier, result["status"], round(time.perf_counter() - started, 1), "seconds",
              result.get("error", ""), flush=True)
        # Cooling/idle observations in later stress reports do not infer fan comfort
        # from an isolated short synthesis. No GPU clocks, limits, or fans are changed.
    if manifest["production_env_sha256"] is not None:
        if digest(ROOT / ".env") != manifest["production_env_sha256"]:
            raise RuntimeError("Production .env changed during shootout; inspect before reporting")


def terminate_group(process):
    """Bound cleanup of an evaluation worker and nested Chatterbox worker."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def sustained(directory: Path, configurations: list[str], seconds: float):
    """Run each representative independently; retain completed reports and immutable logs."""
    from scripts.tts_shootout.common import CONFIGURATIONS, ROOT, evaluation_python

    directory = directory.resolve()
    for identifier in configurations:
        result = directory / f"sustained-{identifier}.json"
        if result.exists():
            print("Retaining sustained result:", identifier, flush=True)
            continue
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "src"),
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                            "NLTK_DATA": str(ROOT / "data/voice/cache/nltk"),
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        if CONFIGURATIONS[identifier]["engine"] == "neutts":
            environment["HF_HOME"] = str(ROOT / "data/voice/cache/neutts-hf")
        if CONFIGURATIONS[identifier]["device"] == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        print("Sustained", identifier, seconds, "seconds", flush=True)
        with (directory / f"sustained-{identifier}.log").open("x") as log:
            process = subprocess.Popen(
                [str(evaluation_python(CONFIGURATIONS[identifier])), "-m",
                 "scripts.tts_shootout.stress", str(directory), "--configuration", identifier,
                 "--seconds", str(seconds)], cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                process.wait(timeout=seconds + 900)
            except BaseException:
                terminate_group(process)
                raise
        if process.returncode != 0:
            print("Sustained run failed:", identifier, "inspect its log", flush=True)
        else:
            print("Sustained complete:", identifier, flush=True)


def main():
    from scripts.tts_shootout.common import CONFIGURATIONS

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("directory", type=Path)
    preparation.add_argument("--reference", type=Path,
                             default=Path("data/voice/references/ciel/reference-1.wav"))
    benchmark = commands.add_parser("run")
    benchmark.add_argument("directory", type=Path)
    benchmark.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS,
                           default=[name for name, value in CONFIGURATIONS.items()
                                    if value["engine"] != "neutts"])
    benchmark.add_argument("--iterations", type=int, default=3)
    benchmark.add_argument("--retry-failed", action="store_true")
    page = commands.add_parser("page")
    page.add_argument("directory", type=Path)
    stress = commands.add_parser("stress")
    stress.add_argument("directory", type=Path)
    stress.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, required=True)
    stress.add_argument("--seconds", type=float, default=300)
    reporter = commands.add_parser("report")
    reporter.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, args.reference)
    elif args.command == "run":
        if args.iterations < 2:
            parser.error("At least two warm iterations are required")
        run(args.directory, args.configurations, args.iterations, args.retry_failed)
    elif args.command == "page":
        from scripts.tts_shootout.page import render

        print(render(args.directory))
    elif args.command == "stress":
        if args.seconds < 1:
            parser.error("Sustained duration must be positive")
        sustained(args.directory, args.configurations, args.seconds)
    elif args.command == "report":
        from scripts.tts_shootout.report import run as report

        print(report(args.directory))


if __name__ == "__main__":
    main()
