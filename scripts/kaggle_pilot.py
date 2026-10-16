"""Run/poll/fetch the private pilot with the official Kaggle CLI; never handle secrets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_pilot_notebook import PILOT_DIRECTORY, ROOT, build


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def auth_environment() -> dict:
    env = os.environ.copy()
    if not env.get("KAGGLE_CONFIG_DIR"):
        legacy = Path.home() / ".kaggle"
        modern = Path.home() / ".config/kaggle"
        env["KAGGLE_CONFIG_DIR"] = str(legacy if legacy.exists() else modern)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate local options before creating output or invoking the Kaggle CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "run", "monitor", "fetch", "finish"]
    )
    parser.add_argument(
        "--kaggle",
        default=shutil.which("kaggle") or str(ROOT.parent / ".venv-kaggle/bin/kaggle"),
    )
    parser.add_argument(
        "--owner", default=os.environ.get("KAGGLE_USERNAME"),
        help="Notebook owner (or set KAGGLE_USERNAME)",
    )
    parser.add_argument("--run-directory", type=Path,
                        default=PILOT_DIRECTORY / "runs/pilot-01")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--poll-seconds", type=int, default=45)
    args = parser.parse_args(argv)
    if not args.owner or not args.owner.strip():
        parser.error("provide --owner or set KAGGLE_USERNAME for your notebook account")
    if not 5 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 5 and 60")
    return args


def main() -> None:
    """Run the requested pilot operation using explicit local configuration."""
    args = parse_args()
    directory = args.run_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    env = auth_environment()
    ref = f"{args.owner}/ciel-xtts-kaggle-pilot"
    state_path = directory / "run-state.json"
    state = (
        json.loads(state_path.read_text()) if state_path.exists() else {"kernel": ref}
    )
    if state["kernel"] != ref:
        raise ValueError("Run directory belongs to another notebook")

    def cli(*tokens: str, check=True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [args.kaggle, *tokens],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        with (directory / "cli.log").open("a") as log:
            log.write(
                f"\n{datetime.now(timezone.utc).isoformat()} kaggle "
                + " ".join(tokens)
                + "\n"
            )
            log.write(result.stdout)
        if check and result.returncode:
            raise RuntimeError(
                f"Official Kaggle CLI failed: {' '.join(tokens)}. "
                f"Inspect {directory / 'cli.log'}"
            )
        return result

    def status() -> str:
        result = cli("kernels", "status", ref)
        match = re.search(r'has status "([^"]+)"', result.stdout)
        if not match:
            raise RuntimeError(
                "Unrecognized official CLI status; inspect cli.log, do not resubmit"
            )
        value = match[1].split(".")[-1].lower()
        state.update(
            status=value, status_checked_at=datetime.now(timezone.utc).isoformat()
        )
        save(state_path, state)
        print(f"{state['status_checked_at']} {ref}: {value}", flush=True)
        return value

    def fetch() -> None:
        logs = cli("kernels", "logs", ref, check=False)
        (directory / "kaggle-execution.log").write_text(logs.stdout)
        files = cli("kernels", "files", ref, "--csv", "--page-size", "200", check=False)
        (directory / "output-files.csv").write_text(files.stdout)
        # CLI 2.2.4's output implementation buffers response.content for each entire
        # file. Use its official SDK output listing and stream those same supplied URLs.
        interpreter = str(Path(args.kaggle).resolve().parent / "python")
        pushed = directory / "push-result.txt"
        match = re.search(r"Kernel version (\d+)", pushed.read_text()) if pushed.exists() else None
        output_ref = ref + "/" + match[1] if match else ref
        with (directory / "download.log").open("a") as log:
            result = subprocess.run([
                interpreter, str(ROOT / "scripts/kaggle_pilot_download.py"), output_ref,
                "-p", str(directory / "outputs")], env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False)
        state["outputs_downloaded"] = result.returncode == 0
        state["download_method"] = "official_Kaggle_API_output_URLs_streamed"
        save(state_path, state)
        if result.returncode:
            raise RuntimeError(
                "Output download failed; inspect download.log before retrying fetch"
            )

    def finish() -> None:
        outputs = directory / "outputs"
        completed = list(outputs.rglob("pilot-complete.json"))
        if len(completed) != 1:
            raise ValueError(
                "Missing unique completion manifest; diagnose logs before rerunning"
            )
        manifest = json.loads(completed[0].read_text())
        from scripts.tts_finetune.artifacts import import_experiment
        from scripts.tts_finetune.common import digest

        archive = completed[0].parent / manifest["archive"]
        if digest(archive) != manifest["sha256"] or manifest["epochs"] != 2:
            raise ValueError("Final export checksum or total epoch count differs")
        imported = directory / "imported-experiment"
        registration = imported / "experiment.json"
        if not imported.exists():
            registration = import_experiment(archive, imported)
        else:
            checksums = json.loads((imported / "checksums.json").read_text())
            for name, expected in checksums.items():
                path = (imported / name).resolve()
                if not path.is_relative_to(imported) or digest(path) != expected:
                    raise ValueError("Existing imported experiment changed")
        python = args.repo / "data/voice/envs/shootout/bin/python"
        # CPU-only structural checkpoint check; no training, no production provider changes.
        check_code = (
            "import json,sys,torch; "
            's=torch.load(sys.argv[1],map_location="cpu",weights_only=True,mmap=True); '
            'assert s["epoch"]>=1 and s["step"]>=1; '
            'assert any(k.startswith("xtts.gpt.") for k in s["model"]); '
            'assert "optimizer" not in s; '
            'print(json.dumps({"epoch":s["epoch"],"step":s["step"],'
            '"tensor_keys":len(s["model"]),"inference_only":True}))'
        )
        checked = subprocess.run(
            [str(python), "-c", check_code, str(imported / "final/model.pth")],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (directory / "local-checkpoint-check.log").write_text(checked.stdout)
        checked.check_returncode()
        with (directory / "local-full-resume-check.log").open("w") as log:
            full_check = subprocess.run([
                str(python), str(ROOT / "scripts/kaggle_pilot_verify.py"), str(directory)],
                env={**env, "CUDA_VISIBLE_DEVICES": ""}, stdout=log,
                stderr=subprocess.STDOUT, check=False)
        full_check.check_returncode()
        # Use the downloaded cloud WAVs and unchanged Chatterbox baseline for immediate listening.
        # This is a quality audition; no local latency/RTF claims are derived from cloud samples.
        from scripts.kaggle_pilot_compare import comparison

        blind = comparison(imported, directory / "blind-comparison")
        state.update(
            local_checkpoint_checked=True,
            local_full_resume_checked=True,
            imported_experiment=str(registration),
            blind_comparison=str(blind),
            production_changed=False,
        )
        save(state_path, state)
        print("Imported/checkpoint-checked:", registration, flush=True)
        print("Blind comparison:", blind, flush=True)

    if args.command == "prepare":
        print(build(PILOT_DIRECTORY, args.owner))
        return
    if args.command == "run":
        if state.get("submission_attempted"):
            raise RuntimeError(
                "Submission already attempted. Use monitor/fetch; no blind resubmission."
            )
        # Official authenticated API identifies the owner; print only the username, never config.
        interpreter = str(Path(args.kaggle).resolve().parent / "python")
        identity = subprocess.run(
            [
                interpreter,
                "-c",
                'from kaggle import api; print(api.get_config_value("username"))',
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if identity.returncode or identity.stdout.strip() != args.owner:
            raise RuntimeError(
                "Authenticated Kaggle account does not match the requested notebook owner"
            )
        mine = cli(
            "kernels",
            "list",
            "--mine",
            "--search",
            "Ciel XTTS Kaggle Pilot",
            "--csv",
            "--page-size",
            "100",
        )
        refs = {r.get("ref") for r in csv.DictReader(io.StringIO(mine.stdout))}
        if ref in refs:
            prior = status()
            if prior not in {"complete", "error", "failed", "cancelled", "canceled"}:
                raise RuntimeError(
                    "An existing run is active or unknown; monitor it before any push"
                )
            # Preserve existing notebook source/metadata and logs before a reviewed new version.
            cli("kernels", "pull", ref, "-p", str(directory / "prior-kernel"), "-m")
            prior_logs = cli("kernels", "logs", ref, check=False)
            (directory / "prior-execution.log").write_text(prior_logs.stdout)
        bundle = PILOT_DIRECTORY
        bundle.mkdir(parents=True, exist_ok=True)
        if not (bundle / "kernel-metadata.json").exists():
            cli("kernels", "init", "-p", str(bundle))
        build(bundle, args.owner)
        meta = json.loads((bundle / "kernel-metadata.json").read_text())
        if not (
            meta["is_private"] is True
            and meta["enable_gpu"] is True
            and meta["enable_internet"] is True
            and meta["machine_shape"] == "NvidiaTeslaT4"
            and meta["dataset_sources"] == ["hexarion0/ciel-xtts-training"]
        ):
            raise ValueError(
                "Notebook security/runtime metadata differs from requested settings"
            )
        state.update(
            submission_attempted=True,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            notebook_sha256=__import__("hashlib")
            .sha256((bundle / meta["code_file"]).read_bytes())
            .hexdigest(),
        )
        save(state_path, state)
        pushed = cli(
            "kernels", "push", "-p", str(bundle), "--accelerator", "NvidiaTeslaT4"
        )
        (directory / "push-result.txt").write_text(pushed.stdout)
        print(pushed.stdout, flush=True)
        state["push_acknowledged"] = True
        save(state_path, state)
        # Read back server metadata; reject any privacy/accelerator mismatch.
        readback = directory / "server-kernel"
        cli("kernels", "pull", ref, "-p", str(readback), "-m")
        actual = json.loads((readback / "kernel-metadata.json").read_text())
        def truth(value):
            return str(value).lower() == "true"
        if not all(
            truth(actual[k]) for k in ("is_private", "enable_gpu", "enable_internet")
        ):
            raise RuntimeError(
                "Server notebook settings differ; inspect server-kernel metadata"
            )
        if actual.get("machine_shape") != "NvidiaTeslaT4":
            raise RuntimeError(
                "Server accelerator differs; inspect metadata before continuing"
            )
    if args.command in {"run", "monitor"}:
        while True:
            current = status()
            if current == "complete":
                fetch()
                finish()
                return
            if current in {"error", "failed", "cancelled", "canceled"}:
                fetch()
                lines = (directory / "kaggle-execution.log").read_text().splitlines()
                failures = [
                    line
                    for line in lines
                    if re.search(
                        r"(Error|Exception|Traceback|FAILED|Storage safety stop)", line
                    )
                ]
                save(
                    directory / "failure-diagnosis.json",
                    {
                        "status": current,
                        "evidence": failures[-30:],
                        "automatic_retry": False,
                        "next_step": "Diagnose logs and preserve state before a safe patched rerun",
                    },
                )
                raise RuntimeError(
                    f"Kaggle run {current}; outputs/logs fetched to {directory}"
                )
            if current not in {"queued", "running", "new", "submitted", "initializing"}:
                raise RuntimeError(
                    "Unknown run state; preserve submission and inspect CLI logs"
                )
            time.sleep(args.poll_seconds)
    elif args.command == "fetch":
        fetch()
    elif args.command == "finish":
        finish()


if __name__ == "__main__":
    # Keep local experiment modules importable when invoked directly from scripts/.
    sys.path.insert(0, str(ROOT))
    main()
