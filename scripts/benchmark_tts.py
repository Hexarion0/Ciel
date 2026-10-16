"""Compare actual TTS engines in isolated processes; never label fallback as success."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

SENTENCES = {
    "short": "Yes. I'm here.",
    "normal": (
        "You changed the configuration again, didn't you? "
        "Show me what you changed and I'll take a look."
    ),
    "technical": (
        "Your graphics driver is loaded correctly, but the service configuration still "
        "needs to be updated before the change will persist after reboot."
    ),
    "conversation": (
        "Everything looks steady so far. Let's check the remaining settings together, "
        "then we can leave the system alone for a while. "
        "You don't need to keep watching it; I'll tell you if something needs attention."
    ),
}


def cpu_name() -> str:
    """Retain the CPU model on Linux, where platform.processor() can be empty."""
    if name := platform.processor():
        return name
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def gpu_snapshot(pids: set[int]) -> dict:
    """Read NVML through nvidia-smi, without importing CUDA into the test process."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        cells = raw.splitlines()[0].split(",")
        values = [float(cell.strip()) for cell in cells]
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        allocated = 0.0
        for row in apps.splitlines():
            pid, memory = row.split(",")
            if int(pid.strip()) in pids:
                allocated += float(memory.strip())
        return dict(zip(
            ("gpu_util_percent", "gpu_total_vram_mib", "gpu_power_w", "gpu_temp_c"), values,
            strict=True,
        ), engine_vram_mib=allocated)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


class ResourceSampler:
    """Sample process-tree RSS and CPU, plus total-device GPU observations."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.halt = threading.Event()
        self.rows: list[dict] = []
        self.cpu_by_pid: dict[int, float] = {}
        self.start_cpu = self._cpu_seconds()
        self.started = time.perf_counter()
        self.baseline_gpu = gpu_snapshot({self.process.pid})
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self):
        return [self.process, *self.process.children(recursive=True)]

    def _cpu_seconds(self) -> float:
        for process in self._tree():
            try:
                if process.name() == "nvidia-smi":
                    continue
                cpu = process.cpu_times()
                self.cpu_by_pid[process.pid] = cpu.user + cpu.system
            except psutil.Error:
                continue
        return sum(self.cpu_by_pid.values())

    def _sample(self) -> None:
        rss = 0
        pids = set()
        for process in self._tree():
            try:
                if process.name() != "nvidia-smi":
                    rss += process.memory_info().rss
                    pids.add(process.pid)
            except psutil.Error:
                continue
        self.rows.append({"rss_mib": rss / 1024**2, **gpu_snapshot(pids)})

    def _run(self) -> None:
        while not self.halt.is_set():
            self._sample()
            self.halt.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.halt.set()
        self.thread.join(timeout=5)
        self._sample()
        elapsed = time.perf_counter() - self.started
        self.metrics = {
            "cpu_average_percent": 100 * (self._cpu_seconds() - self.start_cpu) / elapsed,
            "rss_peak_mib": max(row["rss_mib"] for row in self.rows),
            "gpu_baseline": self.baseline_gpu,
            "gpu_samples": len([row for row in self.rows if "gpu_util_percent" in row]),
        }
        for key in ("gpu_util_percent", "gpu_total_vram_mib", "engine_vram_mib",
                    "gpu_power_w", "gpu_temp_c"):
            values = [row[key] for row in self.rows if key in row]
            self.metrics[key + "_peak"] = max(values) if values else None


def measure(tts, text: str, *, wav: Path | None = None) -> dict:
    """Time audio-ready and full synthesis separately, excluding disk writes."""
    import numpy as np
    import soundfile as sf

    waves = []
    rate = 0
    first = None
    with ResourceSampler() as sampler:
        start = time.perf_counter()
        for wave, rate in tts.synthesize_stream(text):
            if first is None:
                first = time.perf_counter() - start
            waves.append(wave)
        elapsed = time.perf_counter() - start
    if not waves or tts.last_engine != tts.engine:
        raise RuntimeError(f"Requested {tts.engine}; actual synthesis engine was {tts.last_engine}")
    wave = np.concatenate(waves)
    duration = wave.size / rate
    if wav is not None:
        sf.write(wav, wave, rate)
    return {
        "audio_ready_seconds": first, "synthesis_seconds": elapsed,
        "audio_seconds": duration, "rtf": elapsed / duration,
        "actual_engine": tts.last_engine, **sampler.metrics,
    }


def run_engine(args) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    from ciel.audio.tts import TextToSpeech
    from ciel.config import get_settings

    result = {"engine": args.engine, "status": "error", "rows": []}
    settings = get_settings().audio
    result["configuration"] = {
        "pocket_language": settings.tts_pocket_language,
        "pocket_quantization_requested": settings.tts_pocket_quantization,
        "pocket_reference": settings.tts_pocket_neutral_reference,
        "pocket_references": settings.tts_pocket_references,
        "pocket_builtin_voice": settings.tts_pocket_builtin_voice,
        "piper_voice": args.piper_voice,
    }
    result["versions"] = {}
    for package in ("pocket-tts", "torch", "piper-tts", "psutil"):
        try:
            result["versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    tts = None
    try:
        with ResourceSampler() as sampler:
            start = time.perf_counter()
            tts = TextToSpeech(
                engine=args.engine,
                voice_name=args.piper_voice if args.engine == "piper" else "ciel",
            )
            if not tts.enabled or (args.engine != "piper" and not tts.warmup()):
                raise RuntimeError(f"{args.engine} initialization unavailable; see stderr logs")
            result["cold_startup_seconds"] = time.perf_counter() - start
        result["startup_resources"] = sampler.metrics
        result["cold_synthesis"] = measure(tts, SENTENCES["short"])
        for iteration in range(args.iterations):
            for name, text in SENTENCES.items():
                wav = args.output.parent / f"{args.engine}-{name}.wav" if iteration == 0 else None
                result["rows"].append({
                    "sentence": name, "iteration": iteration + 1,
                    **measure(tts, text, wav=wav),
                })
        result["status"] = "ok"
        if tts._pocket is not None:
            result["pocket_quantization_effective"] = tts._pocket.effective_quantization
        result["summary"] = {}
        for name in SENTENCES:
            rows = [row for row in result["rows"] if row["sentence"] == name]
            result["summary"][name] = {
                key + "_median": statistics.median(row[key] for row in rows)
                for key in ("audio_ready_seconds", "synthesis_seconds", "rtf")
            }
    except Exception as err:
        result["error"] = str(err)
    finally:
        if tts is not None:
            tts.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["pocket", "chatterbox_turbo", "piper"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/tts/report.json"))
    parser.add_argument("--piper-voice", default="en_US-ciel-medium")
    parser.add_argument("--pocket-reference", type=Path)
    parser.add_argument("--builtin-pocket", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int8"], default="none")
    parser.add_argument(
        "--engine", choices=["pocket", "piper", "chatterbox_turbo"], help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("At least two warm iterations are required")
    if args.builtin_pocket and args.pocket_reference:
        parser.error("Choose either --builtin-pocket or --pocket-reference")
    os.environ["TTS_POCKET_QUANTIZATION"] = args.quantization
    if args.pocket_reference:
        os.environ["TTS_POCKET_NEUTRAL_REFERENCE"] = str(args.pocket_reference.resolve())
    elif args.builtin_pocket:
        os.environ["TTS_POCKET_NEUTRAL_REFERENCE"] = ""
        os.environ["TTS_POCKET_REFERENCES"] = "data/voices/pocket-builtin-preview"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.engine:
        args.output.write_text(json.dumps(run_engine(args), indent=2) + "\n")
        return
    report = {
        "platform": platform.platform(), "cpu": cpu_name(),
        "logical_cpus": psutil.cpu_count(), "python": sys.version,
        "notes": (
            "Cold = fresh process/model with filesystem cache retained; downloads excluded "
            "only when pre-cached. Warm rows follow one untimed-in-summary synthesis. "
            "CPU is process-tree CPU time/wall time (100% = one logical core). "
            "RAM is summed RSS, not unique RAM. GPU utilization/power/temperature and total "
            "VRAM describe the whole device, including other applications. Engine VRAM "
            "is compute-process allocation. Polling can miss short power spikes. "
            "No playback, STT, LLM, or listening-quality measurement is included."
        ),
        "sentences": SENTENCES, "engines": [],
    }
    for engine in args.engines:
        if engine not in {"pocket", "piper", "chatterbox_turbo"}:
            parser.error(f"Unknown engine: {engine}")
        partial = args.output.parent / f"{engine}.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--engine", engine,
            "--iterations", str(args.iterations), "--output", str(partial),
            "--piper-voice", args.piper_voice, "--quantization", args.quantization,
        ]
        print(f"Benchmarking {engine} ({args.iterations} warm iterations)...", flush=True)
        try:
            subprocess.run(command, check=True, timeout=1800)
            report["engines"].append(json.loads(partial.read_text()))
        except (subprocess.SubprocessError, OSError) as err:
            report["engines"].append({"engine": engine, "status": "error", "error": str(err)})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    for item in report["engines"]:
        print(item["engine"], item["status"], item.get("summary", item.get("error")))


if __name__ == "__main__":
    main()
