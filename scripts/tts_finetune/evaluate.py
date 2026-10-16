"""Experimental checkpoint streaming/telemetry and rating-gated blind comparisons."""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import secrets
import shutil
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .common import EVALUATION_TEXTS, ROOT, digest, dump, safe_path


def evaluate(engine: str, output: Path, checkpoint: Path | None = None, device: str = "cuda",
             iterations: int = 3, sustained_seconds: int = 0,
             experiment: Path | None = None) -> Path:
    """No provider substitution; require multiple warm trials and optional five-minute load."""
    started = time.perf_counter()
    model_directory = None
    experimental_reference = None
    if experiment:
        if checkpoint or engine != "xtts":
            raise ValueError("Use an experimental XTTS manifest OR --checkpoint")
        value = json.loads(experiment.read_text())
        if value.get("production_enabled") is not False or value.get("engine") != "xtts":
            raise ValueError("Expected an experimental XTTS registration")
        checkpoint = safe_path(experiment.parent, value["checkpoint"])
        model_directory = safe_path(experiment.parent, value["model_directory"])
        experimental_reference = safe_path(experiment.parent, value["reference"])
    if iterations < 3 or (sustained_seconds != 0 and not 300 <= sustained_seconds <= 600):
        raise ValueError("Use at least 3 warm iterations and a 300–600 second finalist workload")
    if checkpoint and engine != "xtts":
        raise ValueError("Only supported XTTS checkpoints can be evaluated by this adapter")
    if checkpoint:
        sidecar = checkpoint.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError("Run inference-xtts first to omit optimizer tensors for local loading")
        metadata = json.loads(sidecar.read_text())
        if (metadata.get("inference_sha256") != digest(checkpoint)
                or metadata.get("optimizer_omitted") is not True
                or metadata.get("epoch", 0) < 1):
            raise ValueError("Checkpoint does not match a trained inference export")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # Match the established offline shootout environment before optional imports.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      NLTK_DATA=str(ROOT / "data/voice/cache/nltk"))
    if device == "cpu" and engine != "chatterbox":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    from scripts.tts_shootout.providers import load
    from scripts.tts_shootout.telemetry import Sampler
    from scripts.tts_shootout.worker import measure

    reference = ROOT / "data/voice/references/ciel/reference-1.wav"
    if experimental_reference and digest(experimental_reference) != digest(reference):
        raise ValueError("Experimental reference differs from the apples-to-apples baseline")
    configuration = {"engine": engine, "device": device, "precision": "fp32"}
    if engine == "chatterbox":
        configuration.update(device="cuda", precision="existing unchanged")
    if engine == "styletts2":
        configuration["steps"] = 5
    if checkpoint:
        configuration["checkpoint"] = str(checkpoint)
    if model_directory:
        configuration["model_directory"] = str(model_directory)
    torch.set_num_threads(4)
    result = {"status": "running", "configuration": configuration,
              "label": f"{engine} {'fine-tuned' if checkpoint else 'zero-shot baseline'}",
              "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
              "reference_sha256": digest(reference), "texts": EVALUATION_TEXTS, "rows": [],
              "resource_scope": "GPU power/utilization/temperature/fan are whole-device; "
                                "engine VRAM and CPU/RAM use process tree. Fan is %, not RPM.",
              "listening_acceptance": None}
    result["versions"] = {}
    for name in ("torch", "coqui-tts", "coqui-tts-trainer", "transformers"):
        try:
            result["versions"][name] = version(name)
        except PackageNotFoundError:
            pass
    provider = None
    try:
        with Sampler() as sampler:
            provider = load(configuration, reference, ROOT / "data/voice/finetuning/cache")
        result["cold_startup_seconds"] = time.perf_counter() - started
        result["startup_resources"] = sampler.metrics
        result["conditioning"] = {"state": provider.state_path,
                                  "encoded_this_run": provider.conditioning_encoded,
                                  "loads": provider.conditioning_loads}
        torch.manual_seed(73)
        result["cold_synthesis"] = measure(provider, EVALUATION_TEXTS["short"])
        for iteration in range(iterations):
            for name, text in EVALUATION_TEXTS.items():
                torch.manual_seed(73 + iteration)
                random.seed(73 + iteration)
                np.random.seed(73 + iteration)
                destination = output / "audio" / f"{name}.wav" if iteration == 0 else None
                result["rows"].append({"sentence": name, "text": text,
                                       "iteration": iteration + 1,
                                       **measure(provider, text, destination)})
                dump(output / "result.json", result)
        if sustained_seconds:
            deadline = time.monotonic() + sustained_seconds
            result["sustained"] = []
            index = 0
            with Sampler() as workload:
                while time.monotonic() < deadline:
                    name = list(EVALUATION_TEXTS)[index % len(EVALUATION_TEXTS)]
                    row = {"sentence": name, **measure(provider, EVALUATION_TEXTS[name])}
                    result["sustained"].append(row)
                    idle = max(0, row["audio_seconds"] - row["synthesis_seconds"]) + 1
                    time.sleep(min(idle, 30, max(0, deadline - time.monotonic())))
                    index += 1
                    dump(output / "result.json", result)
            result["sustained_resources"] = workload.metrics
            dump(output / "sustained-telemetry.json", {"samples": workload.rows})
        result["summary"] = {
            name: {key: statistics.median(row[key] for row in result["rows"]
                                         if row["sentence"] == name)
                   for key in ("first_audio_seconds", "synthesis_seconds", "rtf")}
            for name in EVALUATION_TEXTS
        }
        result["conditioning"]["loads_after_synthesis"] = provider.conditioning_loads
        result["status"] = "complete_awaiting_listening"
    except BaseException as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if provider is not None:
            provider.close()
        result["cuda_initialized_in_evaluator"] = torch.cuda.is_initialized()
        result["engine_cuda_initialized"] = (torch.cuda.is_initialized()
                                             or engine == "chatterbox")
        dump(output / "result.json", result)
    return output / "result.json"


def listening(output: Path, results: list[Path]) -> Path:
    """Build new rounds only from matching reference/text, never imagined checkpoint audio."""
    from scripts.tts_shootout.page import RATINGS

    ratings = {**RATINGS, "conversational_feel": "Conversational feel"}
    output.mkdir(parents=True, exist_ok=True)
    reference = ROOT / "data/voice/references/ciel/reference-1.wav"
    reference_hash = digest(reference)
    shutil.copy2(reference, output / "reference.wav")
    values = []
    for path in results:
        value = json.loads(path.read_text())
        if value["status"] != "complete_awaiting_listening":
            raise ValueError("Only complete synthesis runs may enter a blind comparison")
        if value["reference_sha256"] != reference_hash or value["texts"] != EVALUATION_TEXTS:
            raise ValueError("Comparison must use identical speaker reference and seven texts")
        values.append({"key": digest(path), "result": value, "directory": path.parent.resolve()})
    manifest_path = output / "assignments.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"rounds": []}
    run_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    settings, sections = {"run_id": run_key, "identities": {}, "require_ratings": True}, []
    batch_key = hashlib.sha256("".join(sorted(v["key"] for v in values)).encode()).hexdigest()[:12]
    if len(values) >= 2:
        for name, text in EVALUATION_TEXTS.items():
            identifier = f"{batch_key}-{name}"
            existing = next((r for r in manifest["rounds"] if r["id"] == identifier), None)
            if existing is None:
                shuffled = values.copy()
                secrets.SystemRandom().shuffle(shuffled)
                existing = {"id": identifier, "sentence": name, "samples": []}
                for index, value in enumerate(shuffled):
                    label = chr(65 + index)
                    row = next(r for r in value["result"]["rows"]
                               if r["sentence"] == name and r["iteration"] == 1)
                    if row["text"] != text:
                        raise ValueError("Audio text differs from the common evaluation sentence")
                    audio = Path(row["audio"])
                    if not audio.is_absolute():
                        audio = safe_path(value["directory"], row["audio"])
                    elif not audio.resolve().is_relative_to(value["directory"]):
                        raise ValueError("Result audio must remain inside its synthesis run")
                    destination = output / "blind" / identifier / f"sample-{label}.wav"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(audio, destination)
                    existing["samples"].append({"label": label,
                        "identity": value["result"]["label"],
                        "checkpoint_sha256": value["result"]["checkpoint_sha256"],
                        "audio": str(destination.relative_to(output)),
                        "audio_sha256": digest(destination)})
                manifest["rounds"].append(existing)
    for round_ in manifest["rounds"]:
        identifier = round_["id"]
        settings["identities"][identifier] = {}
        samples = []
        for sample in round_["samples"]:
            label = sample["label"]
            settings["identities"][identifier][label] = (
                sample["identity"] + (f" · {sample['checkpoint_sha256'][:12]}"
                                      if sample["checkpoint_sha256"] else "")
            )
            fields = ''.join(
                f'<label>{html.escape(title)}<select data-rating="{key}" '
                f'aria-label="Sample {label}: {html.escape(title)}">'
                '<option value="">Unrated</option><option value="uncertain">Unsure</option>'
                + ''.join(f'<option>{score}</option>' for score in range(1, 6))
                + '</select></label>' for key, title in ratings.items()
            )
            samples.append(f'''<article class="sample" data-label="{label}"><h3>Sample {label}</h3>
<audio controls preload="none" src="{html.escape(sample['audio'], quote=True)}"
 aria-label="{round_['sentence']} Sample {label}"></audio>
<p class="audio-error" hidden>Audio unavailable; keep the blind audio folder beside this page.</p>
<details><summary>Rate this sample</summary><div class="ratings">{fields}</div></details>
<p class="identity" aria-live="polite"></p></article>''')
        choices = ''.join(f'<option>{s["label"]}</option>' for s in round_["samples"])
        sections.append(f'''<section class="round" data-round="{identifier}">
<h2>{round_['sentence'].capitalize()}</h2>
<p class="text">{html.escape(EVALUATION_TEXTS[round_['sentence']])}</p>
<div class="samples">{''.join(samples)}</div><div class="decision">
<label>Your choice<select data-choice><option value="">Choose after listening</option>{choices}
<option value="tie">No meaningful difference</option>
<option value="none">None is acceptable</option>
</select></label><button data-reveal disabled>Reveal after ratings and choice</button></div>
<label class="notes">Listening notes<textarea data-notes rows="2"></textarea></label></section>''')
    if not manifest["rounds"]:
        sections.append('<p class="empty">No fine-tuned checkpoint is available. Fine-tuning and '
                        'checkpoint synthesis have not run. No quality improvement is claimed.</p>')
    dump(manifest_path, manifest)
    template = Path(__file__).with_name("listening.html").read_text()
    output_html = template.replace("__ROUNDS__", ''.join(sections)).replace(
        "__RUN__", html.escape(output.name),
    ).replace("__SETTINGS__", json.dumps(settings).replace("<", "\\u003c"))
    path = output / "index.html"
    path.write_text(output_html, encoding="utf-8")
    return path
