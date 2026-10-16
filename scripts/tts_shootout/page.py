"""Blind comparisons with persisted random assignments and explicit choice-before-reveal."""

from __future__ import annotations

import html
import json
import secrets
import shutil
from pathlib import Path

from .common import SENTENCES, digest, dump

RATINGS = {
    "naturalness": "Naturalness", "similarity": "Speaker similarity",
    "robotic": "Robotic quality (higher = more robotic)", "warmth": "Warmth",
    "prosody": "Prosody", "pronunciation": "Pronunciation", "expression": "Expression",
    "overall": "Overall preference",
}


def available(directory: Path) -> dict:
    """Only completed valid inference audio can enter listening rounds."""
    manifest = json.loads((directory / "manifest.json").read_text())
    result = {}
    for identifier in manifest["configurations"]:
        path = directory / f"{identifier}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            if value["status"] in {"ok", "impractical_cpu"}:
                expected = manifest.get("reference_sha256")
                if expected and value.get("reference_sha256") != expected:
                    raise ValueError(f"{identifier} used a different reference; reject comparison")
                result[identifier] = value
    return result


def build_rounds(directory: Path, manifest: dict, results: dict) -> None:
    """Keep A/B/C assignments stable across page refreshes; never reveal filenames."""
    definitions = []
    primary = ["style-cpu-5", "xtts-cuda-fp32", "chatterbox-baseline"]
    if all(identifier in results for identifier in primary):
        for name in SENTENCES:
            definitions.append((f"main-{name}", name, f"{name.capitalize()} · main comparison",
                                primary))
    variants = [
        ["style-cpu-3", "style-cpu-5"], ["style-cuda-3", "style-cuda-5"],
        ["style-cpu-5", "style-cuda-5"],
        ["xtts-cuda-fp16", "xtts-cuda-fp32", "xtts-cpu-fp32"],
    ]
    for index, configurations in enumerate(variants, 1):
        for name in ("conversational", "technical"):
            valid = [identifier for identifier in configurations if identifier in results
                     and any(row["sentence"] == name for row in results[identifier]["rows"])]
            if len(valid) >= 2:
                definitions.append((f"extra-{index}-{name}", name,
                                    f"Additional comparison {index} · {name}", valid))
    optional = ["neutts-air-cpu-q8", "chatterbox-baseline"]
    if all(identifier in results for identifier in optional):
        for name in SENTENCES:
            definitions.append((f"optional-{name}", name,
                                f"New candidate comparison · {name}", optional))
    previous = {item["id"] for item in manifest["rounds"]}
    for identifier, name, title, configurations in definitions:
        if identifier in previous:
            continue
        shuffled = list(configurations)
        secrets.SystemRandom().shuffle(shuffled)
        samples = []
        for label, configuration in zip("ABC", shuffled):
            row = next(row for row in results[configuration]["rows"]
                       if row["sentence"] == name and row["iteration"] == 1)
            source = Path(row["audio"])
            destination = directory / "blind" / identifier / f"sample-{label}.wav"
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            samples.append({"label": label, "configuration": configuration,
                            "audio": str(destination.relative_to(directory)),
                            "audio_sha256": digest(destination)})
        manifest["rounds"].append({"id": identifier, "sentence": name, "title": title,
                                   "samples": samples})
    manifest["rounds"].sort(key=lambda row: (
        not row["id"].startswith("main-"),
        list(SENTENCES).index(row["sentence"]) if row["id"].startswith("main-") else row["id"],
    ))
    manifest["status"] = "awaiting_blind_listening" if definitions else "awaiting_measurements"
    dump(directory / "manifest.json", manifest)


def identity(configuration: str, manifest: dict) -> str:
    value = manifest["configurations"][configuration]
    engine = {"styletts2": "StyleTTS2", "xtts": "XTTS-v2", "chatterbox": "Chatterbox Turbo",
              "neutts": "NeuTTS-Air"}[
        value["engine"]
    ]
    detail = f"{engine} · {value['device'].upper()} · {value['precision']}"
    if "steps" in value:
        detail += f" · diffusion_steps={value['steps']}"
    return detail


def render(directory: Path) -> Path:
    """Create a local page containing no visible engine identity before a choice."""
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    results = available(directory)
    build_rounds(directory, manifest, results)
    sections, identities = [], {}
    for round_ in manifest["rounds"]:
        identifier = round_["id"]
        identities[identifier] = {}
        sample_html = []
        for sample in round_["samples"]:
            label = sample["label"]
            identities[identifier][label] = identity(sample["configuration"], manifest)
            fields = []
            for key, title in RATINGS.items():
                fields.append(
                    f'<label>{html.escape(title)}<select data-rating="{key}" '
                    f'aria-label="Sample {label}: {html.escape(title)}">'
                    '<option value="">Unrated</option>'
                    + ''.join(f'<option>{score}</option>' for score in range(1, 6))
                    + '</select></label>'
                )
            sample_html.append(f'''
<article class="sample" data-label="{label}">
<h3>Sample {label}</h3>
<audio controls preload="none" src="{html.escape(sample['audio'], quote=True)}"
 aria-label="{html.escape(round_['title'], quote=True)} sample {label}"></audio>
<p class="audio-error" hidden>Audio unavailable. Keep this page beside its audio folders.</p>
<details><summary>Rate this sample</summary><div class="ratings">{''.join(fields)}</div></details>
<p class="identity" aria-live="polite"></p>
</article>''')
        choices = ''.join(f'<option>{sample["label"]}</option>' for sample in round_["samples"])
        sections.append(f'''
<section class="round" id="{identifier}" data-round="{identifier}">
<h2>{html.escape(round_['title'])}</h2>
<p class="text">{html.escape(SENTENCES[round_['sentence']])}</p>
<div class="samples">{''.join(sample_html)}</div>
<div class="decision"><label>Your choice<select data-choice>
<option value="">Choose after listening</option>{choices}
<option value="tie">No meaningful difference</option>
<option value="none">None is acceptable</option></select></label>
<button data-reveal disabled>Reveal identities after choosing</button></div>
<label class="notes">Listening notes<textarea data-notes rows="2"
 placeholder="What sounded human? What felt robotic or inconsistent?"></textarea></label>
</section>''')
    pending = (
        '<p class="empty">The main three-engine comparison is still being generated. '
        'Completed configuration comparisons appear below.</p>'
        if not any(round_["id"].startswith("main-") for round_ in manifest["rounds"]) else ''
    )
    settings = {"run_id": manifest["run_id"], "identities": identities}
    template = Path(__file__).with_name("listening.html").read_text()
    output = template.replace("__ROUNDS__", pending + ''.join(sections)).replace(
        "__RUN__", html.escape(manifest["run_id"]),
    ).replace("__SETTINGS__", json.dumps(settings).replace("<", "\\u003c"))
    path = directory / "index.html"
    path.write_text(output, encoding="utf-8")
    return path
