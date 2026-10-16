"""Summarize measured engine behavior without making subjective quality decisions."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .common import CONFIGURATIONS, ROOT, STYLE_MODEL_REVISION, STYLE_REVISION, digest, dump


def number(value, digits=2):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def median(rows: list[dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def peak(rows: list[dict], key: str):
    values = [row["resources"][key] for row in rows
              if row.get("resources", {}).get(key) is not None]
    return max(values) if values else None


def run(directory: Path) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    feedback = manifest.get("user_feedback", {}).get("main-conversational", {})
    optional_feedback = manifest.get("user_feedback", {}).get("optional-conversational", {})
    quality_title = ("Chatterbox preferred; replacements rejected" if feedback
                     else "listening acceptance pending")
    results, summaries = {}, []
    for identifier in CONFIGURATIONS:
        path = directory / f"{identifier}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        results[identifier] = result
        common = [row for row in result["rows"] if row["sentence"] == "conversational"]
        cpu = [row["resources"]["cpu_average_percent_one_core_100"] for row in common]
        reserved = [row["torch_cuda_peak_reserved_mib"] for row in result["rows"]
                    if row.get("torch_cuda_peak_reserved_mib") is not None]
        summaries.append({
            "configuration": identifier, "status": result["status"],
            "cold_startup_seconds": result.get("cold_startup_seconds"),
            "warm_trials_common": len(common),
            "first_audio_median_seconds": median(common, "first_audio_seconds"),
            "synthesis_median_seconds": median(common, "synthesis_seconds"),
            "rtf_median": median(common, "rtf"),
            "cpu_average_percent_one_core_100_median": statistics.median(cpu) if cpu else None,
            "ram_peak_mib": peak(result["rows"], "rss_mib_peak"),
            "nvml_engine_vram_peak_mib": (
                None if manifest.get("hardware_telemetry_problem")
                else peak(result["rows"], "engine_vram_mib_peak")
            ),
            "torch_reserved_peak_mib": max(reserved) if reserved else None,
            "gpu_power_and_fan_comparison": "pending matching NVIDIA driver/library versions",
            "error": result.get("error"),
        })
    lines = [
        f"# Ciel quality shootout — {quality_title}\n",
        "Pocket's subjective quality is rejected. No Pocket reference tuning or inference "
        "was performed in this shootout. Chatterbox, Pocket, Piper, active settings, "
        "and the existing reference remain intact. No replacement engine has been accepted.\n",
        "## Exact implementations and common inputs\n",
        f"- StyleTTS2 LibriTTS: NeuralVox importable implementation `{STYLE_REVISION}`; "
        f"model `{STYLE_MODEL_REVISION}`. CPU/CUDA FP32, diffusion_steps=1/3/5, "
        "alpha=0.3, beta=0.7, embedding_scale=1.\n"
        "- XTTS-v2: maintained `coqui-tts==0.27.5`, model "
        "`6c2b0d75eae4b7047358e3b6bd9325f857d43f77`, PyTorch 2.6.0+cu124, "
        "Transformers 4.57.6. Native streaming with stream_chunk_size=20, "
        "FP32 CUDA, FP16 CUDA autocast with FP32 weights, and FP32 CPU.\n"
        "- Existing Chatterbox Turbo: `chatterbox-tts==0.1.7` from implementation "
        "`5de7a54aa4e5e2baadb0182dde554908b48b85c2`, model "
        "`749d1c1a46eb10492095d68fbcf55691ccf137cd`; existing CUDA worker, inference, "
        "sampling defaults, primary/event references and selected cue assets unchanged.\n",
        "Optional follow-up: NeuTTS-Air `neutts==1.4.1`, `neucodec==0.0.6`, CPU-only "
        "PyTorch 2.8.0, llama-cpp-python 0.3.36 compiled with CUDA disabled; Air Q8_0 model "
        "`855f828ba33d1885443a7a4971785e318414dfd2`, FP32 ONNX decoder "
        "`ac01424f74ea25b245132252168e2191b3d859fa`; one-time native encoder "
        "`a940021324df13a07f44185f0a17c6dac3698179`. Native streaming, original transcript "
        "and cached reference codes, native watermark retained. Cloning quality is unaccepted. "
        "NeuTTS-2E was excluded before inference because it has four fixed speakers. "
        "Air has no explicit emotion controls.\n",
        "NeuCodec's Hub loader passes removed download keywords to the Hub 1.x version "
        "required by Transformers 5.1. Reference preparation therefore loads pinned local "
        "weights using its native constructor/filter/load logic; encoding inference is "
        "unchanged. Runtime uses the supported local `.onnx` path and cached reference "
        "codes, without carrying the reference encoder through every synthesis.\n",
        "Air reference preparation runs once and caches discrete codes. When prepared "
        "before measurement, reported cold startup uses that cache; it does not hide repeat "
        "reference encoding inside per-sentence timings. Its native thread settings are "
        "six llama.cpp generation threads, twelve prefill threads, and ONNX's default "
        "CPU pool; the worker's four-thread setting controls PyTorch only.\n",
        f"Reference: `{manifest['reference']}`. SHA-256: `{manifest['reference_sha256']}`. "
        "The audition copy is byte-identical. No new speaker recordings were downloaded. "
        "StyleTTS2's native reference preprocessing and XTTS's default six-second GPT "
        "conditioning window remain engine-specific; the source waveform is identical.\n",
        "StyleTTS2 and XTTS cache conditioning tensors in `data/voice/states/shootout/`. "
        "Chatterbox prepares conditionals once in the persistent existing worker. "
        "All test outputs retain native audio levels and rate; no post-synthesis "
        "denoising, pitch changes, smoothing, loudness processing, or compression.\n",
        "## Conversational performance\n",
        "Multiple warm iterations per text; actual coverage and repetitions are recorded "
        "in configuration JSON. Cold means a fresh process loading local "
        "assets and preparing/loading conditioning, including imports; no network download. "
        "First audio means the first finite usable waveform/chunk. StyleTTS2 and Turbo "
        "return whole utterances; XTTS streams chunks. Production Ciel's sentence buffering "
        "and playback are not measured here. RTF <1 is faster than spoken audio. "
        "CPU 100% means one logical core; 12-core machine utilization is that value /12. "
        "RAM peaks are process-tree RSS over all warm trials.\n",
        "| Configuration | Warm n | Cold s | First audio s | Total s | RTF | CPU % | RAM MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for value in summaries:
        keys = ("cold_startup_seconds", "first_audio_median_seconds", "synthesis_median_seconds",
                "rtf_median", "cpu_average_percent_one_core_100_median", "ram_peak_mib")
        lines.append("| " + value["configuration"] + f" | {value['warm_trials_common']} | "
                     + " | ".join(number(value[key]) for key in keys) + " |")
    lines += [
        "\nComplete per-sentence times, resource samples, model versions, waveform levels, "
        "and errors are in the configuration JSON files. A mode is never silently "
        "replaced by a fallback. StyleTTS2 one-step inference fails numerically on CPU "
        "and CUDA: the upstream Karras schedule divides by (num_steps - 1). "
        "This is not evidence of a GTX-specific artifact.\n",
        "XTTS model.half() failed with a Float/Half layer-norm mismatch: its upstream "
        "null-position embedding creates FP32 zeros. The separate successful FP16 AMP "
        "test uses standard PyTorch autocast, preserving FP32 parameters/normalizations. "
        "Do not describe it as a half-sized model. Earlier failures and partial outputs "
        "are preserved in `failed-attempts/`.\n",
        "The original XTTS CPU probe had RTF 3.2–5.3. It is not a practical conversational "
        "finalist on this i5; remaining repetitions stopped after every text had "
        "at least two completed trials. CUDA modes cover three trials of all seven "
        "texts. The long text can exceed XTTS's "
        "advisory 250-character English limit while remaining below its 400-token hard "
        "limit. Check its transcript/listening for completion; do not assume the warning "
        "proves truncation or that finite audio proves correct pronunciation.\n",
        "## GPU, VRAM, power, temperature, and fans\n",
        "GPU: GTX 1660 SUPER 6GiB. Loaded kernel: 7.2.4, NVIDIA 610.57.04. Installed "
        "kernel: 7.2.9, NVIDIA module/libraries 615.78.08. `nvidia-smi` reports "
        "Driver/library version mismatch. The installed newer kernel's module matches "
        "the installed libraries; a normal reboot into it should resolve the mismatch. "
        "No system packages, drivers, clocks, power limits, or fan settings were changed.\n",
        "NVML observations were intermittent and process VRAM queries failed; **power, "
        "temperature, fan, and GPU-utilization comparisons are unresolved**. Historical "
        "samples remain in raw JSON but cannot establish reduced fan spikes. StyleTTS2 CPU "
        "left CUDA uninitialized; the stopped XTTS CPU probe lacks its final check. "
        "Torch allocator VRAM, where recorded, covers that "
        "process's reserved tensor memory and omits context/driver/other-process VRAM. "
        "XTTS CUDA FP32 reserves about 5GiB of this 6GiB card in isolation, leaving "
        "limited room for CUDA context, desktop use and concurrent STT. "
        "Fan RPM and audible comfort are unmeasured. STT/LLM were excluded; total-device "
        "activity can still come from unrelated applications.\n",
        "| Configuration | NVML process VRAM peak MiB | Torch reserved peak MiB |",
        "| --- | ---: | ---: |",
    ]
    for value in summaries:
        lines.append(f"| {value['configuration']} | "
                     f"{number(value['nvml_engine_vram_peak_mib'])} | "
                     f"{number(value['torch_reserved_peak_mib'])} |")
    lines += ["\n## Sustained workload\n"]
    for path in sorted(directory.glob("sustained-*.json")):
        value = json.loads(path.read_text())
        rss = [row["rss_after_mib"] for row in value["rows"]]
        width = max(1, len(rss) // 4)
        delta = statistics.median(rss[-width:]) - statistics.median(rss[:width])
        conditioning = (
            "Existing child-worker reference re-encoding count not instrumented. "
            if value["configuration"]["engine"] == "chatterbox"
            else f"{value['conditioning_calls_after_warmup']} extra reference-conditioning calls. "
        )
        lines.append(
            f"- `{value['configuration_id']}`: {value['wall_seconds']:.1f}s paced dialogue, "
            f"{value['turns']} turns, RSS first/last-quartile delta {delta:+.1f}MiB, "
            f"{conditioning}"
            f"Torch reserved peak {number(value.get('torch_cuda_peak_reserved_mib'))}MiB. "
            "Streaming/batch inference stability is measured; power/fan acceptance is pending."
        )
        if delta > 256:
            lines.append(
                "  Memory grew substantially over this run. Allocator retention and a leak "
                "have not been distinguished; do not claim continuous-use memory stability."
            )
    if not list(directory.glob("sustained-*.json")):
        lines.append("Sustained runs have not finished. Do not infer five-minute stability.")
    for identifier, limitation in manifest.get("sustained_workload_limits", {}).items():
        lines.append(f"\n`{identifier}`: {limitation}")
    lines += [
        "\nThese are provisional technical representatives, not listening-selected winners. "
        "The workload paces synthesis against generated audio duration; it does not play "
        "a full conversation, initialize STT/LLM, or prove absence of long-term leaks.\n",
        "## Blind listening and decision\n",
        "Open [index.html](index.html). The main A/B/C comparisons use the same reference "
        "and seven texts. Additional rounds compare diffusion steps, hardware, and XTTS "
        "precision/CPU. Labels shuffle per round and reveal only after a choice. Original "
        "ratings/choice are snapshotted before reveal; download notes for review. The "
        "private manifest contains mappings for reproducibility, so this is casual blinding.\n",
        ("**User result: Sample A in `main-conversational` was Chatterbox, preferred as "
         "most natural/least robotic. The other two samples (StyleTTS2 CPU five-step and "
         "XTTS CUDA FP32) were both rejected as too robotic for everyday use.** "
         "This does not assign ratings to every hardware/precision variant or sentence. "
         if feedback else "**No blind listening preference is recorded for this run.** ") +
        "Cloning similarity, warmth/prosody/expression ratings remain incomplete. "
        "Chatterbox remains the preferred quality baseline. StyleTTS2 CPU avoids "
        "TTS CUDA usage but has significant batch latency; XTTS CPU is impractical. "
        "CUDA alternatives require listening plus reliable sustained power/fan telemetry. "
        "NeuTTS-Air's CPU cloning and GGUF streaming justify one further optional "
        "comparison, not an assumption of naturalness or acceptable latency. "
        "No permanent everyday engine or architecture is selected.\n",
        ("**Optional conversational result: the user described both samples as natural, "
         "with no robotic sound. Sample A was again Chatterbox and described as perfect. "
         "Sample B was NeuTTS-Air; its perceived older/heavier timbre did not match the "
         "preferred Ciel identity. Air passes this naturalness check but fails identity "
         "acceptance.** Its measured CPU RTF is about 2.05, so immediate playback cannot "
         "keep up without buffering/gaps. It is a CPU naturalness candidate, not an "
         "accepted everyday replacement.\n" if optional_feedback else
         "NeuTTS-Air listening and identity acceptance are pending.\n"),
        "Recommendation: Chatterbox remains the best accepted quality/identity candidate. "
        "StyleTTS2 and XTTS main representatives are rejected for robotic delivery. "
        "Air approaches naturalness, but not the required identity or conversational speed. "
        "There is no accepted low-GPU everyday replacement from this evaluation. Keep "
        "production unchanged. Restore matching NVIDIA driver/library versions before "
        "any reliable fan/power decision. A separate Chatterbox power-profile experiment "
        "could preserve the accepted voice; Chatterbox was not optimized here.\n",
        "## Implementation provenance and limitations\n",
        "The upstream-endorsed NeuralVox StyleTTS2 implementation last changed in April "
        "2024. It is a reproducible evaluation baseline, not a claim of active production "
        "maintenance. Its boot adapter changes only explicit device, local checkpoint "
        "paths, and offline tokenizer provisioning; inference functions remain verbatim. "
        "The newer native ultrapg wrapper was inspected but excluded because its steps "
        "argument is unused, so it cannot substantiate this requested step sweep.\n",
        "Primary sources: [StyleTTS2 upstream](https://github.com/yl4579/StyleTTS2), "
        "[importable implementation](https://github.com/NeuralVox/StyleTTS2), "
        "[maintained Coqui implementation](https://github.com/idiap/coqui-ai-TTS), "
        "[XTTS model API](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html), "
        "[native wrapper source](https://github.com/ultrapg/style2tts/blob/"
        "2ceb37dacc9620f6bc8f09cb2765a9914e8c2c57/src/cpp/styletts2_engine.cpp).\n",
        "Optional engine sources: [NeuTTS upstream](https://github.com/neuphonic/neutts), "
        "[Air Q8 model](https://huggingface.co/neuphonic/neutts-air-q8-gguf).\n",
    ]
    transcripts = directory / "transcript-checks.json"
    if transcripts.exists():
        value = json.loads(transcripts.read_text())
        flagged = [row for row in value["rows"] if row["review_flag"]]
        lines.append(
            f"\nCPU ASR review: {len(value['rows'])} samples checked, {len(flagged)} flagged. "
            "This checks expected ending words and adjacent repeated phrases; it cannot "
            "establish naturalness or speaker similarity. See `transcript-checks.json`."
        )
        for row in flagged:
            lines.append(f"- `{row['configuration']}` / `{row['sentence']}`: listen for "
                         "missing/mispronounced ending or repeated words; "
                         "an ASR error is possible.")
    assert digest(Path(manifest["reference"])) == manifest["reference_sha256"]
    if manifest.get("production_env_sha256") is not None:
        assert digest(ROOT / ".env") == manifest["production_env_sha256"]
    dump(directory / "performance-summary.json", {
        "rows": summaries, "quality_decision": manifest.get("user_feedback") or None,
        "production_unchanged": True,
        "hardware_telemetry_problem": manifest.get("hardware_telemetry_problem"),
    })
    path = directory / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
