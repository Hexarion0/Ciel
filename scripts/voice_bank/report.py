"""Produce a concrete draft bank proposal and an audition handoff report."""

from __future__ import annotations

import json
from pathlib import Path

from .analyze import dump
from .audio import sha256


def run(directory: Path) -> Path:
    """Report measured facts separately from pending listening/activation decisions."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("human_acceptance"):
        raise ValueError("An accepted bank needs an explicit new review run")
    candidates = {row["id"]: row for row in manifest["candidates"]}
    proposal = {
        "status": "draft_requires_user_listening_and_acceptance", "active": False,
        "quantization_default": "fp32", "human_acceptance": None, "styles": {},
        "runtime_directory_after_acceptance": "data/voice/references/ciel/bank/" + directory.name,
        "existing_runtime_reference_preserved": "data/voice/references/ciel/reference-1.wav",
    }
    for style, selection in manifest["recommendation"].items():
        candidate = candidates.get(selection["candidate"])
        proposal["styles"][style] = {
            **selection,
            "state_after_acceptance": candidate["states"]["fp32"]["path"] if candidate else None,
            "provenance_reference": candidate["reference"] if candidate else None,
        }
    dump(directory / "bank-proposal.json", proposal)
    lines = [
        "# Ciel voice-bank audition — unaccepted draft\n",
        "No new bank has been activated. The existing reference and FP32 default remain active. "
        "Listen before choosing references, delivery labels, or numerical mode.\n",
        "## Sources and exclusions\n",
        "User permission covers all four supplied videos. All four best-available original "
        "English audio streams are archived. Metadata identifies the same Clem Whispers "
        "channel. All four contain retained reference material. No whole source was "
        "rejected as a different speaker or dub.\n",
        "| Source | Full duration | Windows screened | Selected clips | Excluded opening |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for source in manifest["sources"]:
        lines.append(
            f"| [{source['id']}]({source['url']}) | {source['duration_seconds']}s | "
            f"{source['windows_screened']} | {', '.join(source['shortlisted_candidates'])} | "
            f"{source['excluded_intro_seconds']:.0f}s |"
        )
    lines += [
        "\nPromotional introductions were excluded from the first/fourth sources. Initial "
        "opening fragments were avoided in the others. Caption windows with explicit sound "
        "effects/kissing/shouting context and clipped/unstable screening measurements were "
        "not put into the automatic shortlist. Source 2 is a thunderstorm roleplay: listen "
        "for residual environmental sound. Waveform scores do not prove absence of music, "
        "overlap, reverb, breathiness, or whispering.\n",
        "## Candidate files and exact source intervals\n",
        "All paths below are relative to this report. Every reference has an unprocessed "
        "comparison and complete provenance in `manifest.json`. No prior sample was deleted.\n",
        "| Candidate | Intended style | Source | Exact interval | Original / prepared |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in manifest["candidates"]:
        original = row.get("original_duration_seconds")
        prepared = row.get("processed_duration_seconds")
        duration = (
            f"{original:.3f}s / {prepared:.3f}s"
            if original is not None else "Existing 12.26s"
        )
        lines.append(
            f"| [{row['id']}]({row['reference']}) | {row['intended_style']} | "
            f"{row['source_id']} | {row['source_timestamp']} | {duration} |"
        )
    lines += [
        "\nWarm-03 keeps 280ms more original audio than warm-01 to include its final "
        "syllable decay and quiet tail. Warm-01 remains for comparison, with its edge-energy "
        "flag recorded. No pitch, formant, or tempo manipulation was performed.\n",
        "## Preprocessing\n",
        "Raw candidates retain decoded original-rate stereo speech. Preparation consists "
        "only of quiet-edge trimming with 100ms padding, mono downmix, polyphase resampling "
        "to 24kHz, and bounded linear gain (at most +6dB; peak at most 0.85). No denoising, "
        "source separation, high-pass, click removal, compression, or synthetic emotion "
        "processing was applied. This prioritizes natural raw speech; inspect noise by ear.\n",
        "## Speaker consistency\n",
    ]
    identity = json.loads((directory / "speaker-candidates.json").read_text())
    values = [value for key, row in identity["pairwise_cosines"].items()
              for other, value in row.items() if other != key]
    within = list(identity["within_clip_cosines"].values())
    lines += [
        f"CPU speaker-encoder candidate-to-candidate cosines range **{min(values):.3f}–"
        f"{max(values):.3f}**; within-clip halves range **{min(within):.3f}–"
        f"{max(within):.3f}**. These are uncalibrated proxies, not calibrated probabilities "
        "or proof of real-world identity. Together with consistent original-track/channel "
        "metadata they support auditioning one speaker across the clips. Human listening "
        "still must confirm identity and acceptable role/microphone variation.\n",
        "Only the existing cached 5.5MiB Chatterbox speaker encoder performed screening, "
        "on CPU. It generated no speech; all test audio is Pocket. No unrelated speakers "
        "or reference states were averaged together.\n",
        "## Provisional best choices / proposed bank\n",
        "These are starting points selected by waveform, context, and boundary checks. "
        "They are **not listening-confirmed winners**.\n",
        "| Delivery | Candidate | Current acceptance |",
        "| --- | --- | --- |",
    ]
    for style, value in proposal["styles"].items():
        choice = value["candidate"] or "Neutral fallback"
        lines.append(f"| {style} | {choice} | Pending listening |")
    lines += [
        "\nFirm and focused need better controlled/analytical material or a listening-confirmed "
        "reclassification of an existing clip. No file was manufactured to complete the set. "
        "The other four labels are also provisional; reject or relabel them if their delivery "
        "does not meet Ciel's composed adult feminine target.\n",
        "## FP32 / INT8 and audition page\n",
        "Open [the local listening page](index.html). Every candidate, including the baseline, "
        "has identical-common-text FP32/INT8 samples and a style-specific line. The page "
        "shows anonymous A/B samples; labels reveal after both finish playing. Both modes "
        "used the same CPU RNG seed and model language. Preview generation left CUDA "
        "uninitialized. Speed does not establish equivalent quality. FP32 stays the default "
        "until listening finds INT8 effectively unchanged in similarity, diction, artifacts, "
        "warmth, prosody, naturalness, and emotional character.\n",
        "Exported `.safetensors` states live in `data/voice/states/ciel/`, keyed by "
        "recording SHA-256, Pocket 3.3.0, language, and numerical mode. Each reference "
        "encodes once per mode, and both sentences reuse its state with `copy_state=True`. "
        "The proposal records exact FP32 state paths for acceptance-time activation.\n",
        "## Sustained and actual playback checks\n",
    ]
    stress = json.loads((directory / "stress-candidate-neutral-01.json").read_text())
    delta = (stress["warm_rss_last_quartile_median_mib"]
             - stress["warm_rss_first_quartile_median_mib"])
    lines += [
        f"Provisional neutral-01 ran for **{stress['wall_seconds']:.1f} wall-clock seconds**, "
        f"**{stress['turns']} turns**, with **{stress['interrupted_generations']} "
        f"canceled generations** and **{stress['total_generated_audio_seconds']:.1f}s** "
        "of produced audio. Streaming remained active, canceled workers unwound, "
        "one reference state stayed cached, and there were zero additional reference "
        "loads/encodes after warmup. CUDA stayed uninitialized; sampled TTS-attributed "
        f"VRAM peaked at **{stress['engine_vram_peak_mib']:.0f}MiB**.\n",
        f"Warm RSS first/last-quartile medians were **"
        f"{stress['warm_rss_first_quartile_median_mib']:.1f} / "
        f"{stress['warm_rss_last_quartile_median_mib']:.1f}MiB** (delta **{delta:+.1f}MiB**); "
        f"observed peak **{stress['rss_peak_mib']:.1f}MiB**. There was no obvious sustained "
        "growth in this short check. It is not proof against long-term leaks or a fan test. "
        "The paced simulation did not play its entire conversation; STT was excluded.\n",
        "Separately, the existing real Ciel streamed-reply playback and real interruption "
        "integration checks passed with neutral-01's exported state (2 tests). These used "
        "real Pocket inference and speaker output, with a local token fixture. The runtime "
        "reference was not changed. Desktop/mobile page renders and blind-gate/notes/filter "
        "interaction checks also passed.\n",
        "## Better than baseline? Permanent everyday voice?\n",
        "**Not determined by listening yet.** The preserved `reference-1.wav` is included "
        "as the control. Speaker cosines, ASR, and cleaner boundaries cannot establish "
        "subjective superiority. Pocket's CPU engine is technically ready; this new "
        "identity/style bank is not accepted. Keep the existing baseline and FP32 until "
        "the user auditions, selects references, and approves activation.\n",
        "Download listening notes from the page and identify the neutral/warm/amused/soft "
        "choices and A/B preferences. No final bank files or `.env` entries have been "
        "changed. Normal Pocket→Piper and HQ Chatterbox→Pocket→Piper remain intact.\n",
    ]
    baseline = candidates["baseline-reference-1"]
    current_baseline = Path("data/voice/references/ciel/reference-1.wav")
    unchanged = sha256(current_baseline) == baseline["reference_sha256"]
    if not unchanged:
        raise RuntimeError("Existing runtime baseline changed; stop before accepting anything")
    proposal["baseline_integrity_sha256"] = baseline["reference_sha256"]
    proposal["baseline_integrity_verified"] = unchanged
    dump(directory / "bank-proposal.json", proposal)
    report = directory / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
