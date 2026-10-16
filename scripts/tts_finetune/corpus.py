"""Build non-overlapping single-speaker candidates; verification is never inferred from ASR."""

from __future__ import annotations

import csv
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from scripts.voice_bank.analyze import permission
from scripts.voice_bank.audio import metrics, vtt_words

from .common import INTRO_END, ROOT, SOURCE_IDS, digest, dump


def caption_text(text: str) -> str:
    """Remove non-spoken YouTube cue markup without changing the spoken words."""
    return " ".join(re.sub(r"\[[^]]+\]|>>", " ", text).split())


def utterances(words: list[dict], duration: float, minimum: float) -> list[dict]:
    """Prefer sentence/pause boundaries, using every caption word at most once."""
    rows, index = [], 0
    while index < len(words):
        if words[index]["start"] < minimum:
            index += 1
            continue
        start = max(minimum, words[index]["start"] - 0.2)
        end_index = index
        chosen = None
        while end_index < len(words):
            word = words[end_index]
            next_start = (words[end_index + 1]["start"]
                          if end_index + 1 < len(words) else duration)
            # Leave lead-in before the next captioned word. Padding after its
            # start would put its first phoneme in this clip and truncate it in
            # the following one. Captions still require audible boundary QA.
            end = min(duration, next_start - 0.1, word["start"] + 0.92)
            if end - start > 11.5:
                break
            chosen = (end_index, end)
            gap = next_start - word["start"]
            punctuation = bool(re.search(r"[.!?]$", word["text"]))
            if ((end - start >= 5.0 and punctuation)
                    or (end - start >= 3.2 and gap > 1.0)):
                break
            if gap > 2.0:
                break
            end_index += 1
        if chosen is None:
            index += 1
            continue
        last, end = chosen
        caption = " ".join(word["text"] for word in words[index:last + 1]).strip()
        if end - start >= 3.0 and len(caption.split()) >= 4:
            # Cut at the following word's start if padding would overlap its clip.
            if rows and start < rows[-1]["end"]:
                start = rows[-1]["end"]
            if end - start >= 3.0:
                rows.append({"start": round(start, 3), "end": round(end, 3), "caption": caption})
        index = last + 1
    return rows


def prepare_audio(wave: np.ndarray, rate: int, target: int = 22050) -> tuple[np.ndarray, dict]:
    """One resample, padded quiet-edge trim, and bounded linear level adjustment."""
    mono = wave.mean(axis=1) if wave.ndim == 2 else wave
    divisor = math.gcd(rate, target)
    audio = resample_poly(mono, target // divisor, rate // divisor).astype(np.float32)
    width = max(1, round(target * 0.02))
    count = len(audio) // width
    if count == 0 or not np.isfinite(audio).all():
        raise ValueError("Empty/non-finite audio")
    rms_frames = np.sqrt(np.mean(audio[:count * width].reshape(count, width) ** 2, axis=1))
    active = np.flatnonzero(rms_frames > max(0.001, float(rms_frames.max()) * 0.02))
    if not active.size:
        raise ValueError("No audible speech candidate")
    first = max(0, int(active[0]) * width - round(target * 0.1))
    last = min(len(audio), (int(active[-1]) + 1) * width + round(target * 0.1))
    original_samples = len(audio)
    audio = audio[first:last]
    rms_db = 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(audio**2)))))
    gain_db = min(3.0, max(-3.0, -23.0 - rms_db))
    gain_db = min(gain_db, 20 * np.log10(0.95 / max(1e-9, float(np.abs(audio).max()))))
    audio *= 10 ** (gain_db / 20)
    return audio, {
        "mono": "equal channel mean", "resample": f"polyphase {rate} -> {target}",
        "trim_left_seconds": first / target,
        "trim_right_seconds": (original_samples - last) / target,
        "edge_padding_seconds": 0.1, "linear_gain_db": float(gain_db),
        "normalization_target_dbfs_rms": -23, "normalization_gain_cap_db": 3,
        "denoising": False, "compression": False, "pitch_or_formant_changes": False,
    }


def prepare(source: Path, output: Path) -> Path:
    """Retain raw candidates and source provenance; produce no training metadata yet."""
    rights = permission(source)
    output = output.resolve()
    if (output / "dataset.json").exists():
        raise FileExistsError("Retain the existing dataset or use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "clips"):
        (output / name).mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1, "dataset_id": output.name, "speaker": "ciel",
        "status": "candidate_dataset_requires_listening_and_transcript_verification",
        "segmentation": "caption boundaries v2: next-word lead-in, prefer longer utterances",
        "permission": rights, "sources": [], "clips": [], "rejections": [],
        "production_env_sha256": digest(ROOT / ".env") if (ROOT / ".env").is_file() else None,
        "baseline_sha256": digest(ROOT / "data/voice/references/ciel/reference-1.wav"),
        "verification_policy": "ASR/caption agreement is a review aid. Only explicit reviewed "
                               "transcript, single-speaker identity, and acoustic-quality "
                               "acceptance can enter training/validation exports.",
    }
    for sid in SOURCE_IDS:
        metadata = json.loads((source / f"source-{sid}.info.json").read_text())
        original = source / f"source-{sid}.decoded.wav"
        archive = source / f"source-{sid}.webm"
        caption_file = source / f"source-{sid}.en-orig.vtt"
        info = sf.info(original)
        source_hash = digest(archive)
        manifest["sources"].append({
            "id": sid, "url": f"https://www.youtube.com/watch?v={sid}",
            "channel": metadata.get("channel"), "title": metadata["title"],
            "raw_seconds": info.duration, "archive_sha256": source_hash,
            "decoded_sha256": digest(original), "caption_sha256": digest(caption_file),
            "excluded_intro_seconds": INTRO_END.get(sid, 12.0),
        })
        proposals = utterances(vtt_words(caption_file), info.duration, INTRO_END.get(sid, 12.0))
        with sf.SoundFile(original) as recording:
            for proposal in proposals:
                identifier = f"{sid}-{round(proposal['start'] * 1000):07d}"
                recording.seek(round(proposal["start"] * info.samplerate))
                frames = round((proposal["end"] - proposal["start"]) * info.samplerate)
                wave = recording.read(frames, dtype="float32", always_2d=True)
                raw_metrics = metrics(wave, info.samplerate)
                reason = None
                if raw_metrics["clipping_fraction"] > 0.001:
                    reason = "clipped original"
                elif raw_metrics["active_fraction_proxy"] < 0.18:
                    reason = "too little waveform activity; speech quality unconfirmed"
                try:
                    audio, operations = prepare_audio(wave, info.samplerate)
                    if not 3.0 <= len(audio) / 22050 <= 11.6:
                        reason = "prepared duration outside 3–11.6 seconds"
                except ValueError as error:
                    reason = str(error)
                if reason:
                    manifest["rejections"].append({"id": identifier, **proposal, "reason": reason})
                    continue
                raw = output / "raw" / f"{identifier}.wav"
                clip = output / "clips" / f"{identifier}.wav"
                if raw.exists() or clip.exists():
                    raise FileExistsError(identifier)
                sf.write(raw, wave, info.samplerate, subtype="FLOAT")
                sf.write(clip, audio, 22050, subtype="PCM_16")
                flags = []
                if re.search(r"\[[^]]+\]", proposal["caption"]):
                    flags.append("caption includes non-speech event")
                if raw_metrics["voiced_fraction_proxy"] < 0.12:
                    flags.append("possible whisper/low periodicity; listen")
                if sid == "iP0AY_BnlHE":
                    flags.append("source includes thunderstorm; inspect background")
                manifest["clips"].append({
                    "id": identifier, "source_id": sid, "source_url": metadata["webpage_url"],
                    "source_sha256": source_hash, **proposal,
                    "original_seconds": len(wave) / info.samplerate,
                    "prepared_seconds": len(audio) / 22050,
                    "raw": str(raw.relative_to(output)), "audio": str(clip.relative_to(output)),
                    "audio_sha256": digest(clip), "raw_sha256": digest(raw),
                    "preprocessing": operations, "metrics": raw_metrics, "flags": flags,
                    "transcript": caption_text(proposal["caption"]),
                    "transcript_origin": "unverified original YouTube automatic captions",
                    "review": {"decision": "pending", "transcript_verified": False,
                               "same_speaker_verified": False, "audio_quality_verified": False},
                })
        dump(output / "dataset.json", manifest)
        print(sid, "candidate clips", sum(row["source_id"] == sid for row in manifest["clips"]),
              flush=True)
    manifest["candidate_seconds"] = sum(row["prepared_seconds"] for row in manifest["clips"])
    dump(output / "dataset.json", manifest)
    return output / "dataset.json"


def normalized(text: str) -> str:
    """Ignore ASR casing/punctuation differences for review flags, not training text."""
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower().replace("’", "'")))


def transcribe(directory: Path, model_name: str = "medium.en",
               reuse_from: Path | None = None) -> Path:
    """Cross-check cached CPU ASR against captions; keep explicit human verification pending."""
    from faster_whisper import WhisperModel

    directory = directory.resolve()
    manifest = json.loads((directory / "dataset.json").read_text())
    path = directory / "asr-checks.json"
    checks = json.loads(path.read_text()) if path.exists() else {"rows": {}}
    reusable = {}
    if reuse_from is not None:
        prior = json.loads((reuse_from / "asr-checks.json").read_text())["rows"]
        reusable = {value["audio_sha256"]: value for value in prior.values()
                    if value["model"] == model_name}
    snapshots = list((Path.home() / ".cache/huggingface/hub" /
                      f"models--Systran--faster-whisper-{model_name}/snapshots").glob("*/model.bin"))
    if not snapshots:
        raise FileNotFoundError(f"{model_name} is not cached; no implicit model download")
    model = WhisperModel(str(snapshots[0].parent), device="cpu", compute_type="int8", cpu_threads=4)
    for index, row in enumerate(manifest["clips"]):
        previous = checks["rows"].get(row["id"], {})
        if (previous.get("model") == model_name
                and previous.get("audio_sha256") == row["audio_sha256"]):
            continue
        if row["audio_sha256"] in reusable:
            check = dict(reusable[row["audio_sha256"]])
            agreement = SequenceMatcher(None, normalized(row["caption"]),
                                        normalized(check["text"])).ratio()
            check["caption_character_agreement"] = agreement
            check["flags"] = [flag for flag in check["flags"]
                              if flag != "caption/independent ASR disagreement"]
            if agreement < 0.9:
                check["flags"].append("caption/independent ASR disagreement")
            check["reused_identical_audio_from"] = str(reuse_from)
            checks["rows"][row["id"]] = check
            dump(path, checks)
            continue
        audio, rate = sf.read(directory / row["audio"], dtype="float32")
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        segments, _ = model.transcribe(wave, language="en", beam_size=5, word_timestamps=True,
                                       vad_filter=False, condition_on_previous_text=False)
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments)
        agreement = SequenceMatcher(None, normalized(row["caption"]), normalized(text)).ratio()
        check = {
            "model": model_name, "device": "cpu", "precision": "int8", "text": text,
            "audio_sha256": row["audio_sha256"], "human_verified": False,
            "caption_character_agreement": agreement,
            "avg_logprob": float(np.mean([s.avg_logprob for s in segments])) if segments else None,
            "no_speech_probability_max": max((s.no_speech_prob for s in segments), default=1),
            "compression_ratio_max": max((s.compression_ratio for s in segments), default=0),
            "words": [{"start": w.start, "end": w.end, "word": w.word,
                       "probability": w.probability} for s in segments for w in (s.words or [])],
        }
        check["flags"] = []
        if agreement < 0.9:
            check["flags"].append("caption/independent ASR disagreement")
        if not text or check["no_speech_probability_max"] > 0.5:
            check["flags"].append("ASR speech detection uncertain")
        checks["rows"][row["id"]] = check
        checks["interpretation"] = "Independent ASR/caption cross-check, not human certification"
        dump(path, checks)
        if (index + 1) % 25 == 0:
            print("ASR checked", index + 1, "/", len(manifest["clips"]), flush=True)
    return path


def write_metadata(path: Path, rows: list[dict]) -> None:
    """Use the actual maintained Coqui formatter's pipe-delimited header."""
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name", "emotion_name"])
        for row in rows:
            writer.writerow([row["audio"], row["transcript"], "ciel", "neutral"])
