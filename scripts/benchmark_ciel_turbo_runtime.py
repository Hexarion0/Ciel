"""Exercise CIEL's real streamed-reply path with local Turbo and configured STT."""

from __future__ import annotations

import argparse
import html
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

from ciel.audio.stt import SpeechToText
from ciel.audio.tts import TextToSpeech
from ciel.config import get_settings
from ciel.providers.base import ChatMessage, LLMStreamChunk


class FixedStream:
    """Small local stand-in for a provider stream; no network or API is used."""

    def __init__(self, text: str, *, delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay

    def stream(self, *_args, **_kwargs):
        for chunk in self.text.split(" "):
            if self.delay:
                time.sleep(self.delay)
            yield LLMStreamChunk(chunk + " ", "local-runtime-test", "fixture")


class MemorySampler:
    """Poll total GPU allocation and the worker's resident RAM while tests run."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.gpu_peak_mib = 0
        self.worker_rss_peak_mib = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _gpu_used() -> int | None:
        import subprocess

        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
            return int(output.strip().splitlines()[0])
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _rss_mib(pid: int | None) -> int | None:
        if not pid:
            return None
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError):
            return None
        return None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            gpu = self._gpu_used()
            rss = self._rss_mib(getattr(self, "worker_pid", None))
            if gpu is not None:
                self.gpu_peak_mib = max(self.gpu_peak_mib, gpu)
            if rss is not None:
                self.worker_rss_peak_mib = max(self.worker_rss_peak_mib, rss)
            self.stop_event.wait(0.25)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> dict[str, int | None]:
        self.stop_event.set()
        self.thread.join(timeout=2)
        return {
            "total_gpu_peak_mib": self.gpu_peak_mib or None,
            "turbo_worker_peak_rss_mib": self.worker_rss_peak_mib or None,
        }


def _word_error(reference: str, hypothesis: str) -> dict[str, float | int]:
    expected, actual = reference.lower().split(), hypothesis.lower().split()
    row = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, candidate in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                row[j] + 1,
                row[j - 1] + (word != candidate),
            ))
        row = current
    errors = row[-1]
    return {"word_errors": errors, "reference_words": len(expected),
            "wer": round(errors / max(1, len(expected)), 4)}


def _save_page(out_dir: Path, results: dict) -> None:
    rows = []
    for case, filenames in results.get("samples", {}).items():
        for filename in filenames:
            rows.append(
                f"<section><h2>{html.escape(case)}</h2><audio controls preload='none' "
                f"src='{html.escape(filename)}'></audio><p>{html.escape(filename)}</p></section>"
            )
    report = html.escape(json.dumps(results, indent=2))
    page = """<!doctype html><meta charset="utf-8"><title>CIEL Turbo runtime</title>
<style>body{font:16px system-ui;max-width:900px;margin:2em auto;padding:0 1em}
section{padding:1em 0;border-bottom:1px solid #ccc}audio{width:min(100%,600px)}
pre{white-space:pre-wrap;background:#f5f5f5;padding:1em}</style>
<h1>CIEL Chatterbox Turbo runtime samples</h1>
<p>Locally generated through CIEL's sentence pipeline. Playback controls below.
Metrics and test notes follow.</p>""" + "\n".join(rows) + (
        f"<h2>Runtime report</h2><pre>{report}</pre>"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-playback", action="store_true", help="Save generated audio only")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = args.output or root / "data/voice/benchmarks/ciel-turbo-runtime"
    out_dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_settings = get_settings().audio
    reference = root / "data/voice/references/ciel/reference-1.wav"
    transcript = (root / "data/voice/references/ciel/reference-1.txt").read_text().strip()
    reference_audio, reference_rate = sf.read(reference, dtype="float32")
    if reference_audio.ndim > 1:
        reference_audio = reference_audio.mean(axis=1)
    if reference_rate != 16000:
        reference_audio = resample_poly(reference_audio, 16000, reference_rate).astype("float32")
    result: dict = {"notes": ["Local provider fixture; no LLM/API request was made."],
                    "stt": {}, "cases": {}, "samples": {}}
    sampler = MemorySampler()
    sampler.start()
    stt = SpeechToText(
        model_size=audio_settings.stt_model,
        device=audio_settings.stt_device,
        compute_type=audio_settings.stt_compute_type,
        language=audio_settings.stt_language,
        initial_prompt="Ciel is the name of the assistant.",
        retry_model=audio_settings.stt_retry_model,
        retry_min_free_mb=audio_settings.stt_retry_min_free_mb,
    )
    if not stt.wait_ready(timeout=180):
        raise TimeoutError("CIEL STT did not finish loading within 180 seconds")
    result["stt"]["configured_model"] = audio_settings.stt_model
    result["stt"]["device"] = stt.device
    result["stt"]["compute_type"] = stt.compute_type
    t0 = time.perf_counter()
    baseline_text = stt.transcribe(
        reference_audio, language="en", beam_size=audio_settings.stt_beam_size
    )
    result["stt"]["baseline_latency_seconds"] = round(time.perf_counter() - t0, 3)
    result["stt"]["baseline_transcript"] = baseline_text
    result["stt"]["reference_word_error"] = _word_error(transcript, baseline_text)
    result["stt"]["reference_sample_rate"] = 16000

    tts = TextToSpeech(
        voice_name="ciel", engine="chatterbox_turbo", models_dir="models/tts",
        speed=audio_settings.tts_speed, output_device=audio_settings.output_device,
        enabled=True, tts_fallback_voice=audio_settings.tts_fallback_voice,
        voice_profiles_dir=audio_settings.tts_voice_profiles,
        chatterbox_model_dir=audio_settings.tts_chatterbox_model,
        chatterbox_python=audio_settings.tts_chatterbox_python,
        min_free_vram_mib=audio_settings.tts_min_free_vram_mb,
        audio_queue_size=audio_settings.tts_audio_queue_size,
    )
    current_case = ["sample"]
    sample_index = [0]
    playback_original = tts.speak

    def record_speak(text, *call_args, _prepared=None, **kwargs):
        if _prepared is not None:
            audio, rate, _timeline = _prepared
            case = current_case[0]
            sample_index[0] += 1
            name = f"{len(result['samples'].get(case, [])) + 1:02d}-{sample_index[0]:03d}.wav"
            sf.write(out_dir / name, audio, rate, subtype="PCM_16")
            result["samples"].setdefault(case, []).append(name)
        if args.no_playback:
            return _prepared is not None
        return playback_original(text, *call_args, _prepared=_prepared, **kwargs)

    tts.speak = record_speak
    try:
        cold_started = time.perf_counter()
        warm = tts.warmup()
        result["cold_worker_ready_seconds"] = round(time.perf_counter() - cold_started, 3)
        result["turbo_startup"] = (
            tts._chatterbox.startup_metrics.copy() if tts._chatterbox else {}
        )
        result["turbo_ready"] = warm
        sampler.worker_pid = (
            tts._chatterbox.startup_metrics.get("pid") if tts._chatterbox else None
        )
        if not warm:
            result["notes"].append(
                "Turbo did not initialize; remaining samples use the configured fallback."
            )

        def run_case(name: str, text: str, *, cancel_after_audio: bool = False) -> dict:
            current_case[0] = name
            event = threading.Event()
            response = FixedStream(text, delay=0.005 if cancel_after_audio else 0.0)
            started = time.perf_counter()
            if cancel_after_audio:
                def on_sentence(_sentence: str) -> None:
                    event.set()

                callback = on_sentence
            else:
                callback = None
            reply = __import__("ciel.audio.streaming", fromlist=["stream_reply"]).stream_reply(
                response, [ChatMessage("user", "runtime benchmark")], tts,
                lambda: not event.is_set(), max_tokens=300, cancel_event=event,
                on_sentence_start=callback,
            )
            row = {
                "llm_first_text_seconds": round(reply.first_token_seconds or 0, 3),
                "first_audio_seconds": round(reply.first_audio_seconds or 0, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
                "spoken": reply.spoken,
                "cancelled": reply.canceled,
                "pipeline": reply.speech_pipeline,
            }
            saved = result["samples"].get(name, [])
            for sentence, filename in zip(row["pipeline"]["sentences"], saved):
                duration = sf.info(out_dir / filename).duration
                sentence["audio_duration_seconds"] = round(duration, 3)
                sentence["real_time_factor"] = round(
                    sentence["synthesis_seconds"] / max(duration, 1e-6), 3
                )
            result["cases"][name] = row
            return row

        run_case("01 Of course", "Of course.")
        run_case("02 Normal response", "Your server is online and everything looks normal. "
                 "I checked the important services, and they are responding as expected.")
        long_response = (
            "I have checked the system, and nothing needs your attention right now. "
            "The server is online, the important services are responding, and storage has enough "
            "free space for normal use. I will keep monitoring the machine and let you know if "
            "anything changes. You can continue with what you were doing."
        )
        run_case("03 Long response", long_response)
        run_case("04 [chuckle]", "[chuckle] I had a feeling you would say that.")
        run_case("05 [sigh]", "[sigh] You're still awake. You should probably get some rest soon.")
        run_case("05a Inline [giggle]", "Oh! [giggle] You got me.")
        run_case("05b [moan] alias", "[moan] Not again. We can fix this.")
        run_case("05c Event budget", "[chuckle] Fair enough. [sigh] Let's try again. "
                 "[groan] Same problem.")
        run_case("06 Rapid request one", "Of course.")
        run_case("07 Rapid request two", "Would you like me to take care of that for you?")
        run_case("08 User interruption", "I have checked the system. Everything is operating "
                 "normally. I will continue monitoring it and tell you if anything changes.",
                 cancel_after_audio=True)

        # Exercise the exact selected fallback after a simulated worker synthesis error.
        current_case[0] = "09 Piper fallback"
        tts._chatterbox_failed = True
        fallback = tts.synthesize("The local voice fallback is ready.")
        if fallback is not None:
            sf.write(out_dir / "fallback-piper.wav", fallback[0], fallback[1], subtype="PCM_16")
            result["samples"].setdefault("09 Piper fallback", []).append("fallback-piper.wav")
            result["cases"]["fallback"] = {
                "audio_seconds": round(len(fallback[0]) / fallback[1], 3),
                "sample_rate": fallback[1],
                "available": True,
            }
        else:
            result["cases"]["fallback"] = {"available": False}
        tts._chatterbox_failed = False

        # Actually run CUDA STT on the reference while Turbo synthesizes and plays a reply.
        stt_result: dict = {}
        stt_started = threading.Event()

        def concurrent_stt() -> None:
            stt_started.set()
            started = time.perf_counter()
            try:
                text = stt.transcribe(
                    reference_audio, language="en", beam_size=audio_settings.stt_beam_size
                )
                stt_result.update(
                    ok=True, text=text,
                    latency_seconds=round(time.perf_counter() - started, 3),
                    word_error=_word_error(transcript, text),
                )
            except Exception as err:
                stt_result.update(ok=False, error=f"{type(err).__name__}: {err}")

        stt_thread = threading.Thread(target=concurrent_stt, name="ciel-stt-overlap")
        stt_thread.start()
        stt_started.wait(timeout=1)
        run_case("10 STT plus TTS", "Your GPU is working normally. I am checking the system "
                 "while you speak.")
        stt_thread.join(timeout=120)
        result["stt"]["concurrent"] = stt_result

        # Closing then re-creating forces the true cold model/reference reload path.
        tts.close()
        reload_tts = TextToSpeech(
            voice_name="ciel", engine="chatterbox_turbo", models_dir="models/tts",
            speed=audio_settings.tts_speed, output_device=audio_settings.output_device,
            voice_profiles_dir=audio_settings.tts_voice_profiles,
            chatterbox_model_dir=audio_settings.tts_chatterbox_model,
            chatterbox_python=audio_settings.tts_chatterbox_python,
            min_free_vram_mib=audio_settings.tts_min_free_vram_mb,
            audio_queue_size=audio_settings.tts_audio_queue_size,
        )
        t0 = time.perf_counter()
        result["reload_ready"] = reload_tts.warmup()
        result["reload_cold_seconds"] = round(time.perf_counter() - t0, 3)
        result["reload_startup"] = (
            reload_tts._chatterbox.startup_metrics.copy() if reload_tts._chatterbox else {}
        )
        t0 = time.perf_counter()
        audio = reload_tts.synthesize("Of course.")
        result["reload_first_synthesis_seconds"] = round(time.perf_counter() - t0, 3)
        if audio is not None:
            sf.write(out_dir / "11-after-reload.wav", audio[0], audio[1], subtype="PCM_16")
            result["samples"].setdefault("11 After reload", []).append("11-after-reload.wav")
        reload_tts.close()
    finally:
        tts.close()
        result["memory"] = sampler.close()
        result["output_directory"] = str(out_dir)
        (out_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        _save_page(out_dir, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
