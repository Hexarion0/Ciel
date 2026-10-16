# Pocket TTS validation on the target PC

Run date: 2026-10-14. Hardware: Intel i5-10400F, 12 logical CPUs; GTX 1660 SUPER 6 GB, NVIDIA driver
610.57.04; Arch Linux 7.2.4-arch1-2. Python 3.14.7, Pocket TTS 3.3.0, torch 2.14.1+cu130 in the main
environment. Chatterbox retains its separate Python 3.12 environment.

Three warm iterations were measured for each of four sentences, after a fresh process/model startup
and one first synthesis. Model files were cached. Pocket used the existing Ciel neutral recording,
Chatterbox used its original Ciel profile, and Piper used en_US-ciel-medium. These are consistent
reference choices, not proof of equivalent acoustic identity, expression, or quality.

Pocket custom runs were recorded separately after successful authentication in the default Hugging
Face cache. The earlier Chatterbox/Piper measurements were retained to avoid another GPU workload.
Earlier provisional Alba runs also remain available locally. No test-suite or other benchmark ran
concurrently with the reported custom synthesis runs.

## Timing

Cold startup measures model/reference initialization with filesystem cache retained, not a cold disk
or first download. First audio-ready means the first generated chunk, not audible speaker latency.
Piper and Chatterbox expose complete waveforms in this pipeline.

| Engine | Cold model/reference startup |
| --- | ---: |
| Pocket FP32 | 5.422 s |
| Pocket INT8 | 6.154 s |
| Chatterbox Turbo | 14.946 s |
| Piper | 1.178 s |

Warm medians (three iterations per row):

| Engine | Sentence | First audio-ready | Total synthesis | RTF |
| --- | --- | ---: | ---: | ---: |
| Pocket FP32 | short | 111 ms | 0.844 s | 0.429 |
| Pocket FP32 | normal | 143 ms | 2.504 s | 0.412 |
| Pocket FP32 | technical | 172 ms | 3.572 s | 0.417 |
| Pocket FP32 | conversation | 159 ms | 5.186 s | 0.414 |
| Pocket INT8 | short | 63 ms | 0.586 s | 0.255 |
| Pocket INT8 | normal | 90 ms | 1.572 s | 0.259 |
| Pocket INT8 | technical | 108 ms | 2.501 s | 0.263 |
| Pocket INT8 | conversation | 98 ms | 3.523 s | 0.256 |
| Chatterbox Turbo | short | 926 ms | 0.926 s | 0.503 |
| Chatterbox Turbo | normal | 2096 ms | 2.096 s | 0.433 |
| Chatterbox Turbo | technical | 3092 ms | 3.092 s | 0.385 |
| Chatterbox Turbo | conversation | 4376 ms | 4.376 s | 0.364 |
| Piper | short | 60 ms | 0.060 s | 0.042 |
| Piper | normal | 408 ms | 0.408 s | 0.042 |
| Piper | technical | 412 ms | 0.412 s | 0.042 |
| Piper | conversation | 514 ms | 0.514 s | 0.037 |

RTF = synthesis time / generated audio duration; less than one is faster than real time. Generated
durations and speaking rates differ across engines.

## Resources

CPU is summed process-tree CPU time / sampled wall time: 100% means one logical core. The table
gives median warm CPU usage. RAM is maximum summed process-tree RSS; shared pages can be
double-counted. Startup RSS is separate. Telemetry adds overhead, particularly for short utterances.

| Engine | Warm CPU median | Warm RSS peak | Startup RSS peak | Engine VRAM peak |
| --- | ---: | ---: | ---: | ---: |
| Pocket FP32 | 160% | 1627 MiB | 1683 MiB | 0 MiB |
| Pocket INT8 | 183% | 1661 MiB | 1837 MiB | 0 MiB |
| Chatterbox Turbo | 107% | 2455 MiB | 4920 MiB | 4000 MiB |
| Piper | 552% | 430 MiB | 171 MiB | 0 MiB |

INT8 was actually enabled through upstream native torch quantization. It reduced measured synthesis
time here but did not reduce peak RSS. No listening test has accepted INT8 as equal in quality; FP32
remains the conservative default. Direct real-inference checks reported `torch.cuda.is_initialized()
== False` after custom FP32 and provisional built-in INT8 generation.

GPU telemetry describes the whole device, including desktop/other applications. Pocket starts with
substantial background GPU activity. These are not isolated TTS utilization or power readings. Piper
ran after Chatterbox, so its temperature includes cooldown.

| Engine | Sampled GPU utilization peak | Total VRAM peak | Power peak | Temperature peak |
| --- | ---: | ---: | ---: | ---: |
| Pocket FP32 | 40% | 602 MiB | 39.69 W | 62 C |
| Pocket INT8 | 39% | 602 MiB | 38.88 W | 61 C |
| Chatterbox Turbo | 93% | 4553 MiB | 124.64 W | 91 C |
| Piper | 34% | 580 MiB | 55.71 W | 72 C |

Zero TTS-attributed VRAM and an uninitialized CUDA context support CPU-only Pocket inference.
Polling can miss brief power spikes. These short synthesis/playback checks do not establish
sustained fan, power, temperature, or listening comfort improvements. STT retains its own
configuration and can still use the GPU.

## Reproduce

Use the authenticated default Hugging Face cache for the current custom voice. Do not set `HF_HOME`
to the earlier provisional cache unless that cache is also authenticated and populated.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_tts.py --engines pocket \
  --pocket-reference data/voice/references/ciel/reference-1.wav --iterations 3 \
  --output benchmarks/tts/custom-fp32/report.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_tts.py --engines pocket \
  --pocket-reference data/voice/references/ciel/reference-1.wav --quantization int8 \
  --iterations 3 --output benchmarks/tts/custom-int8/report.json
```

To compare all engines in one new run, omit `--engines pocket`. Offline mode assumes all required
assets are cached. The benchmark fails the requested engine if fallback is used.

Raw measurements and first-iteration WAV samples remain under ignored `benchmarks/tts/custom-fp32/`,
`benchmarks/tts/custom-int8/`, and the earlier `benchmarks/tts/fp32/` and `benchmarks/tts/int8/`.
`benchmarks/tts/comparison.json` combines the custom Pocket and earlier HQ/Piper results with the
timing distinction recorded.

## Normal-path verification and remaining acceptance

The local `.env` now selects Pocket CPU FP32 with the existing Ciel neutral reference. Actual
custom-reference inference passed. Two real integration checks passed with that configuration:
normal `stream_reply` sentence/token playback and interruption that stopped PortAudio output and
unwound the Pocket worker. The token source was a local fixture; inference and speaker playback were
real. Built-in preview playback also passed, and the custom-model access failure successfully used
the installed Ciel Piper fallback.

The six delivery slots exist in the software; only the existing neutral recording is currently
supplied. Missing warm/amused/firm/focused/soft references use that same neutral identity. Distinct
emotional delivery has not been validated. Do not treat the voice bank or subjective
naturalness/identity/listening comfort as accepted yet.

Microphone/wake/real-LLM acceptance and sustained fan/power comparison remain open. Upstream
sounddevice/NumPy shape-deprecation warnings and native torch INT8 deprecation warnings occurred
without functional test failures. See [setup and listening instructions](pocket-tts.md).


## Checks and changed files

Final ordinary suite: **1,150 passed, 5 integration tests deselected**. The two
Pocket real playback/interruption checks passed separately using the configured
custom reference. Ruff, dependency consistency (`pip check`), shell syntax,
whitespace checks, and wheel build passed. The wheel contains the new audio
modules and the pinned Pocket optional dependency metadata.

Runtime:

- `src/ciel/__main__.py`
- `src/ciel/config.py`
- `src/ciel/setup_wizard.py`
- `src/ciel/audio/tts.py`
- `src/ciel/audio/pocket.py` (new)
- `src/ciel/audio/chunk_playback.py` (new)
- `src/ciel/audio/voice_style.py` (new)

Installation, configuration, documentation, and benchmark:

- `pyproject.toml`
- `setup.sh`
- `.env.example`
- `.env` (ignored local file; only TTS selection/Pocket options updated)
- `README.md`
- `docs/pocket-tts.md` (new)
- `docs/pocket-tts-validation.md` (new)
- `scripts/benchmark_tts.py` (new)

Tests:

- `tests/conftest.py` (new; isolates ordinary TTS tests from private live settings)
- `tests/test_pocket_tts.py` (new)
- `tests/test_tts.py`
- `tests/test_voice_expression.py`
- `tests/test_config.py`
- `tests/test_cli.py`
- `tests/test_setup_wizard.py`
- `tests/test_playback_captions.py`

An empty ignored `data/voices/ciel/` bank was created for future recordings.
Private source references, Chatterbox model/worker code, downloaded model files,
and unrelated application behavior were preserved. The repository already had
extensive uncommitted changes; this list describes the TTS upgrade's additions
and edits, not every change visible in `git status`.
