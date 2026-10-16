# Chatterbox Nano CPU audition

This is an offline listening experiment. Chatterbox Turbo remains the accepted
production voice. Pocket and Piper remain available. No production configuration,
worker, reference recording, or installed package is changed by the audition.

The native [Nano implementation](https://huggingface.co/ResembleAI/chatterbox-nano)
uses `ChatterboxTurboTTS.from_local(..., device="cpu", nano=True)`.
The model is pinned to `71ccd1d0081b430592cea481f4307e764e07bc64` and uses the
existing isolated clone environment: Chatterbox 0.1.7 at implementation commit
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`, Torch 2.6.0+cu124,
Transformers 5.2.0. Its exact dependency list is `scripts/voice_requirements/clone.txt`.
The CUDA-enabled Torch wheel does not require running inference on CUDA.

## Run locally

Run from the repository root. No package installation is needed in the existing
clone environment. Download only the required pinned weights/tokenizer:

```bash
HF_HUB_DISABLE_XET=1 data/voice/envs/clone/bin/python -m scripts.chatterbox_nano_audition download
```

Synthesize all seven fixed evaluation texts, then repeat each for three warm
measurements. Choose a new output directory for each run; existing runs are never
overwritten:

```bash
CUDA_VISIBLE_DEVICES='' data/voice/envs/clone/bin/python -m scripts.chatterbox_nano_audition run \
  --output data/voice/shootouts/nano-20261016-01 --iterations 3 --threads 4
```

The script hides CUDA before importing Torch, explicitly loads every model on
CPU, and checks that CUDA stays uninitialized. The model remains loaded across
sentences. Native `Conditionals.save/load` caches the unchanged Ciel reference
once under `data/voice/states/nano/`, keyed by model inventory, reference hash,
dependency versions and conditioning settings. There is no fallback to Turbo.
An error is recorded as failure instead of silently changing engines.

Nano's native `generate()` returns a complete utterance. This audition does not
claim chunk streaming: first usable audio is available after full synthesis.
No production interruption or streaming implementation is modified.

Create the blind listening page using the preserved accepted Turbo samples:

```bash
data/voice/envs/clone/bin/python -m scripts.chatterbox_nano_audition page \
  --result data/voice/shootouts/nano-20261016-01/result.json \
  --output data/voice/shootouts/nano-20261016-01/comparison
python -m http.server 8772 --bind 127.0.0.1 \
  --directory data/voice/shootouts/nano-20261016-01/comparison/listening
```

Open <http://127.0.0.1:8772/>. Each round uses the identical text and unchanged
`reference-1.wav`. Letters are randomized separately per round. Engine names stay
hidden until ratings and a choice are entered. Listening notes can be exported.
For each pair, constant gain attenuates the louder recording to the quieter
recording's integrated loudness, with at least 1 dB peak headroom. Original WAVs
remain intact; there is no denoising, EQ, compression, pitch or formant change.

## Measurements and acceptance

`result.json` includes startup, cold synthesis, three warm trials per text,
first audio, total synthesis, audio duration, RTF, process CPU/RAM, GPU telemetry,
versions, reference hash, conditioning cache behavior and CUDA initialization.
GPU power/utilization/temperature/fan percentage describe the whole device;
desktop or STT activity must not be attributed to Nano. NVML is read for
telemetry without creating a CUDA context. Fan percentage is not RPM or noise.
RTF below 1 means synthesis finished faster than the generated audio duration.

Quality is pending user listening; neither smaller model size nor a benchmark
establishes equivalence to Turbo. A sustained conversational test and production
integration should follow only if the user accepts the voice. Nothing is activated
automatically.

### Measured on this i5-10400F

Run: `data/voice/shootouts/nano-20261016-01/result.json`, CPU FP32, four Torch
threads, three warm trials per text (21 calls), plus a cold short synthesis.
First usable audio equals complete utterance synthesis; no chunk streaming.

| Text | Median first audio / synthesis | Median RTF |
| --- | ---: | ---: |
| Short | 2.82 s | 1.87 |
| Conversational | 8.50 s | 1.43 |
| Warm | 6.31 s | 1.50 |
| Amused | 5.95 s | 1.50 |
| Technical | 6.53 s | 1.48 |
| Soft | 7.05 s | 1.49 |
| Long | 24.65 s | 1.45 |

Initial startup including imports/model load/reference encoding: 17.21 s.
Reference encoding took 1.45 s and happened once across all calls. A fresh-process
cache reload took 9.18 s including imports/model load, skipped reference encoding,
and produced a further short smoke sample. CUDA remained uninitialized in both
processes; engine compute VRAM measured 0 MiB throughout.

CPU use was 30.8–34.3% of the whole 12-logical-CPU machine during warm calls
(approximately four busy cores). Process RSS peaks increased from 2,872 MiB
on the first short call to 4,368 MiB on the last long call. This run does not
establish a memory plateau or exclude a leak. A longer stability test is pending.

Whole-device power had a per-call median near 38.5 W and a maximum of 43.7 W;
the GPU already showed unrelated utilization before inference. Sampled device
temperatures were 59–64 C and fan readings varied up to 32%. These are shared
GPU readings, not measurements of Nano's own power draw or fan noise. No NVIDIA
power limit, clock, fan setting, or Turbo configuration was changed.

Native CPU inference was slower than realtime in this run, so it is not yet a
fast conversational replacement. The user subsequently rejected Nano's voice:
delivery lacked liveliness and the desired fuller, deeper feminine character.
This is user listening feedback, not an independent listening assessment.
The private feedback record is `data/voice/shootouts/nano-20261016-01/user-feedback.json`.
Turbo remains the accepted production voice. The blind
page is `data/voice/shootouts/nano-20261016-01/comparison/listening/index.html`.
Its seven pairs use the first warm outputs, while `result.json` contains the
completed repeated benchmark. Source WAVs and reference hashes remain unchanged.

Validation: eight Nano unit tests passed, related focused checks/Ruff and the
existing blind-rating JavaScript gate passed, and all 15 page/reference/audio
links passed HTTP/WAV checks. The full suite retained five pre-existing failures
in `tests/test_memory_routing.py` (date/wording/controller expectations), with no
Nano failure. Browser visual inspection was unavailable in this session.

Mocked correctness checks, without model downloads:

```bash
.venv/bin/python -m pytest tests/test_chatterbox_nano.py tests/test_tts_shootout.py -q
node tests/frontend/tts_finetune.cjs
```

## Training cleanup

On 2026-10-16 the user requested removal of the XTTS training data. Removed:
`data/voice/finetuning/`, the dedicated `data/voice/envs/cloud-preflight/`, and
the external Kaggle automation workspace's training transfer and run artifacts.
This reclaimed 17.4 GiB in the repository before the new Nano download; hardlinks
mean apparent directory sizes overstate the space reclaimed.

Preserved: all downloaded source audio and provenance under
`data/voice/source/ciel/`, all existing references under
`data/voice/references/ciel/`, reusable voice states, runtime models/environments,
source code, credentials, and prior zero-shot shootout recordings.
The seven accepted Turbo baseline WAVs are preserved under
`data/voice/shootout-research/chatterbox-accepted-baseline/`.
The cleanup report verifies 397 protected file hashes and records exact removed
paths: `data/voice/shootout-research/training-cleanup-20261016.json`.
Remote Kaggle datasets/notebooks have not been deleted.
