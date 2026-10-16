# Ciel TTS quality shootout

The next experiment is [Chatterbox Nano on CPU](chatterbox-nano-audition.md),
using the existing Ciel reference and a blind comparison against accepted Turbo
samples. No production switch has been made.

Pocket's implementation remains available, but the user rejected its output as
too robotic. This evaluation does not tune Pocket, change production settings,
or select a permanent everyday engine. It preserves Chatterbox Turbo and Piper.

The current audition is `data/voice/shootouts/20261014-01/index.html`:

```bash
xdg-open data/voice/shootouts/20261014-01/index.html
```

The 2026-10-17 cleanup removed the rejected XTTS, NeuTTS-Air, and StyleTTS2
downloads, their two experimental environments, the StyleTTS2 vendor checkout,
and download/build caches. Reports, audition audio, original recordings, and
reproduction scripts remain. Repeating those experiments requires running their
installers again. Turbo, Pocket, Piper, and the Nano audition assets were retained.

In the main conversational round the user chose Sample A (Chatterbox) and
rejected both other samples as too robotic for everyday use. Those samples
were StyleTTS2 CPU five-step and XTTS CUDA FP32. This feedback rules them out
as recommended replacements; it does not supply ratings for every other mode.

In the optional conversational comparison the user described both samples as
natural, with no robotic sound. Sample A was again Chatterbox and described
as perfect; Sample B was NeuTTS-Air, whose older/heavier timbre missed Ciel's
preferred identity. Air passed naturalness for that line, but failed identity
acceptance. Its CPU RTF of about 2.05 also cannot sustain immediate streaming
playback without buffering. No replacement has passed everyday acceptance.
The five-minute Air run completed; see the run's report for cache/memory details.
Its RSS increased about 529 MiB between first/last quartiles; allocator retention
and a leak have not been distinguished. It reused the cached reference without
additional encoding and left CUDA uninitialized throughout.

Chatterbox remains the accepted quality and identity baseline. No replacement
has met Ciel's combined identity and conversational requirements. Restore the
matching NVIDIA stack with a normal reboot before further fan/power conclusions.
A separate Chatterbox power-profile experiment is a reasonable next phase;
the existing Turbo path was not optimized or activated by this shootout.

## Scope and exact implementations

All engines receive the same byte-identical `reference-1.wav`, including the
longstanding Turbo quality baseline. Newer candidate clips remain available;
no speaker sources are downloaded again. The texts are fixed in
`scripts/tts_shootout/common.py`: the six supplied lines and a four-sentence
Ciel response. Generated audio retains native levels, with no postprocessing.

- StyleTTS2 LibriTTS, NeuralVox importable implementation pinned to
  `7b3c5e94fb0ef7d2c778f772dd8b671269bf5009`, model revision
  `3aa7ba7f8f275ec13dce21682a61494c35089e2a`. CPU/CUDA FP32, 1/3/5
  diffusion-step parameters, alpha=0.3, beta=0.7, embedding_scale=1.
- XTTS-v2 through maintained `coqui-tts==0.27.5`, model revision
  `6c2b0d75eae4b7047358e3b6bd9325f857d43f77`. PyTorch 2.6.0+cu124,
  Transformers 4.57.6; CPU FP32, CUDA FP32, and CUDA FP16 autocast.
- Existing `chatterbox-tts==0.1.7`, implementation commit
  `5de7a54aa4e5e2baadb0182dde554908b48b85c2`, Turbo model revision
  `749d1c1a46eb10492095d68fbcf55691ccf137cd`. The existing CUDA worker,
  sampling defaults, conditionals, and cue assets remain unchanged.
- Optional NeuTTS-Air: `neutts==1.4.1`, `neucodec==0.0.6`, CPU-only Torch 2.8.0,
  llama-cpp-python 0.3.36 compiled with CUDA disabled. Q8_0 backbone revision
  `855f828ba33d1885443a7a4971785e318414dfd2`, FP32 ONNX decoder revision
  `ac01424f74ea25b245132252168e2191b3d859fa`, one-time reference encoder revision
  `a940021324df13a07f44185f0a17c6dac3698179`. Native streaming, cached codes,
  identical reference and its existing transcript; watermarking retained.

NeuTTS-Air was added only after the two primary alternatives failed listening
acceptance. Its CPU cloning and GGUF streaming address this hardware's main
constraint. It has no explicit emotion controls, and quality remains subject
to listening. NeuTTS-2E was excluded before inference: its current release
has four fixed speakers rather than Ciel's required custom cloning.
See [the upstream capability table](https://github.com/neuphonic/neutts)
and [Air's model card](https://huggingface.co/neuphonic/neutts-air-q8-gguf).

The upstream StyleTTS2 author links the importable NeuralVox implementation,
whose last commit is April 2024. Treat it as an evaluation baseline, not proof
of active production maintenance. Its original inference functions are preserved;
a local boot copy changes only explicit device, pinned local model paths, and
preprovisioned NLTK resources. The newer native ultrapg implementation was
inspected but its advertised steps argument is unused, so it was excluded from
this step comparison. The MIT PyPI implementation uses a different phonemizer;
the upstream author warns that mismatch lowers quality, so it was not preferred.

Primary implementation sources:
[StyleTTS2](https://github.com/yl4579/StyleTTS2),
[NeuralVox](https://github.com/NeuralVox/StyleTTS2),
[maintained Coqui](https://github.com/idiap/coqui-ai-TTS),
[XTTS API](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html),
[native step implementation](https://github.com/ultrapg/style2tts/blob/2ceb37dacc9620f6bc8f09cb2765a9914e8c2c57/src/cpp/styletts2_engine.cpp).

## Isolated installation

Use the existing Arch `uv`, Git, and espeak-ng tools. The installer creates an
independent Python 3.12 environment, installs pinned dependencies, and downloads
only the model assets. It does not mutate Ciel's `.venv` or the working clone
environment. CUDA wheels match the existing Turbo environment's PyTorch version.
Source/model artifacts and the installed dependency snapshot remain ignored.

The user accepted XTTS's CPML terms for this personal non-commercial evaluation;
that attestation is recorded privately. A fresh installation requires explicit
agreement to [the model license](https://huggingface.co/coqui/XTTS-v2/raw/main/LICENSE.txt).
If accepted, run:

```bash
bash scripts/setup_tts_shootout.sh --accept-xtts-license
```

The optional Air candidate has incompatible dependencies and uses a separate
CPU-only environment. Its installer does not update either existing TTS
environment. This machine lacks OpenBLAS development metadata, so llama.cpp
uses its native CPU kernels with CUDA disabled; no system package is installed.
Its Apache-2.0 model is nevertheless access-gated on Hugging Face. Review and
accept [the model access form](https://huggingface.co/neuphonic/neutts-air-q8-gguf)
using the account already logged in locally. The installer reuses the login
token in place; credentials are not copied into project data. A 403 GatedRepo
means model access is still missing, rather than a reason to reinstall Python.
The [reference encoder](https://huggingface.co/neuphonic/neucodec) and
[FP32 CPU decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder)
also have separate forms. The user accepted all three for this evaluation.

NeuCodec 0.0.6's Hub loader still passes removed download keywords to Hub 1.x,
which NeuTTS's Transformers 5.1 requires. Reference preparation uses the native
codec constructor/filter/load logic with pinned local weights; it preserves
encoding inference and records the checkpoint compatibility receipt. Runtime
loads the officially supported local `.onnx` decoder path. TorchAO is pinned
to 0.13.0 because newer unbounded releases import Torch 2.11-only APIs.
The installer prepares the baseline reference once before timed cache-hit
startup. This one-time setup uses the native codec's public encoder and its
MIT `facebook/w2v-bert-2.0` feature backbone, pinned to
`da985ba0987f70aaeb84a80f2851cfac8c697a7b`. Runtime retains only cached reference
codes, the Q8 speech model, and FP32 decoder.

```bash
bash scripts/setup_neutts_shootout.sh
.venv/bin/python scripts/tts_shootout.py run data/voice/shootouts/NEW-RUN \
  --configurations neutts-air-cpu-q8 --iterations 3
.venv/bin/python scripts/tts_shootout.py page data/voice/shootouts/NEW-RUN
```

Main model caches are verified against pinned revisions, then used offline.
The optional comparison adds new randomized rounds against the existing Turbo
samples; it retains all earlier letters, recordings, and listening notes.

Existing assets are retained and checked against pinned revisions. A broken PyPI
alignment release omitted its Cython source; requirements instead pin the
original upstream Git revision containing the required files. NLTK `punkt` and
`punkt_tab` are provisioned before inference. Offline workers load only the
trusted pinned project checkpoints with the compatibility setting required by
their older PyTorch checkpoint format; it does not change global Python settings.

## Reproduce a new run

```bash
.venv/bin/python scripts/tts_shootout.py prepare data/voice/shootouts/NEW-RUN
.venv/bin/python scripts/tts_shootout.py run data/voice/shootouts/NEW-RUN --iterations 3
.venv/bin/python scripts/tts_shootout.py page data/voice/shootouts/NEW-RUN
.venv/bin/python scripts/tts_shootout.py report data/voice/shootouts/NEW-RUN
```

The runner tests StyleTTS2, XTTS, then Turbo in separate persistent-model
processes. Conditioning caches are keyed by the reference, model inventory,
implementation, and encoding device. Each sentence reuses conditioning;
CPU/GPU and numerical comparisons do not blend different speakers.

`--configurations` selects cases, and `--retry-failed` archives unsuccessful
attempts rather than replacing their files. Successful results are retained;
use a new run for new settings. Worker groups, including nested Turbo workers,
are terminated on timeout/interruption. No inference fallback is allowed to
pretend to be a requested engine.

Cold startup includes a new process, imports, model loading, and reference-state
preparation/loading. It excludes model downloads. Warm measurements include
three iterations of every selected text. First-audio latency is time until the
first usable waveform/chunk; StyleTTS2 and existing Turbo return whole utterances,
whereas XTTS uses native `inference_stream` with chunk size 20. This benchmark
does not include Ciel's LLM, STT, sentence buffer, playback, or device buffering.

The original XTTS CPU probe measured RTF 3.2–5.3, so it is not a practical
conversational finalist on this i5. Future default CPU trials are limited to
short/conversational repeats; explicit `--sentences` on the worker overrides
that limit. All seven lines are covered in both CUDA modes. `report.md` and
the actual rows show which texts ran in the current invocation.

### Numerical limitations

StyleTTS2's one-step API parameter fails on CPU and CUDA: its Karras schedule
divides by `(num_steps - 1)`. No invalid clip is added to the listening page,
and one step is not silently replaced by another value. This failure does not
establish a GTX-specific defect. Listen to the successful CUDA and CPU clips
for the upstream-documented older-GPU high-frequency noise or other artifacts.

XTTS `.half()` failed because the upstream null-position embedding creates FP32
zeros, promoting inputs into a Half layer norm. The separate FP16 AMP evaluation
uses PyTorch autocast while retaining FP32 model weights and normalization. It
does not claim to halve model VRAM. Native streaming without text splitting
avoids a large optional tokenizer dependency; the long test remains below the
400-token hard limit but exceeds the advisory 250-character English limit.
Listen/check transcripts for completeness rather than assuming that warning
either proves truncation or proves safety.

## Blind listening

The main seven A/B/C rounds compare CPU five-step StyleTTS2, CUDA FP32 XTTS-v2,
and existing Turbo. These are predetermined quality representatives, not chosen
by benchmark speed. Additional rounds compare steps, CPU/CUDA, and precision.
Letters shuffle per round and remain stable after refresh. Engine names reveal
only after a choice; original ratings and preference are snapshotted beforehand.

Rate naturalness, speaker identity, robotic quality, warmth, prosody,
pronunciation, expression, and overall preference. Higher robotic scores mean
more robotic; all other higher scores are better. Leave uncertain scores blank.
Download notes before closing if browser storage is unavailable. Audio, voice
state, and notes stay local; no upload or production activation exists here.
The private manifest embeds assignments for reproducibility, so blinding is
casual rather than tamperproof.

## Sustained synthesis and telemetry

```bash
.venv/bin/python scripts/tts_shootout.py stress data/voice/shootouts/NEW-RUN \
  --configurations style-cpu-5 style-cuda-5 xtts-cuda-fp32 chatterbox-baseline \
  --seconds 300
.venv/bin/python scripts/tts_shootout.py report data/voice/shootouts/NEW-RUN
```

Each representative runs independently with dialogue/audio pacing and a
30-second idle recovery observation. This generates streamed/batch audio but
does not play the entire conversation. It observes reference-cache calls,
worker identity, RSS, CPU, and GPU resources. It does not establish subjective
comfort or prove long-term leak absence. Rerun finalists selected by listening.
The completed StyleTTS2 CPU run grew about 2.1 GiB between first/last RSS
quartiles. Allocator retention and leaks have not been distinguished; this
result does not establish memory stability for continuous use.

NVML samples whole-device utilization, watts, temperature, and fan percentage
at 100ms intervals. Process-tree compute VRAM is attributed where supported;
Torch allocator measurements omit context/driver/other-process allocations.
CPU 100% means one logical core; machine percentage divides by logical cores.
Fan percentage is neither RPM nor noise measurement. Unsupported fields and
telemetry failures remain explicit instead of becoming zero.

This machine currently runs kernel 7.2.4 with NVIDIA 610.57.04, while installed
kernel 7.2.9 has matching NVIDIA module/libraries 615.78.08. NVML reports a
driver/library mismatch. A normal reboot into the installed newer kernel should
resolve it. No package or driver changes/reboot were made by this evaluation.
Reliable GPU power/fan conclusions remain pending that repair and repeated
measurements. Do not attribute unrelated desktop/STT GPU activity to CPU TTS.

Optional cached CPU ASR checks flag incomplete/repeated text for listening review:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.tts_shootout.verify data/voice/shootouts/NEW-RUN
```

ASR is not a naturalness, pronunciation, or speaker-similarity acceptance test.
CPU verification is run separately from timed inference to avoid contention.

## Checks and decision

```bash
.venv/bin/python -m pytest tests/test_tts_shootout.py
node tests/frontend/tts_shootout.cjs
.venv/bin/python -m ruff check src tests scripts/tts_shootout scripts/tts_shootout.py
```

Ordinary tests require no downloaded TTS model or GPU. They verify sample
integrity, failed inference, cache separation, stable anonymization, incomplete
results, choice-before-reveal, rating snapshots, and partial telemetry handling.

Choose by naturalness, cloning quality, conversational feel, and expression
before latency, fan behavior, and memory. Neither faster inference nor finite
waveforms select the everyday engine. Production integration is a later accepted
step; all current engines, references, configuration, and architecture remain.
