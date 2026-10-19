# Pocket TTS for Ciel

Pocket is the recommended CPU speech path for continuous conversation. Chatterbox
Turbo remains the optional GPU HQ engine, and Piper remains the final local
fallback. This integration uses the **released `pocket-tts==3.3.0` API**, rather
than assuming that GitHub main matches the installed package.

## Install

From the repository root, with the existing `.venv`, run setup and select Pocket:

```bash
.venv/bin/ciel setup
```

The [installer](../README.md#install-and-start) installs the lightweight core first.
After you review and save, setup installs only the components needed by your choices
and prepares their models. Pocket is the fresh voice choice; existing selections
are preserved. When Pocket is selected, missing PyTorch is installed from its CPU
wheel index. Amy downloads only when selected, and CUDA libraries require choosing
NVIDIA recognition. Rerun setup to change selections or retry missing downloads.

For manual CPU installation with benchmark tools, create a virtualenv, then run:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[voice,pocket-tts,benchmark]'
```

Pocket's upstream dependency requires NumPy 2 or newer. Keep the separate
Chatterbox environment: its pinned dependency stack is independent of this one.
Python 3.10–3.14 is supported by the released Pocket package; availability of torch/torchao
wheels depends on the Python version and platform.

API sources: [released package](https://pypi.org/project/pocket-tts/3.3.0/) and
[upstream Python API](https://github.com/kyutai-labs/pocket-tts/blob/main/docs/API%20Reference/python-api.md).
The implementation was checked against the 3.3.0 wheel; the main-branch docs can change.

INT8 is an optional upstream optimization:

```bash
.venv/bin/python -m pip install -e '.[pocket-int8]'
```

That extra provides upstream's optional torchao backend. It is not required when
the installed torch native backend works: the recorded INT8 runs used native
torch quantization without torchao. Check wheel compatibility before changing it.

Start with FP32 (`none`) for stability and listening comparisons. Set
`TTS_POCKET_QUANTIZATION=int8` to request upstream `load_model(quantize=True)`.
Ciel does not patch model internals or quantize the decoder itself. If INT8 model
loading is unavailable, Ciel logs the exception and retries CPU FP32 once. The
benchmark records requested and effective quantization separately. Failures
later during inference use Piper for the affected turn, with a Pocket retry
on a later turn after a 30-second cooldown.
Initialization/reference errors stay on Piper until restart, avoiding repeated
downloads or reference encoding during conversation. Failed initialization
releases model/reference tensors instead of retaining an unusable model.

## Configuration and model download

Ciel continues to use its existing `.env` settings, rather than adding YAML.
Fresh settings, `.env.example`, and Enter in a new setup select Pocket/Ciel.
Existing explicit engines and legacy voice-only Piper settings are preserved;
`auto` remains Piper for backward compatibility. Pocket stays an optional extra.
If it is absent, playback tries installed local Piper voices without downloading
fallback assets during a reply; choose Amy in setup to install that CPU voice.

```dotenv
TTS_ENGINE=pocket
TTS_VOICE=ciel
TTS_POCKET_DEVICE=cpu
TTS_POCKET_QUANTIZATION=none
TTS_POCKET_STREAMING=true
TTS_POCKET_REFERENCES=data/voices/ciel
TTS_POCKET_LANGUAGE=english_2026-09
TTS_POCKET_BUILTIN_VOICE=alba
TTS_FALLBACK_VOICE=en_US-ciel-medium
TTS_STREAMING=true
```

Setup warms the selected Pocket model before the playback test and reports
preparation failures so you can retry. Normal listening startup also warms Pocket
before enabling the listener. Construction itself stays lazy. The fixed language
config in 3.3.0 refers to revision-pinned model/tokenizer files. Models and voice
states download through Hugging Face on first use. Set `HF_HOME` to an ignored
local directory if desired, consistently for login, warmup, and normal launch:

```bash
export HF_HOME="$PWD/data/voice/pocket-cache"
.venv/bin/python -m ciel.audio.tts --engine pocket --warmup-only
```

Custom cloning requires accepting the terms at
[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts), then logging in with
a **Read** token from the same account. Run the following locally and paste the
token only into the terminal prompt:

```bash
.venv/bin/python -c 'from huggingface_hub.cli.hf import main; main()' auth login
```

This uses the default Hugging Face cache unless you exported `HF_HOME` above.
Use the same environment for login, warmup, previews, and Ciel launch. A login
in one cache does not automatically authenticate a different cache.

This form also works when an old virtualenv's `hf` launcher points to a renamed
repository. Answer no to adding a Git credential unless you need one yourself.
No token belongs in `.env.example`, Git, benchmark reports, or chat.

Without cloning-model access, upstream can load its public model and predefined
voices. A configured custom reference then fails explicitly and Ciel falls back
to local Piper; it does not silently replace your configured Ciel voice with
Alba. When no neutral reference is configured, Ciel logs that Alba is a
**provisional built-in identity**, using it for every style. Alba has not been
accepted as the finished custom Ciel voice.

After a successful download, `HF_HUB_OFFLINE=1` can avoid network checks during
normal startup. Enable it only after all required model and voice assets have
been cached. Initial network/model loading cannot be forcibly interrupted in
process; playback returns promptly on cancellation and bounded synthesis slots
prevent unlimited workers. Preloading assets avoids this first-run situation.

## One identity, six delivery references

Place recordings of the **same adult feminine speaker** in:

```text
data/voices/ciel/
    neutral.wav
    warm.wav
    amused.wav
    firm.wav
    focused.wav
    soft.wav
```

WAV is recommended; FLAC and OGG are also supported by Pocket's audio loader.
Pocket also accepts exported `.safetensors` voice states. They are model-dependent;
regenerate them when changing language/model version. Ciel prefers recordings
when both are present and caches the resulting states in memory. Upstream
reference processing resamples and downmixes the audio and limits long prompts
to 30 seconds. Audio input does not require a transcript.

Restart Ciel after replacing a reference at the same path so its cached state
is rebuilt from the new recording.

To reuse Ciel's existing neutral recording without copying it:

```dotenv
TTS_POCKET_NEUTRAL_REFERENCE=data/voice/references/ciel/reference-1.wav
```

That setting overrides the bank's neutral file. It is also the fallback for all
missing styles. A missing explicit neutral path is an error. If no neutral
reference exists, style files are ignored and one built-in speaker is used for
all styles, preventing accidental identity mixing.

Ciel cannot establish that arbitrary files share a speaker identity. Only add
style recordings that have been checked by listening to the same speaker. Do
not use six preset voices or unrelated speakers as an emotion bank.

The target is an adult feminine, smooth, slightly lower-pitched, composed voice:
roughly 60% calm/intelligent, 20% elegant/soft, 15% warm, and 5% understated playful
delivery. Clear diction and natural pacing matter more than obvious emotion.
Avoid announcer delivery, childish pitch, exaggerated anime acting, excessive
breathiness, seductive styling, and large pitch swings. The references must
supply that acoustic delivery; Pocket has no textual personality/style-prompt
API in this integration. These characteristics require listening acceptance,
not a benchmark claim.

| Style | Reference delivery |
| --- | --- |
| neutral | Calm, intelligent, composed everyday conversation |
| warm | Gentle reassurance and controlled warmth |
| amused | Dry, quiet teasing; no bubbly acting |
| firm | Serious, composed authority |
| focused | Precise technical diction and slightly brisker pacing |
| soft | Quiet, relaxed, slower late-night conversation |

Missing or incomplete style assets do not create synthetic substitutes.
`TTS_SPEED` continues to control Piper; Pocket retains reference/model pacing
without pitch-changing resampling. Existing Chatterbox vocal-event behavior is
preserved; Pocket/Piper receive words with those markers removed.

## Pipeline and delivery routing

The existing LLM sentence buffer and visible-text cleanup remain in use.
`voice_style.select_voice_style` runs before synthesis, independent of Pocket:
explicit `style=` metadata wins; existing intentional `[chuckle]`, `[laugh]`, or
`[giggle]` cues route to amused; otherwise the default is neutral. It does not
infer the user's mood or run an extra classifier. All six styles are available
through the internal API and preview CLI; automatic contextual routing beyond
amusement is left to future conversation metadata.

```python
tts.speak("Let's check the configuration together.", style="warm")
tts.synthesize("The environment variable is missing.", style="focused")
```

Pocket reuses one lazy CPU model and cached reference states, passing
`copy_state=True` for each utterance. Inference is serialized because upstream
stream generation is not thread-safe. Each CPU chunk reaches a bounded playback
queue and a continuous PortAudio output stream. Playback begins on the first
chunk; remaining audio generates concurrently. No torch import or CUDA probe
occurs during construction. Even a CUDA-enabled torch installation stays on CPU.

`stop()` invalidates the existing reply generation, cancels Pocket inference,
and aborts playback. The adapter drains the canceled upstream iterator so its
internal decoder/generator threads can exit before the model is reused. Stale
synthesis never reaches a new reply. Shutdown cancels active work and drops
cached tensors. Streaming captions and interruption text are estimates without
forced alignment; streaming captions reveal from played samples and finish
with the complete sentence only after playback drains.

Pocket failure before first audio selects installed Ciel Piper, then installed
Amy if the first local Piper voice is unavailable. A streaming failure after
chunks have been emitted stops that sentence; replaying the entire sentence
through another voice would repeat already heard words. Later turns can use
Piper. Fallback never downloads a Piper model during a reply.

HQ preserves Chatterbox's original sentence-prefetch/playback path and native
vocal events. Its failure chain now includes Pocket before Piper. Prepared HQ
fallback sentences are complete waveforms; normal Pocket replies stream chunks.

## Test, preview, and switch engines

Test real Pocket speaker playback; fail the command if a fallback was used:

```bash
.venv/bin/python -m ciel.audio.tts --engine pocket --voice ciel --style neutral \
  --require-engine "Everything is running normally. I can check the rest of the system too."
```

Save a private sample without using the speakers:

```bash
.venv/bin/python -m ciel.audio.tts --engine pocket --voice ciel --style warm \
  --require-engine --output data/voices/previews/warm.wav \
  "You don't have to rush. We'll work through it together."
```

Examples are preview fixtures, never production dialogue. Repeat with all six
styles and compare identity, diction, pacing, comfort, and subtle emotion by
ear. Missing styles intentionally sound like neutral.

Select the normal path in `.env` with `TTS_ENGINE=pocket`. Run
`.venv/bin/python -m ciel --listen` after model/reference setup. `ciel setup`
offers **Ciel Pocket** under **Customize**. Use `TTS_ENGINE=chatterbox_turbo` (or `hq`)
and `TTS_VOICE=ciel` for the existing HQ assets, or `TTS_ENGINE=piper` and
`TTS_VOICE=en_US-ciel-medium`/`en_US-amy-medium` for direct Piper.

Ordinary tests require no real Pocket model:

```bash
.venv/bin/python -m pytest tests/test_pocket_tts.py tests/test_tts.py tests/test_streaming.py
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests scripts/benchmark_tts.py
```

With cached models and a working output device, test the real normal sentence
pipeline and interruption (these commands speak aloud):

```bash
.venv/bin/python -m pytest -m integration tests/test_pocket_tts.py \
  -k 'real_playback or real_barge_in'
```

## Benchmark

See [target-PC measurements and acceptance limits](pocket-tts-validation.md) for
the recorded FP32/INT8/Piper/HQ comparison.

Pre-cache models first. The comparison intentionally activates the GPU for
Chatterbox HQ, so it is separate from normal CPU conversation.

```bash
.venv/bin/python scripts/benchmark_tts.py --iterations 3 \
  --pocket-reference data/voice/references/ciel/reference-1.wav \
  --output benchmarks/tts/custom/report.json
```

Without custom-model access, use `--builtin-pocket` instead of
`--pocket-reference`. For a Pocket-only INT8 comparison add
`--engines pocket --quantization int8` and select a separate output directory.

Each engine runs in a fresh process. The report separates model/reference cold
startup, first synthesis, and three or more warm runs of short, normal,
technical, and longer conversational text. Warm medians include audio-ready
latency, total synthesis time, and real-time factor (synthesis/audio duration;
less than one is faster than real time). It also records process-tree CPU time,
summed RSS, sampled total GPU utilization/VRAM/power/temperature, and
compute-process VRAM attributed to the tested engine. Sample WAVs are retained
for listening. Engine fallback is reported as an error, never as a successful
measurement of the requested engine.

Cold means fresh process/model with filesystem cache retained, not a cold disk
or fresh download. First audio-ready means a generated chunk, not audible
speaker latency. CPU 100% means one logical core. RSS sums may double-count shared
pages. GPU utilization and total VRAM include desktop/other processes; nvidia-smi
polling can miss short power spikes. Results do not prove quieter fans or lower
temperature during sustained conversation. Do a longer real conversation and
listening comparison on the target PC after the custom voice is accessible.

## Troubleshooting

- **Voice cloning unavailable / 401 / 403:** accept model terms and log in using
  the same account and `HF_HOME`; restart Ciel after login to reload model state.
- **Fallback used:** check logs and `actual_engine` in the test command. Restore
  the Pocket extra, cached model, and configured reference. Piper fallbacks must
  already exist under `models/tts/`.
- **INT8 fails:** set `TTS_POCKET_QUANTIZATION=none`; check matching torch/torchao
  wheel support before enabling the optional extra. Do not assume FP32 recovery
  is an INT8 result.
- **Stutters:** inspect streaming empty/underflow warnings, close competing CPU
  workloads, and adjust `TTS_PLAYBACK_LATENCY`. Streaming may outrun a small CPU
  under contention; whole-waveform playback is available with
  `TTS_POCKET_STREAMING=false`.
- **Unexpected GPU activity:** confirm `TTS_ENGINE=pocket` and inspect STT/wake/
  other application settings separately. This upgrade does not change STT's
  independently configured CUDA use.
- **Bad identity or style:** verify reference files and model-compatible states;
  inspect missing-style warnings and listen to neutral first. No automatic
  speaker-identity validation or voice-quality improvement is claimed.
