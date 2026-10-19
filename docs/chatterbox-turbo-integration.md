# Chatterbox Turbo CIEL voice

## Setup requirements

The [installer](../README.md#install-and-start) installs the lightweight core, then
opens setup to review and save choices before preparing selected components.
Fresh setup selects CPU Pocket. Choose **Customize**, then **Ciel Turbo** in
`ciel setup` to configure Turbo. Setup saves the choice and checks existing assets;
it does not download the Turbo model or create its inference environment.
Turbo requires an NVIDIA CUDA GPU and a separate inference environment. Wheels
include the worker and safe default manifest; private references and model weights
must be supplied locally.

Required local assets:

| Asset | Default path | Included in Git? |
| --- | --- | --- |
| Voice manifest | `voice_profiles/ciel/voice.json` (bundled copy in `ciel/voice_profiles/`) | Yes |
| Turbo worker | `ciel/audio/turbo_worker.py` (checkout wrapper in `scripts/`) | Yes |
| Inference environment | `data/voice/envs/clone/` | No |
| Pinned model | `data/voice/models/chatterbox-turbo/` | No |
| Reference audio | `data/voice/references/ciel/reference-1.wav` | No |
| Reference transcript | `data/voice/references/ciel/reference-1.txt` | No |
| Optional effect style | `data/voice/references/ciel/event-style.wav` | No |
| Selected cue clips | `data/voice/references/ciel/sigh.wav` and `groan.wav` | No |
| Optional custom Piper fallback | `models/tts/en_US-ciel-medium.onnx` and `.onnx.json` | No |

Relative asset paths use the checkout root during source development. An installed
wheel uses `$XDG_DATA_HOME/ciel`, or `~/.local/share/ciel` when unset, for writable
voice assets. Absolute path overrides retain their configured locations. A local
`voice_profiles/ciel/voice.json` overrides the bundled manifest.

The manual preparation commands below use the checkout's helper scripts, with
Python 3.12 installed:

```bash
python3.12 -m venv data/voice/envs/clone
data/voice/envs/clone/bin/python -m pip install -r scripts/voice_requirements/clone.txt
data/voice/envs/clone/bin/python scripts/fetch_voice_model.py chatterbox-turbo --inventory-only
data/voice/envs/clone/bin/python scripts/fetch_voice_model.py chatterbox-turbo
```

The downloader requests the pinned production revision, including when replacing
a cached inventory for a different revision. The full model is about 2.8 GiB;
the installed inference environment measured about 6 GiB on the development PC.
Download sizes and installation footprints can differ across machines.

Restore your private reference WAV and matching transcript to the paths above
from a separate backup. The downloader supplies model weights, not the original
CIEL voice reference. Copy the custom Piper model/config separately if you want
that fallback too. Amy is downloaded only when selected in setup; choose Amy once,
then rerun setup and select Turbo to retain it as an installed fallback.
Run `.venv/bin/ciel setup`, select **Ciel Turbo**, and check actual playback.
Environment/model locations can be overridden with `TTS_CHATTERBOX_PYTHON`,
`TTS_CHATTERBOX_MODEL`, and `TTS_VOICE_PROFILES`.

The requirements file records the tested environment. A fully automated,
clean-machine custom install remains a roadmap item; file presence checks do not
prove CUDA compatibility or synthesis quality. Installed Turbo startup resolves its
bundled worker without a checkout, but still requires the inference environment,
model, and private reference assets listed above.

## Runtime

CIEL can use the pinned local Chatterbox Turbo checkpoint with the selected primary
reference. The production request path is:

```text
LLM stream -> visible text filter -> sentence buffer -> bounded synthesis queue
           -> persistent Chatterbox Turbo worker -> ordered audio playback
```

The model runs in a persistent Python 3.12 child process so it does not conflict with
CIEL's Python 3.14 dependencies. It loads once in the background after STT is ready,
keeps the primary reference conditioning warm, and runs offline. A free-VRAM guard skips
Turbo startup below `TTS_MIN_FREE_VRAM_MB`. A synthesis or initialization failure is logged
and latched to the local Piper CIEL voice; Amy is tried if that fallback model is absent.
The existing Piper backend remains selectable with `TTS_ENGINE=piper` and its selected
voice remains `TTS_VOICE`.

The voice manifest is [voice_profiles/ciel/voice.json](../voice_profiles/ciel/voice.json).
It names the exact reference WAV/transcript and records the pinned model revision
`749d1c1a46eb10492095d68fbcf55691ccf137cd`. Model files, audio, generated samples, and
benchmarks are kept under ignored `data/` paths. The inference environment is pinned in
`scripts/voice_requirements/clone.txt`.

Turbo's nine verified event markers are `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`,
`[groan]`, `[sniff]`, `[gasp]`, `[chuckle]`, and `[laugh]`. They can appear at the start of
a sentence or inline at a reaction point. CIEL also accepts `[giggle]` as `[chuckle]`
and `[moan]` as `[groan]`. These are application aliases: Turbo has no verified native
moan token, so the latter is an approximation, not a distinct moaning voice.
The native set and inline placement follow the
[official Turbo demo](https://github.com/resemble-ai/chatterbox/blob/ed27b95ee46b95be201147bafe5ca85ac57ac4f2/gradio_tts_turbo_app.py).

`TTS_EXPRESSIVENESS=expressive` is the default. It permits up to two events per reply,
including across streamed sentences; `natural` permits one, and `off` removes them all.
The limits also apply to direct batch synthesis. Stacked cues are collapsed. The prompt
asks for at most one cue per sentence and usually zero or one in a short reply. It pairs
chuckles with amusement, sighs with relief or shared weariness, gasps with surprise, and
groans with light frustration. Serious or sensitive moments stay calm; coughs, sniffs,
and shushing need a specific reason. No events are inserted automatically from keywords.

All spoken engines receive guidance for warmer wording, varied short clauses, natural
pauses, and tone suited to the conversation. Piper does not perform Turbo events; they
are removed before Piper synthesis, including after a Turbo failure. Captions, web
replies, logs, and newly saved assistant turns omit cues while Turbo receives them.
Existing archived turns are not rewritten. A standalone supported event can be synthesized
directly with Turbo, but model replies should still contain dialogue.

These settings control event frequency and spoken phrasing, not sustained acoustic moods.
The pinned Turbo engine ignores the original Chatterbox `exaggeration` and CFG controls,
so CIEL does not expose a misleading emotion-strength slider. The speaking voice
reference and sampling defaults are retained. Restart after changing `.env` or the manifest.

### Faint sighs and groans

The selected reference can make native effects faint even when the tokenizer and audio
pipeline accept their tags. CPU checks on 2026-10-10 found the original standalone sigh
at RMS 0.00042 and groan at 0.00111; greatly amplifying these produced poor listening results.
Using a separate speaking-style prompt while keeping CIEL's speaker embedding and
waveform-decoder reference produced RMS 0.02364 and 0.00805 at seed 42.
These measurements establish signal level, not human sound quality.

The optional `event_reference_audio` manifest entry enables separate conditioning for
sighs and groans. The worker renders each effect separately, balances it with at most
3x gain, and caches it in memory. Normal speech uses the original conditionals, including
after an effect failure. Very faint, invalid, or excessively long effects are omitted with
a warning rather than amplified into noise. Inline effects retain their position with a
60 ms joining pause. Other native cues keep the original synthesis path.

The `event_audio` manifest entries prefer selected local 24 kHz WAVs for each cue. On this
machine they contain the sigh at seed 73 (variant B) and the groan at seed 42 selected during
listening. These exact clips are reused across restarts, avoiding new variants from GPU
sampling. The clips work even if the optional style reference is absent. Invalid files
fall back to synthesis. Generated cue defaults use the same seeds and retain the clone's
speaker identity; waveform level alone does not establish the right sound.

The local effect-style file comes from the public reference used by the
[official Turbo demo](https://github.com/resemble-ai/chatterbox/blob/master/gradio_tts_turbo_app.py).
To reproduce this optional asset from the checkout root:

```bash
curl --fail --location --output data/voice/references/ciel/event-style.wav \
  https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav
```

The WAVs stay outside Git. You can replace the style with a clean reference over five
seconds long and restart. If it is missing or cannot be conditioned, the worker reports
that it is using native inline effects when selected clips are also unavailable. Without
selected clips, the first use of each separately rendered effect adds a synthesis call;
later replies reuse its cached cue. `[moan]` still maps to a groan.

## Sentence queue and interruption

Turbo produces sentence waveforms rather than a live audio stream. CIEL starts synthesis
as soon as a complete sentence is available, while a separate player emits earlier audio in
order. One sentence may wait in the text queue, and the audio queue defaults to two waveforms.
Backpressure limits memory use on long replies. A barge-in invalidates the active generation,
stops playback, and discards queued audio; it cannot be replayed by a later turn.

The benchmark uses a local fixed text stream through CIEL's `stream_reply` implementation,
so it measures actual text filtering, sentence preparation, worker IPC, and local playback
without making an LLM/API request. Live LLM time before the first sentence is available will
add to the reported first-audio times. Device output must be available for playback timing.

## Reproduction

To save just three expression previews using the local clone environment on CPU:

```bash
data/voice/envs/clone/bin/python scripts/preview_voice_expression.py
```

This uses the installed model and primary reference, makes no network request, and
does not play audio or allocate CUDA memory. WAVs and a timing report go under ignored
`data/voice/expressive-previews/`. `--expressiveness off` generates the same phrases
without effects for comparison; `--output PATH` selects another local output directory.
This is a synthesis check, not a judgment of conversational timing or audible quality.
`--cues-only` saves standalone clips. When setting up without existing selected clips,
install the style reference first, generate standalone previews, listen, then copy the
chosen sigh and groan files to the manifest paths. The groan preview is named
`moan-alias.wav`. Existing selected clips take precedence over new generation.

Run the full on-device TTS/STT and playback test with:

```bash
.venv/bin/python scripts/benchmark_ciel_turbo_runtime.py
```

It plays and saves the short response, a multi-sentence response, a longer reply, `[chuckle]`,
`[sigh]`, inline cues and aliases, a cue-budget example, two rapid requests, an interrupted
reply, the Piper fallback, a CUDA STT overlap,
and a post-restart sample. It writes a private `index.html` listening page, WAVs, and
`report.json` below `data/voice/benchmarks/ciel-turbo-runtime/`. Use `--no-playback` to
save samples without sending them to the audio device.

## GTX 1660 SUPER validation (2026-10-05)

Run on the CIEL machine with its configured CUDA `small.en` STT model, `int8_float16`,
the pinned Turbo worker, the primary reference, and the default audio output. The benchmark
used a local provider fixture (no paid or remote LLM request):

- Turbo cold ready after STT was resident: **13.13 s** (model load **7.69 s**, reference
  conditioning **0.68 s**). A new process after shutdown took **13.03 s** to become ready.
- Short “Of course.” warm text-to-first-playback: **1.00 s**. Other warm first-audio times
  were **0.73–1.56 s**, depending on sentence length and GPU load.
- The two-sentence reply had one measured inter-sentence gap of **70 ms**. The four-sentence
  longer reply had gaps of **71–74 ms**. The synthesis queue ran ahead while playback was
  active; maximum observed audio queue depth was **2**. Sentence real-time factors ranged
  from **0.39–0.61** in this run.
- Total GPU memory peaked at **4,347 MiB of 6,144 MiB** with STT and Turbo active. The
  Chatterbox worker's peak resident RAM was **2,357 MiB**. No OOM or inference failure occurred.
- STT decoded the 31-word primary reference with **0 word errors**. STT latency was **0.56 s**
  alone and **1.32 s** during simultaneous Turbo synthesis/playback; overlap worked, with the
  expected GPU contention increasing STT latency.
- Two rapid requests completed in order. A barge-in stopped the active response in under
  **0.93 s** from request start and no second queued sentence started playing. A forced Turbo
  failure produced audio through local Piper fallback. Cold worker reload and synthesis both
  succeeded.

These are one on-device run with a local text-stream fixture; real LLM first-token delay and
environmental variation are not included. The listening page from this run is stored in the
ignored benchmark directory alongside the measurements.
