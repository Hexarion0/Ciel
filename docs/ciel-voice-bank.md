# Ciel reference-bank audition workflow

Pocket passed the technical checks but the user rejected all auditioned outputs
as too robotic on 2026-10-14. Keep the implementation and recordings as optional
low-resource assets; further Pocket reference tuning is stopped. The next evaluation
is documented in [the TTS quality shootout](tts-quality-shootout.md).

This phase prepares candidates and listening comparisons. It does not train a
model, change the TTS architecture, replace `reference-1.wav`, change the FP32
default, or activate a new bank before listening acceptance.

Current local audition:

```text
data/voice/references/ciel/candidates/20261014-02/index.html
```

Open it locally:

```bash
xdg-open data/voice/references/ciel/candidates/20261014-02/index.html
```

The page includes the existing baseline, unprocessed and prepared clips, identical
Pocket test sentences, style-specific lines, concealed FP32/INT8 A/B pairs,
listening notes, and a JSON feedback download. Numerical labels reveal only after
both samples finish playing. This is a casual blind comparison, not a secure
clinical study: the private manifest contains the mapping for reproducibility.
No audio or notes are uploaded. The page does not apply a voice or numerical mode.

## Permission and source provenance

Use the four supplied sources only within the permission scope confirmed by the
user. The current `data/voice/source/ciel/permission.json` records the user's
confirmation covering all four. This is an attestation, not independent rights
verification. The downloader and source-processing commands require that record;
they do not infer permission from a publicly accessible video.

Original best-available English audio streams, descriptions, `.info.json`
metadata, original-language captions, and lossless decoded working files remain
under `data/voice/source/ciel/`. Each candidate records the exact source URL,
start/end timestamps, source and reference SHA-256, original/processed duration,
preprocessing, intended style, and review status. Sources are not overwritten.
The supplied sources all resolve to the Clem Whispers channel's original English
tracks. Channel attribution alone does not prove that every clip has one speaker.

The original archive was downloaded with system yt-dlp **2026.08.19**, Node's JS
runtime, and the `bestaudio` selector. Translated caption requests hit HTTP 429
on two videos; their audio downloads completed after disabling those optional
requests. Original English captions were retained for all four. This failure
was not evidence that the audio sources were unusable.

## Install and reproduce

```bash
.venv/bin/python -m pip install -e '.[voice-bank,dev]'
```

FFmpeg, ffprobe, Node, and yt-dlp were already available on this Arch installation.
A fresh Arch system needs the FFmpeg and Node system packages. The existing clone
Python environment and cached 5.5 MiB speaker-encoder weights are used only for
CPU identity screening; **Chatterbox generates none of the reference or test audio**.
See the existing [HQ environment documentation](chatterbox-turbo-integration.md)
if those optional analysis assets need restoration.

Archive confirmed sources:

```bash
.venv/bin/python scripts/ciel_voice_bank.py download
```

Screen complete sources into a **new** directory; existing runs remain intact:

```bash
.venv/bin/python scripts/ciel_voice_bank.py analyze \
  --output data/voice/references/ciel/candidates/NEW-RUN

data/voice/envs/clone/bin/python -m scripts.voice_bank.speaker \
  data/voice/references/ciel/candidates/NEW-RUN \
  --weights data/voice/models/chatterbox-turbo/ve.safetensors \
  --baseline data/voice/references/ciel/reference-1.wav

.venv/bin/python scripts/ciel_voice_bank.py shortlist \
  data/voice/references/ciel/candidates/NEW-RUN
```

Caption word boundaries and waveform measurements screen the entire sources.
Promotional introductions are excluded. Candidate style labels are context hints,
not acoustic emotion classifications. The shortlist uses reference-to-anchor and
within-clip speaker-encoder cosines. Thresholds are uncalibrated: they flag
possible inconsistencies, not establish real-world identity. No references or
speaker vectors are blended into a new voice.

Prepare state caches and both numerical modes, using the authenticated default
Hugging Face cache on this machine:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/ciel_voice_bank.py preview \
  data/voice/references/ciel/candidates/NEW-RUN
```

Offline mode requires previously cached Pocket models. WAV references encode once
per recording SHA-256, Pocket version, language, and numerical mode. The actual
Pocket 3.3.0 `export_model_state` API writes reusable `.safetensors`; later runs
load them through `get_state_for_audio_prompt`. Both test sentences reuse the
same in-memory state, with `copy_state=True`. All model generation runs on CPU.
The common sentence and CPU RNG seed are identical for FP32 and INT8. Speaker
states for the two modes are kept separate. No quality equivalence is assumed.

Check boundaries and render the page:

```bash
data/voice/envs/clone/bin/python -m scripts.voice_bank.speaker \
  data/voice/references/ciel/candidates/NEW-RUN --candidates \
  --weights data/voice/models/chatterbox-turbo/ve.safetensors \
  --baseline data/voice/references/ciel/reference-1.wav

HF_HUB_OFFLINE=1 .venv/bin/python scripts/ciel_voice_bank.py verify \
  data/voice/references/ciel/candidates/NEW-RUN

.venv/bin/python scripts/ciel_voice_bank.py page \
  data/voice/references/ciel/candidates/NEW-RUN
```

Boundary checks use cached Whisper `small.en` on CPU. They read NumPy audio
rather than the file decoder, because the installed PyAV/faster-whisper pair
has an incompatible `metadata_errors` file-opening argument. No runtime STT or
package dependency was changed. ASR and edge-energy flags are review aids;
listen to the original/reference to judge clipped words, music, overlap, reverb,
whispering, breathiness, and artifacts.

## Conservative preparation and manual refinements

Unprocessed candidates preserve the original decoded rate and stereo channels
in WAV files. Prepared references use quiet-edge trimming with 100ms padding,
mono downmixing, polyphase resampling to 24 kHz, and bounded linear gain toward
-23 dBFS RMS. Gain is capped at +6 dB and peak at 0.85. No high-pass, denoiser,
source separation, compression, click removal, pitch, formant, or tempo changes
were applied. This does not prove the sources lack noise; raw clips remain
available to compare naturalness and possible downmix issues by ear.

Add an explicitly selected 5–15 second interval without replacing a prior file:

```bash
.venv/bin/python scripts/ciel_voice_bank.py extract \
  data/voice/references/ciel/candidates/NEW-RUN \
  --id candidate-warm-03 --source 3mZ2ZbciDPM \
  --start 652.76 --end 663.50 --style warm \
  --notes "Longer quiet tail; preserve earlier cut for comparison."
```

Regenerate only missing previews, rerun candidate speaker screening and boundary
checks, then rebuild the page. Avoid concurrent commands that write the manifest.
The current warm-03 is a refinement of warm-01 with 280ms more source audio;
the earlier cut remains available and its positive tail-energy flag is documented.

## Provisional proposal and acceptance

The current private `report.md` and `bank-proposal.json` describe a provisional
neutral/warm/amused/soft bank. Firm and focused remain **neutral fallbacks**;
no poor or unrelated voice was created to fill their names. All style labels,
including the four supplied styles, await listening confirmation.

After previews, candidate speaker screening, and the sustained check finish,
write the findings and inactive proposal:

```bash
.venv/bin/python scripts/ciel_voice_bank.py report \
  data/voice/references/ciel/candidates/NEW-RUN
```

During audition, compare the baseline against every new common-sentence output.
Assess speaker similarity, pronunciation, artifacts, warmth, prosody,
naturalness, emotional character, and long-conversation comfort. Record A/B
preferences before revealing modes. Keep FP32 unless listening finds INT8
quality effectively unchanged. Download the listening notes and supply the
candidate choices before activation. No final filenames or `.env` settings
are changed by this workflow.

After acceptance, use the existing configuration and router: select the accepted
neutral state with `TTS_POCKET_NEUTRAL_REFERENCE`, and put same-speaker style
states under `TTS_POCKET_REFERENCES` as `warm.safetensors`, `amused.safetensors`,
etc. Keep the original recording and provenance. Missing styles reuse neutral.
Normal Pocket → Piper and HQ Chatterbox → Pocket → Piper remain intact.

## Sustained check and validation

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/ciel_voice_bank.py stress \
  data/voice/references/ciel/candidates/NEW-RUN \
  --candidate candidate-neutral-01 --seconds 300
```

This simulates five minutes of wall-clock dialogue pacing, using exported states,
streamed CPU chunks, periodic actual inference cancellation, thread-unwind
checks, cache-use counters, and RSS observations. It does not play the full
conversation. GPU compute memory is attributed to the test process; global GPU
activity can come from unrelated applications or Ciel STT. CUDA initialization
is checked after every turn. A short check cannot prove absence of long-term
leaks, subjective comfort, or sustained fan behavior. Rerun it on the chosen
winner after listening acceptance.

```bash
.venv/bin/python -m pytest tests/test_voice_bank.py
node tests/frontend/voice_bank.cjs
.venv/bin/python -m ruff check src tests scripts/voice_bank scripts/ciel_voice_bank.py
```

The page was rendered and inspected in isolated headless Firefox at desktop and
390px widths. Browser interaction checks cover the blind reveal gate, single
active player, notes export, and filtering. Existing real Pocket integration
checks were also run with the provisional neutral **exported state**, without
changing the active voice or microphone/provider configuration.

All source audio, generated previews, voice states, and local feedback stay in
ignored `data/` directories. Only tooling, template, tests, and documentation
are intended for version control.
