# Historical stabilization notes — v0.2 and early v0.3

This document preserves earlier investigations and their results. It is not the
current task list; some referenced engines and training scripts have since been
removed. Use the [roadmap](roadmap.md) for current priorities and
[release checks](release-checks.md) for outstanding live acceptance.

Scope: polish the existing v0.2.0 assistant before adding features. Items below
come from the current source, focused tests, and local reproductions.

## Fix First

- [x] **Make installation and CLI startup self-contained.**
  `audio/tts.py` imports `edge_tts`, `msgpack`, and `requests`; `audio/trainer.py`
  imports `onnx`. These packages are not declared directly in `pyproject.toml`
  or `requirements.txt`. `onnx` is absent from the inspected environment, and
  `audio/__init__.py` imports the trainer even for ordinary runtime commands.
  Declare required dependencies and load training dependencies only for training
  commands. Verify a clean installation can run `python -m ciel --help` and
  use Piper without installing the training toolchain.

- [x] **Restore and clear summaries by session identity.**
  `memory/manager.py:get_summary()` searches summary *content* for the session
  ID, although the ID is saved in metadata. A newly created manager failed to
  restore a saved summary in a local reproduction. `clear_session()` deletes
  turns but leaves stored summaries. Retrieve summaries by exact session ID and
  remove that session's summaries when clearing it. Test restart restoration,
  isolation between sessions, and clearing followed by a restart.

- [x] **Preserve configuration when running setup.**
  `setup_wizard.py` rewrites `.env` with fixed defaults and drops custom STT,
  memory, and Fish Speech settings. The selected voice now saves its matching
  engine, and an omitted engine follows the voice on normal startup.
  Preserve existing values and report
  voice-test success only when `tts.speak()` succeeds. Verify setup followed by
  a normal launch uses the same voice and settings.

- [x] **Parse audio device indices consistently.**
  Setup writes numeric device choices to `.env`, but `int | str` settings retain
  values such as `"3"` as strings. `platform/linux.py:resolve_device()` treats
  strings as name searches; TTS passes the string directly to sounddevice.
  Normalize numeric strings to integers while preserving device-name queries.
  Test microphone and speaker selection by both index and name.

## Reliability and Response Quality

- [x] **Keep slow work outside the microphone callback.**
  `audio/listener.py:_audio_callback()` runs wake detection and invokes `on_wake`
  while holding the listener lock. Wake detection can run Whisper inference;
  `__main__.py:on_wake()` synthesizes an acknowledgement before starting playback.
  Queue this work outside the audio callback and measure callback latency against
  the 80 ms frame interval. Test wake acknowledgement, barge-in, failed stream
  startup, shutdown during processing, and restarting the listener.

- [x] **Bound summarization work and avoid repeated requests.**
  After the threshold is exceeded, `summarize_older_turns()` rereads and summarizes
  all older turns on each call, saving another summary. The call in `__main__.py`
  is synchronous despite its background-work comment. Track the last summarized
  turn and update the summary incrementally. Verify unchanged history causes no
  additional request and long conversations keep summary input bounded.

- [x] **Make failed voice preparation clean and retryable.**
  `audio/voice_dataset.py` removes `_scratch` only after extraction and
  transcription succeed. Exceptions leave temporary audio and partial clips;
  reruns start numbering at `0001` in the same output folder. Use guaranteed
  scratch cleanup and prepare into a staging directory before publishing the
  dataset. Test extraction/transcription failures and a retry against an existing
  dataset; existing valid metadata and clips must remain consistent.

- [x] **Recognize farewell intent without matching ordinary questions.**
  `__main__.py` searches for farewell words anywhere in user input and AI replies.
  The existing regex matches `How do you say goodbye in Spanish?`, causing a
  canned farewell instead of answering. Restrict session-ending detection to
  deliberate sign-offs. Test actual farewells, quoted words, translation requests,
  and explanatory replies mentioning goodbye.

## Verification Before Moving On

- [x] Separate deterministic unit tests from audio/model integration tests;
  ordinary tests should not require a microphone, model downloads, or services.
- [x] Resolve the original 62 Ruff findings: 54 long lines, six unused imports, one
  unnecessary f-string, and one import-order issue.
- [x] Update README installation instructions to include the editable package
  installation and describe the actual `src/ciel/` layout and supported engines.
- [ ] Run the unit suite, Ruff, a clean-install CLI check, and a manual microphone
  session covering wake, follow-up, interruption, silence, provider failure, and
  exit. Treat live audio checks separately from unit-test results.

## Original Audit Evidence

- Focused configuration, logging, provider, router, and memory suite: **24 passed**.
- Local checks reproduced missing summary restoration, retained summaries after
  clearing, numeric device strings, and farewell false positives.
- The audio-inclusive test run stalled; an isolated import trace stopped during
  sounddevice initialization. It was stopped, so audio behavior is unverified here.
- This checklist proposes fixes; application code was not changed during the audit.


## Selected Fixes Completed

The dependency/startup, failed-preparation cleanup, farewell detection, and
microphone callback items above are implemented. Voice preparation stages clips
and metadata before replacing the destination, restores the previous dataset on
publication failure, and retains a recoverable backup if restoration itself fails.

The listener now uses bounded audio ingestion, asynchronous wake inference,
separate notification and conversation workers, and recent audio for commands
spoken immediately after wake detection. Old responses cannot cancel new
barge-in recordings. Wake greetings no longer delay recording. Silence timing
uses audio samples; defaults are a one-second pause and STT beam size three,
configurable through `UTTERANCE_SILENCE_SECONDS` and `STT_BEAM_SIZE`.

Summary requests run separately from replies. The remaining memory and setup
items were completed in the stabilization pass below.

Validation: **118 unit tests passed**, with three integration tests excluded by
default. The local Piper synthesis integration test also passed. Ruff passes for
changed modules and tests. A wheel built successfully;
its isolated installation starts CLI help without importing native audio,
Whisper, or training dependencies. In a synthetic stress check with detection
blocked, callback latency was 0.0036 ms median and 0.0224 ms maximum, with at most
eight queued frames. This measures ingestion overhead, not live recognition or
end-to-end reply latency. Live microphone checks still require the target hardware.

The reported listen failures are also covered: punctuation after a bare wake
phrase no longer triggers a provider request; automatic TTS engine selection
uses the selected Piper voice; Fish Speech tries installed local fallback voices
before network speech. NIM handles null content, requests non-thinking Nemotron 3
answers, and checks the backup response rather than returning empty text.

Live service verification: a short NIM request to the configured Nemotron 3
model returned a visible answer in 0.95 seconds. Loading the configured
`en_US-amy-medium` voice and synthesizing a short sentence locally took 1.33
seconds. These checks did not record microphone audio or play speech. The
non-thinking request option follows
[NVIDIA's Nemotron 3 documentation](https://docs.nvidia.com/nim/large-language-models/2.0.4/turbo/get-started-nemotron-3-super-120b-a12b.html#reasoning).

## Stabilization Pass — 2026-10-02

- Summaries restore by exact session metadata, including summaries from older
  releases. Clearing deletes the session's summaries and turns atomically and
  invalidates in-flight summary results. Other sessions remain intact.
- Summarization updates one stored summary from at most 32 new older turns per
  request, with at most 1,000 characters per turn and 2,000 characters of prior
  context/output. The persisted turn ID resumes progress across restarts;
  unchanged history sends no request. This bounds input by truncating very long
  turns, so details beyond those limits may be omitted.
- Background summaries also work with an in-memory SQLite database.
- Setup preserves existing configuration, comments, and unchanged quoting,
  including STT, Fish Speech, and memory settings. Enter keeps device selections;
  `default` resets them. A changed voice saves its matching engine, and failed
  playback no longer reports success.
- Numeric audio device strings become integer indices; name queries remain names.
  Regression tests cover both settings and direct backend/TTS calls.
- The entire source and test tree passes Ruff. Unit engine-selection tests are
  independent of the local `.env` engine choice.

Validation: 128 unit tests passed; three integration tests were excluded by
selection. The local Piper synthesis integration test passed separately. Wheel
build and fresh dependency installation succeeded with Python 3.14.7. The isolated
wheel starts CLI help without importing PortAudio, Whisper, or training modules,
and synthesizes Piper speech with the existing local model and no `onnx` training
package. `pip check` verifies the installed dependency set.

Live microphone and speaker checks remain outstanding: this execution environment
has no `/dev/snd`. Follow [the manual release checks](release-checks.md) on the
assistant's target machine before tagging a release. Earlier recorded service and
callback timings above are historical audit evidence, not measurements from this
pass.

## STT Follow-Up — 2026-10-02

The supplied live log demonstrates wake detection, replies, follow-up, barge-in,
and clean Ctrl+C shutdown with zero dropped frames, but also shows incorrect
transcripts. It does not establish complete hardware acceptance or reliable STT.
The user reported PC hum without competing speech.

Utterance decoding now uses a short name hint instead of a command-word list,
removes repetition suppression, preserves confident short responses, and skips
inference for digital silence. Wake spotting uses speech filtering and rejects
low-confidence/no-speech segments. Local recognition configuration was changed
from cached `base.en` to cached `small.en`, with beam size five; CPU/int8 and the
selected Piper voice were retained.

Both configurations recovered all words of five generated Piper phrases in a
synthetic comparison. CPU decoding took about 0.6 seconds per phrase with the
previous base configuration and 1.6–2.0 seconds with the small configuration.
This clean generated speech comparison does not prove improved recognition of
the user's voice. Both tiny and small models produced no transcript for a
synthetic 60 Hz hum through speech filtering. Live retesting is still required.

## Wake Handoff and STT Quality — 2026-10-03

A new live report showed "Hey Ciel" being transcribed as "I hate Ciel."
The listener had been re-transcribing wake audio after the keyword detector
already recognized it. Keyword recognition now returns word timing in original
sample coordinates; the listener removes the greeting audio, retaining commands
spoken immediately afterward. Sample coordinates account for asynchronous
inference delay. A known greeting is used only for that initial utterance;
follow-up speech is not relabeled. Mentioning Ciel inside a sentence no longer
counts as a greeting. Recent audio retention was increased to three seconds.

STT now reports word/segment quality, retries ambiguous short utterances once
with a wider beam (optionally a separately loaded larger model), and requests a
repeat when quality remains low. The listener carries the quality result to the
CLI, which does not contact a provider for unclear speech. The larger retry model
is cached after first use and adds memory and latency. Local configuration keeps
small.en on CPU/int8 with beam five and uses cached medium.en only for retries.

Generated speech plus synthetic 60 Hz hum confirmed that a bare wake leaves no
command after timestamp trimming. The medium model took roughly 5–10 seconds
for the tested CPU decodes, so it was not selected for every utterance. This is
synthetic evidence only: word timing, quality scores, and larger models cannot
guarantee correct transcription of the user's voice. Live retesting remains
necessary.

The separate keyword spotter was also upgraded from tiny.en to base.en, exposed
as `WAKE_STT_MODEL`. The first synthetic end-to-end run with tiny missed one
wake-and-command greeting. Bare wakes with no post-greeting speech now bypass
command decoding entirely, including brief leftover greeting sounds.

Additional synthetic runs exposed imperfect wake word-end timestamps: decoding
only a greeting's leftover syllable could still invent a confident command.
For a snapshot recognized as a bare greeting, the whole recognized snapshot is
now consumed as the greeting. Only newer audio can become a command; snapshots
containing a recognized command still use word timing to retain that command.
Regression coverage checks this distinction, preservation of genuine negative
statements, one-use greeting context, repeat requests without provider calls,
retry model reuse/failure, and rejection of repetition-heavy hallucinations.

Final validation for this pass: 153 unit tests passed, with three integration
tests excluded; Ruff and whitespace checks passed. With generated Alan voice
speech plus synthetic hum, the final handoff returned no command for a bare
wake, retained the question in a combined wake-and-command utterance, and retained
a genuine "I hate Ciel" statement following a greeting. Normal small-model
command decoding took about 2.2–2.3 seconds in that final sample. These checks are
not a substitute for live recognition on the user's microphone.

## Personalized Persona and GPU Setup — 2026-10-03

The user selected a warm, mature, confident feminine companion, with casual
flirting, light wit, focused task assistance, patience during frustration, and
direct explanations of inconsistencies. English is the default language and
the local preferred name is hexarion. `CIEL_PREFERRED_NAME` keeps that preference
separate from the desktop login; the persona remains in Python for now.

The prompt uses configured speech engines and database paths. It treats unknown
GPU telemetry as unknown hardware, distinguishes user corrections from unsupported
assistant claims, and does not promise successful actions or saved facts without
application confirmation. Historical replies and summaries cannot establish the
persona's tone or invent log evidence. Summary generation now preserves claim
sources, including role labels in its local fallback.

The sandbox hid the host GPU. A host check identified a GTX 1660 SUPER with 6 GB
VRAM and CUDA support in the installed CTranslate2 library. Initial GPU decoding
failed because cuBLAS was not on the library search path. Required CUDA 12/cuDNN 9
libraries already exist in the Piper training environment. The new
`scripts/launch_ciel_gpu.sh` exposes existing local libraries before Python
starts, without installing libraries or starting training. Local STT is configured
for small.en with CUDA/int8_float16; the wake spotter and Piper playback remain on
CPU, and the existing larger retry setting is retained.

Using those libraries, a GPU check loaded the cached small.en model in 1.38 seconds
and decoded 3.13 seconds of generated Amy speech in 1.21 seconds. The transcript
was "Hey Siel, what did we build together?" for a phrase spoken as "Hey Ciel.
What did we build together?" This confirms functioning GPU decoding, rather than
live microphone accuracy. Active training can change available VRAM.

Validation: 160 unit tests passed, with three integration tests excluded; Ruff,
whitespace checks, launcher shell syntax, and launcher CLI help passed. Prompt
adherence and the trained voice's naturalness still need a live conversation check.

## Legacy Persona Carryover — 2026-10-03

A live retest loaded the new prompt but still produced the old commanding style.
The runtime reused `desktop_session`, which contained eight assistant turns with
cold catchphrases. Those recent replies were replayed as assistant examples despite
the prompt's warnings. Broad keyword memory recall also admitted conversation
summaries from other sessions into the personal-fact block.

The companion persona now uses a versioned desktop context. Legacy turns and
summaries remain archived in the database; saved facts and preferences remain
available. General recall excludes conversation summaries before applying its
result limit, while the active context retains its own recent turns and summary
across restarts. Future persona replacements can advance the context version.

Two live NIM checks with isolated synthetic legacy history returned warm replies
to "What do you do?" and "What's good?" in 1.56 and 4.16 seconds. These checks used
the configured conversational model, rather than mocked provider responses, and
did not record audio or write to the user's database. Regression tests verify the
actual CLI request excludes legacy replies and summaries, includes saved facts,
and leaves both archived and current conversations in SQLite.

## Dated Project Memory and Practical Answers — 2026-10-03

The next live run stopped using the commanding tone but substituted relationship
monologues for development questions. It also reused an old saved fact saying
"this is the third day" without its recorded timestamp. The September 28
correction had existed in chat history rather than a durable project anchor.

The user's latest explicit correction is now saved locally as a project start
date of September 28, 2026. Recall includes recorded timestamps, always includes
the bounded project-anchor set, and calculates elapsed calendar days and the
inclusive development-day number. On October 3 this yields five elapsed days
and development day six. Original facts and conversation archives remain stored.

The practical companion revision uses a new context version, keeps the user's
chosen warmth and casual playfulness, and prioritizes concrete answers for dates,
code, and features. Positive examples clarify the likely feature/future speech
confusion and continuation requests without substituting philosophical monologues.

Final live NIM checks answered development day six from September 28, interpreted
"add a future" as a possible feature request with a concrete project-journal idea,
and gave the current time directly. Their request latencies were 10.17, 1.14,
and 5.24 seconds; provider latency remains variable. Validation: 168 unit tests
passed with three integration tests excluded, plus Ruff and whitespace checks.
Regression tests cover restart recall, informal queries, relative-fact timestamps,
deduplication, and invalid or future date anchors.

## Structured Memory and Task Routing — 2026-10-03

The complex-model request following a simple clock answer was the background
summarizer. Its transcript was classified as ordinary conversation, where code
keywords or length selected the complex model. Summaries now explicitly use the
economical route and log their task purpose. They wait for six new older messages
instead of requesting a summary after each exchange. A bounded pending tail keeps
context available between summary batches. Standalone clock/date questions are
answered locally and do not schedule provider work.

Conversation routing now reserves the complex model for implementation/debugging,
architecture, embedded code, or explicit overrides. Generic technical explanations
and long conversations use the normal model. Repeated continuation requests keep
the original task's route. Missing providers fall back to configured providers.
Cloud latency still depends on the configured provider and model.

The application recognizes conservative direct statements about preferred names,
favorites, project start dates, and first commit dates. Corrections update a unique
key atomically, retaining up to twenty previous revisions. Existing structured
metadata is indexed additively; arbitrary historical notes are not automatically
converted into structured facts. Questions, uncertain statements, third-party
claims, and unrelated follow-ups do not silently overwrite a fact. Explicit
`remember ...` remains available for other notes. Recall ranks words and aliases,
excluding conversation summaries; it does not provide embedding search.

Forgetting removes the selected durable fact and persists suppression patterns for
its wording, values, and revisions. Those patterns redact recent history, prior
summaries, and new summary inputs sent to providers, including after restart.
Explicit relearning clears suppression for that key. Archived chats remain stored,
and unrecognized paraphrases are outside this literal suppression mechanism.

Validation: 215 unit tests passed, with three integration tests excluded. Coverage
includes the actual CLI local clock/fact callbacks, economical summary routing,
batched context, concurrent keyed updates, restart recall, ambiguous corrections,
and forgetting. Ruff and whitespace checks passed. No live microphone acceptance
test or provider latency measurement was performed for this change.

## Natural Conversation and Wake Capture — 2026-10-03

The built-in prompt and editable personality preferences now favor expressive,
relaxed conversation, varied phrasing, and warmth in short answers. They reduce
canned reassurance, unsolicited advice, repeated closing questions, and routine
AI disclaimers while retaining truthful identity and capability instructions.
Examples cover casual greetings, tone feedback, frustration, and practical help.

The Whisper wake path previously required RMS above 0.012, included quiet speech
in its noise estimate, and retained only 1.5 seconds. The minimum is now configurable
with a default of 0.006, and candidate speech no longer raises its own noise floor.
The default buffer retains three seconds. After 160 ms below the speech gate, a
pending greeting gets one complete snapshot even within the normal check interval.
Reset clears that pending check. Punctuation such as `Hey, Ciel` is accepted
without broadening detection to unrelated mentions. Startup logs explicitly report
when the CPU keyword model is ready. VAD and transcript-confidence filtering remain.

Validation: 230 unit tests passed, three integrations excluded; Ruff and whitespace
checks passed. Synthetic tests cover quieter frame submission, complete slower
greetings, background hum below the gate, punctuation, and reset behavior. These
tests establish capture behavior, not recognition accuracy on the user's voice.
The release checklist includes repeated normal/quiet wakes and actual PC-hum trials;
live microphone acceptance remains pending.

## Ambient Listening, Pauses, and Speech Interpretation — 2026-10-03

`--ambient` starts optional continuous speech capture; `AMBIENT_LISTENING` also
enables it with a normal listening launch. The installed local Silero VAD accepts
arbitrary audio frame sizes through a 512-sample accumulator, detects onset, and
drives recording without relying solely on volume. Startup and voice toggles
require 16 kHz for this path. "Ciel, stop listening" returns to wake-word mode;
"Hey Ciel, listen continuously" enables ambient capture again.

Direct addresses are accepted locally. Recent possible follow-ups are judged on
the economical `speech_gate` route using the last interaction, including local
acknowledgements. The default window is twenty seconds, measured from when speech
began so a long follow-up can still qualify. Uncertain intent, provider errors,
invalid JSON, and explicit family addresses default to silence. Model confidence
is an estimate, not a calibrated probability or speaker-identification result.

Background excerpts are bounded to six entries of 300 characters, expire after
ninety seconds on context access, and remain in RAM. They can enter the configured
provider's prompt but are not directly saved as chat or personal facts. Inferred
ambient follow-ups cannot mutate durable memory; direct addresses and reliable
recognition are required. This implementation has no speaker enrollment,
diarization, or acoustic echo cancellation. Ambient transcription pauses during
TTS playback; direct wake detection remains available for interruption.

Recording adds 0.8 seconds of pause grace to the existing one-second silence
setting. Short utterances use a 0.12-second minimum and initial silence waits up
to five seconds. A resumed-speech onset during processing invalidates the earlier
generation, preserves onset audio, and suppresses its stale response before
playback or assistant-history writes. In-flight provider requests still run to
completion; the next queued utterance may wait for that request.

The prompt interprets small STT wording mistakes with recent context but preserves
the original transcript. Failed recognition retains diagnostic raw text for
address decisions and still asks for a repeat; inferred wording is not passed to
memory saving. Ambiguous names, dates, numbers, negation, and commands require
clarification rather than an invented correction.

Validation: 258 unit tests passed with three integration tests excluded. Ruff,
whitespace, and CLI help checks passed. A local smoke check using the actual
installed Silero model rejected silence and a quiet synthetic 120 Hz hum.
Regression tests cover ambient mode switching, background persistence boundaries,
follow-up judgments and failures, original transcript preservation, thinking
pauses, resumed speech, stale responses, VAD framing, and playback exclusion.
Live microphone/speaker trials and real-world intended-listener accuracy remain
pending; reproduction steps are in `docs/release-checks.md`.

## Ambient Silence Diagnosis and Wake Replay — 2026-10-03

The reported run logged recordings ending but no transcription or reply. INFO
logging previously hid ambient rejections, STT completion, quality retries, and
transcripts superseded by a new recording. Those stages now log explicit metadata
and reply/silence reasons without dumping background text. DEBUG exposes raw
candidates when deliberately enabled. Command STT also announces readiness.

The onset pre-roll increased from 240 ms to 800 ms. Local replay through the old
capture path lost "Hey" in one of four existing wake samples. Another defect
discarded a direct address when fresh speech arrived during STT. The old reply
still stays cancelled, but a confidently decoded direct address may enter the
bounded temporary ambient context so a following utterance can be judged. No old
command or fact is executed or saved by that observation callback.

Recognizable low-confidence speech retains a raw diagnostic transcript even when
the command text is rejected; ambient mode can ask for a repeat when clearly
addressed. Bare wake phrases above the existing acceptance threshold skip the
larger decoder retry. The original configured CUDA check spent 6.83 seconds on
its first retry; ambiguous commands still retain the stronger retry path.
Exact fenced JSON from the ambient judge is accepted, while other invalid results
remain silent and produce a diagnostic warning.

Validation: 266 unit tests passed, three integrations excluded; Ruff and whitespace
checks passed. Four existing local wake recordings were replayed through actual
Silero capture and configured `small.en` CUDA/int8_float16 STT with beam size five.
All four yielded "Hey Ciel" and a direct-address decision, without quality
retry; decode times were 0.94, 0.83, 0.77, and 0.89 seconds. These are transcription
times, excluding recording pauses and model startup. No audio was uploaded or new
audio recorded. Live microphone/device verification remains pending.
