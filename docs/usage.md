# CIEL usage and tuning

For installation and voice selection, start with the [README](../README.md).
Run commands from the checkout root. This guide describes the current runtime;
older benchmark reports describe the settings used for their recorded runs.

## Providers and routing

The current default NIM model is `nvidia/nemotron-3.5-lightning-30b-a3b` for both
conversation and complex tasks. `NIM_MODEL` and `NIM_COMPLEX_MODEL` override those
choices. The [availability and latency report](conversation-latency.md) records the
earlier Super comparison and the later endpoint failure that prompted this change.
Nemotron 3 and 3.5 requests disable thinking by default for conversational replies.
Missing or retired models (HTTP 404/410) immediately try a different configured NIM backup.
`NIM_FALLBACK_MODEL` defaults to the same model, so an unavailable endpoint is not
retried through an implicit larger backup. Explicit backup choices remain supported.
Empty or reasoning-only responses also try that backup once; reasoning is never
used as the spoken answer. Fallback never splices a new answer into partial speech.
Standalone time and date questions are answered from the local clock without an
AI request. Bounded clock questions ending in the assistant's name also work,
including “What time is Ciel?” when speech recognition drops “it”. That recovery
retains the raw transcript and does not authorize memory commands. Background summaries use the economical route (Groq when configured,
otherwise the default NIM model), with separate `summary` routing logs. Technical
words or long messages alone do not select the complex model; implementation,
debugging, architecture, and explicit `/strong` requests do. `/fast` and `/local`
remain available, and continuation requests retain the preceding task's routing.

`/local` requires Ollama and never falls back to cloud providers. After a local
request, follow-up replies, summaries, and ambient intent classification for that
persisted conversation also require Ollama, even after context trimming or restart.
Clearing the conversation resets the restriction. A local
provider outage reports failure or defers the background request. This setting
controls provider routing; it does not disable microphone capture or persistence.

## Personality and memory

See [memory and personality](memory-persona.md) for confirmed goals and profiles,
contextual recall, structured conversation state, and example conversations.

The voice persona is a warm, mature, confident feminine companion: playful in
casual conversation, focused during tasks, and patient when you're frustrated.
She uses expressive, natural conversational phrasing, with concise answers that
stay warm, specific curiosity about your interests, and reasoned opinions of her own.
She can give one friendly challenge, then respect your decision; personal questions
follow openings in what you say. During an existing conversation, she may revisit a
relevant thread from supplied history or offer a fresh thought, leaving room for the
conversation to end. Human-like delivery does not require invented personal experiences
or repeated AI disclaimers in ordinary small talk.
Set `CIEL_PREFERRED_NAME` to the name you want her to use. English is the
default language. Edit `persona.txt` in the project folder to customize her tone,
humor, affection, language, and reply length. It is reloaded before every AI reply;
restart once after installing this feature, then edits take effect while running.
Copy `persona.example.txt` to `persona.txt` for a fresh template. Your personal
file is excluded from Git. `CIEL_PERSONA_FILE` selects another path (relative
to the working directory), or an empty value disables custom preferences.
You can also say “be more playful” or “change your tone to be calmer” to persist
a style adjustment. “Can you update your persona?” starts a short follow-up where
CIEL asks what to change, then saves your next clear style instruction. Say
“reset your persona” to remove CIEL's managed adjustment while keeping the rest
of `persona.txt` intact. The initial request must be reliably transcribed and
explicitly address CIEL; the follow-up must also be reliable.
The pending edit expires after 60 seconds. Cancellation, farewell, a listening
mode change, or an unrelated question clears it; `/stop` also clears pending edits.
Missing, unreadable, invalid UTF-8, or oversized files fall back to the built-in
personality. Keep the file under 32 KiB; `#` comment lines are ignored. These
preferences shape AI replies; they do not change TTS voices, STT models, local
clock/memory acknowledgements, or saved facts.
The prompt uses configured audio settings and supplied
telemetry; unavailable GPU telemetry does not establish which hardware is present.
Historical replies and summaries are context, rather than verified facts or style
instructions. Generated replies cannot run commands or update saved facts:
the application handles common direct facts (preferred name, favorite items, and
project dates), their corrections, and explicit `remember ...` / `forget ...`
commands. Confirmed facts update one keyed record with a bounded revision
history. Other facts can be saved with `remember ...`. Saved name and favorite
queries can be answered locally. Recall ranks related words and a small set of
aliases; it does not use embeddings or an additional model.
For example: `Ciel, my favorite game is CS2`, then `Ciel, yes` to confirm.
A correction such as `Actually, my favorite game is Valorant` also needs
confirmation. `Ciel, forget my favorite game` removes the saved fact. Forgetting removes the durable fact
and masks its known wording and previous values in history sent to the model,
including after a restart. Archived conversation text remains in SQLite;
paraphrases outside the known wording can still require explicit cleanup.
Forgetting a supported topic that has no saved fact leaves other facts intact.
Saving a new value keeps older forgotten values suppressed unless you explicitly
save that same value again.
Conversation context is scoped to the persona version. The companion persona
starts a separate context while the legacy desktop chat stays archived in SQLite.
Saved facts and preferences remain shared; conversation summaries are excluded
from general memory recall and only the active session's summary is replayed.
Recalled facts include their recorded timestamp so relative statements such as
"this is day three" don't become claims about today. Saved project anchors are
included even for informal questions, with elapsed calendar days and inclusive
development-day counts calculated from a confirmed start date.

## GPU speech recognition

The default command recognizer is `small.en` on CUDA with `int8_float16` and beam
size 5. The installer includes the runtime libraries; existing installations can
add them with `.venv/bin/python -m pip install -e ".[gpu]"` (Linux).
This keeps speech recognition independent of the voice-training environment.
Start with `ciel`, or `./scripts/launch_ciel_gpu.sh`. The launcher adds existing NVIDIA
library directories from the application or Piper training environment before
starting Python. It does not install libraries or start training. CUDA 12 cuBLAS
and cuDNN 9 are required by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu).
Startup runs a short silent inference before reporting CUDA STT ready, because
these libraries may not load until the first transcription. If it fails, the
startup log reports the CPU fallback before you speak.
`STT_STARTUP_TIMEOUT_SECONDS=120` bounds waiting for command-model startup.
Loading errors do not report readiness; a request that times out can be retried
after loading completes. `TTS_SYNTHESIS_TIMEOUT_SECONDS=60` bounds waiting for a
Turbo synthesis response and terminates an unresponsive worker so Piper can take
over. These deadlines do not interrupt native Whisper or Piper inference.
Keep the smaller command model when Turbo or other work shares limited GPU memory.
Both launch commands run the same application. The script selects `.venv/bin/python`,
switches to the project folder, and prepares `LD_LIBRARY_PATH`. Running
`python -m ciel --listen` uses the current Python and working directory without
that preparation; activate `.venv` first. The script does not force GPU settings:
`STT_DEVICE` and `STT_COMPUTE_TYPE` still control transcription.

## Listening, captions, and interruptions

For the desktop web interface, run `ciel web`. It starts the same voice
listener and opens `http://127.0.0.1:8765`. The browser displays a voice orb,
the current reply in a speech bubble, status, and command feedback.
The bubble stays readable until the next request or until you dismiss it; saved
conversation history is not displayed on this page. Text messages use the same queue and memory as
microphone requests. Mute, stop, and end-session
buttons run `/mute`, `/stop`, and `/exit`. Voice capture and playback use the
PC's devices; the page does not request the browser's microphone. You can draft
text while disconnected; runtime actions become available when the page reconnects.

Use `--no-open-browser` for manual opening or `--web-port 8766` for a different
port. Run one Ciel process at a time. The server binds to `127.0.0.1`, so
this version does not accept connections from a phone or another computer.

Keyboard input is enabled automatically when listening in an interactive terminal.
Type at the owner-name prompt (for example, `[hexarion]: `) and press Enter;
no wake phrase is needed. The label uses `CIEL_PREFERRED_NAME` when set,
otherwise the desktop account name. Logs and speech appear above the input
row, preserving your draft while you type. Backspace removes a character,
Ctrl-U clears the draft, and Ctrl-W removes the last word.
Commands report their result with a `[system]:` message, including in normal
console mode. Typed requests use the
same conversation, memory confirmations, and validated actions as voice requests.
A new typed message cancels the previous request. Replies can still be spoken.

| Command | Behavior |
| --- | --- |
| `/mute` | Toggle microphone input off/on. Typed messages and spoken replies still work. |
| `/stop` | Cancel the current recording, queued request, reply generation, and playback. |
| `/exit` | Shut down the listener and voice worker and return to the shell. |
| `/help` | Show keyboard controls. |

Provider overrides such as `/fast explain this` also work in typed messages.
Use `ciel start --no-text-input` to disable the keyboard reader, or
`--text-input` to enable it explicitly (including redirected stdin). EOF closes
keyboard input while leaving voice listening active. Microphone mute is temporary
and does not change `.env`; it discards pending voice capture, not conversation memory.

Listening keeps microphone ingestion separate from wake inference, transcription,
and callbacks. It records immediately after wake detection without a spoken
wake greeting. Recent audio is retained to catch the beginning of your command.
A wake phrase alone, including `Hey Ciel.`, receives a local acknowledgement
without calling an AI provider.
Whisper wake detection retains three seconds of audio by default and submits a
complete snapshot after a short trailing pause, including for slower greetings.
`WAKE_MIN_RMS=0.006` controls the minimum audio level considered for a wake check;
raise it if noise causes excessive checks. `WAKE_WINDOW_SECONDS=3.0` controls
retained greeting audio. Whisper's VAD and transcript confidence checks still
filter wake candidates. `WAKE_THRESHOLD` applies to openWakeWord models, not the
Whisper keyword spotter. Wait for the `Wake keyword spotter ready` log on startup.
After `ciel train-wake`, set `WAKE_MODELS` to the JSON list printed by training
(for example, `WAKE_MODELS=["models/custom/hey_ciel.onnx"]`) and restart.
An empty list keeps the Whisper keyword spotter. Audio capture currently supports
only 16 kHz mono; unsupported rates or channel counts fail during configuration.
`STT ready` reports that command transcription has loaded. Development mode (`ciel start dev`) shows audio
being transcribed, STT word count/confidence, quality retries, superseded speech,
and each ambient reply/silence reason. DEBUG logs include raw candidates, including
background speech. For a single diagnostic run, use `ciel start dev --show-transcripts` to show
the recognized words at INFO level, including rejected ambient speech. Set
`SHOW_TRANSCRIPTS=true` in `.env` to make this the default, or override it for a run
with `--no-show-transcripts`. Recognizable but low-confidence
direct speech can prompt a repeat without executing or saving the uncertain text.
Live AI captions are enabled by default and reveal letters as the voice
plays, including local greetings and clock answers. For a single run, use
`--show-ai-transcripts` or `--no-show-ai-transcripts` to override this setting.
The terminal appends letters to one `CIEL: ` line for the whole reply,
wrapping naturally when the terminal row fills. Ordinary logs use separate lines,
and interruptions leave only the reached prefix.
[Piper's native phoneme/audio alignments](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/ALIGNMENTS.md)
provide spoken-word timing from the same generated waveform; letters are
interpolated within each word's audio span. The playback cursor accounts for the
output device's reported latency. This follows the spoken pace, including speed
changes, without a separate fixed typing rate. Spelling is not a direct map to
sounds, so individual letters remain approximate. Unsupported timing or uncertain
text normalization falls back to an estimate based on the audio duration.
Piper's alignment dependency is included in normal installation; voices are
patched in memory, leaving downloaded and custom model files unchanged.
Both streaming and batch TTS use progressive captions. Redirected output writes
only a final or interrupted snapshot, avoiding control sequences and letter spam.
Captions show intended TTS text; they do not detect a model's mispronunciation.
Complete responses remain at DEBUG level while live captions are enabled and
are saved once in conversation history. If audio fails before any caption
progress, the intended response appears as a plain text fallback.
`SHOW_TRANSCRIPTS` separately displays recognized microphone speech.
Bare wake phrases meeting the acceptance threshold skip the larger quality retry.
If a quality retry runs out of memory, its cached model is released and further
quality retries on that device are disabled for this process. Primary transcription
continues. A CPU fallback discards any cached GPU retry model.
The default `STT_BEAM_SIZE=5` favors recognition accuracy; a smaller beam reduces
decoding work. `UTTERANCE_SILENCE_SECONDS=0.7` plus
`UTTERANCE_PAUSE_GRACE_SECONDS=0.2` gives 0.9 seconds before an utterance finishes;
increase the grace if you pause longer mid-sentence. Recording uses local VAD
instead of relying only on volume, so quiet speech and louder steady noise can
be distinguished. Speech resumed during STT or
reply generation invalidates the old response, retains the new speech onset, and
suppresses stale playback/history. Addressed speech superseded during STT is kept
temporarily and included with the next utterance so additions do not lose the
original request. An in-flight provider request can still finish before the next
queued utterance is processed; its canceled reply is discarded. Linked additions
such as `and ...`, `also ...`, and `no, I meant ...` inherit reply permission from
the exact unfinished request they canceled. Pending fragments are sent as one user
message, while their original wording remains separate in chat history. Unrelated
recordings do not inherit this permission, and inherited permission does not
automatically authorize saving personal facts.

Summaries run separately
from voice replies, wait for at least six new older messages, process bounded
batches, and resume across restarts. Context keeps a bounded tail of messages
awaiting a summary so batching does not immediately lose the preceding exchange.
Provider failures and empty summaries leave the cursor unchanged for a later
retry. The providerless fallback advances only through the turns it describes.
Clearing a conversation also removes its summaries. Sign-off commands such as `goodbye` end follow-up mode;
questions that merely contain farewell words do not.

## Ambient conversation

Ambient listening is the default for `ciel`, `ciel start`, and the GPU launcher.
Existing explicit `.env` choices still apply; set `AMBIENT_LISTENING=true` to
enable it in an older configuration. No additional launch flags are needed.
`--no-ambient` selects wake-word mode for one run. Local Silero VAD captures
speech without a wake phrase at 16 kHz. Direct addresses are handled locally;
possible follow-ups within `AMBIENT_FOLLOWUP_SECONDS` (300 seconds by default)
use an economical `speech_gate` request before a reply.
`AMBIENT_FOLLOWUP_POLICY=conversation` (the default) treats recent dialogue as
continuing: ordinary questions, answers and topic changes can receive replies
without repeating the wake phrase. The classifier distinguishes the assistant,
other listeners and uncertainty. During an active window, valid uncertainty favors
continuation; clear evidence of another listener closes the window. Fresh sessions, malformed judgments and provider failures stay silent. After the
active window, recent dialogue can still support a high-confidence contextual
follow-up for up to 30 minutes after interaction. Clear speech to another listener
closes both windows; ordinary room speech is not automatically authorized.
Set `AMBIENT_FOLLOWUP_POLICY=strict` to require high classifier confidence on every
inferred follow-up. Intent confidence and the chosen policy appear in local logs.

Bounded feedback about recent spoken delivery, such as “Why are you talking so
fast right now?”, “Could you speak more slowly?” and “You sound like a robot, do
you know?”, is accepted locally during that window. Quoted speech and sentences
addressed to someone else still go through the gate. Feedback permits a response;
it does not itself change `TTS_SPEED`. This estimates the intended listener from
text and recent conversation; it does not identify speakers. Conversation mode
can therefore respond to ambiguous nearby speech during an active window; strict
mode favors silence in those cases. Inferred follow-ups do not authorize saved
personal facts or memory commands.
Ambient onset keeps 800 ms of preceding audio to preserve the greeting while VAD
decides speech has begun. The independent keyword spotter also checks ambient audio;
its confirmed address permits a reply even if command STT misses the name. Greetings
such as `What's up Ciel?`, `So what's good Ciel?`, and `So Ciel, what's on
your mind?` are recognized locally, including known name spellings and short leading
fillers such as "so" or "well". A name
at the end of a question does not trim away the preceding question. Keyword evidence
belongs to one utterance and does not restart its recording or authorize the next
background conversation. If new speech supersedes an in-flight STT result, a clear
direct address from that result can still enter temporary context. Its canceled
reply is not executed or spoken; the original user fragment enters history only
when a subsequent addressed utterance is handled.

Say `Ciel, stop listening` to return to wake-word mode, or `Hey Ciel, listen
continuously` to turn ambient capture back on. This switches modes rather than
closing the microphone. With headphones, set `BARGE_IN_MODE=speech`: local VAD
stops playback after `BARGE_IN_SPEECH_SECONDS=0.24` of sustained speech and records
the interruption with onset pre-roll. Saying her name still works. With speakers,
select `BARGE_IN_MODE=wake` to avoid treating CIEL's loudspeaker output as your
voice. Speech mode does not provide acoustic echo cancellation or identify speakers.

The follow-up window starts after playback finishes. Interruptions during a long
reply can still be judged as follow-ups even if that window would otherwise have
expired. A canceled synthesis cannot play later. The next AI reply receives the
previous reply's text and estimated playback progress; asking `continue` can resume
the unfinished explanation. The resume point is estimated from elapsed audio time,
not exact word timestamps.

Up to six short background excerpts are held in RAM for temporary context and
expire after 90 seconds. Raw background turns are not written to chat or saved
as personal facts. Relevant excerpts and candidate follow-ups may be sent to the
configured AI provider for interpretation. Ambient capture uses more STT work,
and judging follow-ups can add cloud latency. Inferred follow-ups do not modify
durable memories: address CIEL directly for memory changes.

The AI uses recent dialogue to interpret small recognition mistakes while retaining
the original transcript. Names, dates, numbers, negation, and commands stay uncertain
when recognition is unclear; the assistant asks rather than inventing missing
details. Low-confidence transcripts do not automatically save personal facts.

Setup keeps existing custom settings and device choices. Press Enter to keep a
device, enter an index or name to change it, or enter `default` to reset it to the
system default. Numeric device indices in `.env` are parsed as integers.

See [release checks](release-checks.md) for automated validation and the
live microphone checks required before tagging a release.

## Memory confirmation and local actions

Say “Ciel, my favorite game is CS2” and she proposes the fact. Say “Ciel, yes”
to save it or “Ciel, no” to reject it. Corrections also require confirmation;
“Ciel, remember my favorite game is CS2” saves immediately. Proposals expire
after 60 seconds, a topic change, cancellation, or restart. In ambient mode, the
confirmation must address Ciel directly. Rejecting a fact does not erase the
conversation transcript or summaries; it prevents a structured long-term fact write.

These commands run locally without an AI request:

- “Ciel, how much RAM am I using?”
- “Ciel, what's my GPU temperature?”
- “Ciel, how many CPU cores do I have?”
- “Ciel, open Discord.”
- “Ciel, can you open Discord for me?”

Application launching supports Discord (with Vesktop fallback), Vesktop, Firefox,
Chromium, Steam, and VS Code/VSCodium on Linux when their executable is on PATH.
Natural requests can include “can you,” “please,” and “for me.” Exact spelling
such as “D I S C O R D” is also recognized. After “Ciel, I meant D I S C O R D,”
you can say “Ciel, can you open it for me?” within 60 seconds. An unrelated
request, cancellation, or uncertain transcription clears that reference. Unknown
or ambiguous names prompt a clarification rather than a guessed launch.
It accepts no shell commands, paths, URLs, or flags. In ambient mode, launching
requires a direct address. A launch acknowledgement means the process was started;
it does not confirm that a window appeared. See [action development](actions.md).

## Expressive voice

With the custom Turbo voice selected, CIEL can chuckle at a joke, sigh with relief,
gasp at a surprise, or groan at a playful frustration. The voice prompt also encourages
natural pauses and emotional wording that fits the conversation. Effects are optional;
they do not appear in captions or newly saved dialogue.

Set `TTS_EXPRESSIVENESS=expressive` in `.env` for up to two vocal cues per reply (the
default), `natural` for at most one, or `off` for none. Restart after changing it.
Piper uses the spoken phrasing but cannot render Turbo's effects. `[giggle]` maps to
`[chuckle]`; `[moan]` maps to `[groan]`, an approximation. For example, a spoken reply
can use `[chuckle] You got me.` or `Oh! [groan] Not again.`
See [Turbo voice details](chatterbox-turbo-integration.md) for the supported events
and local listening benchmark. The setting controls cue frequency, not emotion intensity.
If sighs or groans are missing or robotic, the voice reference may weaken them. The
optional effect-style reference in the voice manifest enables separate, cached cues
while retaining the original speaker embedding and normal speech reference. See the
same guide for the local asset and reproduction steps; `[moan]` remains a groan alias.

## Streaming replies and shared GPU use

`TTS_STREAMING=true` (the default) connects provider tokens to sentence-sized speech.
Generation continues during playback, so the first sentence can play before the
whole response finishes. Set `TTS_STREAMING=false` to restore batch replies.
Playback uses 1,024-frame blocks and requests a 120 ms output buffer so brief
Python scheduling delays are less likely to interrupt audio. If speech stutters,
set `TTS_PLAYBACK_LATENCY=0.20` in `.env` and restart CIEL. Larger buffers add
some delay before sound starts. Playback warnings report output-buffer underruns;
this setting affects playback buffering, not model synthesis time or speaking speed.
Reasoning tags are filtered across token boundaries. Code remains in conversation
history and is omitted from spoken output. Provider fallback happens before text
arrives; a connection failure after a partial answer is reported rather than
splicing a second provider's answer onto it.

New speech cancels pending network reads, synthesis, and queued sentences. An old
synthesis job may finish internally, but its audio cannot play; canceled callers
return promptly. Voice inference is serialized, and worker queues are bounded.
Interruption context includes completed sentences and the estimated position in
the current sentence. An interrupted generated reply is archived only if playback
started, with an interruption note. Archived status notes are separated from
assistant dialogue when building model context; echoed internal markers are
filtered from output and speech. Logs separate first text, first audio, provider
generation time, and total reply time; these are measurements, not latency guarantees.

While Turbo or other work shares the GPU, `small.en` with `int8_float16` is the
default command STT profile. Fresh configuration also selects `medium.en` as the
quality retry. Before allocating a separate CUDA quality-retry
model, CIEL checks free VRAM on its default GPU. `STT_RETRY_MIN_FREE_MB=2048`
sets the minimum; if memory is lower or cannot be checked, the already loaded
primary model receives a stronger decode instead. Temporary CUDA retry models
release their allocation after use. CPU retry models remain cached. Set the
threshold to `0` to disable the probe. This reduces avoidable allocations; another
process can still change free memory between the check and inference.


## Speech recognition accuracy

Command STT defaults to `STT_MODEL=small.en` and `STT_BEAM_SIZE=5`.
Larger models and wider beams need more inference time; compare recognition on
your microphone before changing them. Use an explicit microphone index
or name if the system default is ambiguous.

The wake keyword spotter applies speech filtering and ignores low-confidence
segments. Utterance decoding retains natural repetitions such as "no, no" and
confident short replies such as "okay" or "bye". These safeguards reduce specific
failure modes; they do not identify the speaker or guarantee accuracy in fan
noise. A close microphone and sensible input gain still matter.

Wake recognition now supplies word timestamps to the listener. The detected
wake greeting is removed from the recorded audio before command transcription,
then supplied as known context for that first utterance. Bare wakes receive the
local acknowledgement; follow-up speech and genuine negative statements are
not rewritten. Three seconds of recent audio are retained to cover delayed wake
inference. These timestamps come from the recognizer and may be imperfect.

For ambiguous speech up to 12 seconds long, STT makes at most one quality retry
with a wider beam. `STT_RETRY_MODEL=medium.en` uses a separately loaded model for
that retry; an empty value retries the primary model. The retry model is loaded
only when needed. CPU retry models remain cached; temporary CUDA retry models
are released after use. A failed retry
keeps the primary result and quality check. Longer unclear recordings request a
repeat without a second inference pass. Scores below `STT_MIN_CONFIDENCE=0.4`
ask the speaker to repeat instead of sending a guessed command to the provider.
`STT_RETRY_CONFIDENCE=0.55` controls when to try the second pass. Scores combine
word and segment likelihoods; they are diagnostic scores rather than calibrated
probabilities that a sentence is correct.

`WAKE_STT_MODEL=base.en` controls the separate CPU/int8 keyword spotter. It now
uses base instead of tiny and requires a greeting at the beginning of its
transcript, rather than triggering on every mention of Ciel. Change this
setting independently of `STT_MODEL` if wake recognition needs tuning.

When a keyword snapshot contains only the greeting, the entire recognized
snapshot is consumed; its trailing syllables are not decoded as a new command.
Audio arriving afterward is still retained for the question. If the snapshot
contains both greeting and command, word timing separates them instead.
