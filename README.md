# CIEL

A Python desktop voice assistant for Linux: speak to her, get spoken replies,
remember useful facts, search public information, and launch supported applications.

**Current version: 0.3.6 development.** Core voice, persistent memory, public web
retrieval, and the first local actions are implemented. Live microphone, restart,
internet, and desktop acceptance checks remain open in
[release checks](docs/release-checks.md).

## What works today

- Local wake recognition and speech-to-text with Whisper; optional openWakeWord models.
- NVIDIA NIM, Groq, OpenRouter, and Ollama clients with routing and provider fallback.
- CPU Pocket TTS with streaming and a same-speaker delivery bank, local Piper
  fallback, and optional custom Chatterbox Turbo HQ speech.
- Sentence-by-sentence speech while the provider is generating, interruption
  handling, and optional captions that follow playback.
- SQLite conversation history, background summaries, and persistent facts.
  Conversational facts need confirmation; explicit `remember ...` commands save directly.
- Local time/date, RAM usage, GPU temperature, CPU core count, and supported Linux app launching.
- Free public web search and HTML/text page reading, with bounded requests and source links.
- Editable personality preferences, voice-requested style adjustments, and concise
  or development console output.

Reminders, model-selected tool calls, messaging, Windows support, tone detection,
and presence detection are **planned**.
See the [roadmap](docs/roadmap.md) for implementation status and priorities.

## Install and start

Use **Python 3.10+** on Linux with a working microphone and audio output.
A GPU is optional: command STT prefers CUDA and recovers on CPU if unavailable;
the default Pocket voice runs on CPU. Initial package and model
installation needs internet access. Provider replies need a configured cloud
account or a running local Ollama server with a model installed.

```bash
git clone https://github.com/Hexarion0/ciel.git Ciel
cd Ciel
bash setup.sh
ciel start
```

The installer creates `.venv`, installs CIEL with Pocket support and CUDA STT libraries,
downloads fallback Amy, the configured command STT model (default `small.en`),
and the separate wake model (`base.en`), then opens configuration for API keys, voice, and
audio devices. If `.env` already exists, it offers to reconfigure it.
It also installs a `ciel` launcher in `~/.local/bin` that uses this checkout's
`.venv` and CUDA libraries automatically. Keep the checkout in place and add
`~/.local/bin` to your shell's PATH if it is not already there.

For a manual installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[pocket-tts,gpu]"
.venv/bin/ciel setup
.venv/bin/ciel start
```

Run these commands from the repository root. The wizard writes `.env` and tests
voice playback; it preserves unrelated settings on reconfiguration. Amy and Whisper
models download when first needed if they are not already present. Alternatively,
copy `.env.example` to `.env` and edit it; fresh settings, the wizard, and the example
use CPU Pocket with the CIEL profile (provisional Alba without private references).
Command STT uses `small.en` on CUDA with `int8_float16`, beam size 5, and a guarded
`medium.en` retry. Ambient listening, speech interruption, and live AI captions
start automatically. Existing explicit settings keep their selected values.
Existing explicit voice choices are preserved. If Pocket is unavailable, playback
tries installed local Piper voices; choose Amy in setup to download its fallback model.

Activation is optional with the commands above. In Bash/Zsh use
`source .venv/bin/activate`; in fish use `source .venv/bin/activate.fish`.
After activation, `ciel start` is equivalent to `.venv/bin/ciel start`.
To add the standalone command to an existing installation, run
`bash scripts/install_ciel_command.sh`. Then `ciel` starts listening from
any directory without activation; options such as `ciel setup` and
`ciel start dev` are forwarded unchanged.
If audio initialization reports a missing PortAudio library, install your
Linux distribution's PortAudio runtime and rerun setup.

### Desktop web interface

Run `ciel web` to start the assistant and open its desktop page at
`http://127.0.0.1:8765`. The page centers on a voice orb, with the current reply
in a temporary speech bubble and an input for typed messages and commands.
Microphone, stop, help, and exit controls report their results on the page.
It uses the PC's microphone and speakers, with the same memory and validated
actions as the terminal. You can also type `/mute`, `/stop`, `/exit`, and `/help`.

Use `ciel web --no-open-browser` to open the page yourself, or
`ciel web --web-port 8766` if the default port is in use. Stop an existing
terminal Ciel session before starting web mode so two processes do not
share the microphone and voice models. This first interface serves the PC;
same-Wi-Fi phone access and Telegram integration are planned next.

### Choose a voice

Run `.venv/bin/ciel setup`:

| Choice | Voice | Requirements |
| --- | --- | --- |
| **1** | CIEL custom voice | Chatterbox Turbo model, separate Python 3.12 environment, private reference audio/transcript, NVIDIA GPU |
| **2** | Amy | Piper voice model; CPU playback synthesis |
| **3** | CIEL CPU voice | Pocket extra; same-speaker references or provisional built-in Alba |

New installations default to **Pocket on CPU**. Enter preserves an existing selection.
See [Pocket setup, styles, and benchmarks](docs/pocket-tts.md) for installation,
custom-model access, reference files, streaming tests, and optional INT8.
Custom voice setup saves the choice and checks local assets; it does **not**
install Turbo or recreate your private voice reference. Missing assets are listed
and the custom playback test is skipped until they are supplied.

The current local Turbo model occupies about **2.8 GiB**, and its Python environment
about **6 GiB**, in addition to the main application environment. These are
observed installation sizes, not minimum requirements or download-size guarantees.
The current worker requires CUDA and checks for at least **3,000 MiB free VRAM**
before loading. See [custom voice setup](docs/chatterbox-turbo-integration.md)
for the exact paths, model revision, and recorded hardware results.

The custom voice files are **not in Git**. Back up your reference audio/transcript
and custom Piper model separately if you want the same voice on another computer.
Turbo failures try CPU Pocket, then the installed custom Piper fallback, then installed Amy;
a missing fallback does not trigger an automatic download during a voice reply.

### Provider configuration

The wizard accepts NIM, Groq, and OpenRouter API keys. Keep them in `.env`.
The current NIM default for both ordinary and complex requests is
`nvidia/nemotron-3-super-120b-a12b`, selected in the recorded
[latency comparison](docs/conversation-latency.md). Override it with `NIM_MODEL`
and `NIM_COMPLEX_MODEL`; availability and latency depend on your provider account.
The model's official API example is available from [NVIDIA](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/build).

Ollama is an optional local fallback, not bundled with CIEL. It must be running
at `OLLAMA_HOST` with the requested model installed (currently `llama3.2` by default).
No provider setup is needed for local clock, memory, or supported desktop actions.

## Everyday use

```bash
.venv/bin/ciel start                    # ambient conversation and live AI captions
.venv/bin/ciel start dev                # detailed diagnostics
.venv/bin/ciel start --no-ambient        # use wake-word mode for this run
.venv/bin/ciel start --no-show-ai-transcripts # hide live AI captions for this run
.venv/bin/ciel setup                    # change voice, keys, or audio devices
```

Try:

- “Hey Ciel, what time is it?”
- “Ciel, how much RAM am I using?”
- “Ciel, can you open Discord for me?”
- “Ciel, my favorite game is CS2.” Then “Ciel, yes” to save it.
- “Ciel, what's my favorite game?”
- “Ciel, what's the weather in Paris today?”
- “Ciel, compare current Linux gaming laptops.”
- “Ciel, be more playful.”

Local app launching supports Discord/Vesktop, Firefox, Chromium, Steam, and
VS Code/VSCodium when installed on PATH. “Open it” can resolve a recent app name
or exact spelling correction for 60 seconds. Launches require a direct address
in ambient mode and accept no executable paths, shell commands, or flags.
A launch acknowledgement confirms process creation, not that a window appeared.

Copy `persona.example.txt` to ignored `persona.txt` to customize her style.
Preferences reload before each AI reply. Voice-requested changes update a managed
block; “reset your persona” clears that block while preserving your own text.
These preferences change reply style, not the actual TTS engine or playback speed.
The default style uses simple, direct replies. Bracketed stage directions describing
breathing, pauses, mood, or gestures are removed before speech and from saved replies.

### Internet access

Explicitly addressed, reliably transcribed clock, memory, and supported desktop
commands run through local handlers without an AI request. Other transcripts and
typed messages go to an AI controller, which chooses silence, a reply,
a clarification, web search, page reading, or an existing local tool. By default it uses
`nvidia/nemotron-3-super-120b-a12b` through your NIM account. Conversation replies
use the normal provider settings. Stop and mute controls remain immediate.
Existing `CONTROLLER_MODEL` overrides are preserved; update yours if it still
selects Lightning, then restart CIEL.

For searches, CIEL gives a short spoken acknowledgment, retrieves pages using an
AI-written query, and reports the supported answer. The controller uses recent user
context and relevant saved memories to keep the subject when you say “try again.”
“Why can't you search?” asks for an explanation. If sources are insufficient, the
controller can refine the query or read another page, up to three lookups per turn.
Local tools keep their existing allowlists and confirmation rules.

Current facts can trigger automatic searches. Say “search” for an explicit lookup;
research requests can use the larger retrieval budget. The terminal shows each query,
and the terminal and desktop page show source links. When you ask for a page URL,
CIEL points to the matching source link without reading the address aloud.
Unrelated results are discarded;
source previews alone do not establish an exact answer.
Web replies use a short answer with matching source excerpts, which the application
checks before speech. Invalid output, invented quotes, and unsupported numbers receive
an uncertainty reply. Exact anime-to-novel reading points require a page excerpt,
not only a search preview. Web replies wait for validation before playback; ordinary
conversation still streams when enabled.

Search uses DuckDuckGo and Brave through the free `ddgs` package and needs no search API key.
Public search and sites can block or fail requests. When no useful sources are found, CIEL gives
a local verification-failure response and offers a retry or general background.
The first version reads public HTML and plain text. Account actions, purchases,
file downloads, browser automation, and PDF/media extraction remain future work.

Edit `.env` and restart to change these controls:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CONTROLLER_ENABLED` | `true` | Use AI decisions; `false` restores the earlier intent handlers |
| `CONTROLLER_MODEL` | `nvidia/nemotron-3-super-120b-a12b` | Dedicated NIM decision and web report model |
| `CONTROLLER_TIMEOUT_SECONDS` | `8.0` | Deadline for each action decision |
| `CONTROLLER_MAX_TOOL_STEPS` | `3` | Maximum sequential web lookups per turn |
| `INTERNET_ENABLED` | `true` | Enable public search and page reading |
| `INTERNET_AUTOMATIC` | `true` | Search current facts without an explicit search command |
| `INTERNET_HOME_CITY` | Empty | Default city and country for local questions |
| `INTERNET_TIMEOUT_SECONDS` / `INTERNET_RESEARCH_TIMEOUT_SECONDS` | `8.0` / `20.0` | Budgets for each individual lookup |
| `INTERNET_MAX_RESULTS` / `INTERNET_RESEARCH_MAX_RESULTS` | `5` / `8` | Search result limits |
| `INTERNET_MAX_PAGES` / `INTERNET_RESEARCH_MAX_PAGES` | `2` / `4` | Page reading limits; `0` uses snippets only |

A location in your request overrides the saved default. Set `INTERNET_HOME_CITY`
to your city and country, or tell CIEL the location when asked; a timezone does
not establish your home city. `/local` keeps the conversation session private:
internet retrieval stays disabled for follow-ups and after restarting CIEL,
until that session's history is cleared.
Private-session decisions use Ollama. Local commands remain available without NIM.
For other requests, an unavailable controller produces a short clarification when
addressed and silence for background speech. It does not
silently fall back to a larger cloud model. Each web report has a deadline of at least
12 seconds; several lookups and speech playback can take longer than one lookup budget.

### Listening and performance settings

| Setting | Fresh-install value | Purpose |
| --- | --- | --- |
| `STT_MODEL` / `STT_DEVICE` | `small.en` / `cuda` | Command transcription; CPU recovery if unavailable |
| `STT_COMPUTE_TYPE` / `STT_BEAM_SIZE` | `int8_float16` / `5` | Command decoding precision and beam width |
| `STT_RETRY_MODEL` | `medium.en` | Quality retry when free memory permits |
| `WAKE_STT_MODEL` | `base.en` | Separate CPU wake recognizer |
| `UTTERANCE_SILENCE_SECONDS` | `0.7` | Silence before ending a turn |
| `UTTERANCE_PAUSE_GRACE_SECONDS` | `0.2` | Extra pause tolerance; total default 0.9 seconds |
| `AMBIENT_LISTENING` | `true` | Capture speech and classify intended-listener context |
| `AMBIENT_FOLLOWUP_SECONDS` | `300` | Active ambient conversation window |
| `BARGE_IN_MODE` | `speech` | Interrupt with speech; use `wake` to require a name |
| `SHOW_TRANSCRIPTS` | `false` | Raw STT diagnostics in development output |
| `SHOW_AI_TRANSCRIPTS` | `true` | Captions during playback |

Increase pause grace if CIEL cuts you off while thinking. The previous
1.8-second endpoint was more tolerant of pauses in recorded reference passages;
the current shorter default still needs live microphone acceptance.

Existing installations can add the CUDA STT libraries with
`.venv/bin/python -m pip install -e ".[gpu]"`. The normal launch already selects
CUDA transcription; `./scripts/launch_ciel_gpu.sh` also prepares library paths.
Explicit `STT_DEVICE=cpu` with `STT_COMPUTE_TYPE=int8` selects CPU operation.
`small.en` is the model used
in the recorded GPU tests. Turbo has its own separate environment.

See [usage and tuning](docs/usage.md) for ambient follow-ups, strict mode,
interruption recovery, captions, memory behavior, and STT retry settings.

## Data and privacy

API keys, `persona.txt`, model weights, recordings, databases, and generated
benchmarks are excluded from Git. Accepted conversations are stored in
`data/ciel.db` by default. Forgetting a fact removes the structured memory
and suppresses known wording in provider context; it does not erase archived chats.

Wake recognition, transcription, and speech synthesis run locally. Cloud replies
send conversation context and relevant saved facts to the selected provider.
Web searches send query terms and relevant user context to public search services,
and page reads contact the requested public sites. Use `/local` to keep the session
private or set `INTERNET_ENABLED=false` to disable retrieval.
In ambient mode, short background excerpts stay temporarily in RAM and are sent to
the lightweight NIM controller with bounded dialogue context to judge who is being
addressed. `/local` keeps those decisions local too. There is no speaker
identification or acoustic echo cancellation. Diagnostic transcript logging can
expose nearby speech, so enable it deliberately.

## Development

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pip wheel . --no-deps -w dist
```

Ordinary tests use mocked devices/providers and exclude hardware integrations.
Run `.venv/bin/pytest -m integration` only with the required devices/models.
Optional wake training needs `pip install -e ".[train]"`; this is separate from
custom TTS inference. Wheels include the Turbo worker and default voice manifest.
Installed apps resolve relative voice asset paths under `$XDG_DATA_HOME/ciel`, or
`~/.local/share/ciel` when unset. Source checkouts keep their existing asset paths.
Supply the separate Turbo environment, model, and private reference files; absolute
paths and custom voice profiles remain supported.

```text
src/ciel/
├── actions/        # trusted local actions, validation, execution
├── audio/          # wake recognition, recording, STT, TTS, wake training
├── providers/      # AI clients, routing, fallback
├── memory/         # SQLite facts, conversations, summaries
├── platform/       # Linux audio and system telemetry
├── config.py       # environment settings
├── internet.py     # bounded public search, page reading, source evidence
├── persona.py      # conversational style
└── __main__.py     # CLI and voice loop
scripts/            # launchers, downloads, diagnostics
tests/              # unit and integration tests
voice_profiles/     # versioned manifests; referenced private assets stay local
models/             # local Piper and wake models (ignored)
data/               # private runtime data and Turbo assets (ignored)
```

Documentation: [roadmap](docs/roadmap.md) · [release checks](docs/release-checks.md) ·
[action development](docs/actions.md) · [changelog](CHANGELOG.md).

## License

[MIT](LICENSE)
