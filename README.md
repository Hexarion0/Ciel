# RAPHAEL

A Python desktop voice assistant for Linux: speak to her, get spoken replies,
remember useful facts, search public information, and launch supported applications.

**Current version: 0.3.6 development.** Core voice, persistent memory, public web
retrieval, and the first local actions are implemented. Live microphone, restart,
internet, and desktop acceptance checks remain open in
[release checks](docs/release-checks.md).

## What works today

- Local wake recognition and speech-to-text with Whisper; optional openWakeWord models.
- NVIDIA NIM, Groq, OpenRouter, and Ollama clients with routing and provider fallback.
- Default Amy speech through Piper, or RAPHAEL's custom Chatterbox Turbo voice
  when its separate model, environment, and reference assets are installed.
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
A GPU is optional for the default Amy/CPU setup. Initial package and model
installation needs internet access. Provider replies need a configured cloud
account or a running local Ollama server with a model installed.

```bash
git clone https://github.com/Hexarion0/Raphael.git
cd Raphael
bash setup.sh
raphael start
```

The installer creates `.venv`, installs RAPHAEL, downloads default Amy and
Whisper `base.en`, and opens the configuration wizard for API keys, voice, and
audio devices. If `.env` already exists, it offers to reconfigure it.
It also installs a `raphael` launcher in `~/.local/bin` that uses this checkout's
`.venv` and CUDA libraries automatically. Keep the checkout in place and add
`~/.local/bin` to your shell's PATH if it is not already there.

For a manual installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/raphael setup
.venv/bin/raphael start
```

Run these commands from the repository root. The wizard writes `.env` and tests
voice playback; it preserves unrelated settings on reconfiguration. Amy and Whisper
models download when first needed if they are not already present. Alternatively,
copy `.env.example` to `.env` and edit it; the example uses Amy and CPU transcription.

Activation is optional with the commands above. In Bash/Zsh use
`source .venv/bin/activate`; in fish use `source .venv/bin/activate.fish`.
After activation, `raphael start` is equivalent to `.venv/bin/raphael start`.
To add the standalone command to an existing installation, run
`bash scripts/install_raphael_command.sh`. Then `raphael` starts listening from
any directory without activation; options such as `raphael setup` and
`raphael start dev` are forwarded unchanged.
If audio initialization reports a missing PortAudio library, install your
Linux distribution's PortAudio runtime and rerun setup.

### Desktop web interface

Run `raphael web` to start the assistant and open its desktop page at
`http://127.0.0.1:8765`. The page centers on a voice orb, with the current reply
in a temporary speech bubble and an input for typed messages and commands.
Microphone, stop, help, and exit controls report their results on the page.
It uses the PC's microphone and speakers, with the same memory and validated
actions as the terminal. You can also type `/mute`, `/stop`, `/exit`, and `/help`.

Use `raphael web --no-open-browser` to open the page yourself, or
`raphael web --web-port 8766` if the default port is in use. Stop an existing
terminal Raphael session before starting web mode so two processes do not
share the microphone and voice models. This first interface serves the PC;
same-Wi-Fi phone access and Telegram integration are planned next.

### Choose a voice

Run `.venv/bin/raphael setup`:

| Choice | Voice | Requirements |
| --- | --- | --- |
| **1** | RAPHAEL custom voice | Chatterbox Turbo model, separate Python 3.12 environment, private reference audio/transcript, NVIDIA GPU |
| **2** | Default Amy | Piper voice model; CPU playback synthesis |

New installations default to **Amy**. Enter preserves an existing selection.
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
Turbo failures use the installed custom Piper fallback, then installed Amy;
a missing fallback does not trigger an automatic download during a voice reply.

### Provider configuration

The wizard accepts NIM, Groq, and OpenRouter API keys. Keep them in `.env`.
The current NIM default for both ordinary and complex requests is
`nvidia/nemotron-3-super-120b-a12b`, selected in the recorded
[latency comparison](docs/conversation-latency.md). Override it with `NIM_MODEL`
and `NIM_COMPLEX_MODEL`; availability and latency depend on your provider account.
The model's official API example is available from [NVIDIA](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/build).

Ollama is an optional local fallback, not bundled with RAPHAEL. It must be running
at `OLLAMA_HOST` with the requested model installed (currently `llama3.2` by default).
No provider setup is needed for local clock, memory, or supported desktop actions.

## Everyday use

```bash
.venv/bin/raphael start                    # concise console, wake-word mode by default
.venv/bin/raphael start dev                # detailed diagnostics
.venv/bin/raphael start --ambient          # optional continuous speech capture
.venv/bin/raphael start --show-ai-transcripts
.venv/bin/raphael setup                    # change voice, keys, or audio devices
```

Try:

- “Hey Raphael, what time is it?”
- “Raphael, how much RAM am I using?”
- “Raphael, can you open Discord for me?”
- “Raphael, my favorite game is CS2.” Then “Raphael, yes” to save it.
- “Raphael, what's my favorite game?”
- “Raphael, what's the weather in Paris today?”
- “Raphael, compare current Linux gaming laptops.”
- “Raphael, be more playful.”

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

RAPHAEL automatically recognizes common requests for changing facts such as news,
weather, prices, and current recommendations, and reads public HTTP(S) links you
provide. Say “search” to request
a lookup; “research,” “compare,” and “double-check” use the larger research budget.
She uses relevant user memory and recent conversation context to refine queries,
keeps spoken replies concise, and shows source links in the terminal and desktop page.
Follow-up searches keep the active user topic, and the terminal shows the query
before retrieval. Unrelated results are discarded; source previews alone do not
establish an exact answer.
Web replies use a short answer with matching source excerpts, which the application
checks before speech. Invalid output, invented quotes, and unsupported numbers receive
an uncertainty reply. Exact anime-to-novel reading points require a page excerpt,
not only a search preview. Web replies wait for validation before playback; ordinary
conversation still streams when enabled.

Search uses DuckDuckGo and Brave through the free `ddgs` package and needs no search API key.
Public search and sites can block or fail requests. When no useful sources are found, RAPHAEL gives
a local verification-failure response and offers a retry or general background.
The first version reads public HTML and plain text. Account actions, purchases,
file downloads, browser automation, and PDF/media extraction remain future work.

Edit `.env` and restart to change these controls:

| Setting | Default | Purpose |
| --- | --- | --- |
| `INTERNET_ENABLED` | `true` | Enable public search and page reading |
| `INTERNET_AUTOMATIC` | `true` | Search current facts without an explicit search command |
| `INTERNET_HOME_CITY` | Empty | Default city and country for local questions |
| `INTERNET_TIMEOUT_SECONDS` / `INTERNET_RESEARCH_TIMEOUT_SECONDS` | `8.0` / `20.0` | Total lookup budgets |
| `INTERNET_MAX_RESULTS` / `INTERNET_RESEARCH_MAX_RESULTS` | `5` / `8` | Search result limits |
| `INTERNET_MAX_PAGES` / `INTERNET_RESEARCH_MAX_PAGES` | `2` / `4` | Page reading limits; `0` uses snippets only |

A location in your request overrides the saved default. Set `INTERNET_HOME_CITY`
to your city and country, or tell RAPHAEL the location when asked; a timezone does
not establish your home city. `/local` keeps the conversation session private:
internet retrieval stays disabled for follow-ups and after restarting RAPHAEL,
until that session's history is cleared.

### Listening and performance settings

| Setting | Fresh-install value | Purpose |
| --- | --- | --- |
| `STT_MODEL` / `STT_DEVICE` | `base.en` / `cpu` | Command transcription |
| `WAKE_STT_MODEL` | `base.en` | Separate CPU wake recognizer |
| `UTTERANCE_SILENCE_SECONDS` | `0.7` | Silence before ending a turn |
| `UTTERANCE_PAUSE_GRACE_SECONDS` | `0.2` | Extra pause tolerance; total default 0.9 seconds |
| `AMBIENT_LISTENING` | `false` | Capture speech without a wake phrase |
| `AMBIENT_FOLLOWUP_SECONDS` | `300` | Active ambient conversation window |
| `BARGE_IN_MODE` | `wake` | Interrupt by name; use `speech` with headphones |
| `SHOW_TRANSCRIPTS` | `false` | Raw STT diagnostics in development output |
| `SHOW_AI_TRANSCRIPTS` | `false` | Captions during playback |

Increase pause grace if RAPHAEL cuts you off while thinking. The previous
1.8-second endpoint was more tolerant of pauses in recorded reference passages;
the current shorter default still needs live microphone acceptance.

For NVIDIA CUDA transcription, install `.venv/bin/python -m pip install -e ".[gpu]"`,
set `STT_DEVICE=cuda` and `STT_COMPUTE_TYPE=int8_float16`, then run
`./scripts/launch_raphael_gpu.sh`. This launcher prepares CUDA library paths;
it does not install models or force GPU settings. `small.en` is the model used
in the recorded GPU tests. Turbo has its own separate environment.

See [usage and tuning](docs/usage.md) for ambient follow-ups, strict mode,
interruption recovery, captions, memory behavior, and STT retry settings.

## Data and privacy

API keys, `persona.txt`, model weights, recordings, databases, and generated
benchmarks are excluded from Git. Accepted conversations are stored in
`data/raphael.db` by default. Forgetting a fact removes the structured memory
and suppresses known wording in provider context; it does not erase archived chats.

Wake recognition, transcription, and speech synthesis run locally. Cloud replies
send conversation context and relevant saved facts to the selected provider.
Web searches send query terms and relevant user context to public search services,
and page reads contact the requested public sites. Use `/local` to keep the session
private or set `INTERNET_ENABLED=false` to disable retrieval.
In ambient mode, short background excerpts stay temporarily in RAM and may be
sent to that provider to judge who is being addressed. There is no speaker
identification or acoustic echo cancellation. Diagnostic transcript logging can
expose nearby speech, so enable it deliberately.

## Development

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/python -m pip wheel . --no-deps -w dist
```

Ordinary tests use mocked devices/providers and exclude hardware integrations.
Run `.venv/bin/pytest -m integration` only with the required devices/models.
Optional wake training needs `pip install -e ".[train]"`; this is separate from
custom TTS inference. The supported custom Turbo path currently uses a source
checkout: its worker, manifest, and private assets are not bundled in the wheel.

```text
src/raphael/
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
