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
  Conversational facts ask for confirmation by default; explicit `remember ...`
  commands save directly.
- Local time/date, RAM usage, GPU temperature, CPU core count, and supported Linux app launching.
- Free public web search and HTML/text page reading, with bounded requests and source links.
- Editable personality preferences, voice-requested style adjustments, and concise
  or development console output.
- Owner-paired Telegram text chat, automatically started when configured, with
  conversation continuity across voice and chat without loading voice models.
- On-request Hyprland screen understanding across all monitors, including mouse
  position, temporary image context, and explicit screenshot delivery in Telegram.
  See [screen awareness](docs/screen-awareness.md) for setup and acceptance status.

Reminders, additional chat services, model-selected tool calls, Windows support, tone detection,
and presence detection are **planned**.
See the [roadmap](docs/roadmap.md) for implementation status and priorities.

## Install and start

Use **Python 3.10+** on Linux. Voice input needs a working microphone, and spoken
replies need audio output. Typed chat remains available without a microphone.
A GPU is optional: fresh setup offers CPU recognition, and CUDA is an explicit
choice. The default Pocket voice runs on CPU. Initial package and model
installation needs internet access. Provider replies need a configured cloud
account or a running local Ollama server with a model installed.

```bash
git clone https://github.com/Hexarion0/ciel.git Ciel
cd Ciel
bash setup.sh
ciel start
```

The installer creates or reuses `.venv` and installs CIEL's lightweight core, then
opens setup for your name, preferences, API keys, voice, and audio devices. Review
and save your choices before setup installs the selected voice and recognition
packages and downloads their configured models. CPU recognition avoids the optional
CUDA libraries; choose **NVIDIA GPU** in setup if you want GPU recognition. Existing
saved settings are preserved by **Quick start**.
It also installs a `ciel` launcher in `~/.local/bin` that uses this checkout's
`.venv` and CUDA libraries automatically. Keep the checkout in place and add
`~/.local/bin` to your shell's PATH if it is not already there.

For a manual installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ciel setup
.venv/bin/ciel start
```

Run these commands from the repository root. Setup offers **Quick start** to reuse
your settings or **Customize** to choose AI keys, voice, audio devices, and listening
preferences. Both paths ask what CIEL should call you, an optional city and country
for weather and local searches, and how it should learn supported facts: ask first
(`confirm`), save only explicit remember commands (`explicit`), or save supported clear
statements automatically (`automatic`). Enter keeps each current answer; `-` clears
your name or city. An empty name uses your computer username, and an empty city means
CIEL asks for a location when needed. Review the choices before saving, installing
selected components, and checking voice playback; you can replay the test or change
speakers. Canceling before saving leaves `.env` untouched and skips these downloads.
Run `ciel setup` again to change choices or install missing components after a
selection change or failed download. Saved choices survive a download failure.
Both paths first let you choose an API-key provider: NVIDIA NIM, Groq, or
OpenRouter. Setup asks only for that provider's key, then lets you add another
provider or continue. You can also skip keys or keep saved connections. With
multiple providers configured, Ciel routes requests automatically. Both paths
offer an optional AI personality interview after API-key entry.
Saved API keys stay hidden, and unrelated settings survive reconfiguration. Setup
installs Pocket only when selected, CUDA libraries only for GPU recognition, and
openWakeWord only when configured. It prepares the configured command and enabled
wake recognition models, plus the selected voice; it does not download Amy for
every voice choice. Pocket's model is prepared before the setup playback test.
Alternatively, copy `.env.example` to `.env` and edit it, then run `ciel setup` to
prepare its selections. Fresh setup uses CPU Pocket with the CIEL profile
(provisional Alba without private references) and offers CPU/`int8` command STT.
The CUDA option uses `int8_float16`; command STT uses `small.en`, beam size 5, and a
guarded `medium.en` retry. Ambient listening, speech interruption, and live AI
captions start automatically. Existing explicit settings keep their selected values.
If Pocket is unavailable, playback tries installed local Piper voices; choose Amy
in setup to download its model.

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
actions as the terminal. Both accept the in-session commands listed below.

Use `ciel web --no-open-browser` to open the page yourself, or
`ciel web --web-port 8766` if the default port is in use. Stop an existing
terminal Ciel session before starting web mode so two processes do not
share the microphone and voice models. This first interface serves the PC;
same-Wi-Fi dashboard access remains planned. Telegram chat runs separately below.

## Telegram chat

Run `ciel chat setup` to select Telegram and enter its bot token locally. Like AI
provider selection, setup asks for credentials only after you choose a service.
Create your bot through [Telegram's BotFather](https://core.telegram.org/bots/tutorial),
then pair your account using the temporary code shown on your PC:

```bash
ciel chat setup
ciel chat pair telegram
```

Selected chat starts automatically after setup, after pairing, whenever Ciel opens,
and when you log in. There is no manual-start or startup-toggle choice. Check its
state with `ciel chat status`; deselect the service in setup to disable it.

Chat uses your existing AI providers, saved facts, and the paired owner's recent
voice conversation. You can continue a voice conversation in Telegram and return
to it by voice. The archives retain their channel and time; memory confirmations
stay in the channel where they were proposed. Chat does not load speech models.
Use `/help`, `/memory`, `/cancel`, and `/reset` in the paired chat.

Automatic startup uses Linux systemd user services. The PC must be on, logged in,
and connected. Pairing automatically pauses and resumes chat; no separate start
or stop commands are needed. Only the paired account's private text messages are accepted.

This first version supports text. Reminders, Discord, Signal, voice messages, and
remote desktop actions come later. Live Telegram delivery and login autostart still
need acceptance with your own bot. See [chat behavior and limits](docs/telegram-chat.md).

### Choose a voice

Run `ciel setup`, select **Customize**, then choose your voice:

| Voice | Engine | Requirements |
| --- | --- | --- |
| CIEL CPU voice | Pocket | Pocket extra; same-speaker references or provisional built-in Alba |
| Amy | Piper | Piper voice model; CPU playback synthesis |
| CIEL custom voice | Chatterbox Turbo | Turbo model, separate Python 3.12 environment, private reference audio/transcript, NVIDIA GPU |

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
Normal requests select `nvidia/nemotron-3.5-super-vl-120b-a12b`; complex requests select
`nvidia/nemotron-3-ultra-550b-a55b`. When NIM is configured, ordinary conversation
uses it too. `NIM_MODEL`, `NIM_COMPLEX_MODEL`, and `NIM_FALLBACK_MODEL` override these
choices. The fallback defaults to Ultra 550B, and replies report the actual served
model. On 2026-10-28, Super 3.5 VL passed direct inference, streaming, native-tool,
and reasoning on/off checks. It replaces the older Super 120B endpoint that returned
404. See the [availability checks](docs/conversation-latency.md).

Use `/reasoning auto`, `/reasoning on`, or `/reasoning off` in the terminal or
paired text chat. `auto` enables reasoning for complex work and disables it for
ordinary replies; `NIM_REASONING` sets the startup preference. Supported Nemotron
models receive the thinking flag, while internal reasoning stays out of replies.

OpenRouter defaults to its official [Free Models Router](https://openrouter.ai/docs/guides/routing/routers/free-router),
`openrouter/free`. Its available underlying model and response speed can vary.

Ollama is an optional local fallback, not bundled with CIEL. It must be running
at `OLLAMA_HOST` with the requested model installed (currently `llama3.2` by default).
No provider setup is needed for local clock, memory, or supported desktop actions.

## Everyday use

The terminal clears the visible screen on launch, shows a boxed status panel,
and keeps an editable `❯` composer. Typed replies
stay visible in chat, with room for explanations, lists, and copyable code. Ciel
also speaks typed replies; longer answers get a brief excerpt and a pointer to the
full text. Voice requests keep their brief spoken style. `/status` shows requested
models, reasoning mode,
and the last served model; `/help` lists controls. Plain and narrow terminals remain
supported.

Each new `ciel` or `ciel web` launch starts a fresh conversation. Saved personal
memories remain available, and earlier conversations stay archived without being
automatically replayed. During that launch, paired Telegram and desktop voice/text
share the current conversation; a running Telegram worker follows the new launch.

`ciel` opens a compact launch screen with the current listening mode, voice,
recognition device, and caption setting. A microphone being active does not imply
every model is ready: deferred voice preparation and unavailable recognition are
shown in the panel. `ciel start dev` keeps detailed diagnostics. Setup supports
arrow keys and Enter in interactive terminals, with numbered choices for plain
terminals and redirected input. Set `NO_COLOR=1` for uncolored output.

If the microphone cannot open, an interactive `ciel` session or `ciel web` stays
open for typed chat. The launch panel and `/status` report that the microphone is
unavailable; spoken replies can still work when enabled. Voice capture stays off
for that run. Reconnect the microphone or select another input in setup, then
restart for voice. Terminal mode with `--no-text-input` still exits on microphone
failure; the web interface has its own text input.

```bash
.venv/bin/ciel start                    # ambient conversation and live AI captions
.venv/bin/ciel start dev                # detailed diagnostics
.venv/bin/ciel start --no-ambient        # use wake-word mode for this run
.venv/bin/ciel start --no-show-ai-transcripts # hide live AI captions for this run
.venv/bin/ciel setup                    # change voice, keys, or audio devices
```

Use the command toolkit to inspect your setup:

| Command | Purpose |
| --- | --- |
| `ciel status` | Show the configured listening mode, voice, STT, providers, and paths |
| `ciel doctor` | Check local setup and show warnings, errors, and suggested fixes |
| `ciel devices` | List audio device indexes and defaults; filter with `--input` or `--output` |
| `ciel config` or `ciel config show` | Show effective settings, with API keys shown as configured or unconfigured |
| `ciel help [command]` | Show general help or help for one command |

Add `--json` to `status`, `doctor`, `devices`, or `config` for machine-readable output.
`status` describes configuration; it does not check a running CIEL process or live
provider health. `doctor` runs offline without downloads, model inference, or network
requests. It exits with code 1 for errors and code 0 for warnings alone.

While CIEL is running, type these commands in the terminal or desktop web input:

| In-session command | Purpose |
| --- | --- |
| `/status` | Show microphone, listening mode, voice, recognition readiness, and memory status |
| `/mute [on\|off]` | Toggle the microphone, or explicitly mute/unmute it; typed messages still work |
| `/unmute` | Unmute the microphone |
| `/mode [ambient\|wake]` | Show or change the listening mode |
| `/memory [status\|pause\|resume]` | Show or pause/resume saved-fact recall |
| `/stop` | Cancel the current recording and reply |
| `/help [command]` | Show controls or help for one command, such as `/help memory` |
| `/exit` or `/quit` | Shut down CIEL |

Microphone and listening-mode changes apply to the current session; restarting uses
your saved configuration. Memory recall pause/resume persists across restarts and
keeps saved facts, recent conversation, and learning active. These controls run
locally without an AI request. Prefix a message with `/fast`, `/strong`, or `/deep`
to select a provider route for that message.

Try:

- “Hey Ciel, what time is it?”
- “Ciel, how much RAM am I using?”
- “Ciel, can you open Discord for me?”
- “Ciel, my favorite game is CS2.” Then “Ciel, yes” to save it.
- “Ciel, what's my favorite game?”
- “Ciel, update my favorite game to Hades.” Then “Ciel, yes, update that.”
- “Ciel, what do you remember about me?”
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

For guided customization, run `ciel setup` and choose **Build a custom personality with AI**
at the personality step. Describe the character, tone, or reference text you have in
mind. The selected cloud provider uses your answers to ask up to three follow-up
questions, then shows a draft. Type `/done` to draft early or `/skip` to discard the
interview. You can request up to two revisions before accepting the personality.
Questions and answers are sent to that provider using your configured API key,
including a key just entered in setup; reference links are not fetched.

An accepted draft is saved to `CIEL_PERSONA_FILE` (default `persona.txt`) only when
you save the setup. It uses a separate managed block and preserves your own text and
voice-requested style block. Returning to **Customize** keeps an accepted draft until
you discard it. Skipping without accepting a draft, or canceling setup before saving,
leaves the personality file unchanged. Without a cloud key, setup skips the interview.
If writing the personality fails, setup reports the failure; saved settings remain
available and the old personality file is preserved.

### Internet access

Explicitly addressed, reliably transcribed clock, memory, and supported desktop
commands run through local handlers without an AI request. Other transcripts and
typed messages admitted to the conversation go directly to the main response model.
It answers or requests a native web, memory, or screen tool, then continues with
the real result. Ordinary chat has no separate AI decision request.
Inactive room speech
and clear speech to another person are filtered before the reply; stop and
mute controls remain immediate.

Native tools use normal conversation routing and existing model overrides.
NIM, Groq, OpenRouter, and Ollama adapters preserve structured tool requests and
results; tool arguments never enter speech. Invalid, unknown, repeated, or stale
requests are rejected. A provider cannot splice a fallback after publishing text
or a tool fragment. Existing local-only conversations still require Ollama.

For searches, CIEL gives a short spoken acknowledgment, retrieves pages using an
AI-written query, and reports the supported answer. The main model uses recent user
context and relevant saved memories to keep the subject when you say “try again.”
Brief corrections such as “the light novel” retain the original title and question.
The requested outcome is sent separately from search keywords, so an episode lookup
cannot replace a question about where to continue reading. When the title is known
but the season or episode is missing, CIEL asks for that detail instead of guessing it.
“Why can't you search?” asks for an explanation. If sources are insufficient, the
main model can refine the query or read another page within the bounded tool loop.
Requests allow three to five model rounds; repeated tool requests stop the loop.
Individual network/model requests also expire.
Telegram keeps receiving messages during research, so “stop” cancels the active request.
Local tools keep their existing allowlists and confirmation rules.

Current facts can trigger automatic searches. Say “search” for an explicit lookup;
research requests can use the larger retrieval budget. The terminal shows each query,
and the terminal and desktop page show source links. When you ask for a page URL,
CIEL points to the matching source link without reading the address aloud.
Unrelated results are discarded;
source previews alone do not establish an exact answer.
Web replies use a short answer with matching source excerpts, which the application
checks for quotation support and relevance to the actual task before speech. Invalid
output, unrelated claims, invented quotes, and unsupported numbers receive
an uncertainty reply or the supported subset with its gaps. News requests aim for 3–5
recent stories with individual article links and dated source metadata. Mongolia news
checks English and Mongolian reporting and replies in English. If a fuller summary
cannot be supported, CIEL can show the dated article headlines she actually retrieved.
Publication, modification, and provider-reported dates remain distinct. Exact
anime-to-novel reading points, including requests written as “lightnovel” or “LN,”
require a page or provider excerpt,
not only a search preview. Web replies wait for validation before playback; ordinary
conversation still streams when enabled.

Bare public domains such as `retro.mn` work without typing `https://`. Follow-ups such as
“tell me what you saw” or “read the second source” use retained source observations
across the currently paired voice/chat context, including after restart. Observations
remain separate from personal facts; a failed lookup cannot replay an older topic as
its evidence. A later unrelated conversation ends implicit research continuation.

Search defaults to [Parallel's free search service](https://docs.parallel.ai/integrations/mcp/search-mcp),
which supplies ranked excerpts without an account or API key. The anonymous service
has lower usage limits than paid access. Setup also offers the free `ddgs` package
(DuckDuckGo and public Brave backends), Brave Search, and Tavily. For the latter two,
choose the service first, then enter its masked API key. Only
the selected search provider is used; saving another key never enables automatic
fallback or spending that service's quota. Switching providers preserves saved keys.
Optional services may charge according to your account plan. Search connections are
not tested or charged during setup. The integrations use the official
[Brave web/news APIs](https://api-dashboard.search.brave.com/app/documentation/news-search)
and [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search).
Public search and sites can block or fail requests. When no useful sources are found, CIEL gives
a local verification-failure response and offers a retry or general background.
The reader supports public HTML, plain text, and news RSS/Atom links. Account actions, purchases,
file downloads, browser automation, and PDF/media extraction remain future work.

Edit `.env` and restart to change these controls:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CONTROLLER_ENABLED` | `true` | Use the main response model with native tools; `false` retains the legacy desktop intent path |
| `CONTROLLER_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Legacy decision/progress utility model; unused for main response tool selection |
| `CONTROLLER_TIMEOUT_SECONDS` | `8.0` | Web claim relevance check budget; legacy decision deadline |
| `CONTROLLER_MAX_TOOL_STEPS` | `3` | Main model rounds per request, clamped to 3–5 |
| `INTERNET_ENABLED` | `true` | Enable public search and page reading |
| `INTERNET_PROVIDER` | `parallel` | Use only this search service: `parallel`, `ddgs`, `brave`, or `tavily` |
| `BRAVE_SEARCH_API_KEY` / `TAVILY_API_KEY` | Empty | Credentials for the selected optional service |
| `INTERNET_AUTOMATIC` | `true` | Search current facts without an explicit search command |
| `INTERNET_HOME_CITY` | Empty | Default city and country for local questions |
| `INTERNET_TIMEOUT_SECONDS` / `INTERNET_RESEARCH_TIMEOUT_SECONDS` | `8.0` / `20.0` | Budgets for each individual lookup |
| `INTERNET_MAX_RESULTS` / `INTERNET_RESEARCH_MAX_RESULTS` | `5` / `8` | Search result limits |
| `INTERNET_MAX_PAGES` / `INTERNET_RESEARCH_MAX_PAGES` | `2` / `4` | Page reading limits; `0` uses snippets only |

A location in your request overrides the saved default. Set `INTERNET_HOME_CITY`
to your city and country, or tell CIEL the location when asked; a timezone does
not establish your home city. `INTERNET_ENABLED=false` disables internet retrieval.
Local commands remain available without a cloud provider. If every eligible
response provider fails, an admitted request receives a short failure message.
Background speech stays outside the conversation. Tool-assisted requests can use
additional model rounds; ordinary chat answers in one main request.

### Listening and performance settings

| Setting | Fresh setup value | Purpose |
| --- | --- | --- |
| `STT_MODEL` / `STT_DEVICE` | `small.en` / `cpu` | Command transcription; optional CUDA choice in setup |
| `STT_COMPUTE_TYPE` / `STT_BEAM_SIZE` | `int8` / `5` | Command decoding precision and beam width |
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

Run `ciel setup` and choose **NVIDIA GPU** to save CUDA recognition settings and
install its libraries. For manual configuration, install
`.venv/bin/python -m pip install -e ".[voice,gpu]"`, then set `STT_DEVICE=cuda` and
`STT_COMPUTE_TYPE=int8_float16`. The launcher prepares installed CUDA library paths.
Explicit `STT_DEVICE=cpu` with `STT_COMPUTE_TYPE=int8` selects CPU operation.
`small.en` is the model used
in the recorded GPU tests. Turbo has its own separate environment.

See [usage and tuning](docs/usage.md) for ambient follow-ups, strict mode,
interruption recovery, captions, memory behavior, and STT retry settings.

## Editable memory

CIEL keeps durable notes in `data/ciel.workspace/` beside the default
`data/ciel.db`. `MEMORY.md` holds curated memory, `USER.md` holds an optional
profile, and `memory/*.md` holds dated working notes and detailed archives.
`DREAMS.md` records reflection results and proposed changes for owner review.
Existing durable SQLite facts migrate once into managed Markdown blocks;
conversation archives and settings stay in SQLite.

New foreground launches start fresh conversations with bounded curated memory.
Ask `Search my memory for Atlas recovery` or `Read memory/atlas.md lines 10-30`
to retrieve real source excerpts; past conversations remain available through
same-owner memory lookup. Keyword search works without an embedding model.
Optional hybrid recall uses an accepted local Ollama model. Source edits and
deletions refresh the derived index without rereading unchanged note text.

While CIEL runs, a local 03:00 reflection checks source evidence, recall frequency,
query diversity, and relevance before proposing durable changes. The default
`confirm` mode queues additions for review; merges and conflicts require approval
in all modes. Automatic daily working-note capture has a separate setting.

```bash
ciel memory status
ciel memory search Atlas recovery
ciel memory get MEMORY.md --from 1 --lines 40
ciel memory review
ciel memory apply RUN_ID --action 0
ciel memory backup /path/to/new-snapshot.db
```

Keep both `new-snapshot.db` and `new-snapshot.workspace/` from a backup. See
[memory and personality](docs/memory-persona.md) for migration, recall controls,
nightly review, configuration, and forgetting limits.

## Data and privacy

API keys, `persona.txt`, model weights, recordings, databases, and generated
benchmarks and the default `data/` memory workspace are excluded from Git. Keep
custom workspace paths outside tracked source files. Accepted conversations
stay in `data/ciel.db` by default; durable Markdown, provenance, and private
preimages live beside it. Forgetting removes the selected fact, redacts known
matching wording from workspace notes and derived lifecycle data, and suppresses
it in provider context. Archived raw chats, prior backups, and unknown paraphrases
need separate cleanup.

Wake recognition, transcription, and speech synthesis run locally. Cloud replies
send conversation context and bounded curated/retrieved memory to the selected provider.
Web searches send query terms and relevant user context to public search services,
and page reads contact the requested public sites. Set `INTERNET_ENABLED=false`
to disable retrieval.
In ambient mode, short background excerpts stay temporarily in RAM. Ambiguous
follow-ups can go to the configured AI provider with bounded dialogue context to
choose a response or action. Clear inactive or third-party speech is filtered
before the action decision. There is no speaker identification or acoustic echo
cancellation. Diagnostic transcript logging can
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
├── memory/         # Markdown workspace, indexed recall, SQLite archives, lifecycle
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
