# CIEL development roadmap

Last reviewed: **2026-10-30**. Package version: **0.3.6 development**.

The goal is a personal companion with natural voice conversation, persistent memory,
shared context across voice and chat, and useful awareness of what the user is doing.
Linux is the current target. Build one feature at a time and verify its behavior
before treating it as ready for everyday use.

## Current status

| Area | Implemented | Still pending |
| --- | --- | --- |
| Voice loop | Wake recognition, local STT, provider streaming, sentence playback, interruptions, captions, actionable microphone startup errors | Sustained microphone/speaker acceptance of the current combined changes |
| Voices | Default CPU Pocket streaming; local Piper fallback; optional custom Turbo HQ; bundled worker and manifest | Reproducible custom-voice installation and private asset import |
| Providers | NIM, Groq, OpenRouter, Ollama, provider-specific models, contextual routing, empty-reply failover, latency tracing | Usage totals, budget controls, broader live fallback verification |
| Controller | Exact local handlers, then native reads selected by the main responding model; shared task context, bounded tool rounds, and final synthesis; guarded legacy diagnostics | Broader real-conversation quality and outage evaluation |
| Memory | Markdown primary notes, migration/paired backups, bounded source retrieval, fresh-launch curated context, indexed same-owner archives, optional local hybrid recall, guarded nightly review/apply, current facts/revisions, forgetting | Live voice/chat restart acceptance, local embedding evaluation, overnight scheduler and review acceptance |
| Actions | Trusted module discovery, validation, local telemetry, allowlisted Linux app launching | Model-selected calls, reminders, live desktop launch acceptance |
| Internet | Free Parallel search; optional DDGS/Brave/Tavily; article/news/feed reading, dates, supported partial answers, shared source observations, full research goals, relevance checks, and a shared 30-second native research budget | Broader live news quality, site compatibility, PDF/media extraction |
| Personality | Modular persona prompt, continuity rules, opinions/initiative, editable preferences, voice-requested style updates/reset | Continued evaluation during real conversations |
| Setup | Configure/review/save before selected package/model installation; CPU recognition choice by default; preserved existing settings; missing-asset reporting | Automatic Turbo setup, managed desktop runtime, clean-machine validation |
| Desktop web interface | PC-local voice/text chat, live captions, status, and controls | Same-Wi-Fi phone access and desktop acceptance |
| Messaging | Owner-paired Telegram text, automatic startup, shared conversation with separate archives | Live Telegram acceptance, reminders, additional services |
| Screen awareness | On-request all-monitor/cursor capture, selected vision model, temporary shared images, Telegram screenshot output, opt-in desktop observation and repeated-error coding hints | Live image quality, pointer targeting, sustained observation load and hint quality, paired voice/Telegram acceptance, verified player pause signals |
| Tone, Windows, presence | Planned | Implementation |

A checked item below means the implementation exists. It does not establish
microphone accuracy, speaker quality, or a completed release. Historical evidence
is in [release checks](release-checks.md), [latency findings](conversation-latency.md),
and the [changelog](../CHANGELOG.md). The repository has a `v0.2` tag; no stable
`v0.3` tag is recorded yet.

## Agreed next upgrade — 2026-10-25

The [smarter CIEL design](smarter-ciel.md) adds a quality-first sequence before
broader computer control: representative conversation checks and a shared normal
research budget of 30 seconds, then optional temporary observation of both monitors
and useful initiative. Zen pause detection and sustained desktop acceptance remain
unverified. Anime comments wait for a pause; brief coding hints may be offered
while anime plays elsewhere after evidence of a repeated unresolved problem.

The default native conversation path now shares a bounded task projection across
voice and chat, handles retries/corrections, and reserves a final sourced answer
within its normal 30-second research budget. The [conversation/search evaluation](
conversation-search-evaluation.md) records automated, synthetic hosted-model, and
public-retrieval checks. The [optional observation coordinator](screen-initiative.md)
and coding hints are implemented. Sustained microphone/desktop acceptance and verified
Zen pause detection remain pending; legacy controller diagnostics retain their old bounds.

Screen capture remains on request unless observation is explicitly enabled for the
current desktop session. Anime comments, desktop edits, commands, clicking, and typing
remain deferred.

## Proposed feature order

This is the proposed direction after the conversation, memory, search, and routing
upgrades. [Screen awareness](screen-awareness.md) is implemented; its target-desktop
and live paired-chat acceptance remain open.
The version headings below group existing work and backlog, rather than promise dates.

| Order | Feature | First useful outcome |
| --- | --- | --- |
| 1 | Screen awareness | Discuss all monitors and the mouse pointer through requested captures, with fresh visual follow-ups across voice and chat |
| 2 | Image and document conversations | Send a screenshot through Telegram, discuss it by voice, then add text/PDF attachments |
| 3 | Useful action workflows | Connect requested research, explanations, and supported desktop actions through the existing validated action registry |
| 4 | Reminders and optional initiative | Persistent reminders with Telegram delivery, followed by quiet-hours-aware check-ins tied to user-approved topics |
| 5 | More chat services | Add Discord text using the same owner, conversation, and memory rules; assess Signal separately |
| 6 | Everyday desktop app | Managed installation, launcher, updates, recovery, and useful dashboard controls; Windows follows Linux acceptance |

Tone and camera presence remain later experiments. They are separate from screen
awareness and are not needed for the first screenshot feature.

For every feature, keep voice/chat continuity, immediate cancellation, and owner
boundaries working. Run the relevant [release checks](release-checks.md) alongside
development: real conversations, owner pairing/startup, restart recovery, provider
outages, and clean installation still need broader desktop acceptance. Existing
automated checks and small live probes do not complete that release acceptance.

## Screen awareness — implementation and acceptance

The [agreed design and implementation plan](screen-awareness.md) covers on-request
PC capture through voice, the desktop interface, and paired Telegram. Images sent
by the user belong to the next image/document milestone. The first version looks
and explains; continuous watching and automatic clicking or typing are out of scope.

- [x] Capture all monitors on request, including the mouse pointer and its position;
      use the focused monitor as the main context.
- [x] Route image requests only to a configured provider/model that supports images;
      use existing connections first and explain when no compatible model is available.
- [x] For “look at my screen” without a specific question, give a brief observation
      and ask a natural follow-up question.
- [x] Answer questions about visible errors, pages, code, or game menus; distinguish
      visible details from guesses, crop/enlarge small text, and ask if still unreadable.
- [x] Take fresh captures for clear visual questions, “what about now?”, and contextual
      next-step follow-ups; compare with the previous image when useful. Explicit
      questions about an earlier capture reuse it while available.
- [x] Share temporary captures and brief timestamped text observations across the
      paired voice/chat conversation. Keep only the last two captures for up to
      15 minutes; clear images on reset, owner change, and capture-service shutdown.
- [x] Reply with text in Telegram by default and send a screenshot only when requested.
- [x] Preserve stop/cancel and owner boundaries. Text inside an image is context,
      never permission to run a command or save a personal fact.
- [ ] Verify actual capture quality and cursor targeting across all monitors, then
      test fresh comparison and paired voice/Telegram handoff on the target desktop.

The first milestone is complete when CIEL can explain an actual all-monitor capture
including the pointer, take a fresh follow-up, compare a visible change, and continue
across paired voice/chat without inventing unseen screen contents.

## Foundation and core voice — v0.1 / v0.2

- [x] Python `src/` package, CLI, centralized settings, secrets handling, logging.
- [x] Linux audio device discovery, recording, playback, and error handling.
- [x] Whisper keyword spotting and optional openWakeWord integration.
- [x] Local faster-whisper transcription, VAD, confidence checks, bounded retries.
- [x] NIM, Groq, OpenRouter, and Ollama provider clients.
- [x] Heuristic request routing and `/fast`, `/strong`, `/deep` overrides.
- [x] Explicit local time/date, memory, and desktop commands without an AI decision.
- [x] Shared terminal/web session runtime outside CLI argument and startup handling.
- [x] Ambient listening and live AI captions by default; setup offers CPU recognition
      and explicit CUDA while preserving saved recognition settings.
- [x] Provider fallback before visible output; no splicing after a partial answer.
- [x] Use provider-specific model defaults and bounded recent context for referring
      work follow-ups; reject empty replies and preserve actual returned model metadata.
- [x] Piper speech, persistent Turbo worker, bounded sentence queues and interruption.
- [x] Contextual vocal cues, inline placement, giggle/moan aliases, and adjustable cue
      frequency; plain captions/history and Piper fallback, with regression coverage.
- [x] Playback captions; native Piper timing with duration estimates when unavailable.
- [x] Foreground voice requests take priority over background summaries.
- [x] Concise normal console and explicit development diagnostics.
- [x] Keyboard messages alongside voice, with microphone mute, reply cancellation,
      and clean shutdown commands.
- [x] Stage-level latency instrumentation and recorded model comparisons.
- [ ] Repeat live acceptance on the current combined build, including PC noise and pauses.
- [ ] Record sustained-session behavior and real provider-failure recovery.
- [ ] Track request/token totals and configurable provider budget limits.

Keep provider costs configurable; hosted free tiers and model availability can
change. Ollama requires a separately installed server/model and is not guaranteed
to be available just because the client is configured.

## Memory and conversation — v0.3, in development

- [x] SQLite conversations, durable facts, metadata, and keyed revisions.
- [x] Bounded short-term context and incremental background summaries across restarts.
- [x] Explicit `remember ...` and targeted `forget ...` commands.
- [x] Confirmation before conversational facts or corrections are saved.
- [x] Proposal cancellation on rejection, unrelated requests, expiry, or restart.
- [x] Ranked word/alias recall and local answers for supported saved facts.
- [x] Confirmed current project, goal, and occupation with source attribution and short corrections.
- [x] Resolve short recall references from recent user dialogue with a total context budget.
- [x] Validated summary state retaining decisions, unfinished threads, and superseded plans.
- [x] Suppress known forgotten wording from model context; retain archived chat separately.
- [x] Automated restart, correction, and memory isolation tests.
- [x] Indexed recall across the complete durable collection, Unicode aliases, and duplicate suppression.
- [x] Configurable automatic/confirmed/explicit learning and confirmation before conflicting replacements.
- [x] Persistent pause/resume for saved-fact recall while retaining conversation continuity.
- [x] Project decision/constraint journals, current next steps, and owner inspection/backup commands.
- [x] Owner-editable Markdown source of truth with sibling-workspace migration and paired backups.
- [x] Curated `MEMORY.md`/`USER.md` bootstrap; detailed notes and past same-owner launches on demand.
- [x] Bounded `memory_search`/`memory_get` with actual source paths, line ranges, and privacy checks.
- [x] Rebuildable FTS5 workspace index, fresh edit/delete reconciliation, and bounded text/hash caches.
- [x] Keyword access during optional local hybrid inference, without implicit model downloads.
- [x] Authenticated daily checkpoints before compaction/close, with private-context exclusions.
- [x] Runtime 03:00 reflection with recall gates, tool-free evidence plans, and `DREAMS.md` review.
- [x] Explicit owner apply for merges/conflicts, guarded preimages, and stale-owner/reset rejection.
- [x] Shared character budgets and atomic summary commits across independent managers.
- [x] Editable personality file and managed style updates/reset by voice.
- [x] Modular personality and continuity prompt with bounded initiative and reasoned opinions.
- [x] Ambient follow-ups with application admission; accepted turns receive a reply
      or clarification instead of a model-selected silent response.
- [x] Linked interruption fragments and estimated playback context for continuation.
- [x] Keep internal interruption status out of spoken dialogue and provider reply examples.
- [ ] Teach a fact by voice, confirm it, restart, and recall it on the target desktop.
- [ ] Verify rejection and forgetting behavior through the actual microphone.
- [ ] Verify a live scheduled reflection, owner review/apply, and paired backup/restore on Linux.
- [ ] Evaluate an installed local embedding model before enabling optional semantic/hybrid recall.
- [ ] Test active follow-ups and filtering of speech addressed to other people after
      removing model-selected silence; check the earlier optional intent path separately.
- [ ] Finish [release acceptance](release-checks.md) before tagging stable v0.3.

Memory recall uses local keyword indexes by default. Optional semantic/hybrid
recall requires an accepted local Ollama embedding model and remains disabled by
default. The structured fact vector index has an explicit indexing command;
workspace search refreshes its own derived vectors in the background. Fresh launches
retain curated memory without replaying old conversation. Reflection runs while a
CIEL process is alive and applies the configured learning/review gates. Forgetting
retains raw chat archives and prior backups. Ambient classification estimates the
intended listener without identifying speakers.

## Installation and custom voice

- [x] Setup menu: CPU Pocket, Amy/Piper, or custom Turbo HQ voice.
- [x] Preserve existing settings and selected audio devices during setup.
- [x] Lightweight core install, followed by review/save and selected speech packages/models.
- [x] CPU recognition offered by default; CUDA libraries, Pocket, and openWakeWord only
      installed when selected/configured; preserve existing explicit choices.
- [x] Rerun setup to install missing selected components and retry failed downloads
      with saved settings.
- [x] Report missing custom reference, model files, or inference environment.
- [x] Download the exact Turbo model revision expected by the runtime.
- [x] Keep private references, recordings, weights, environments, and databases out of Git.
- [ ] Create the separate Python 3.12 Turbo environment automatically when selected.
- [ ] Add private reference import/backup with validation and actionable recovery steps.
- [x] Align Pocket fresh-install defaults across README, configuration, and setup.
- [ ] Validate a fresh machine with Pocket, Piper fallback, then optional Turbo HQ.
- [x] Bundle the Turbo worker and default manifest; resolve installed voice assets
      under the user's data directory, with absolute path overrides.
- [ ] Reduce optional inference dependencies after measuring what the worker actually needs.
- [ ] Build a desktop installer that bundles or automatically manages Python and
      dependencies, so users do not create or activate a virtual environment.
- [ ] Add a desktop launcher that works outside the source checkout and starts
      the required voice worker automatically.
- [ ] Manage model downloads and voice selection during first launch; keep the
      heavier custom voice optional and import its private reference separately.
- [ ] Validate the packaged app on a clean machine without a preinstalled Python
      environment, including Amy, custom voice requirements, and missing-GPU recovery.
- [ ] Verify packaged-app upgrades preserve configuration, memory, and private voice assets.

The existing custom voice needs its original reference audio/transcript. A Git clone
contains the manifest, not those assets. The heavier Turbo installation should remain
optional. See [custom voice requirements](chatterbox-turbo-integration.md).

Fresh setup selects Pocket and offers CPU recognition. `setup.sh` installs the core
first; the wizard saves reviewed choices, installs their required extras and models,
then tests voice playback. It preserves saved settings and installs newly needed
components when rerun. Verify the selected voice after restart with the target
desktop's installed models and audio devices. Clean-machine and live audio checks
remain open.

The current source installation still uses `.venv` and the separate Turbo
environment. The desktop installer should manage these runtime needs internally;
developers can keep virtual environments for development. Packaging simplifies
setup but does not remove the storage required for dependencies and models.

## Skills and actions — v0.4

- [x] Discover trusted installed modules exporting `ACTION` without editing the dispatcher.
- [x] Validate action names and arguments; reject ambiguous requests.
- [x] Enforce direct-address authorization for ambient side effects.
- [x] Support confirmation tied to exact arguments, expiry, and cancellation.
- [x] Bound blocking work with cooperative deadlines and report failures locally.
- [x] Read RAM usage, GPU temperature, and logical CPU core count.
- [x] Launch Discord/Vesktop, Firefox, Chromium, Steam, and VS Code/VSCodium by allowlist.
- [x] Recognize polite app requests, exact spelling corrections, and short-lived app references.
- [ ] Observe actual window opening and missing-app behavior on the target desktop.
- [ ] Accept model-selected calls through the same validated action registry.
- [ ] Return tool results to the provider without treating generated text as execution proof.
- [ ] Add relative reminders, persistence, listing, cancellation, and restart recovery.
- [ ] Add timezone-aware absolute reminders and recurring reminders.
- [x] Add application-side web search, source links, bounded public page reads,
      timeouts, and honest failure handling.
- [x] Search current facts by default, use relevant user memory/context, and support
      larger research/compare/double-check budgets.
- [x] Configure an optional home city; explicit request locations take priority.
- [x] Free DuckDuckGo/public Brave retrieval through DDGS, with optional Brave/Tavily
      API selection before credentials; use only the selected search provider.
- [x] News/article discovery, source date provenance, supported partial findings,
      and research that stops when useful leads are exhausted, without an overall timer.
- [x] Retain bounded source observations across paired voice/chat sessions and restart,
      separately from personal facts; failed lookups cannot revive an older result.
- [ ] Validate live search, page reading, location questions, and voice/text source display.
- [x] AI decisions for voice actions and a shared voice/text research path, with
      spoken acknowledgment, AI-written queries, bounded requests, and validated reports.
- [x] Configured-provider decision fallback and one malformed-JSON repair within the
      original deadline; the action model cannot silence an admitted user turn.
- [x] Keep private-session routing, immediate stop/mute, action allowlists, and
      explicit memory confirmation enforced outside the model.
- [ ] Add PDF/media extraction and broader public-site compatibility.
- [ ] Test end-to-end action chaining only after single actions are reliable.

No destructive action ships today. Test explicit confirmation end-to-end before
adding one. Action modules are trusted Python code, not sandboxed third-party plugins.
The interface and existing behavior are documented in [actions.md](actions.md).
Public web retrieval runs before answer generation in the application. It does
not use model-selected action calls or grant website account access.

## Dashboard — v0.5

- [x] Add a PC-local HTTP server and live status polling (`ciel web`).
- [x] Share the desktop microphone, speakers, conversation worker, memory, and controls.
- [x] Display voice/text chat, current-session history, and live speech captions.
- [ ] Show idle/listening/thinking/speaking state, selected provider/model, and errors.
- [ ] Display conversation history, routing decisions, and measured response stages.
- [ ] Add useful settings controls and audio visualization.
- [ ] Review access controls before exposing the dashboard beyond localhost.
- [ ] Complete UI and accessibility checks against real desktop use.

The first desktop interface uses Python's HTTP server and a small HTML/CSS/JS
frontend. PC voice input stays in the existing listener. Add same-Wi-Fi phone
access after desktop acceptance. Telegram now runs independently using shared
persona, memory, and provider components. Reassess FastAPI/WebSocket if later
dashboard requirements need them.

## Messaging — v0.6

- [x] Telegram text request/reply without audio initialization.
- [x] Chat-service selection before credentials, with automatic Linux login and setup startup.
- [x] Expiring owner pairing, private-chat restrictions, duplicate claims, and bounded retries.
- [x] Shared durable facts, recent dialogue, historical recall, and summaries across channels.
- [x] Preserve channel archives and keep pending confirmations channel-specific.
- [ ] Accept live Telegram delivery, login startup, and provider/network recovery.
- [ ] Telegram voice input and optional spoken output.
- [ ] Discord text request/reply, permissions, and channel restrictions.
- [ ] Signal text support, subject to a maintained integration path.
- [ ] Broader rate limits, configurable request budgets, and multi-adapter loop prevention.
- [ ] Consider Discord voice after text messaging is stable.
- [ ] Reassess WhatsApp as an optional extension, based on maintenance cost.

Opening the Discord desktop app is already implemented; a Discord messaging bot is not.

## Voice tone awareness — v0.7

- [ ] Extract pitch, energy, speech rate, pauses, and volume.
- [ ] Estimate tone with confidence rather than assigning a definitive emotional state.
- [ ] Supply useful estimates to conversation generation without overreacting.
- [ ] Evaluate across speakers, microphones, noise, sarcasm, and gaming audio.

Turbo's contextual vocal event markers, adjustable cue frequency, and conversational
personality are existing features; they are not an acoustic mood classifier. The moan
alias renders a native groan, not a separately trained moan. Live listening still needs
to judge the sound and timing of effects in real conversations.

## Desktop integration and Windows — v0.8

- [x] Automatic Linux login service for configured owner-paired Telegram chat.
- [ ] Managed startup and recovery for the full desktop voice runtime.
- [ ] Desktop notifications through a platform abstraction.
- [ ] Manual activation hotkey for noisy rooms or missed wake phrases.
- [ ] Windows audio, notifications, startup, launching, and hotkeys.
- [ ] Validate Windows parity for voice, memory, actions, and later integrations.

## Presence and proactive behavior — v0.9

- [ ] Optional webcam presence/motion detection.
- [ ] Event → trigger → condition → action execution.
- [ ] A bounded welcome-back greeting after room entry.
- [ ] Presence-triggered reminders.
- [ ] Quiet hours, Do Not Disturb, and independent camera/voice notification controls.
- [ ] Confidence thresholds and cooldowns tested against shadows, pets, screens, and noise.

## Integration and release — v1.0

- [ ] Exercise the combined voice, memory, actions, messaging, tone, and presence system.
- [ ] Repeat supported-platform acceptance and long-session failure recovery.
- [ ] Verify clean installation, upgrade, private-data backup, and uninstall behavior.
- [ ] Audit unused configuration/dependencies and misleading capability claims.
- [ ] Update user docs, changelog, and a demo with actually observed behavior.
- [ ] Tag releases only after their automated checks and live acceptance are complete.

## Working rules

- Keep commits focused and describe behavior and validation, not just file changes.
- Work on and push to `main` for this repository unless the user requests a branch.
  Fetch first and preserve remote changes; never force-push to resolve divergence.
- Keep models, API keys, recordings, transcripts, and conversation databases local.
- Record benchmark conditions and dates. Old measurements are evidence about that
  run, not guarantees for current models, hardware, or configuration.
- Keep automated tests distinct from real microphone, GPU, speaker, and desktop checks.
- Update this checklist when functionality changes; preserve detailed historical
  results in release notes rather than accumulating contradictory current claims.

## Local web interface upgrade

Persistent chat, archive browsing/continuation, the retained orb, browser voice, local
HTTPS pairing, and temporary image questions are implemented. Automated and synthetic
checks are documented in [web interface](web-interface.md). Final real-phone HTTPS,
microphone/speaker, mobile keyboard, and interruption acceptance remains pending.
