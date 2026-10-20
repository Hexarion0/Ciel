# CIEL development roadmap

Last reviewed: **2026-10-21**. Package version: **0.3.6 development**.

The goal is a reliable personal desktop assistant with natural voice conversation,
persistent memory, useful actions, and eventually remote messaging and presence.
Linux is the current target. Build one feature at a time and verify its behavior
before treating it as ready for everyday use.

## Current status

| Area | Implemented | Still pending |
| --- | --- | --- |
| Voice loop | Wake recognition, local STT, provider streaming, sentence playback, interruptions, captions | Live microphone/speaker acceptance of the current combined changes |
| Voices | Default CPU Pocket streaming; local Piper fallback; optional custom Turbo HQ; bundled worker and manifest | Reproducible custom-voice installation and private asset import |
| Providers | NIM, Groq, OpenRouter, Ollama, routing, retries/fallback, latency tracing | Usage totals, budget controls, broader live fallback verification |
| Controller | Local handlers for explicit commands without NIM; lightweight AI decisions for other turns; bounded retries and cancellation | Live microphone acceptance and broader decision-quality evaluation |
| Memory | SQLite conversations, shared voice/chat context, indexed older user dialogue, structured state, current facts/revisions, optional local semantic recall, forgetting | Live cross-channel and voice-based restart acceptance, summary quality evaluation |
| Actions | Trusted module discovery, validation, local telemetry, allowlisted Linux app launching | Model-selected calls, reminders, live desktop launch acceptance |
| Internet | Free automatic search, public HTML/text reading, bounded retrieval, relevant user context, source links | Live voice/text retrieval acceptance, broader site compatibility, PDF/media extraction |
| Personality | Modular persona prompt, continuity rules, opinions/initiative, editable preferences, voice-requested style updates/reset | Continued evaluation during real conversations |
| Setup | Configure/review/save before selected package/model installation; CPU recognition choice by default; preserved existing settings; missing-asset reporting | Automatic Turbo setup, managed desktop runtime, clean-machine validation |
| Desktop web interface | PC-local voice/text chat, live captions, status, and controls | Same-Wi-Fi phone access and desktop acceptance |
| Messaging | Owner-paired Telegram text, automatic startup, shared conversation with separate archives | Live Telegram acceptance, reminders, additional services |
| Tone, Windows, presence | Planned | Implementation |

A checked item below means the implementation exists. It does not establish
microphone accuracy, speaker quality, or a completed release. Historical evidence
is in [release checks](release-checks.md), [latency findings](conversation-latency.md),
and the [changelog](../CHANGELOG.md). The repository has a `v0.2` tag; no stable
`v0.3` tag is recorded yet.

## Next work, in order

The next feature priority is Telegram chat before reminders, alongside acceptance
of the current v0.3 desktop build.

1. **Accept Telegram text chat on the target PC.** Configure it through the chat
   service selector, pair the owner, check ordinary replies and confirmed memory
   changes, continue a voice conversation through chat, and test automatic startup,
   network recovery, unpairing, and restart. Keep Discord
   and Signal as later adapters. See [chat behavior and limits](telegram-chat.md).
2. **Validate the combined build on the target desktop.** Run wake, Pocket
   playback, barge-in, confirmed memory across restart, and Discord launch checks.
   Include provider and network outages, PC noise, and a sustained conversation.
   Repeat interrupted long turns and personal corrections: stale fragments must
   not delay typed input, and memory replies must accurately report whether a
   change was saved or still needs direct address and confirmation.
   Confirm the 0.9-second endpoint preserves normal pauses. Exercise web retrieval
   and source display through voice and text using [release checks](release-checks.md).
3. **Prove clean installation with real voice assets.** Test core installation,
   configure-first CPU/Piper and optional Pocket/CUDA selections, cancellation,
   failed-download retry, and restart outside the checkout. Import
   the private Turbo reference, install its separate environment and model, and
   verify missing-model and missing-GPU recovery. Keep Turbo optional.
4. **Add persistent reminders after chat acceptance.** Reuse the paired Telegram
   channel for delivery and shared memory/conversation components for confirmation.
   Start with relative times, list/cancel, and restart recovery before recurrence.

The current code handles explicit local commands without NIM, aligns Pocket
defaults across configuration and setup, bundles the Turbo worker and manifest,
and shares terminal/web session handling in `ciel.runtime`. Automated regressions
also cover canceled streams, nonblocking superseded speech, guarded memory
corrections, and one shared controller decision deadline. Microphone and speaker
acceptance remains open; these changes do not establish hosted-provider availability.

The broader milestones below remain the plan. New priority choices should update
this section rather than silently marking later milestones complete.

## Foundation and core voice — v0.1 / v0.2

- [x] Python `src/` package, CLI, centralized settings, secrets handling, logging.
- [x] Linux audio device discovery, recording, playback, and error handling.
- [x] Whisper keyword spotting and optional openWakeWord integration.
- [x] Local faster-whisper transcription, VAD, confidence checks, bounded retries.
- [x] NIM, Groq, OpenRouter, and Ollama provider clients.
- [x] Heuristic request routing and `/fast`, `/strong`, `/deep`, `/local` overrides.
- [x] Explicit local time/date, memory, and desktop commands without an AI decision.
- [x] Shared terminal/web session runtime outside CLI argument and startup handling.
- [x] Ambient listening and live AI captions by default; setup offers CPU recognition
      and explicit CUDA while preserving saved recognition settings.
- [x] Provider fallback before visible output; no splicing after a partial answer.
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
- [x] Shared character budgets and atomic summary commits across independent managers.
- [x] Editable personality file and managed style updates/reset by voice.
- [x] Modular personality and continuity prompt with bounded initiative and reasoned opinions.
- [x] Optional ambient follow-ups with intended-listener classification.
- [x] Linked interruption fragments and estimated playback context for continuation.
- [x] Keep internal interruption status out of spoken dialogue and provider reply examples.
- [ ] Teach a fact by voice, confirm it, restart, and recall it on the target desktop.
- [ ] Verify rejection and forgetting behavior through the actual microphone.
- [ ] Test the five-minute active ambient window and later context-gated follow-ups
      against speech addressed to other people.
- [ ] Finish [release acceptance](release-checks.md) before tagging stable v0.3.

Memory recall uses words and aliases by default. Optional semantic recall uses a
configured local Ollama embedding model and an explicit indexing step; it remains
disabled by default. Forgetting retains the chat archive. Ambient classification
estimates the intended listener without identifying speakers.

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
      larger research/compare/double-check budgets with session privacy through `/local`.
- [x] Configure an optional home city; explicit request locations take priority.
- [ ] Validate live search, page reading, location questions, and voice/text source display.
- [x] Put an AI decision before normal voice/text handlers, with spoken search
      acknowledgment, AI-written queries, bounded retries, and validated reports.
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

- [ ] Linux user service and optional autostart.
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
