# Changelog

## Unreleased — 0.3.6 development

- Automatically start selected chat services after setup/pairing, on Ciel launch,
  and at login. Remove manual-start choices and coordinate pairing with the worker.
- Bridge recent voice/chat conversation and summaries for the paired owner while
  preserving archives, channel-specific confirmations, and local-only routing.
  Keep a larger bounded unsummarized tail and index older user dialogue by topic,
  excluding forgotten and superseded values from historical recall.
- Improve corrected search scope, tracking-link deduplication, source relevance,
  and citation subject checks. Carry explicit browsing controls across channels.
- Strengthen Ciel's companion voice with reasoned opinions and conversational
  follow-ups. Handle wait/cancel/no-search before model decisions, discard canceled
  fragments, and keep lookups paused until the user explicitly requests another.
- Add optional owner-paired Telegram text chat with service-first setup, separate
  history, shared persona and saved facts, guarded memory corrections, durable
  duplicate suppression, and optional Linux user-service login startup. Keep chat
  independent of audio models; reminders and additional messaging adapters remain planned.
- Launch ambient conversation and live AI captions by default. Align command STT
  across configuration, setup, and installation with small.en on CUDA/int8_float16,
  beam 5, and a guarded medium.en retry. Preserve explicit overrides, CPU recovery,
  the separate CPU wake recognizer, and one-run listening/caption opt-outs.
- Use the measured working Nemotron 3 Super controller route instead of the
  stalled Lightning endpoint. Keep the eight-second decision deadline and
  explicit model overrides; require valid model responses in live diagnostics
  so local safety handling cannot hide provider failures.
- Keep superseded speech handling off the provider path, clear stale fragments
  when typing, and share one controller deadline across follow-up checks. Route
  supported personal corrections locally and explain when an unaddressed memory
  change has not been saved; preserve explicit authorization and exact values.
- Honor stream cancellation independently of playback state, keep installed wake
  assets in writable user storage, hide saved keys in setup prompts, and preserve
  legacy voice-only Ciel configuration consistently with setup.
- Reject aliased archive members and symlinked partial downloads in optional
  voice-training tools. Require explicit Kaggle account selection and keep
  diagnostic downloads under private local data.
- Keep explicitly addressed local clock, memory, and supported desktop commands
  available without NIM; retain authorization, cancellation, and private routing.
  Share terminal/web session handling in a runtime module outside CLI startup.
- Align fresh configuration and setup with the Pocket CPU default while preserving
  existing voices. Bundle the Turbo worker and default manifest in wheels, with
  writable per-user voice paths and explicit asset overrides.
- Extend script lint and shell checks, verify bundled wheel resources in CI, and
  remove machine-specific repository/account defaults from the Kaggle runner.
- Advance memory with full-collection indexed recall, Unicode matching, current-fact
  deduplication, bounded conversation replay, and atomic background summary commits.
  Add selectable automatic learning, confirmed replacements, persistent recall
  pause/resume, project journals, inspection, revision history, and private backups.
  Prevent forgotten wording from leaking through speech gates and local fact reads.
- Rename the project to Ciel, including the `ciel` package and command, `CIEL_*`
  settings, desktop branding, custom voice profile, and default “Hey Ciel” wake phrase.
- Fix faint sigh/groan output using optional separate effect-style conditioning,
  retained speaker identity, cached cues, and bounded level balancing. Restore
  normal conditionals after each effect, including failures; reject inaudible cues.
- Add expressive spoken phrasing and contextual Turbo vocal cues with configurable
  expressive/natural/off frequency, inline placement, and bounded streamed replies.
  Map giggle/moan aliases to native chuckle/groan; keep cues out of Piper fallback,
  captions, and newly stored batch dialogue. Preserve native sampling defaults.
- Add confirmed current-project, goal, and occupation memory with local recall,
  short corrections, source attribution, and targeted forgetting. Use recent user
  dialogue to resolve short references during bounded memory recall.
- Store validated conversation state with user context, decisions, open threads,
  and superseded plans; retain plain-summary compatibility and retry invalid state.
- Compose a shorter persona prompt around continuity, natural reactions, opinions,
  and bounded initiative while preserving custom preferences and existing sessions.
- Mask known forgotten details in other recalled notes as well as chat history.
- Enforce `/local` through reply fallback, running summaries, and ambient speech
  classification, retaining the policy across summary persistence and restart.
- Preserve unrelated facts when forgetting an absent topic, and keep old forgotten
  values suppressed when a different value is saved under the same key.
- Fix microphone stream replacement deadlocks and clean up failed stream starts.
  Require detected wake evidence to interrupt playback in wake mode.
- Expire pending persona edits and clear them on cancellation, farewell, mode
  changes, or unrelated requests. Expose `WAKE_MODELS` and print activation steps
  after custom wake training. Retry failed summaries without skipping dialogue.
- Bound STT startup and Turbo response waits, validate audio configuration earlier,
  hide API key entry, and restrict setup configuration files to owner access.
- Add regression coverage and CI for Python 3.10/3.14, frontend tests, lint,
  packaging, and an installed CLI smoke check.
- Add breathing, orbiting particles, flowing ribbons, and distinct voice-state
  animations to the orb. Cache geometry calculations, batch canvas draws, cap
  high-DPI resolution, and reduce frame rates while waiting and speaking.
- Buffer TTS playback in 1,024-frame blocks with configurable
  `TTS_PLAYBACK_LATENCY=0.12` to tolerate scheduling delays, and report underruns.
- Add `ciel web`, a PC-local desktop page using the existing microphone,
  speakers, conversation worker, memory, and actions. Display current speech,
  status, and mute/stop/shutdown controls without new dependencies.
- Replace the web conversation feed with a voice orb, a temporary current-reply
  speech bubble, and a message/command input. Animate the orb from runtime states;
  respect reduced motion and pause animation when the page is hidden. Show the
  bubble above a layered 3D surface mesh with purple/cyan bands and a luminous core.
- Polish the orb page with stable speech placement, readable replies with manual
  dismissal, larger controls, accessible completed-reply announcements, offline
  drafting, and request deadlines that recover from stalled connections.
- Add keyboard messages alongside voice listening, with `/mute` for microphone
  input, `/stop` for cancellation, `/exit` for shutdown, and `/help` for controls.
  Typed messages share the conversation worker, memory, and validated actions.
- Keep the owner-name keyboard prompt and its draft separate from background
  logs and progressive speech captions; restore terminal input mode on exit.
- Use plainspoken companion defaults, brief greetings and corrections, and
  direct honesty. Start a new conversation context so earlier verbose replies
  are not reused as style examples; preserve archived chats and saved facts.
- Keep archived playback-status notes out of assistant dialogue sent to providers,
  and filter echoed status markers from streamed/batch replies and TTS so they
  cannot be spoken as an answer.
- Add trusted local actions for system information and allowlisted Linux app
  launching, including polite requests and recent spelling corrections.
- Confirm conversational memory proposals before saving; retain explicit
  remember/forget commands and persistent recall.
- Add voice-requested personality preferences, contextual ambient follow-ups,
  and concise/development console modes.
- Integrate the persistent custom Chatterbox Turbo voice with Piper fallback;
  remove retired voice experiments and training tools.
- Offer custom CIEL or default Amy in setup. Check missing custom assets,
  preserve existing settings, and use CPU/Amy for the fresh-install example.
- Pin Turbo downloads to the revision accepted by the worker, refreshing an
  incompatible cached inventory. Keep the custom reference assets private.
- Retain the later measured Nemotron 3 Super routing choice documented in
  [latency findings](docs/conversation-latency.md); earlier model changes below
  describe historical development states.
- Refresh installation instructions, usage documentation, and the roadmap to
  separate implemented behavior from pending live acceptance and future work.

## 0.3.6 — 2026-10-04 (development)

- Reveal AI captions letter by letter during playback, using native Piper audio
  alignments for word spans and interpolating letters within those spans. Include
  the output device's reported latency, and freeze partial captions on barge-in.
- Coordinate caption redraws with console logs. Preserve terminal width, secret
  masking, normal log formatting, and plain final snapshots for redirected output.
- Enable Piper's alignment dependency and patch voices in memory. Preserve model
  files and public synthesis results; unsupported timing uses a duration estimate.
- Keep complete replies out of INFO logs ahead of live captions, and display a
  plain text fallback when speech fails before any visible playback progress.

## 0.3.5 — 2026-10-04 (development)

- Add live AI speech transcripts at playback start, showing each streaming
  sentence while it plays. Include local replies and batch TTS; omit speech
  canceled before playback and clean Markdown using the same speech rules.
- Add `SHOW_AI_TRANSCRIPTS` and `--show-ai-transcripts` /
  `--no-show-ai-transcripts` overrides, independent of raw microphone diagnostics.
  Preserve the final response log and one assistant turn in persistent history.

## 0.3.4 — 2026-10-04 (development)

- Use active conversation state for ordinary follow-ups and topic changes. Valid
  uncertain intent favors continuing a recent exchange; clear evidence of another
  listener closes it. New, expired or reset sessions still require an address.
- Add `AMBIENT_FOLLOWUP_POLICY=conversation|strict`; conversation is the default.
  Strict retains high-confidence follow-up classification. Failed or malformed
  judgments remain silent, and inferred turns do not authorize memory writes.
- Classify assistant/other/uncertain listeners with recent exchange context;
  expose intent confidence and the selected policy in logs.
- Preserve structured NIM JSON before conversational cleanup so interpretation
  strings containing “Direct response:” or reasoning-tag literals stay intact.

## 0.3.3 — 2026-10-04 (development)

- Accept natural descriptions of recent spoken delivery, including “you sound
  like a robot” and brief conversational tags such as “do you know?” or “right”.
  Quoted speech, other listeners and unrelated appended instructions still need
  the ambient intent decision.
- Recover bounded clock questions ending in CIEL's name when STT drops “it”,
  such as “What time is Ciel?”. Use the local clock and retain the original
  wording in history. Inferred address does not authorize memory writes.
- Answer normal trailing-name clock questions locally as well, avoiding a cloud
  request for “What's the time Ciel?”.

## 0.3.2 — 2026-10-04 (development)

- Accept recent feedback about spoken pace and delivery without a cloud speech
  gate or another wake phrase. Keep the original conversation window after an
  uncertain fragment; confirmed speech to someone else still ends it.
- Include speech feedback and answers to the assistant's last question in the
  follow-up classifier instructions.
- Add a Linux `gpu` installation extra for CUDA 12 cuBLAS and cuDNN 9, so command
  STT does not depend on a separate training environment's libraries.
- Run actual CUDA inference during background model loading before reporting
  STT ready; missing lazy-loaded libraries trigger the startup CPU fallback.

## 0.3.1 — 2026-10-03 (development)

- Replace the retired default NIM model with Nemotron 3.5 Lightning and keep
  thinking disabled for conversational replies.
- Try the configured NIM backup immediately on HTTP 404/410 in both batch and
  streaming modes. Preserve transient retries and cancellation without restarting
  a partially emitted answer. Handle usage-only and terminal SSE events.
- Resume an unanswered, directly linked question when a false interruption
  produces empty STT output. Keep original history once, reject expired or
  unrelated requests, and avoid replaying memory writes or partial playback.

## 0.3.0 — 2026-10-03 (development)

- Speak completed sentences while provider generation continues. Filter reasoning
  across token boundaries and retain code in history without reading it aloud.
- Cancel network reads, queued speech, and pending synthesis on interruption.
  Preserve completed sentences and estimated partial playback for continuation.
- Keep interrupted questions and additions together before the ambient reply
  decision. Record original user fragments separately and one assistant response.
- Budget additional CUDA STT retry models against available VRAM; reuse the
  primary model under training load and release temporary CUDA retry models.
- Retain SQLite facts, corrections, forgetting, and restart recall. Batch older
  conversation summaries in the background using economical routing.
- Select ambient listening, transcript diagnostics, and headphone speech barge-in
  through `.env`. The GPU launcher works without flags.
- Align package metadata, runtime version, and startup banner at 0.3.0.

Live microphone, training-load, and restart acceptance remain pending in
[`docs/release-checks.md`](docs/release-checks.md). A stable v0.3 tag has not been created.
