# v0.3 Release Checks

Run from the repository root with the development environment active.

## Repository review before commit — 2026-10-28

- Reviewed pending native response/provider/search, memory, desktop/audio/chat/screen,
  terminal, configuration, evaluation script, and documentation changes.
- Fixed archived recall privacy across paired channel projections, including transitive
  membership before any summary exists. Local-only recall and owner/reset boundaries
  retain regression coverage with temporary synthetic databases.
- Native screen calls require an observation request in the current user wording.
  Capability questions, acknowledgements, and discussion of a prior view cannot authorize
  fresh pixels, even if the model requests the screen tool. Explicit rechecks still work.
- Strict ambient mode checks inferred speech before persistence or tools. Default
  conversation mode and explicit wake/keyboard messages retain their normal response path.
- Translation checks cancel their own provider work at timeout without canceling the
  owner's request or accepting a late result. All four findings have regression tests
  that reproduced the failure before the corresponding fix.
- Lint, Python 3.10 syntax checks, dependency consistency, shell syntax, isolated wheel
  build, and installed-package CLI/resource/behavior checks passed. Credential-pattern
  and pending-file scans found no credentials, recordings, databases, or model artifacts.
  Final ordinary suite: **4,139 passed, 5 hardware tests deselected**, 97.41 seconds.
  Local diagnostics: `data/diagnostics/final-review/`. Hardware acceptance remains separate.

## Spoken desktop typed replies — 2026-10-28

- Desktop typed messages, spelling corrections, local replies, and search results
  now use speech as well as their full written display. Native and legacy response
  paths both retain the text response budget and code in chat. Search work receives
  one audible acknowledgement across repeated lookup steps.
- A deterministic excerpt keeps typed speech within 80 words, including its pointer
  to the full answer. It filters internal notes, reasoning, code, and citation markers
  without generating new factual claims or making another model request. Code-only
  replies receive a short spoken acknowledgement.
- Focused checks cover native and legacy desktop delivery, verified searches,
  screen replies, memory proposals, cancellation, playback failure, and missing
  microphones. The excerpt helper passed **28 tests** without model or audio access.
  The final ordinary suite passed **4,101 tests**, with **5 hardware tests deselected**,
  in 92.33 seconds.
- Disabled speech and canceled turns stay silent. Playback exceptions keep the full
  typed answer visible once without generating another answer. Paired headless chat
  workers retain text delivery. Ruff, whitespace checks, isolated wheel build, and
  packaged excerpt checks passed. Local diagnostics: `data/diagnostics/typed-speech/`.
- Manual acceptance: restart `ciel`, type a short question, type a search and a
  spelling correction, then request a long explanation with code. Expect audible
  replies and one full written answer; code stays in chat. Check `/stop` during
  playback and typed replies with the microphone muted or unavailable. These automated
  checks mock audio hardware and do not establish speaker acceptance.

## Episode-to-reading follow-up regression — 2026-10-28

- Replayed latest episode → novel continuation using the supplied episode title.
  Desktop voice, desktop typing, and headless chat send the new reading outcome to
  retrieval, carrying the series/season without concatenating both full requests.
  Contextual “light node” interpretation affects query data only, not raw transcripts.
  A canceled lookup or independently named topic does not inherit the series.
- Unverified mapping replies skip the episode question when the user supplies an
  episode number or title. Requests without an episode retain that clarification.
  Related plot summaries and chapter mentions remain insufficient mapping evidence.
- Focused reading, episode, chat, legacy runtime, and goal checks: **87 passed**.
  Full ordinary suite: **4,067 passed, 5 hardware tests deselected**, 93.26 seconds.
  Ruff, whitespace checks, isolated wheel build, and packaged behavior checks passed.
- Two real Parallel/model probes sent the corrected continuation goal in one lookup
  each. Neither established a supported reading point. The second returned the
  uncertainty reply in 14.104 seconds without asking for the title again. These are
  public synthetic-history probes, not microphone acceptance or proof of broad
  retrieval quality. Diagnostics: `data/diagnostics/reading-followup/`.

## Latest-episode search regression — 2026-10-28

- Replayed the reported latest-Tensura-episode → “Why?” → spoken season-four
  correction. Ciel retains the title and requested outcome, discards the abandoned
  scope words in its task projection, and preserves the raw transcript. Desktop
  voice and headless chat regressions cover the corrected request.
- Ciel explains recognized application verification failures without another model
  or web call. It does not infer that the retrieved pages lacked the answer.
- Initial queries cannot introduce a season absent from the owner's task. Latest
  episode checks reject unrelated premiere/dub announcements. Reading-point fallback
  requires a reading request and respects manga versus novel wording.
- One bounded report repair can correct invalid JSON/quotation output. Source checks
  recognize hash-numbered rows under an Episode table header, date abbreviations,
  and wording such as “24th and final episode.” Fabricated numbers, missing source
  support, canceled work, and repair loops remain rejected.
- Full ordinary suite: **4,051 passed, 5 hardware tests deselected**, 96.39 seconds.
  Final focused checks passed **127 tests**. Ruff, whitespace checks, isolated wheel
  build, and packaged behavior smoke checks passed.
- Public Parallel/model reproduction returned a cited latest-listed-episode answer
  in 10.110 seconds. The follow-up replay explained “Why?” in 0.006 seconds with no
  lookup, then retained season four and returned cited partial findings after one
  lookup in 16.160 seconds. Partial findings retained an uncertainty note.
  These are sampled public searches, not microphone or provider-availability guarantees.
  Local diagnostics: `data/diagnostics/tensura-season-repair/`.

## Task continuity and useful search answers — 2026-10-28

- Full ordinary suite: **4,026 passed, 5 hardware integration tests deselected**
  in 90.33 seconds. Focused task/search checks: **177 passed**, including **39
  conversation replays** through desktop and chat. Ruff and `git diff --check` passed.
- The eight-case live configured-model evaluation passed **8/8** with synthetic
  dialogue and evidence: title repairs, medium/season amendments, retry, interrupted
  continuation, compound questions, and partial findings. Each case made one lookup
  and three logical model requests, completing in **3.373–12.250 seconds**. Earlier
  probes encountered provider and evidence-check timeouts; this sample does not
  establish continuous provider availability.
- A real Parallel lookup for an overview of Python returned cited findings from
  official `docs.python.org` documentation in **7.043 seconds**. This is a public
  source-retrieval smoke check, not a broad live-search quality benchmark.
- A separate temporary paired-session check passed voice task → Telegram title
  correction → voice retry. Unpairing removed access to the shared projected task.
- Deadline tests cover slow lookup and model code that ignores cancellation, final
  synthesis/checking reserves, source retention, and rejection of late output.
  Control tests verify cancellation is retained in raw dialogue so another channel
  cannot revive canceled work with “again.”
- An actual desktop stream interruption replay verifies retrieved evidence is
  archived with the interrupted answer, then reused by typed “continue” with no
  repeated lookup. The related audio/privacy/runtime group passed 210 checks.
- The isolated wheel build and packaged task/deadline/gap-parser imports passed.
  Diagnostics are under ignored `data/diagnostics/task-search/`. Reproduction,
  coverage, and limits are in [conversation/search evaluation](conversation-search-evaluation.md).
- Real microphone/speaker acceptance remains separate; no personal recordings,
  saved conversations, or screen captures were sent in these evaluation runs.

## Search corrections and recognition hints — 2026-10-28

- The complete ordinary regression run passed **3,901 tests**, with **5 hardware
  integration tests deselected** (85.84 seconds). Ruff, `git diff --check`, the
  wheel build, and packaged correction/STT/compatibility smoke checks passed.
- Replayed the supplied TensorFlow → `Tensura*` exchange with synthetic sources.
  A name correction completes the admitted search directly, preserves the user's
  scope, and avoids a new clarification or alternating generated search subjects.
  Raw user turns retain their original wording.
  A live replay through the configured Super 3.5 VL model, with temporary history
  and synthetic evidence, made one `Tensura` lookup and returned a cited answer.
- Desktop search progress speaks once for voice requests and recent typed subject
  repairs of those requests. Other typed messages and typed final answers remain
  display-only. Tests cover streaming, repeated reads, cancellation before lookup,
  quoted/negated search mentions, unrelated topics, and expired voice preference.
- STT accepts only explicit owner-provided spelling hints, bounded to 12 names and
  five minutes. Tests cover expiry, owner/session/reset revocation, retry decoding,
  unsupported decoder versions, and preserving raw transcripts and confidence.
- Live synthetic checks reproduced a valid web report being withheld after the
  independent relevance check timed out. That check now uses the normal responder
  with reasoning disabled, without changing the eight-second bound or accepting
  unchecked claims. Timeout/provider failures receive accurate empty-source replies.
  A synthetic check under chat reasoning `on` completed successfully in 0.757s.
- During validation, a cancellation deadlock was reproduced with a full audio queue.
  Final sentinel delivery and worker finalization now notice cancellation promptly.
  Deterministic regressions keep active inference resources owned until the worker
  actually finishes, and reject stale playback.
- Focused search/correction checks, STT checks, and verification/routing checks passed.
  Real microphone recognition of the corrected name remains a hardware acceptance
  step; synthetic checks do not measure that accuracy.

## Model selection, reasoning, and terminal chat — 2026-10-28

- The complete ordinary regression run passed **3,821 tests**, with **5 hardware
  integration tests deselected** (266.77 seconds). Ruff passed for `src`, `tests`,
  and `scripts`; `git diff --check` passed.
  The subsequent replacement-model and restored-panel changes passed 312 focused
  provider/routing tests and 109 terminal/startup/CLI tests.
- The saved local configuration and package defaults select Super 3.5 VL 120B for normal
  conversation, Ultra 550B for complex work, and Ultra 550B for NIM fallback.
  Replies and `/status` report the actual served model.
- Synthetic live requests on 2026-10-27 found Super 120B listed but returning 404
  for inference. Ultra 550B answered, served the normal fallback, handled explicit
  reasoning on/off, and completed an automatic-reasoning native tool/result loop.
  No personal histories or screen pixels were used.
  A fresh 2026-10-28 check reproduced Super's inference 404 and a successful
  Ultra inference response despite both models appearing in the catalogue.
- The replacement `nvidia/nemotron-3.5-super-vl-120b-a12b` passed direct synthetic
  inference with reasoning off and on, a native tool/result loop, and a streaming
  reply with the ordinary voice budget on 2026-10-28. No fallback was involved.
  NVIDIA returned backend alias `private/nvidia/nemotron-3.5-super-120b-a12b`.
  Actual CIEL routing also completed reasoning-off greeting, reasoning-on arithmetic,
  and a two-round native tool/result reply using the new public model identifier.
- `/reasoning auto|on|off` is a local session control in the terminal and paired
  text chat. Mode selection is snapshotted for each request. Tests cover the
  supported NIM template flag, hidden-token allowance, and suppression of separate
  reasoning fields and split thinking tags without discarding native tools.
- Voice keeps its brief prompt and ordinary 400-token answer allowance. Typed
  desktop and paired chat allow 4,096-token answers with paragraphs and complete
  code. Desktop typed replies stream as text without initializing speech, including
  when `CONTROLLER_ENABLED=false` selects the retained legacy intent flow.
- The terminal clears the visible interactive screen at launch, restores its boxed
  status panel after initialization, and keeps the editable ruled composer.
  Synthetic PTY checks cover Unicode input, interruption,
  resizing, narrow/plain terminals, safe output, and terminal-mode restoration.
  Fenced code retains indentation and original source line boundaries for copying.
- The wheel built successfully. A smoke check imported its actual packaged modules
  and verified model defaults, reasoning request flags, local toggle handling,
  code rendering, and the text streaming module.
- These checks do not establish real microphone/speaker acceptance or guarantee
  future hosted-model availability and latency.

## Main-model native tools — 2026-10-27

- The complete ordinary suite passed **3,725 tests**, with **5 hardware integration
  tests deselected**. The full run required local socket access for synthetic web
  and Wayland-session tests. Ruff passed for `src`, `tests`, and `scripts`, and
  `git diff --check` passed.
- Default desktop and text-chat paths no longer ask a separate controller to choose
  the next action. Ordinary turns use one main-model request. Native web, memory,
  and contextual screen calls preserve real tool-call IDs and result messages.
- Real callback regressions cover full follow-up context, raw-note search/get,
  original-wording action authorization, local-only histories, cancellation/reset,
  active screen task revisions, speech repairs, and fresh web-pause changes.
- Native adapter tests cover all four providers, fragmented arguments, tool-only
  completions, malformed fields, fallback restrictions, and image/privacy boundaries.
- Web reports retain quote, number, scope, translation, and semantic relevance
  checks before speech. Relevance checks use normal conversation routing and a
  separate bounded worker pool, so an occupied retrieval worker cannot starve them.
- Synthetic live probes against the configured NIM Lightning model passed a
  one-request greeting and both synchronous and streaming two-request tool loops.
  No personal histories or screen pixels were sent. These small checks establish
  native tool compatibility, not live voice latency or general reasoning accuracy.
- The wheel built successfully. A smoke check imported its actual packaged modules
  and executed a synthetic native tool/result/answer loop from the wheel.
- `CONTROLLER_ENABLED=false` retains the legacy desktop intent flow. The old
  controller checker remains a legacy diagnostic rather than the default-path test.

## Controller latency — 2026-10-27

- The complete ordinary suite passed **3,565 tests**, with **5 hardware integration
  tests deselected**. Ruff passed for `src`, `tests`, and `scripts`.
- An eight-request alternating live comparison measured median decision time of
  **2.186 s before** and **1.247 s with compact output**. All eight ordinary/contextual
  sample decisions used the expected action. This measures routing, not total voice latency.
- Four actual-callback regression cases verify screen turns recall memory once
  instead of twice, preserving keyboard/voice behavior and protected history.
- Format repair retains both memory tools. Current follow-ups are presented in
  their own user message, separately from labeled historical context.
- The broader live routing sweep passed **21/24**. Search-failure explanations,
  ambiguous references, and configured-city weather remain unreliable in subsequent
  probes. Prompt changes did not reliably fix these; they are recorded as open model
  behavior rather than passing checks. See [latency findings](conversation-latency.md).

## Markdown memory implementation and pending acceptance — 2026-10-27

The complete ordinary suite passed **3,559 tests**, with **5 hardware integration
tests deselected**, using isolated data and mocked devices/providers. Ruff passed
for `src`, `tests`, and `scripts`; `git diff --check` passed. The focused workspace,
cache, and search run also passed 105 tests. This records automated behavior, not a
live microphone, installed embedding model, or overnight provider run. Reproduce
the complete check and inspect the focused memory/channel cases with:

```bash
.venv/bin/pytest -o addopts="-ra -m 'not integration'" -q
.venv/bin/pytest tests/test_memory_workspace.py tests/test_memory_workspace_cache.py \
  tests/test_workspace_search.py tests/test_memory_workspace_cli.py \
  tests/test_memory_workspace_privacy.py tests/test_memory_workspace_runtime.py \
  tests/test_memory_tools.py tests/test_memory_lifecycle.py \
  tests/test_memory_consolidation_parity.py tests/test_chat_session.py \
  tests/test_conversation_bridge.py
.venv/bin/ruff check src tests scripts
git diff --check
```

The wheel build passed. A headless smoke check imported the built wheel and verified
its actual package path, then exercised status, forced index, search, raw-note get,
reset-index, search after reset, and a paired SQLite/Markdown backup with synthetic
data. This checked installed-package behavior without audio or provider calls.

Use disposable database/workspace overrides and synthetic owner statements for
acceptance. Global CLI options precede the subcommand:

```bash
ciel memory --db /tmp/ciel-memory-check.db --json status
ciel memory --db /tmp/ciel-memory-check.db search Atlas recovery
ciel memory --db /tmp/ciel-memory-check.db get MEMORY.md --from 1 --lines 40
ciel memory --db /tmp/ciel-memory-check.db index --force
ciel memory --db /tmp/ciel-memory-check.db reset-index
```

Check these boundaries before treating the memory upgrade as release-ready:

1. Migrate a stopped synthetic legacy database. Confirm current facts/revisions,
   settings, and dialogue survive, owner prose stays intact, and a second open adds
   no duplicate managed blocks. An explicit `--db` must use its own sibling workspace.
2. Edit, atomically replace, and delete Markdown while CIEL runs. Check source
   lines and snippets after each operation, including equal-size/mtime-preserving
   edits. Invalid UTF-8, revoked permissions, and symlinked notes/cache paths must
   fail without serving cached text. Reset/rebuild must preserve authoritative files.
3. Restart the foreground app. Confirm bounded curated `MEMORY.md`/`USER.md`
   context persists while yesterday's raw dialogue and daily notes stay out of
   ordinary replay. Request earlier owner evidence in voice and paired chat;
   check sources and the three-call/evidence limits. A fresh public question after
   memory lookup must follow its own route. Unpaired or replaced owners receive
   no workspace or prior-owner archive evidence. A reset during bootstrap or tool
   retrieval must prevent another model call with old excerpts and must not
   reinsert the earlier user turn into the reset conversation. Repeat with the
   controller disabled and after streamed audio begins: ordinary interruption can
   retain spoken history, while a reset must leave the erased archive empty.
   Recognized personal-history queries must not reach public search; explicit
   public research about note-taking must remain available.
4. Pause recall, then test ordinary replies, local fact reads, `memory_search`,
   and `memory_get`. Owner CLI inspection should remain available. Repeat privacy
   checks with `/local` and private notes, including a later turn in that context.
   Quoted/tool/assistant text and a note's frontmatter must not manufacture owner
   authority or durable promotion evidence.
5. Check daily checkpoints before compaction and close. Use the lifecycle tests'
   controlled clock for the 03:00 schedule and hard ranking thresholds. Then run a
   separate live scheduled check with the selected provider. Unqueried notes and
   repeated identical same-day queries must remain ineligible. Record provider
   deferral, shutdown, reset, owner-change, and missed-run behavior.
6. Inspect `review RUN_ID` before applying. Check selected zero-based actions,
   partial/repeated apply, all-pending apply, merge/supersede conflicts, curated
   budget rejection, source deletion, and a concurrent owner edit. Verify that
   managed current-fact corrections still use the normal confirmation handler.
   `DREAMS.md` and preimages must remain outside automatic recall/promotion.
7. Back up to a new destination and retain both the `.db` and `.workspace/`
   outputs. Restore the pair into a separate namespace, with a derived/custom
   workspace path matching the restored database. Check private provenance,
   source references, and revisions; rebuilding search must work without copying
   its derived cache. Existing destinations must remain untouched on failure.
8. Forget a synthetic fact that also appears in a daily note and reflection log.
   Verify known wording is suppressed/redacted and pending plans cannot restore it.
   Verify the documented limits: raw conversation archives, old backups, unknown
   paraphrases, and forensic storage recovery require separate handling.

A warm-filesystem synthetic benchmark used 1,000 archive notes of 7,800 bytes plus
one managed curated fact, with three unchanged scans per measurement:

| Operation | Before cache | After cache |
| --- | --- | --- |
| Files reread per unchanged document scan | 1,002 | 0 |
| Median document scan | 118.89 ms | 51.50 ms |
| Median structured-fact signature check | 45.67 ms | 0.18 ms |

Document scans still discover and stat allowed paths for freshness. The text/hash
cache holds at most 4,096 entries and about 16 MiB; evicted notes require rereads.
These timings cover this synthetic local check, not model inference or full reply
latency. With a blocked embedding adapter, regression tests require fresh keyword
access without waiting for it. Optional semantic search gets a 100 ms inference
budget and may finish after the first keyword-only answer. Evaluate a real local
model with `semantic-evaluate` before enabling it; record quality and timing results
apart from mocked tests.

## Search corrections and task grounding — 2026-10-25

- **3,243 tests passed**, with five hardware/model integrations deselected. Ruff
  and whitespace checks passed. Local socket tests ran with the required permission.
- Regressions cover a spoken anime request followed by a typed light-novel
  correction, retained user-provided titles and episode scope, targeted questions
  for missing details, and unrelated new questions that must start a new task.
  Model-invented season/episode numbers are rejected, including written-out numbers;
  explicit user corrections replace stale scope in search keywords and evidence checks.
- Parallel receives the original research objective separately from search keywords.
  An independent answer check rejects related episode/release facts that do not
  answer the requested continuation question. Useful supported partial answers remain
  available; displayed sources and citation numbers reflect the claims actually used.
- Text-only live probes with public anime wording retained a known title through
  short corrections and asked for missing scope when the model guessed a season.
  Four decision samples took 1.54–2.13 seconds. Synthetic evidence probes rejected
  an unrelated release claim in 1.06 seconds and accepted a supported continuation
  claim in 0.96 seconds. These samples do not establish general accuracy or total
  microphone-to-speaker latency. No private history, images, or audio was sent.
- Full-history privacy checks cover both the answer generator and independent
  answer check, including protected context outside the compacted recent dialogue.
  Cancellation, deadlines, malformed judge responses, and provider failures are tested.
- Live microphone acceptance and the factual current Tensura continuation point
  remain unchecked. Restart CIEL, ask where to continue in the light novel, correct
  the medium, and provide a season or episode when asked. CIEL should retain the
  title and requested relation rather than ask for the name again or substitute
  an episode schedule for the reading point.

## Chat after microphone startup failure — 2026-10-24

- **3,169 tests passed**, with five hardware/model integrations deselected. Ruff
  and whitespace checks passed. Local socket tests ran with the required permission.
- Mocked microphone-open failures now exercise real listener processing workers,
  real desktop callbacks, typed replies, saved conversation turns, and clean exit.
  Both interactive terminal input and browser chat remain usable after the failure.
- The text fallback starts no capture or wake workers, stays muted, reports an
  unavailable microphone, and does not retry capture through unmute or mode controls.
  Terminal sessions with text input disabled retain the existing actionable error.
- Actual ALSA hardware recovery remains unchecked. Restart CIEL with the unavailable
  input, type a message, inspect `/status`, and exit with `/exit`. Reconnect or select
  the desired microphone and restart to restore voice capture.

## Parallel free search — 2026-10-24

- **3,163 tests passed**, with five hardware/model integrations deselected. Ruff
  and whitespace checks passed. Local socket tests ran with the required permission.
- The default search provider is now Parallel's anonymous Search MCP. Setup and
  configuration accept it without credentials; saved keys remain inactive unless
  their provider is selected. The local configuration was switched separately.
- Real `InternetService` searches for “Python asyncio documentation” and “OpenClaw
  free web search providers” each returned five sources with usable excerpts in
  1.39 and 1.20 seconds respectively. These samples measure retrieval only, not
  answer generation, microphone-to-speaker latency, or general search accuracy.
- Source excerpts remain labeled through storage and follow-ups. They support
  quote validation without claiming a local page read or requiring another fetch.
- Regressions cover cancellation before query transmission during the handshake,
  JSON/SSE protocol handling, shared deadlines, bounded responses, private requests,
  safe errors, provider isolation, and fresh anonymous sessions after chat resets.

## Contextual follow-ups and fresh launches — 2026-10-24

- **3,104 tests passed**, with five hardware/model integrations deselected. Ruff
  and whitespace checks passed. Local socket tests ran with the required permission.
- The reported sequence (screen request, damaged transcript, correction, “check
  again,” “again”) now has entry-point regressions in chat and both desktop
  controller modes. Tests verify new screen operations, retained original/corrected
  focus, direct clarification, and absence of the ordinary stale-answer fallback.
- Task lifecycle tests cover pauses, canceled requests, late commits, simultaneous
  writes from separate processes, reset between validation and commit, low-confidence
  speech, and explicit new tasks. Cancel/reset cannot restore the old task.
- Real foreground-launch and persistent-chat tests verify fresh prompts and
  summaries, retained saved facts and old archives, current-run voice/chat continuity,
  and discarded old pending confirmations and screen context.
- Complete structured summaries exceeding per-field concision targets are retained
  losslessly within the hard 4,000-character total. Types, schema, nonempty text,
  over-budget rejection, pending turns, and reset/cooldown behavior remain tested.
- A live text-only synthetic intent probe resolved unseen another-look wording in
  1.08 seconds. Two other probes reached the two-second decision limit and returned
  clarification. These samples do not establish live speech accuracy or future
  provider latency. No saved conversation, desktop image, or microphone audio was sent.

Live acceptance: start CIEL, request a screen observation, change something visible,
then correct an utterance and ask “Can you check again? I meant,” “check again,”
and “again.” Each another-look request should obtain a new view and focus on relevant
changes. Repeat after an interruption or pause. Cancel or start a different task and
verify “again” does not revive the screen request. Restart CIEL and verify a fresh
conversation with saved personal facts still available. This microphone/desktop
acceptance remains outstanding.

## Screen timeout and retry recovery — 2026-10-24

- All **3,001 non-integration tests passed**; five hardware/model integrations
  were deselected. Ruff and whitespace checks passed. The full run required
  permission to open temporary local sockets for web and desktop-discovery tests.
- Regressions cover a screen timeout followed by typed retry in both desktop
  controller modes and chat. Retries retain the original question and historical
  image scope; unrelated conversation, resets, and owner changes end the context.
- Failed background summaries retain pending turns and wait 30 seconds before
  retrying. Tests cover expiry, provider/validation failure categories, foreground
  preemption, and reset recovery without skipping turns.
- The previously selected Llama 3.2 11B endpoint timed out on two tiny synthetic
  image probes. Nemotron 3.5 Super VL passed the color check and read an invented
  title and panel colors through CIEL's actual vision client in 2.31 seconds
  including verification, then 1.77 seconds with verification cached. An earlier
  attempt returned HTTP 500. These samples do not guarantee service availability.
  No personal screenshots, conversation history, microphone, or playback were used.
- Default/example settings now select the tested vision model, with reasoning
  disabled for NIM Nemotron vision requests. Existing explicit choices are preserved;
  the reported local configuration was updated separately. Live desktop acceptance
  remains open; restart CIEL and request a new screen observation to verify it.

## Screen-awareness validation — 2026-10-22

Follow-up tool-feedback/grounding change: **2,960 tests passed**, five integration
tests deselected; Ruff passed. Synthetic coverage checks screen announcements before
capture, web announcements before retrieval, cancellation during announcements,
application authorization before progress, transient progress history, private
wording routes, and foreground preemption of background wording. A synthetic
player's invented visible title was read correctly in 5.14 seconds including model
verification. This does not establish real anime identification or desktop latency.
AI wording is prepared in the background; feedback remains immediate if generation
is slow or unavailable. No real desktop pixels or Telegram messages were used.

- 2,943 automated tests passed; five hardware/model integration tests deselected.
- Ruff, local documentation links, and wheel build passed.
- The selected NIM Llama 3.2 11B Vision model passed a synthetic image check in
  1.44 seconds. No personal screenshots were used. The larger 90B candidate's
  inference timed out despite appearing in the model catalog.
- Read-only desktop metadata checks confirmed two active monitors, cursor and lock
  state support, and background discovery without inherited display variables.
- Real all-monitor screenshot quality, pointer targeting, and live paired
  voice/Telegram handoff remain unchecked. See [screen awareness](screen-awareness.md).

For live acceptance, request “Look at my screen,” point to a visible item and ask
what it is, change the screen and ask “What about now?”, then continue in paired
Telegram. Replies should contain text unless “Send me a screenshot” is requested.
Verify stop, reset, image expiry, and unavailable/locked-desktop responses.

## Automated validation

```bash
python -m pytest
python -m ruff check src tests scripts
for script in setup.sh scripts/*.sh; do
  bash -n "$script" || exit
done
python -m pip check
python -m pytest -m integration tests/test_tts.py -k piper_initialization_and_synthesis
python -m pip wheel . --no-deps -w dist
```

Unit tests use mocked devices/providers. The Piper integration test requires its
local voice model; synthesis does not prove speaker playback works. Device and
wake-training integrations have their own hardware/model requirements and can be
run with `pytest -m integration` on an appropriately configured machine.

Install the wheel in a fresh virtual environment outside the checkout. From a
separate directory, run both `python -m ciel --help` and `ciel --help`, then
`pip check`. Verify core installation can open setup without Whisper, Piper,
Pocket, openWakeWord, Torch, or CUDA packages already installed. Normal voice
startup should not require optional training packages. Select an existing local
Piper model after installing the `voice` extra to check synthesis without playback.

Fresh setup should offer CPU recognition by default, with `small.en`/`int8` and
beam 5. **Customize** should offer explicit CUDA/`int8_float16` recognition. Existing
saved recognition settings should remain selected. Launch after setup should use
ambient listening and live AI captions. The separate wake recognizer remains CPU
`base.en` when enabled. Check plain `ciel` as well as `ciel start`; neither needs additional
mode flags. Explicit `--no-ambient` and `--no-show-ai-transcripts` override those
defaults for one run. Older `.env` values remain explicit overrides; use a newly
configured environment when validating fresh defaults.

Check `ciel setup` with both Quick start and Customize. Cancel at the review
and confirm the original configuration is untouched and no optional packages or
models have downloaded. Save, confirm only selected packages and configured models
are prepared, replay the voice test,
change speakers, and return to the system default; confirm the next test and
normal launch use that output. Saved keys must stay hidden. Check narrow windows,
arrow navigation, `TERM=dumb`, and `NO_COLOR=1`. Normal `ciel` should show the compact
status panel before keyboard input begins, while `ciel start dev` retains diagnostics.

Run `bash setup.sh` with both a fresh and existing `.venv`. It should install only
the editable core before opening setup, always offer the wizard, and install the
launcher. Check fresh CPU/Piper setup does not install Pocket, Torch, CUDA, or
openWakeWord; select Pocket or configured openWakeWord separately and verify the
needed extra is installed. No Amy model should download for an unselected voice.
Rerun setup after changing a selection and confirm newly needed components install.
Interrupt a model download, confirm saved choices survive, then retry with Quick
start. Cancellation and package/download failures must produce an incomplete result
and a nonzero exit status, without an installation-success message. Turbo still
requires its separate environment and private assets; verify missing prerequisites
are reported without claiming the voice is ready.

For an installed package's CUDA test, include the `gpu` extra when installing
the current source: run `python -m pip install '.[voice,gpu]'` from the source root
using the isolated environment's Python. Reinstall an earlier wheel before testing changed
defaults. Without usable CUDA, check the logged CPU recovery independently.

## Recorded validation — 2026-10-18 (final review)

- All 1,759 non-integration tests passed, including the local HTTP-server cases.
  Five hardware/model integrations were excluded. Ruff, dependency, shell syntax,
  and whitespace checks passed.
- Review regressions preserve private routing beyond the five-fragment speech tail,
  through both listener gates, reply generation, persisted history, and summaries.
  Rejected or replaced pending chains do not save their unaccepted fragments.
- Empty interruptions recover observer-only questions exactly once without duplicating
  accepted turns or restarting an answer after playback began. Valid model rechecks
  can resolve follow-up references; failed rechecks and policy questions retain their
  existing handling. Restored speech gains no action or memory-write permission.
- The final wheel built and installed outside the checkout. Both installed CLI entry
  points, version output, terminal rendering, and setup review/cancel passed. Package
  contents match the reviewed source and exclude private/generated artifacts.
  Installation reused existing dependencies; a fresh dependency installation and new
  live microphone/speaker acceptance were not performed. The earlier hosted synthetic
  results below remain the latest provider checks.

## Recorded validation — 2026-10-18 (short conversational references)

- The complete non-integration suite passed: 1,733 tests, including the eleven
  local HTTP-server cases previously blocked in this environment. Five hardware/model
  integrations were excluded. Ruff and whitespace checks passed.
- Read-only investigation confirmed the preceding assistant reply remained in
  conversation history. The unwanted questions came from the controller's direct
  clarification path. An invented development-day exchange reproduced the mistake.
- Controller guidance now resolves reactions against recent assistant replies.
  Valid model clarifications in active dialogue can use the ordinary full-context
  reply stage. Application-generated policy questions, outages, pending local
  confirmations, and background speech retain their existing handling. Delegation
  grants no tool or memory-write authority and preserves private routing and cancellation.
- All 25 hosted synthetic checks passed: 24 valid controller decisions and one
  source-validated report. Successful decisions took 1.06–7.30 seconds. The day-reference
  case still selected clarification, but its application-owned context-resolution flag
  now sends it to the reply stage; the checker reports that flag separately. A normal
  chat probe resolved the synthetic day-21 reference. An earlier run hit the existing
  eight-second timeout; these checks do not guarantee future provider latency.
- New runtime regressions cover complete history replay, local timeline replies,
  contextual clarification in both reply modes, forged delegation flags, policy
  boundaries, outages, cancellation, and absence of action or memory execution.
  Live microphone verification remains open.

## Recorded validation — 2026-10-18 (terminal setup and resumed speech)

- 1,689 tests passed with `pytest -k 'not test_http'`; the mocked HTTP-stream
  cancellation test passed separately, for 1,690 passing tests overall. Eleven
  local HTTP-server cases and five hardware/model integrations were excluded. The HTTP checks
  remain pending because this environment could not approve the full test run.
  Ruff, dependency, shell syntax, and whitespace checks passed.
- Guided setup covers review/cancel, hidden keys, atomic private configuration
  saves, preserved custom settings, playback failure/retry, and output-device
  changes. Real pseudo-terminal arrow navigation and terminal restoration passed;
  plain, narrow, ASCII, and no-color rendering also passed.
- Synthetic audio frames and delayed transcription preserve two and three
  consecutive clips. Runtime tests retain linked nameless follow-ups for one
  final decision, reject unrelated or expired fragments, preserve private routing,
  and require the existing authority for memory writes. Streamed text stays visible
  when audio fails, and canceled replies stay hidden.
- These checks used mocked providers and playback. Live microphone/speaker
  acceptance and a new wheel build were not performed for this change.

## Recorded validation — 2026-10-18 (normal startup defaults)

- 1,630 non-integration tests passed; five hardware/model integrations were
  excluded. Ruff, dependency, shell syntax, and whitespace checks passed.
- Fresh configuration, setup, direct STT construction, example configuration, and
  normal launch agree on small.en/CUDA/int8_float16, beam 5, guarded medium.en
  retry, ambient conversation, live AI captions, and speech interruption.
  Explicit CPU, wake-only, and hidden-caption settings remain supported.
- The wheel built and installed outside the checkout. Fresh default settings,
  plain command startup, captions, and explicit opt-outs passed with mocked
  hardware/providers. Installer prefetch tests verify configured distinct command
  and wake downloads without inference; no real installer or new model download
  was run for this change.

## Recorded validation — 2026-10-18 (user confirmation)

The user reported that CIEL was working well after the controller recovery fix
in `e60a62d`. This is confirmation from normal use; individual acceptance-case
results and device timings were not supplied. The broader acceptance checklist
and clean-install validation retain their existing status.

The controller fix's [GitHub checks](https://github.com/Hexarion0/Ciel/actions/runs/39050301565)
also passed on Python 3.10 and 3.14.

## Recorded validation — 2026-10-17 (controller timeout recovery)

- 1,606 non-integration tests passed; five hardware/model integrations were
  excluded. Ruff, dependency, and whitespace checks passed. New regressions cover
  consistent controller defaults and refusing false diagnostic passes after model
  timeouts, malformed replies, low confidence, or late completion.
- The corrected live checker passed all 21 synthetic checks: 20 valid controller
  decisions in 1.08–3.09 seconds each and one report validated against a synthetic
  page. It used the configured NIM service without live search or private history.
- Fresh settings select Super while preserving the eight-second decision deadline
  and explicit custom model overrides. The current local configuration resolves to
  Super and retains the custom Turbo voice.
- A synthetic typed greeting passed the real controller in 6.91 seconds and
  received a real NIM conversation reply in another 1.74 seconds. It loaded no
  conversation database and used no microphone, playback, or desktop actions.
  Restart an existing CIEL process to load the new setting. Live audio acceptance
  and future hosted-service latency remain separate checks.

## Recorded validation — 2026-10-17 (repository review)

- 1,595 non-integration tests passed; five hardware/model integrations were
  excluded. The separate local Amy/Piper synthesis test passed. Ruff covered
  source, tests, and scripts; dependency, shell syntax, and whitespace checks passed.
- New regressions cover superseded-turn handling without provider waits, typed
  fragment reset, guarded memory corrections, the shared controller deadline,
  independent streaming cancellation, setup key privacy, and installed wake paths.
  Archive/download regressions cover canonical duplicate paths and partial-file
  symlinks; Kaggle CLI checks make no submissions or credential reads.
- The final wheel built and installed outside the checkout. CLI help/version,
  standalone Turbo worker help without app dependencies, bundled web/voice assets,
  writable user paths, and absence of private/generated artifacts passed.
  Installation reused existing dependencies; a fresh dependency installation and
  target-desktop microphone, speaker, GPU, and provider acceptance remain open.

## Recorded validation — 2026-10-16 (local commands and installed voice resources)

- 1,559 non-integration tests passed; five integration tests were excluded. Ruff
  covered `src`, `tests`, and `scripts`; dependency, shell syntax, and whitespace
  checks passed. The separate Piper synthesis integration passed with local Amy.
- Local-command regressions cover missing NIM, outages, private-session restart,
  cancellation, action allowlists, confirmations, local topic changes, and long
  sessions without provider summaries. Desktop effects and providers were mocked.
- The wheel built and installed to a temporary target outside the checkout. CLI
  help/version, the standalone worker's help command, bundled manifest/web assets,
  and per-user voice path resolution passed. Installed Piper synthesized locally
  without recording or playback. This check reused existing dependencies rather
  than installing a fresh environment's dependency set.
- The execution environment exposed no `/dev/snd` directory. Live wake detection,
  microphone/speaker playback, CUDA/Turbo synthesis, sustained conversation, and
  actual desktop-window opening still require target-desktop acceptance.

## Live internet acceptance

Run `ciel start dev` with `CONTROLLER_ENABLED=true` and the configured NIM model. Confirm
logs show `AI chose ...` before public retrieval. Explicit local commands use the
local-command log instead and need no NIM request. Search acknowledgment must finish
before the first lookup; the factual report waits for source validation. Test
`Okay, try again` after a failed lookup and `Why can't you search it on the internet?`:
the first should retain the topic, while the second should explain the limitation.
Nearby speech such as `Mom, what's for dinner?` should produce silence and no saved
conversation turn. Also test a background interruption during an active reply.
Unplug the network: addressed decisions should fail briefly; background decisions
should remain silent. Stop/mute must still work without a model response.
While offline, check explicit clock requests, memory confirmation/readback, and
allowlisted app launching. Change to another local task before saying `yes` or
`open it`; expired proposals and unrelated app references must not execute.

Hosted-NIM checks on 2026-10-14 passed 20 synthetic decision cases and one report
validated against a synthetic page. Cases cover greetings after unfinished requests,
ambiguous acceptance, reading-point follow-ups, contextual retries, capability
explanations, background speech, clock lookup, research corrections, city precedence,
and disabled automatic search. Observed decisions took about 1.0–3.0 seconds.
The revised controller also passed 1,114 non-integration tests and Ruff. An
end-to-end public test found and cited the official Python tutorial through the
real search backend. These checks use invented or public text; microphone and
broader live search-site acceptance remain open.
An active-conversation regression check confirms a mistaken silent decision gets
a second lightweight NIM check. Clear speech to another person remains silent;
unresolved listener intent prompts a short clarification.
Run `.venv/bin/python scripts/check_controller.py` for the expanded reproducible
check of decisions, city precedence, retries on different topics, and a validated
report from a synthetic page. This uses the configured NIM API key and makes no
microphone, saved-conversation, desktop-action, or live-search requests.

Use both `ciel web` and a normal voice session. These checks require a working
network and a configured AI provider; mocked tests do not establish search
availability or spoken answer quality.

1. Ask for today's news and a current product price without saying “search.”
   Check the linked sources support the answer and that lookup finishes within
   the applicable per-lookup timeout; retries have separate bounded budgets.
   Ask a greeting and the local clock question;
   neither should perform a web lookup.
2. Ask CIEL to read a public article URL. Expect an answer based on the page
   text, with a source link. A blocked or unsupported page must receive an
   honest explanation rather than an invented summary.
3. Ask for weather without a location and with `INTERNET_HOME_CITY` empty.
   Expect a city question. Reply with a city, then check an explicit location
   overrides the configured home city on a later request.
4. Ask to compare or research a topic, then request a related follow-up.
   Confirm relevant preferences and user context improve the query. Click
   sources by mouse and keyboard, then restart and check their metadata survives
   alongside the saved answer. Source URLs should not be read aloud.
   Ask about a named series or software release, then say “Can you search the
   latest episode/release?” without repeating its name. Check the displayed
   query retains the user topic. Correct an installment or version number;
   the next query must use that correction and check fresh sources. Search
   guides or a different named topic must not be cited as evidence. When
   excerpts do not establish a precise answer, expect an honest limitation or
   clarification rather than a guessed number or a denial of internet access.
   Replay “search Tensura season 4 where it ends,” “so what volume should I read?”,
   “research again, finished season 4 part 2,” then “research.” Queries must retain
   the title, reading question, and part number without old assistant volume guesses.
   Web answers wait for complete source-quote validation before audio starts, even
   with `TTS_STREAMING=true`. Invalid JSON, invented quotes, or a volume unsupported
   by a supplied page excerpt must produce a short uncertainty reply.
5. Interrupt a lookup or its spoken answer. Expect responsive cancellation and
   no stale answer or source links attached to the next request. Disconnect the
   network, ask for current facts, and expect an inability-to-verify explanation
   and a retry offer. Restore connectivity and retry.
6. Verify a fixture containing previously marked private conversation history
   makes no cloud or external retrieval requests, including after restart.
   Check the retired private-session command is rejected locally without an AI
   request. Also check `INTERNET_ENABLED=false` and `INTERNET_AUTOMATIC=false`
   behave as documented.

## Live microphone acceptance

Use the target desktop, microphone, and speakers with the normal `.env` and
installed speech models. Record device names/indices, STT model/device/compute
type, TTS voice/engine, silence duration, and beam size with the results. Keep keys
and private audio out of reports.

1. Run `python -m ciel setup`. Keep custom settings, select your devices and
   local voice, and confirm the voice test actually plays. Restart normally with
   `python -m ciel --listen` and verify the same devices and voice are used.
2. Say “Hey Ciel.” Expect an acknowledgement, then ask a short question.
   Repeat with “Hey Ciel, what time is it?” in one utterance. Verify the start
   of the command is retained and one reply plays.
   Use `ciel start dev` and wait for `Wake keyword spotter ready` before starting.
   Test ten wake attempts
   at normal volume, ten quieter attempts, and several slower greetings. Record
   successful attempts and wake latency, along with `WAKE_MIN_RMS` and
   `WAKE_WINDOW_SECONDS`. Leave the PC humming without speaking for one minute
   and record any false wakes; increase the minimum RMS if needed.
3. Ask a follow-up without the wake phrase. Check that the reply uses the earlier
   context. Pause briefly mid-sentence and verify the recorder does not cut you
   off; adjust `UTTERANCE_SILENCE_SECONDS` if needed.
4. Interrupt a longer spoken reply with a new command. Expect playback to stop,
   the new command to be recorded, and the new reply to complete.
   Add several fragments while routing is pending, then type a new request.
   Superseded speech must not wait for another controller call, and the typed
   request must not inherit the abandoned fragments. Test `/stop` during a
   stalled provider stream as well as during audible playback.
5. Remain silent after wake/follow-up. Expect recording to time out and return
   to standby without invented questions, repeated replies, or a stuck listener.
6. In a cloud-provider session, temporarily disconnect the network. Ask a
   question and verify configured fallback or an error acknowledgement; restore
   connectivity and check the next question succeeds. Also run the mocked
   provider regression tests to cover null answers and failed fallback responses.
7. Ask “How do you say goodbye in Spanish?” and confirm it receives an answer.
   Say “Goodbye” and verify follow-up ends. Say the wake phrase again, then press
   Ctrl+C. Confirm the process exits and the microphone can be opened again on a
   fresh launch.
8. Launch with `./scripts/launch_ciel_gpu.sh --ambient`. Address CIEL directly,
   then ask a related follow-up without her name. Talk to someone else and verify
   she stays quiet and no background turn or personal fact appears in SQLite.
   Check that `Ciel, stop listening` returns to wake-word mode and `Hey Ciel,
   listen continuously` enables ambient mode again. Interrupt playback by name.
   With headphones, also set `BARGE_IN_MODE=speech` and interrupt with ordinary
   speech. Confirm audio stops, the first word is retained, and steady PC hum does
   not interrupt. Pause a long reply, then say `Ciel, continue`; verify the AI
   receives interruption context and resumes instead of starting over. Estimated
   progress may repeat or skip a few words; it is not exact word alignment.
   Add words while STT or routing is pending and confirm one combined response.
   In particular, ask a question by name and add `And good` while routing is
   pending: expect `merged_continuation` and a single combined request. Repeat
   with successive additions and a correction; verify the original question and
   negation survive. Speech addressed to `Mom` must still be rejected.
   After an answered greeting, repeat with two or three closely spaced follow-up
   fragments without saying CIEL again. Expect the full request, in order, to
   reach one final decision. Typed replacement and `/stop` must discard that chain.
   Check `AMBIENT_LISTENING` and `SHOW_TRANSCRIPTS` defaults with a no-argument GPU
   launch, then test `--no-ambient --no-show-transcripts` overrides.
9. Pause for 1.2 seconds mid-sentence, then finish the thought; expect one complete
   turn. Resume speaking during reply generation and verify the old reply is
   suppressed. Check small wording mistakes against recent context; uncertain
   names, dates, quantities, and negations should get a clarification, not a saved
   inferred correction. Repeat with the actual PC hum and record missed/false wakes.

10. With `TTS_STREAMING=true`, ask for a multi-sentence explanation. Confirm the
    first sentence plays before the entire reply arrives and history contains one
    assistant turn. Compare logged first-text and first-audio times with batch mode
    (`TTS_STREAMING=false`) using the same model, voice, and question. Interrupt
    during the first sentence, synthesis, and a gap between sentences: later queued
    audio must not play. Say `Ciel, continue` and check completed sentences are
    represented in interruption context. Ask for code and verify it remains in
    history while spoken output omits the code block. Disconnect after a sentence
    starts: expect a connection-cutoff notice, without a fallback answer being
    appended to the partially spoken reply.
    With `SHOW_AI_TRANSCRIPTS=true`, verify `CIEL: ` reveals letters at the
    spoken pace in the terminal. A canceled sentence that never plays must not
    appear. During barge-in, expect only the reached prefix plus `[interrupted]`.
    Check the local clock answer and greeting, both CLI overrides, batch TTS,
    and two voice speeds. Ordinary logs must appear without breaking the caption
    line. Redirect output to a file and expect final/partial snapshots without
    terminal escapes or per-letter lines. Final history must still contain one
    assistant reply. Native word spans drive letter interpolation; spelling and
    phonemes are not the same, so assess remaining drift on the actual headset.
    Ask "What did we talk about last time?" after a short technical conversation.
    Expect a brief factual recap without poetry or speeches about trust or agency.
    Repeat in batch mode. Bracketed breathing, pause, mood, or gesture descriptions
    must not appear in captions, reach either synthesizer, or enter saved replies.
    Automated tests cover the reported `[soft inhale — steady, focused]` output,
    every token boundary, and descriptions longer than a speech sentence buffer.
11. While voice training is running, ask a short unclear question that triggers
    STT retry. Below `STT_RETRY_MIN_FREE_MB`, expect a primary-model retry without
    allocating `medium.en`. After training releases VRAM, a stronger retry can be
    used again and its temporary CUDA model must be released afterward. Record
    GPU memory and transcript quality; this does not prove hum rejection by itself.

## Memory proposals and first local actions

Run these checks on the target desktop after starting `ciel start`:

1. Say “Ciel, my favorite game is CS2.” Expect a proposal, then say
   “Ciel, no.” Verify no structured favorite-game fact was added or changed.
2. Repeat the statement and say “Ciel, yes.” Close CIEL, restart, then ask
   “Ciel, what's my favorite game?” Expect CS2. Propose a different game and
   reject it; the confirmed value must stay CS2. Use a disposable test fact if
   you do not want to change your real preference.
3. Propose another change, ask for the time, then say “Ciel, yes.” The old
   proposal must not be saved. Repeat after waiting more than 60 seconds and
   after “Ciel, cancel.” An inferred ambient yes must not authorize a write.
4. Ask RAM usage, GPU temperature, and logical CPU core count using the exact
   examples in README. Expect local answers, or an unavailable-metric response,
   with no provider request. Compare against a system monitor.
5. Say “Ciel, open Discord.” Verify a Discord or Vesktop window appears.
   Check a supported but uninstalled app receives an honest missing-app response.
   During ambient follow-up, say “Open Discord” without her name: it must not
   launch. Paths, shell syntax, and command-line flags must never execute.
6. Interrupt a spoken action result and verify voice listening remains responsive.
   A completed launch is not undone by a later interruption.

**Status:** live microphone, speaker, restart, and desktop-window acceptance is
pending. The current execution environment exposes no audio device directory.
Automated callback tests use mocked audio and mocked launches. Update this status
after observing the checks on the target desktop.

## Recorded validation — 2026-10-08 (setup, documentation, and spoken status notes)

- 662 unit tests passed; three integrations were excluded. Ruff, shell syntax,
  whitespace checks, and all 27 local documentation links passed.
- The user reported hearing “Playback was interrupted” aloud. Callback tests
  reproduced a provider returning the internal marker as a spoken answer.
  Archived assistant status annotations now enter provider context as separate
  playback metadata; streamed and batch output suppress the exact markers.
  Tests cover every marker chunk boundary, mixed answers, marker-only replies,
  direct TTS cleanup, restart context, and summary inputs.
- Turbo model downloads now request the revision accepted by the worker and
  refresh incompatible cached inventories. Download tests use a mocked hub;
  no model download or live provider request was needed for validation.
- Fresh-install example settings use CPU transcription and Amy. The installer
  installs runtime dependencies from package metadata and checks Python version
  plus both Amy model/config files. Setup preserves the user's existing settings.
- The wheel built and installed to an isolated target outside the checkout.
  Installed-package import, CLI help/version, and the status filter passed using
  the existing environment's dependencies. This is not a clean dependency install.
- README and roadmap now reflect current defaults and distinguish implemented
  features, custom assets absent from Git, historical measurements, and pending
  live acceptance. Restart CIEL and retest interruption/continuation on the
  actual microphone and speaker; no live playback acceptance was performed here.

## Recorded validation — 2026-10-08 (Discord request routing)

- 642 unit tests passed, with three integrations excluded. Ruff and whitespace
  checks passed. Desktop launching was mocked during these tests.
- The reported conversation contained misheard app names, an exact spelling
  correction, and “Can you open it for me?” The narrow launch matcher sent the
  request to conversational generation instead of the local action. Discord's
  executable and installed host were present.
- Launch matching now accepts polite commands and exact letter-by-letter app
  spelling. Recent user corrections can resolve “open it” for 60 seconds;
  unrelated requests, cancellation, restart, and uncertain speech invalidate it.
  Direct-address authorization and the executable allowlist remain required.
- Regression tests replay natural requests, corrections after interrupted speech,
  and the follow-up launch through real CLI callbacks with mocked processes.
- Retest after restart with “Ciel, can you open Discord for me?” If STT still
  mishears the name, say “Ciel, I meant D I S C O R D,” then “Ciel, open it.”
  Actual microphone recognition and window appearance still require a live retest.

## Recorded validation — 2026-10-07 (memory proposals and local actions)

- 617 unit tests passed; three hardware/model integrations excluded. Ruff and
  whitespace checks passed.
- The previously failing summary-routing test now mocks the streaming provider
  path used by background summaries and still checks the economical model and
  token limit.
- Memory tests cover yes/no, proposal corrections, expiration, unrelated turns,
  unauthorized confirmation, cancellation, restart persistence, and explicit
  commands. Real CLI callbacks verify local acknowledgements and history.
- Action tests cover discovery without core edits, duplicate names, ambiguous
  requests, invalid arguments, executable allowlisting, direct-address permission,
  cancellation, deadlines, unavailable telemetry/apps, and handler failures.
  Application launches and microphone input are mocked in automated tests.
- Piper initialization and synthesis passed using the Alan model after downloading
  the missing model. No speaker playback was performed.
- The 0.3.6 wheel built and installed with its declared dependencies in a fresh
  Python 3.14 virtual environment outside the checkout. Both CLI help entry points
  and `pip check` passed. Installed-package discovery and separate-process SQLite
  restart recall passed. Piper synthesis from that environment also passed without
  playback; optional wake-training code was not imported.
- Live acceptance above is still pending; no stable release tag was created.

## Recorded validation — 2026-10-04 (v0.3.6 patch)

- 567 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  CLI version, and wheel build passed. The wheel includes caption/alignment
  modules and declares Piper's alignment dependency.
- Native timing was checked offline on the installed custom voice and Amy with
  contractions, multiple sentences, `5:32 AM`, and `88°C`. Phoneme durations
  summed exactly to each generated waveform, and original-text character
  schedules were monotonic and bounded by that same audio. No audio was played.
- Playback tests cover reported output latency, character ordering, stop-time
  partial captions, failed/canceled synthesis, replaced playback, and a broken
  caption writer. Renderer tests cover TTY redraws, narrow terminals, plain
  redirected output, secret masking, and coordinated log restoration.
- CLI tests confirm captions do not reveal complete local/batch replies ahead
  of audio, text-only failure fallback remains visible, normal INFO logs remain
  with captions disabled, and assistant history is saved once.
- The installed custom ONNX checksum is unchanged. Native alignments are exposed
  in memory without modifying model files or adding another neural inference.
  Live headset synchronization remains to be checked; letters interpolate inside
  measured word spans, and uncertain mappings use a duration estimate.

## Recorded validation — 2026-10-04 (v0.3.5 patch)

- 501 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  CLI help, and the v0.3.5 wheel build passed.
- Regressions verify that live sentence captions follow playback start and
  precede completed generation, exclude canceled synthesis and queued speech,
  filter reasoning/code, honor environment and CLI controls, and retain one
  assistant history turn. Local replies and batch playback are covered.
- Synthetic custom-voice samples and a separate earlier-checkpoint export were
  produced for diagnosis. Structural ONNX validation and finite synthesis passed;
  those checks do not establish intelligibility. Live caption timing and voice
  listening acceptance remain to be checked on the target desktop.

## Recorded validation — 2026-10-04 (v0.3.4 patch)

- 488 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Before the change, a live NIM gate probe with a synthetic recent exchange
  rejected “What do you want to talk about?” with addressed=false/confidence=0.6.
  After the change it selected listener=assistant/confidence=0.95 and the
  application accepted it as `active_conversation`.
- A second live ordinary follow-up, “Can you suggest something fun?”, was also
  accepted. “Would you like some tea, Mom?” selected listener=other/confidence=0.95
  and remained silent. These are intent checks, not microphone acceptance.
- Configured the local conversation policy with the existing 40-second window.
  During recent dialogue valid uncertainty favors continuation; strict mode
  requires high certainty. Provider failures and malformed judgments remain
  silent in both policies. Inferred turns remain non-explicit and cannot save
  personal facts through the memory-command handler.
- Regressions cover unseen topic changes, other listeners, legacy gate JSON,
  strict mode, expired/reset/user-only contexts, onset-based time windows,
  malformed confidence, and structured NIM JSON containing cleanup markers.
- Reproduce using the ambient GPU launcher: ask a question by name, wait for
  playback, then ask “What do you want to talk about?” without her name. Expect
  `active_conversation` with logged intent confidence. Address Mom and expect
  silence; test again after the window expires and in strict mode.

## Recorded validation — 2026-10-04 (v0.3.4 patch)

- 489 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Before the change, a live NIM gate probe with a synthetic recent exchange
  rejected “What do you want to talk about?” with addressed=false/confidence=0.6.
  After the change it selected listener=assistant/confidence=0.95 and the
  application accepted it as `active_conversation`.
- A second live ordinary follow-up, “Can you suggest something fun?”, was also
  accepted. “Would you like some tea, Mom?” selected listener=other/confidence=0.95
  and remained silent. These are intent checks, not microphone acceptance.
- Configured the local conversation policy with the existing 40-second window.
  During recent dialogue valid uncertainty favors continuation; strict mode
  requires high certainty. Provider failures and malformed judgments remain
  silent in both policies. Inferred turns remain non-explicit and cannot save
  personal facts through the memory-command handler.
- Regressions cover unseen topic changes, other listeners, legacy gate JSON,
  strict mode, expired/reset/user-only contexts, onset-based time windows,
  stale assistant turns after idle, malformed confidence, and structured NIM
  JSON containing cleanup markers.
- Reproduce using the ambient GPU launcher: ask a question by name, wait for
  playback, then ask “What do you want to talk about?” without her name. Expect
  `active_conversation` with logged intent confidence. Address Mom and expect
  silence; test again after the window expires and in strict mode.

## Recorded validation — 2026-10-04 (v0.3.3 patch)

- 428 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Replayed the reported “What time is Ciel?” followed by “You sound like a
  robot, do you know?” through the real CLI callbacks with mocked audio and an
  isolated database. Both receive replies: local clock first, then one model
  answer for the feedback, without cloud speech-gate calls.
- Original transcripts are retained in history. Clock address inferred from a
  missing “it” is non-explicit and does not authorize memory writes. Regression
  coverage rejects quoted speech, other listeners, geographic/compound clock
  requests, custom-assistant mismatches, and appended instructions.
- Feedback descriptions and brief tags require recent assistant speech within
  the original follow-up window. Expired and reset contexts still reject them.
- The reported hardware run already shows `small.en` on CUDA without fallback;
  microphone acceptance of these new reply-gate rules remains pending. Reproduce
  using the GPU launcher in ambient mode, ask the clock question by name, and
  give the voice feedback after the local reply.

## Recorded validation — 2026-10-04 (v0.3.2 patch)

- 391 unit tests passed; three integrations excluded. Ruff, whitespace checks,
  and the runtime version command passed.
- Replayed the reported transcript “Why are you talking so fast right now?”
  after an assistant reply: accepted locally as `speech_feedback`, without
  another wake phrase or cloud intent request. Tests retain room-conversation
  rejection, deadline expiry, and context after uncertain fragments.
- Installed CUDA 12 cuBLAS and cuDNN 9 into the application `.venv` and tested
  the cached `small.en` model on the GTX 1660 SUPER using the launcher's library
  search paths. Actual inference on synthetic silence completed successfully
  with `cuda/int8_float16`; loading plus inference took 3.11 seconds.
- Startup regressions force a lazy CUDA-library failure and verify CPU fallback
  occurs before the ready event. The new `gpu` extra documents reproducible
  installation independent of the voice-training environment.
- Live follow-up/wake acceptance with the user's microphone remains pending.
  Restart the launcher, ask a question by name, then ask “Why are you talking
  so fast right now?” after the reply; expect `speech_feedback` and a response.
  This change does not automatically alter `TTS_SPEED`.

## Recorded validation — 2026-10-03 (v0.3.1 patch)

- 375 unit tests passed; three hardware/model integrations excluded. Ruff,
  `git diff --check`, and `python -m ciel --version` passed.
- Authenticated NIM requests confirmed the old default model returned HTTP 410
  with retirement time 2026-10-03 09:00 UTC (16:00 Asia/Hovd).
- Updated the local `NIM_MODEL` and package defaults to
  `nvidia/nemotron-3.5-lightning-30b-a3b`. A synthetic greeting through the real
  streaming provider produced its first text in 0.98s and completed in 1.02s.
  This measures provider text, not microphone-to-speaker latency.
- Regressions cover missing/retired models, transient retries, authentication
  failures, cancellation, single backup attempts, empty streams, terminal SSE
  events, and refusing fallback after partial output.
- Empty-interruption recovery is tested before STT delivery and during generation,
  without duplicate history, memory writes, expired-request replay, or restarting
  partial playback. Live microphone acceptance remains pending.
- Reproduce the reported audio case with headphones, `small.en` on CUDA with
  `int8_float16`, Piper, and ambient speech barge-in: ask a question by name, then
  let a brief non-speech sound trigger recording before the answer starts. When
  the new STT result is empty, expect `resumed_after_empty_audio` and one answer
  without repeating the question. Meaningful added speech must still merge.

## Recorded validation — 2026-10-03 (v0.3 development)

- 350 unit tests passed; three hardware/model integrations excluded.
- Regression checks cover speech before generation finishes, cancellation during
  network reads, synthesis and playback, merged additions, partial failures,
  cumulative interruption context, and STT retries under simulated VRAM pressure.
- Ruff and `git diff --check` passed.
- Built `ciel-0.3.0-py3-none-any.whl` and installed it into an isolated target
  directory using the existing environment's dependencies. Verified installed
  package imports, version metadata, module CLI and entry-point CLI with `--help`
  and `--version` from outside the checkout.
- Live microphone, cloud latency, GPU training-load and voice restart acceptance
  remain pending. No stable v0.3 release tag was created.

## Recorded validation — 2026-10-02 (v0.2)

- 128 unit tests passed; three integrations excluded by default.
- Ruff and whitespace checks passed across the repository changes.
- Local Piper synthesis integration passed.
- Release wheel built and dependencies installed into a fresh Python 3.14.7 venv.
- Isolated installed CLI and Piper synthesis checked without the ONNX training
  package; installed dependencies checked with `pip check`.
- Live microphone/speaker acceptance is pending because the execution environment
  has no `/dev/snd`. Do not mark hardware acceptance complete or tag the release
  based only on the automated results.
