# Conversation and search evaluation

## Pre-push review (2026-10-30)

Review added regressions for four edge cases and corrected them before publication:

- AI queries can expand explicit numeric shorthand such as `S2`, `ep. 24`, and
  `volume eleven` using the same numeric-scope parser as retrieval. Different
  installment numbers remain inadmissible.
- Current-result ranking requires the subject in the answer passage or its active
  named section. A different show's latest-episode sidebar cannot borrow relevance
  from an earlier paragraph.
- A redirected page keeps its own content, snippet, publication metadata, and URL.
  Earlier excerpts remain attached to their original source rather than becoming
  evidence for the redirect destination.
- The synthetic evaluator checks the primary query and every batch member against
  model proposals already emitted when the lookup starts. Page reads alone do not
  establish that a model-generated search occurred.

Focused regressions live in `tests/test_search_planning.py`,
`tests/test_search_review_regressions.py`, and `tests/test_conversation_search_evaluator.py`.
The rebuilt wheel matched all 110 packaged Python modules and passed packaged
query, confirmation, source-evidence, and text-depth smoke checks.
Final reviewed suite: **4,420 passed, 5 hardware tests deselected**, 106.34 seconds.
Ruff, Python 3.10 syntax checks, and diff whitespace checks passed. Release artifacts
and diagnostics remain outside the commit.

## Typed response depth (2026-10-30)

Generic saved preferences for short speech could still limit written replies, despite
the TUI's 4,096-token response budget. Text delivery now asks for a developed answer
to substantive questions without requiring a follow-up request for more detail. Those
channel rules follow the saved persona. Current requests for brevity and preferences
explicitly about text remain applicable. Voice delivery and spoken excerpt limits
stay unchanged.

The channel-depth tests include the real desktop callback with a temporary short-reply
persona. Related persona, TUI, streaming, speech-excerpt, and runtime checks passed
137 tests. A synthetic hosted-model check used the same coding question and generic
short-reply preference in both channels: text returned 208 words, voice 45 words.
This is a delivery sample, not a guarantee of response length for every question.
The result is under ignored
`data/diagnostics/search-upgrade/text-depth-live.json`.
Final combined suite: **4,400 passed, 5 hardware tests deselected**, 104.91 seconds.
Ruff and diff whitespace checks passed.

## AI-written search queries (2026-10-30)

The main responding model now supplies the retrieval keywords through `web_search`.
Ciel keeps the original task as a separate objective for ranking, excerpt selection,
and answer verification. Valid model queries pass through with whitespace normalization;
the application no longer prepends the conversation, appends fixed search phrases,
or forces a search before the model handles a typed name correction. Query batches
keep the model's order, with the first admitted query available to single-query providers.

An empty call, acknowledgement, superseded spelling, conflicting explicit platform or
medium, or invented numeric scope returns an error to the same model before any lookup.
The model can rewrite the query in its existing tool loop. Ordinary paraphrases and
aliases remain valid, including "how much" to "price" and "LN" to "novel".
The query-generation path adds no separate planning-model request.

If evidence leaves a requested detail unknown, the model can include `search_queries`
in its existing claims report. One remaining recovery lookup can use those queries
within the original budget. Without a usable model proposal, Ciel keeps the supported
partial answer rather than fabricating a retry query. Cancellation, privacy, and
independent answer checks still apply.

Regressions cover outgoing queries and separate objectives in
`tests/test_ai_search_queries.py`, `tests/test_ai_query_retrieval.py`, and the updated
planning, recovery, correction, and desktop/chat replay suites. The synthetic hosted
evaluation records model-emitted queries and checks that each outgoing lookup came
from those proposals.

Search regression suite: **4,397 passed, 5 hardware tests deselected**, 104.88 seconds.
Hosted checks used synthetic dialogue and fixture sources. The corrected-title case
passed in 3.691 seconds and the two-part request in 9.691 seconds. The confirmation
case generated valid queries but its answer timed out on the first attempt; a retry
passed in 3.432 seconds with `Tensura anime mod Minecraft` as the outgoing query.
These samples establish query ownership and context retention, not live web accuracy.
Diagnostics are under ignored `data/diagnostics/search-upgrade/ai-query-*.json`.

## Search-offer confirmation — 2026-10-29

The reported Minecraft conversation reproduced two errors before the fix. Subject
correction replaced the opening greeting or the word "know" instead of the title.
After the assistant offered a lookup, "Yes, sure" became the literal search objective.

Subject correction now skips bounded conversational framing and retains the requested
mod and Minecraft scope. An immediate acceptance of one grounded public-search offer
keeps the corrected user request. Both desktop and chat pass that request to the native
tool even if the model supplies the acknowledgement, a stale spelling, or no query.
The projection does not copy facts, names, or numeric scope from assistant replies.

`tests/test_search_offer_context.py` covers projection and native keywords.
`tests/test_conversation_search_replays.py` runs the sequence across voice, keyboard,
and chat with automatic search disabled, plus reset, cancellation, ambiguous offers,
and private context. The hosted-model evaluation includes
`accepted_corrected_lookup_offer`, using synthetic reference text and temporary memory.
The configured model passed that case in 4.079 seconds with one fixture lookup and
three model requests. The recorded lookup retained Tensura, Minecraft, and the mod
question; it contained neither the stale spelling nor the acknowledgement. This checks
confirmation handling with a hosted model, not live web results or microphone accuracy.
Full regression suite: **4,373 passed, 5 hardware tests deselected**, 119.38 seconds.
Ruff and diff whitespace checks passed.

## Retrieval upgrade — 2026-10-29

The search failures were reproducible at several boundaries: a corrected subject could
lose a platform constraint, a contextual query could hide useful search vocabulary,
keyword ranking could promote an unrelated adaptation mapping, and short excerpts could
suppress the page read needed to answer. The report checker also rejected valid facts
whose version or season appeared in a source heading rather than the selected paragraph.

The native tool now accepts a single query, a bounded query batch, or the current owner's
goal. CIEL retains that goal and sends up to three distinct retrieval probes, preserving
constraints and rejecting invented numeric scope. Equivalent admitted batches reuse
evidence; distinct leads can run. Parallel uses its current anonymous MCP endpoint and
free page-extraction tool. See the provider's [MCP documentation](https://docs.parallel.ai/integrations/mcp/quickstart)
and [query guidance](https://docs.parallel.ai/search/best-practices).

Ranking considers answer-bearing passages, the requested subject, and local installment
scope. Missing relations can trigger extraction of at most two promising pages within
the existing page allowance. Compound evidence must cover the requested requirements.
Source-owned passage IDs avoid model copying errors. Paragraphs retain hard-wrapped
lines and bounded overlapping windows can join a condition with its consequence or a
release heading with its date. Headings can attribute their own facts; sibling sections
and standalone omission markers reset that scope. Earlier source spans survive page
refreshes and catalog changes. The exported evidence uses at most 36,000 encoded
characters so native tool limits cannot discard a large result. Exact quotations,
numbers, source identity, and independent relevance checks remain required.

One additional scoped read can recover a missing requested detail after the initial
lookup. CIEL keeps supported earlier claims and their source attribution if recovery
fails or returns only another partial answer. Recovery shares the original 30-second
budget, read/verification reserves, cancellation, and privacy controls. It does not run
on retained evidence alone or after multiple reads have already occurred.

Regressions live in `tests/test_goal_retrieval.py`, `tests/test_search_planning.py`,
`tests/test_search_recovery.py`, `tests/test_web_evidence_passages.py`, and the existing
transport, correction, runtime, and privacy suites. Real public probes use eight
synthetic questions about anime, Python, SQLite, OpenClaw, and Godot, with temporary
memory databases and no personal conversation context. Their raw diagnostic records
remain under ignored `data/diagnostics/search-upgrade/`; a citation count alone does
not prove that a response answers the question.

The public probes exposed failures that fixture-only evaluation had missed. Recorded
reruns after the corresponding fixes returned:

| Public question | Observed result | Seconds |
| --- | --- | ---: |
| TaskGroup failure and remaining tasks | Cancellation, waiting, and exception grouping, with an official Python source | 10.656 |
| Stable Godot and Linux availability | Both requested parts, citing official pages | 11.176 |
| Stable Python release | Version and release date from Python.org | 8.640 |
| SQLite concurrency | Concurrent readers, one writer, and WAL behavior from SQLite.org | 10.928 |
| OpenClaw search and a free option | Configured search and a key-free self-hosted option from official docs | 14.729 |
| Latest Tensura episode | Supported episode/date; remaining details stayed partial | 21.431 |

These are diagnostic samples from several stages, followed by targeted reruns after
fixes, not an eight-out-of-eight accuracy or availability claim. The exact novel
continuation from the supplied "Hero Awakens" premise remained unverified. Some
probes encountered hosted-model timeouts or weak source choices. No series-specific
episode, date, or volume answer was added to application code.

Final ordinary suite: **4,300 passed, 5 hardware tests deselected**, 115.90 seconds.
Ruff, whitespace checks, wheel build, packaged imports, source-to-wheel comparison,
and packaged query/evidence smoke checks passed.

## Earlier follow-up checks — 2026-10-28

The reading-continuation regression is covered in `tests/test_reading_followup.py`.
It exercises the supplied latest-episode → named-episode novel continuation across
desktop voice, desktop typing, and headless chat. The lookup receives the new reading
goal, not both full questions concatenated. Tests also cover unrelated subjects,
cancellation, raw transcript preservation, and avoiding redundant episode questions.
The final ordinary suite passed **4,067 tests**, with **5 hardware tests deselected**;
focused follow-up and related checks passed **87 tests** on 2026-10-28.
Two real public searches used the corrected goal but did not establish a supported
adaptation mapping. These samples verify query continuity, not a successful reading-point
answer. Diagnostics are under `data/diagnostics/reading-followup/`.

The follow-up latest-episode regression passed **4,051 ordinary tests**, with
**5 hardware tests deselected**, on 2026-10-28. See `tests/test_episode_search_repair.py`
and the [release checks](release-checks.md) for the spoken false-start, failure
explanation, schedule quotation, and live reproduction checks. The acceptance
results below record the preceding task/search upgrade.

The current-task and native search upgrades are exercised through the desktop
runtime and headless `ChatSession`. Tests use invented conversations and sources;
they do not read personal conversation archives or use microphone recordings.

## Behavior covered

| Requirement | Evidence |
| --- | --- |
| Corrections retain title, episode, season, and requested outcome across voice/text | `tests/test_conversation_search_replays.py`, `tests/test_search_correction_runtime.py` |
| Retry uses the full task; continue receives interruption context and retained evidence | `tests/test_task_context.py`, desktop/chat replay tests |
| Wait permits resumption; cancel/no-search do not revive old work | Desktop/chat control replays and `tests/test_conversation_controls.py` |
| New topics, reset, owner changes, and private context remain separate | Task-context tests, `tests/test_dialogue_task_context.py`, `tests/test_research_runtime_privacy.py` |
| Broad named searches receive an overview; compound questions keep all parts | Replay cases, model evaluation below, `tests/test_web_goal_check.py` |
| Partial findings name only missing details actually requested by the user | `tests/test_main_tools.py`, `tests/test_web_answer_gaps.py` |
| Repetition and tool limits reserve one final synthesis without more tool execution | `tests/test_unified_agent.py` and bounded-search replays |
| New sources survive a full source allowance; later lookup failure retains evidence | `tests/test_main_tools.py` |
| A shared native research deadline reserves answer/checking time and rejects late output | Slow-read and slow-answer tests in main-tools/unified-agent tests |
| Raw turns, full typed text, brief speech, and permission checks remain intact | Replay assertions and existing channel/privacy regression suites |

The task projection comes from bounded, current owner dialogue. It is prompt data,
not an extra model decision, durable memory, or permission to act. Trusted output
format instructions are sent separately from untrusted page bytes. Retrieved
claims still pass quotation, scope, source, and independent relevance checks.

Native research allows bounded read rounds and at most one final synthesis with
tools disabled. Its 30-second budget begins when the responding tool session is
created, including the initial model turn for an explicit search. Read work stops
with eight seconds reserved; answer generation leaves four seconds for checking.
Automatic lookups admitted later use that same original start time. Unrelated
ordinary conversation is not assigned this research deadline. The legacy controller
diagnostic path is not the default native tool path and retains its old bounds.

## Reproduce

```sh
pytest tests/test_conversation_search_replays.py tests/test_task_context.py \
  tests/test_dialogue_task_context.py tests/test_main_tools.py \
  tests/test_unified_agent.py tests/test_web_answer_gaps.py
python scripts/evaluate_conversation_search.py
python scripts/evaluate_conversation_search.py --live
```

The default script writes an offline case manifest. `--live` sends only the configured
synthetic dialogues and reference excerpts to the configured model; actual web,
screen, audio, and personal-history access are not used. It checks the objective
at the real lookup boundary, useful supported output, citations, unchanged raw
turns, and the deadline. Each case permits six logical foreground model requests
and 30 seconds, with 240 seconds across the run. Provider fallback may add HTTP
attempts within a logical request. Reports include the synthetic model output for
diagnosis and are written under ignored `data/diagnostics/`.

These checks establish application behavior and a small hosted-model sample.
They do not establish microphone accuracy, live website compatibility, continuous
provider availability, or sustained desktop acceptance.

## Recorded acceptance — 2026-10-28

- Complete ordinary suite: **4,026 passed, 5 deselected**, 90.33 seconds.
- Focused checks listed above: **177 passed**, including **39 desktop/chat replays**.
- Hosted model gate: **8/8 passed**, 3.373–12.250 seconds, one fixture lookup and
  three logical model requests per case. Normal model: Super 3.5 VL 120B.
- A desktop playback interruption followed by typed continuation preserves retrieved
  evidence and citations without rerunning the completed search. The broader related
  interruption, memory, privacy, and ambient checks passed 210 tests.
- One real Parallel search returned a cited Python overview from official Python
  documentation in 7.043 seconds. Temporary paired-session smoke checks verified
  a Telegram repair is used in the next voice retry and unpairing removes access.
- Ruff, whitespace checks, isolated wheel build, and packaged import/behavior smoke
  checks passed. Results are in `data/diagnostics/task-search/pytest.log`,
  `replays.log`, `live-final.json`, `real-search.json`, `paired-smoke.log`,
  `build.log`, and `wheel-smoke.log`.

Earlier live probes exposed a format instruction buried inside a tool result:
the model's useful prose was correctly rejected by the strict report parser.
The current path supplies trusted report rules as system context after retrieval;
untrusted evidence remains tool data. Other probes encountered provider/checking
timeouts. The final gate passed without changing quote or relevance approval to
accept unchecked answers.
