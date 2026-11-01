# CIEL Chat and History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the temporary reply bubble with a welcoming, readable chat interface,
keeping CIEL's animated orb and adding safe archive browsing and explicit continuation.

**Architecture:** Keep `ciel.web.DesktopWebUI` as the compatibility facade. Extract
browser assets, safe rendering, archive admission, and active selection into focused
units. Route switches through the existing runtime cancellation and lifecycle controls.

**Tech Stack:** Existing Python/SQLite service, vanilla browser client, pytest/Node.
**Spec:** [Approved design](../specs/2026-10-30-web-interface-upgrade-design.md).

## Shared requirements and files

Apply the [umbrella plan's global constraints and design contract](2026-10-30-web-interface-upgrade.md).
No personal media, credentials, or downloaded speech models belong in source control.

| File | Responsibility |
| --- | --- |
| `src/ciel/web.py` | Existing UI facade and bounded state; wire extracted routes |
| `src/ciel/webapp/__init__.py` | Package boundary |
| `src/ciel/webapp/http.py` | Explicit static/API route dispatch and body limits |
| `src/ciel/webapp/conversations.py` | Archive catalogue, opaque handles, paginated history |
| `src/ciel/webapp/selection.py` | Active conversation and monotonic activation revision |
| `src/ciel/memory/archive_scope.py` | Shared archive admission and protected-history classification |
| `src/ciel/memory/tools.py` | Reuse factored admission without changing recall behavior |
| `src/ciel/memory/bridge.py` | Publish active selection and revision across channels |
| `src/ciel/memory/store.py` | Atomic active-selection write and consistent paired-record read |
| `src/ciel/runtime.py` | Rebind manager/dialogue/tool state on explicit switches |
| `src/ciel/chat/session.py` | Respect selection revision before commits and delivery |
| `src/ciel/web_assets/index.html` | Semantic application shell |
| `src/ciel/web_assets/chat.css` | Desktop/mobile layout, tokens, interaction states |
| `src/ciel/web_assets/chat.js` | State polling, requests, timeline, drawer, history |
| `src/ciel/web_assets/render.js` | Safe text/Markdown/source/code rendering |
| `src/ciel/web_assets/orb.js` | Existing procedural orb moved intact |

## Review focus

Test streaming scroll/focus preservation, safe links/raw HTML, archive ownership,
switching A→B→A, and stale paired-chat commits in the tasks that own those boundaries.

## Task 1: Split and serve packaged browser assets

**Files:** `web.py`, new `webapp/http.py`, `web_assets/chat.css`, `chat.js`, `orb.js`;
`tests/test_web.py`, `tests/test_web_frontend.py`, and `tests/frontend/orb_animation.cjs`.

**Interfaces:** `load_web_asset(path: str) -> tuple[bytes, str] | None` serves only
an explicit asset map. Preserve the current `DesktopWebUI.start(submit, *, port)`
call signature. `createOrb()` keeps its current browser-facing interface.

- [ ] Add tests that GET `/assets/chat.css`, `/assets/chat.js`, `/assets/orb.js`, and
  `/assets/render.js` returns packaged content; traversal and unknown paths return 404.
- [ ] Run `.venv/bin/pytest tests/test_web.py tests/test_web_frontend.py`; confirm
  the new asset assertions fail before implementation.
- [ ] Move existing styles/script/orb into the named files and add only explicit
  same-origin asset routes. Update the Node harnesses to load actual packaged files.
- [ ] Run the focused tests and `node tests/frontend/orb_animation.cjs`; preserve
  motion, mute, hidden-tab, reduced-motion, resize, fallback, and frame-budget checks.
- [ ] Review the diff as an asset extraction checkpoint; keep existing behavior intact.

## Task 2: Safe complete message rendering

**Files:** `web_assets/render.js`, `chat.js`, `tests/frontend/chat_render.cjs`,
`tests/test_web_frontend.py`.

**Interfaces:** `renderMessageContent(node, text)` and `renderMessageSources(node, sources)`
use DOM nodes, not raw `innerHTML`. `renderTimeline(messages, {followLatest})` keys
nodes by message identity and updates only changed content.

- [ ] Add failing assertions for user and assistant history, paragraphs, lists,
  fenced code/Copy, safe external links, sources, literal `<script>` text, unsafe
  URLs, and an unclosed code fence during streaming. Copy must preserve code newlines.
- [ ] Run `node tests/frontend/chat_render.cjs`; confirm the new behavior fails.
- [ ] Implement a bounded renderer for those specified Markdown structures using
  `textContent` and explicit elements. Reuse source URL validation; do not add a CDN
  dependency or parse raw HTML. Retain node identity for unchanged source links.
- [ ] Run the renderer and Python frontend tests; verify hostile input stays inert,
  source focus survives polling, and full message text remains available to copy.
- [ ] Review supported formatting and document its limits beside the renderer tests.

## Task 3: Welcoming chat layout with persistent animated orb

**Files:** `index.html`, `chat.css`, `chat.js`, `tests/frontend/web_ui.cjs`,
`tests/test_web_frontend.py`.

**Interfaces:** The existing state payload still drives `render(snapshot)`. Keep
the composer/Stop/mute controls operational, add a timeline, history drawer,
speaker control placeholder binding, New messages action, and archive browsing state.

- [ ] Replace the old assertion forbidding a conversation feed with failing checks
  that all loaded messages are readable without automatically playing their audio.
  Assert drafts survive failed/late submissions and scrolling away from the bottom
  prevents forced scrolling. Opening/closing the mobile drawer restores focus.
- [ ] Run the frontend tests and confirm the new timeline/layout behavior fails.
- [ ] Apply the umbrella plan's exact tokens, dimensions, and typography. Keep the
  120px desktop/56px mobile orb visible with all existing states. Show recording,
  thinking, speaking, interrupted, offline, empty, and retry states in plain language.
- [ ] Run frontend tests, then inspect desktop/narrow/phone layouts with long text,
  code, sources, streaming output, reduced motion, and an open mobile keyboard.
- [ ] Fix observed layout/focus issues and capture review images using synthetic data.

## Task 4: Owner-admitted archive catalogue and pagination

**Files:** new `memory/archive_scope.py`, `webapp/conversations.py`, `memory/tools.py`;
`tests/test_web_conversations.py`, existing memory privacy/recall tests.

**Interfaces:** `admitted_archive_sessions(store, active_scope: tuple[str, ...], *,
include_private: bool) -> tuple[str, ...]`; `ConversationCatalog.list(before: int | None,
limit: int = 30) -> dict`; `ConversationCatalog.read(handle: str, before: int | None,
limit: int = 50) -> dict`. Handles are opaque and server-admitted for the current owner.

- [ ] Add failing tests that earlier same-owner desktop launches and the currently
  paired Telegram owner's matching launches appear once; another owner, an arbitrary
  session string, a changed binding, and forged cursors cannot expose conversations.
  Assert 30-row catalogue/50-turn defaults, chronological display, bounded titles,
  stable pagination, and safe display of protected archives without cloud inference.
- [ ] Run `.venv/bin/pytest tests/test_web_conversations.py` and confirm the new APIs fail.
- [ ] Factor admission/privacy classification out of `MemoryTools._past_dialogue`,
  preserving its existing retrieval behavior. Group only proved same-owner launch
  projections. Add explicit ownership records where existing archive metadata cannot
  prove grouping. Treat ambiguous historical projections as isolated.
- [ ] Run the new tests and existing memory-tool privacy tests, including transitive
  private projections and direct private transcripts without summaries.
- [ ] Review SQL bounds and owner checks; archive reads must not call a model or mutate history.

## Task 5: Explicit active selection and cancellation revision

**Files:** new `webapp/selection.py`, `memory/bridge.py`, `memory/store.py`, `runtime.py`,
`chat/session.py`;
`tests/test_web_selection.py`, `tests/test_session_runtime.py`, conversation bridge tests.

**Interfaces:** `ActiveSelection(handle: str, revision: int)`;
`ConversationSelection.current() -> ActiveSelection`;
`ConversationSelection.activate(handle: str | None, expected_revision: int) -> ActiveSelection`.
`None` creates a fresh chat. An unknown handle or stale revision fails without a switch.
Add `MemoryStore.select_desktop_session(session_id: str, expected_revision: int) -> int`
to update the existing active-desktop ID and a separate activation revision in one
SQLite transaction; preserve the existing string ID for legacy readers. New guarded
readers load both records in one query. Never reset the revision when reactivating a chat.

- [ ] Write failing tests for read-only archive browsing, explicit Continue/New chat,
  fresh process startup, saved-fact preservation, lifecycle flushing, and A→B→A:
  the first A request cannot write/speak after the third activation. Test a Telegram
  worker refreshing context and rejecting old-revision results before committing.
- [ ] Run `.venv/bin/pytest tests/test_web_selection.py` and confirm the new behavior fails.
- [ ] Implement serialized selection and publish an independent activation revision.
  Cancel current output/recording/initiative and pending confirmations; close the old
  lifecycle and rebind conversation, dialogue, screen, and tool context. Clear transient
  images/fragments without deleting history. Have all completion guards check revision.
- [ ] Run selection, runtime, bridge, screen lifecycle, and chat tests. Continue a
  protected archive and verify it retains local routing before its first new request.
- [ ] Review every closure bound to the old manager; stale callbacks must not operate
  through newly rebound globals or create duplicate lifecycle workers.

## Task 6: Connect history controls and incremental web state

**Files:** `web.py`, `webapp/http.py`, `webapp/conversations.py`, `webapp/selection.py`,
`web_assets/chat.js`; `tests/test_web.py`, `tests/test_web_conversations.py`, frontend harnesses.

**Interfaces:** GET `/api/conversations?before=&limit=`; GET `/api/conversations/{handle}?before=&limit=`;
POST `/api/conversations/activate` with `{handle, expected_revision}`; GET `/api/state?after=`
includes active handle/revision and incremental message changes. Keep bounded legacy
snapshots for existing internal callers.

- [ ] Add failing tests that sidebar browsing never switches voice context, Continue
  does, New chat creates a fresh handle, draft text survives browsing, and stale
  selection responses cannot overwrite a newer view. Test owner revocation mid-read.
- [ ] Run the web catalogue/selection/frontend tests and confirm the missing wiring fails.
- [ ] Wire routes and the drawer, Continue/New chat buttons, pagination, return-to-active
  affordance, and message cursor reconciliation. Keep cached browser history scoped
  to the authenticated owner and clear it on revocation.
- [ ] Run all web tests and inspect switching on two browser clients. A background
  selection change must be visible and must not auto-send a draft to another chat.
- [ ] Update the desktop-interface README section with the actual completed behavior.

## Plan completion

- [ ] Run the umbrella validation relevant to chat/history and review the final diff.
- [ ] Record actual tests and browser checks. Continue with the local-phone plan;
  phone transport/voice and image input are still required for the full upgrade.
