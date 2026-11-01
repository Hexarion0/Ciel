# CIEL Web Interface Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved welcoming chat interface on PC and local-network phones,
with CIEL's animated orb, resumable conversations, spoken replies, and image input.

**Architecture:** Keep the existing Python assistant and vanilla browser client. Add
explicit owner-scoped conversation selection, structured browser requests, authenticated
local HTTPS, and temporary image/audio transport around the existing model/STT/TTS paths.

**Tech Stack:** Python 3.10+, SQLite, existing NumPy/Pillow/httpx, HTML/CSS/JavaScript,
browser microphone/Web Audio APIs, pytest, Ruff, and Node frontend regression harnesses.

**Spec:** [Approved design](../specs/2026-10-30-web-interface-upgrade-design.md).

## Execution and existing work

The owner approved the design and visual direction. Execution-method selection and
review of these implementation plans remain pending. Recommend native execution in
this chat: the session, device, and cancellation interfaces are tightly connected.
Do not spawn workers before an execution method has been selected.

The working tree already contains conversation/search and screen-initiative changes.
Record the starting diff and untracked source list before implementation. Preserve
those changes. Review exact diffs before staging; do not sweep unrelated work into
a feature commit. Use `codex/web-interface-upgrade` for a new feature branch.

## Global constraints

- Preserve the animated orb and its listening/thinking/speaking states.
- The interface is welcoming, calm, and cozy, using Bear as a spacing/type reference.
- PC and phone work on the local network; the PC hosts CIEL and remains on.
- Phone uses tap-to-talk; each reply plays on the requesting device.
- Typed and voice requests receive spoken replies by default on both devices.
- Show full written answers; speak an excerpt for long answers and provide Read full answer.
- Process startup creates a fresh conversation. Browsing history does not change it.
- Continue chat changes the shared active conversation explicitly.
- Accept images; PDFs and durable raw-image storage remain out of scope.
- Protected history, owner boundaries, direct-address action authority, and cancellation remain intact.
- Observation still refers to the PC screen and starts disabled at every process launch.
- Python code uses four spaces, type hints/docstrings, and a 100-character line limit.
- Package frontend assets locally, without CDN dependencies.

## Plans and delivery order

| Order | Plan | Working outcome |
| --- | --- | --- |
| 1 | [Chat, orb, and conversation history](2026-10-30-web-chat-history.md) | Readable persistent chat, a visible animated orb, owner-scoped history browsing, Continue/New chat |
| 2 | [Local phone connection and voice](2026-10-30-local-phone-voice.md) | Paired HTTPS phone access, phone microphone input, CIEL speech on the right device |
| 3 | [Image conversations and acceptance](2026-10-30-web-image-conversations.md) | Upload/preview images, grounded visual answers, full cross-device validation |

All three plans belong to the approved upgrade. Do not present Plan 1 as completion
of phone voice or image conversations. Later plans depend on the named interfaces
from earlier plans, not on undeclared globals or callbacks.

## Design contract

Use these six base tokens, checking contrast against the actual state colors:

| Token | Value | Use |
| --- | --- | --- |
| Base | `#1C1B20` | Main background |
| Surface | `#26242B` | Sidebar and composer |
| Raised | `#312D37` | Selected items and active controls |
| Text | `#F1EDE8` | Warm primary text |
| Muted | `#B4AEBD` | Secondary text |
| Accent | `#B8A0D9` | Lavender focus, selection, and voice accents |

Use an offline humanist system text stack, sentence case, 16px body text, 1.65 line
height, and a 72-character reading width. Code uses a monospace stack and horizontal
scrolling. Keep text left-aligned. Use spacing and hierarchy to group content,
with restrained borders and no repeated dashboard-card layout.

Desktop uses a 248px history sidebar and a flexible conversation column. The voice
area lives in the sidebar with a visible 120px orb; it never replaces the timeline.
Below 760px the sidebar becomes a focus-managed drawer, and the 56px orb remains
visible in the conversation header beside voice state. Keep 44px touch targets.
Use safe-area insets and dynamic viewport units for the composer and mobile keyboard.

Preserve the existing `createOrb()` drawing and performance limits. Change sizing
and tokens without inventing a different animation or adding background effects.
The visual signature is CIEL's orb against warm charcoal, with a quiet reading area.

```text
PC
┌ History ──────────┬ Active conversation ─────────────────────┐
│ New chat          │ title / connection / return to active    │
│ Recent chats      │                                         │
│                   │ user and CIEL messages, code, images     │
│                   │                                         │
│ Animated orb      │ message composer / attach / mic / send   │
│ voice controls    │ Stop / sound / PC observation status     │
└───────────────────┴─────────────────────────────────────────┘

Phone
┌ menu / title / small animated orb / voice state ┐
│ scrollable conversation                        │
│                                                │
│ message composer / attach / mic / send          │
│ Stop / sound                                   │
└────────────────────────────────────────────────┘
```

## Review focus

1. A→B→A conversation changes must invalidate the first activation's pending work;
   test this in Plan 1 Task 5 and Plan 2 Task 4.
2. Both devices and multiple tabs may be open; audio must play once on its origin,
   with no automatic replay after reconnect. Test in Plan 2 Tasks 5–7.
3. Mobile keyboards, long code, and scrolling during streaming must preserve the
   draft and reading position. Test in Plan 1 Task 3 and final real-browser checks.
4. An attachment can expire while the owner browses history or a provider is busy;
   show expiry and reject late image use. Test in Plan 3 Tasks 1–3.
5. A denied browser permission or blocked autoplay must leave text chat and Stop
   usable, with honest recovery controls. Test in Plan 2 Tasks 6–7.

## Final verification

- [ ] Complete each child plan's focused checks before broadening validation.
- [ ] Run `.venv/bin/pytest` and `.venv/bin/ruff check src tests` after integration.
- [ ] Check changed Python modules against Python 3.10 syntax and run `git diff --check`.
- [ ] Build a wheel with the configured Hatchling backend and inspect its asset list.
  If Hatchling is absent, prepare an isolated build environment; record the actual outcome.
- [ ] Inspect desktop and phone layouts in a browser with realistic conversations
  and fix clipping, focus, overlap, missing controls, and mobile keyboard behavior.
- [ ] Verify real local HTTPS trust, pairing, microphone input, CIEL voice output,
  interruption, history continuation, image input, and reconnect on the owner's phone.
  Report device/model acceptance separately from synthetic tests.
