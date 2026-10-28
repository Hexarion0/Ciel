# Smarter CIEL: conversation, research, and useful initiative

Design agreed through the interview on **2026-10-25**. This document describes
proposed work; it is not a release announcement.

Implementation update, **2026-10-28**: the default native path now has shared task
continuity, supported search overviews/partial answers, and a normal 30-second
research budget. See the [conversation/search evaluation](conversation-search-evaluation.md)
for current evidence and limits. The optional observation and initiative work below
remains proposed.

## Desired experience

CIEL should follow what the user means, figure things out, and contribute useful
thoughts without requiring exact command phrases or repeatedly reciting memories.
The user wants all of those improvements. Computer control is a later milestone;
the present upgrade does not add project edits, commands, clicks, or typing.

Keep the current free-provider approach. Choose improvements using observed
conversation quality and response time rather than parameter counts or more model
requests. Reuse the existing speech, memory, search, vision, and cancellation paths
where their contracts fit the new behavior.

## Agreed behavior

| Situation | Expected behavior |
| --- | --- |
| Messy speech or a short correction | Retain the user-provided subject and intended outcome; clarify only the detail that changes the answer. |
| Difficult research | Aim for a useful result in 10–30 seconds. At the normal 30-second limit, return verified partial findings with uncertainty, or explain that the answer remains unverified. |
| Continuing research | The user can explicitly request further investigation; do not silently restart the time budget to keep going. |
| Ordinary conversation | Contribute a grounded opinion, connection, or useful question. Avoid irrelevant memory lists, repeated answers, and invented background thoughts. |
| Optional screen awareness | Observe both monitors only while explicitly enabled, show its state, and provide an easy pause. |
| Unsolicited conversation | Speak only with a relevant contribution, no more than once per 15 minutes. The interval is a minimum gap, not a quota. |
| An ignored check-in | Suppress further initiative until the owner reengages. Reengagement does not erase the 15-minute gap. |
| Watching anime in Zen | Wait for an established viewing pause before offering an anime comment; keep it brief and avoid spoilers. |
| Coding while anime plays elsewhere | A brief coding hint is allowed. Playback on the other monitor does not impose a blanket silence rule. |
| A coding error appears | Wait for a repeated visible error or evidence of stalled progress over roughly two minutes. Give one short concrete hint rather than comment on every error. |
| Screen-derived information | Keep background observations temporary. A confirmed observation can enter the existing memory flow; clear user statements retain their configured learning policy. |
| Starting a new CIEL process | Keep the existing fresh-conversation behavior and saved memories. Do not resume an old transient observation or unfinished unsolicited comment. |

## 1. Establish conversation quality and a shared research deadline

Start here before adding background observations. Recent regression fixes preserve
research objectives, validate answer relevance, and handle corrections, but their
automated coverage does not prove sustained microphone performance.

Create a small replay set that judges task outcomes rather than exact wording:

- A garbled anime title, a typed light-novel correction, and a follow-up about the
  last episode retain the requested anime-to-novel relationship.
- A missing season or episode produces one targeted clarification, not repeated
  requests for an already supplied title or a guessed episode number.
- A genuinely new GPU or weather question ends the anime task; an explicit retry
  retains it. Interruptions and cancel/reset cannot revive stale work.
- A question about CIEL's opinion produces a reasoned response using the current
  topic. An open conversational prompt does not trigger a saved-profile recital
  or a claim of thinking privately between turns.

Track incorrect task choices, unnecessary clarification, unsupported claims,
time to a useful reply, and provider failures. Use representative synthetic text
first, then actual microphone acceptance with the user's normal voice setup.

Research currently has bounded individual lookups and generation steps, while
`AIController.run_web()` explicitly has no overall elapsed-time limit. Introduce
one monotonic deadline for the admitted research request, including its decision,
retrieval, report generation, repair, relevance check, and final validation. Each
stage receives the remaining time instead of starting a new allowance. Reserve
time for a safe final response and cancel outstanding work when the budget ends.

A quick acknowledgment does not count as the useful answer. Measure first useful
text and first useful audio separately; the duration of reading the final answer
aloud is not the research budget. Keep the best already validated partial result.
Never turn raw search excerpts into an asserted answer just because time expired.
An explicit request to investigate further begins a new user-requested operation.

Verify deadline behavior with slow retrieval, slow generation, an unavailable
provider, weak evidence, and repeated leads. Late work must not speak, commit
history, or replace the completed partial answer. Keep cancellation and the
existing private-conversation rules throughout.

## 2. Implement the initiative policy before connecting screen analysis

Build and verify a coordinator using synthetic observations first. It owns
enable/pause state, observation freshness, owner/session identity, cooldown, and
the requirement to wait for reengagement. It must not pretend an internal event
is a new user message or displace an explicitly requested screen task.

Proposed initial defaults are session-scoped enablement, no observation before
explicit enablement, and quiet behavior whenever activity evidence is ambiguous.
These are implementation defaults rather than additional interview decisions.

The coordinator should:

- Yield immediately to foreground conversation and stop/reset/disable controls.
- Recheck current activity, relevance, owner, and permission immediately before
  delivery; drop a suggestion if its evidence became stale.
- Allow at most one analysis job in flight, with bounded retries and failure
  backoff. Do not queue a backlog of comments.
- Begin the 15-minute gap when a comment is actually delivered. Then wait for an
  admitted owner turn before allowing another unsolicited comment. Audio from
  the anime or speech addressed to someone else does not count as reengagement.
- Keep scans silent and preserve the existing mute/pause controls' meaning.

Tests should establish when CIEL remains quiet as carefully as when she speaks.

## 3. Add temporary awareness of both monitors

Reuse the capture backend and selected vision connection, with per-monitor
geometry and focused-activity context. An unchanged local image should not
automatically trigger another expensive model request. Determine a sustainable
capture and inference frequency by measuring desktop load and provider behavior;
the 15-minute speaking gap does not specify an observation frequency.

Use a bounded, timestamped transient observation buffer. Expire and clear it on
disable, reset, owner change, and shutdown. Do not repeatedly call the ordinary
screen answer-and-save path: it is designed for requested, archived conversation.
Unspoken observations must not enter SQLite conversations, summaries, durable
memory extraction, or ordinary reply logs.

User-visible conversation retains the existing conversation/archive contract,
separate from saved personal facts. A spoken check-in must not cause its hidden
observation buffer to be archived or treated as a confirmed current fact. Restart
must not reload old observations as current activity.

Use targeted readable crops when needed for coding text. Do not assume the mouse
is on the coding monitor or that a combined thumbnail preserves an error message.
Retain the original image timestamps and distinguish visible evidence from guesses.

## 4. Enable context-sensitive speaking

Investigate actual playback signals exposed by Zen on the user's Linux desktop.
Playback availability remains unverified. Microphone silence, a still image, or
an idle voice loop cannot establish that a video is paused. Validate pause/resume
with the actual viewing setup before enabling unsolicited anime comments. If the
signal is unavailable or ambiguous, keep those comments quiet; direct questions
still work normally.

Treat coding help as a separate candidate, so a playing video on another monitor
does not suppress it. Require readable, repeated error evidence or positive signs
that the same problem remains unresolved. Reading, thinking, scrolling, or not
typing is not sufficient evidence of being stuck. Recheck that the error still
exists before delivering one brief hint. No automatic computer changes follow.

Both kinds of contribution obey the same cooldown, reengagement, freshness,
foreground-priority, and cancellation rules. Screen observations must not be
promoted to instructions or to user permission.

## Acceptance before everyday use

1. Run the mixed spoken/typed correction cases through the actual desktop entry
   point. Measure correctness and useful-answer delay, not only test counts.
2. Verify the shared research budget, supported partial output, and immediate
   interruption through both voice and paired chat.
3. Enable awareness, switch activity across both monitors, disable it, and verify
   that new capture and stale output stop. Confirm transient data is not persisted.
4. In Zen, play, pause, resume, and switch tabs. An anime comment requires reliable
   pause evidence and must not expose plot developments beyond observed context.
5. Code on one monitor with anime on the other. A repeated unresolved error can
   produce one brief hint; normal editing, reading, and a fixed error do not.
6. Ignore a check-in and wait beyond 15 minutes: CIEL stays quiet. Reengage before
   15 minutes: she responds normally but does not send another unsolicited comment.
7. Run a sustained session and compare desktop load, request counts, provider
   failures, unnecessary interruptions, and useful contributions with awareness off.

## Later: requested computer actions

Desktop control remains part of the intended direction. Design project edits,
commands, clicking, and typing as explicit requested tasks with reviewable results
and verification. The observation feature does not introduce those capabilities.
