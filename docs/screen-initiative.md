# Optional screen observation and coding hints

CIEL can periodically look at the desktop and offer one brief hint about a repeated,
readable coding error. Observation is **off at every launch**. It uses the existing
Hyprland capture backend and the image provider/model selected in setup. When a
cloud image model is selected, enabling observation permits sending screen images
to that provider; protected local conversations still require a local image model.

## Controls

Use these in the desktop terminal or local web input:

| Command | Result |
| --- | --- |
| `/observe on` | Enable observation for this desktop session |
| `/observe pause` | Stop new captures and cancel unfinished background work |
| `/observe resume` | Resume observation with fresh evidence |
| `/observe off` | Disable observation and clear temporary evidence |
| `/observe` or `/observe status` | Inspect the current state |

Directly address CIEL by voice: “Ciel, start watching my screen,” “Ciel, pause
watching my screen,” or “Ciel, stop watching my screen.” Uncertain transcription,
quoted requests, and speech addressed to someone else do not authorize observation.
The desktop web page shows the current state; `/status` includes it in the terminal.
These controls are local desktop controls, not remote Telegram commands.

`/stop` and spoken stop/wait controls pause observation. Microphone mute suspends
background capture/output while retaining the enabled setting. Typed-only sessions
with an unavailable microphone can still show hints. Owner/pairing changes, history
reset, and shutdown invalidate outstanding work; a new process never restores opt-in.

## When CIEL speaks

- Check the active monitors about every 20 seconds while idle. Unchanged views do
  not cause repeated image-model requests; one scan permits at most one closer view.
- Require the same readable error to remain visible across roughly two minutes.
  Normal editing, reading, an example on a webpage, or inactivity alone is insufficient.
- Verify the error again using a new image-model analysis, then capture once more
  immediately before output. Any pixel change in the checked error area drops the hint.
- Yield to voice/typed requests and newly accepted paired Telegram messages. Suggestions
  expire after 15 seconds, with no backlog. A failed scan uses bounded retry backoff.
- Leave at least 15 minutes between delivered hints and wait for an accepted owner
  turn after each hint. Ignoring a hint keeps CIEL quiet even after the interval expires.
- During uninterrupted observation, do not repeat a hint for the same error unless
  a later scan first sees it disappear. Explicit reenablement starts fresh evidence.

The 20-second scan interval is an initial implementation default, not a measured
desktop performance guarantee. Captures are local; changed views and verification
can make image-model requests. Cooldown and ignored-hint suppression stop scanning.
The feature does not click, type, edit files, execute commands, or save observations
as personal facts. Anime comments remain disabled until actual player pause signals
have been verified on the target desktop.

## Data and validation

Unspoken evidence stays in process memory. Background scans bypass the requested
screenshot cache, SQLite conversation history, memory extraction, and ordinary
reply logs. Only a delivered hint is archived as an assistant turn. No synthetic
user turn is created. Screen text is untrusted context, never instructions or consent.

Automated coverage in `test_screen_initiative.py`, `test_background_output.py`, and
`test_session_runtime.py` uses synthetic pixels, a controllable clock, and fake image
responses. It covers silence, repeated errors, unchanged-view deduplication, closer
views, foreground priority, mute/stop, stale/fixed evidence, owner/reset changes,
privacy routing, retry backoff, reengagement, and shared desktop controls.

Validation on 2026-10-30: **4,502 tests passed, five hardware/model integration
tests deselected**, in 110.82 seconds. Ruff, whitespace checks, and Python 3.10
syntax checks passed. A synthetic local web preview verified the visible off/on/paused
indicator and command submission; it captured no personal desktop pixels. The optional
wheel build could not run because this environment lacks the Hatchling build backend.

Target-desktop acceptance is still required: use the configured speech/model setup,
enable observation, reproduce an error on either coding monitor for two minutes,
check that fixing it suppresses the hint, interrupt delivery, pause/resume, then
measure a sustained session's load and image-request counts. Also verify that an
ignored hint remains quiet and that restart requires a new explicit opt-in.
