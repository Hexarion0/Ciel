# Screen awareness: agreed design and implementation plan

Status: **implemented; live screen/Telegram acceptance pending**. Updated: **2026-10-24**.

The first version lets Ciel look at the user's desktop when asked and discuss it
naturally through voice, the desktop interface, or the paired Telegram account.
It covers games, websites, code, and errors. It explains what is visible and gives
steps the user can follow. Continuous watching and computer control are out of scope.

## Agreed behavior

| Request or situation | Expected behavior |
| --- | --- |
| “Look at my screen.” | Capture all active monitors, give a brief observation, and ask a natural follow-up question. |
| “What does this error mean?” | Take a fresh capture when the request clearly refers to the desktop; ask for clarification when the reference is ambiguous. |
| The user points at something | Include the mouse pointer and its position as context for “this” and “here”; identify the target only when it is visible and clear. |
| “What about now?” | Capture again and compare with the previous available capture. |
| During a screen discussion: “What should I do next?” | Take a fresh capture automatically and answer from the updated screen. |
| Immediately after a screen response: “Try again.” | Repeat the original visual question, including after a model timeout. A request for an earlier image keeps that historical scope. |
| A question explicitly about the previous capture | Reuse that capture while it remains available, preserving its capture time. |
| Small or unreadable text | First enlarge or crop the relevant captured region; ask the user if it remains unreadable. |
| A paired Telegram screen request | Capture the PC desktop and reply with text by default. Send the screenshot only when explicitly requested. |
| Ordinary conversation unrelated to the screen | Do not capture. |

Capture all monitors, using the focused monitor as the main context. Preserve the
layout and mouse position even when an output is left of the primary origin or
uses a different scale. Cropping an image does not authorize changing app zoom,
moving the mouse, clicking, or typing.

## Temporary images and shared conversation

- Keep at most the last **two captures per paired owner**, for up to **15 minutes
  from each capture**. One capture includes all monitors and its derived crops.
- Enforce expiry whenever an image is read and through cleanup while Ciel runs;
  Telegram may keep running independently of the desktop interface.
- Clear images on conversation reset, owner change/unpairing, and capture-service
  shutdown. Do not restore them after restart. Closing one frontend is distinct
  from shutting down the shared capture service.
- Share available captures across the owner's voice and chat conversation, using
  the existing owner and reset boundaries. Retain a brief, timestamped text
  observation for continuity after images disappear.
- If an old image has expired, say so. A clear request to inspect the current
  screen can obtain a new capture; a question about an unavailable old image must
  not silently substitute the current screen.
- Keep raw images, image payloads, and derived crops out of SQLite conversations,
  summaries, durable facts, normal logs, and the repository. Explicit screenshot
  delivery to Telegram or a configured AI provider creates a copy outside Ciel's
  temporary cache; the local expiry rule cannot delete those external copies.

Screen observations provide conversation context. Existing memory authorization
still governs saving or correcting personal facts; text visible in an image does
not grant that authorization.

## Provider and desktop integration

Use existing configured AI services first. Select an image-capable model through
setup; any additional provider requires an explicit setup choice. Do not infer
vision support from the presence of a key, a provider name, or a catalog listing.
Verify the model with a small synthetic image before using personal screenshots.

Implemented integration:

- `providers/base.py` carries typed image attachments, excluded from message
  representations. Text-only serialization remains unchanged; Ollama uses native
  image fields. Generic text routing rejects attachments before any fallback.
- `providers/vision.py` uses one explicitly selected provider/model, verifies it
  with synthetic pixels, checks protected history, and bounds/cancels HTTP calls.
  One labeled contact sheet supports models that accept only one image per request.
- `screen/service.py` resolves clear user requests and visual follow-ups for
  `runtime.py` and `chat/session.py`. Admission, ownership, cancellation, and
  memory authorization remain application decisions. One bounded native crop can
  supply a closer view before asking the user about unreadable details.
- `tool_progress.py` prepares short AI-written intent variations in the background
  and supplies an immediate context-specific fallback when necessary. Screen work,
  initial web retrieval, and app/system actions announce before execution; existing
  AI search speech is reused, and repeated initial research announcements are avoided.
  Progress is transient and grants no action authority. Foreground conversation
  cancels expendable background wording work.
- Telegram sends a bounded PNG only when explicitly requested. Expiry, pairing,
  reset, and cancellation are rechecked before delivery and retry. Desktop/voice
  requests to send an image receive guidance to request it in Telegram.
- `screen/capture.py` reads all active outputs and cursor metadata without audio.
  `screen/cache.py` shares private runtime files between same-owner processes;
  live leases, reset generations, expiry checks, and idle cleanup bound their lifetime.

Configure screen understanding in `ciel setup`, or use `SCREEN_ENABLED`,
`SCREEN_PROVIDER`, `SCREEN_MODEL`, and `SCREEN_TIMEOUT_SECONDS` in `.env`.
The default is the existing NIM connection with
[`nvidia/nemotron-3.5-super-vl-120b-a12b`](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b).
NIM image requests use final-answer mode, matching the ordinary Nemotron text client.
Explicit saved model choices remain selected; change `SCREEN_MODEL` or rerun setup
to select the new default in an existing installation.

On **2026-10-24**, the previous Llama 3.2 11B choice timed out twice on a tiny
synthetic image with a 12-second test deadline. The replacement passed the same
color check in **1.10 seconds**. Through CIEL's actual vision client, two generated
1920×1080 screens returned the correct invented title and panel colors in
**2.31 seconds** including first-use verification, then **1.77 seconds** with
verification cached. One earlier attempt returned HTTP 500. These are small
synthetic samples, not a real-desktop latency or future availability guarantee.
The previous model had passed in 1.44 seconds on 2026-10-22; neither catalog
membership nor an earlier success establishes current inference availability.
No personal screen pixels or conversation history were used in these probes.

An active screen task retains its original question and latest corrected focus
through unclear speech, clarification, pauses, and interruptions. “Again,”
“check again,” and “Can you check again? I meant” take another look immediately.
Other wording reaches the main response model, which can request the native screen
tool or ask a short clarification while retaining the task. The legacy desktop path
(`CONTROLLER_ENABLED=false`) retains its two-second contextual classifier.
The next observation answers
the original question and focuses on relevant changes, briefly acknowledging when
nothing relevant changed. Actual pixels are captured for every fresh observation.

A new task, explicit cancellation, reset, owner change, or new foreground launch
ends the old task. Historical-image retries keep their historical scope. Late
interpretation and capture results cannot restore canceled task state.
Background conversation-summary failures are separate from screen state: failed
batches remain pending with a 30-second retry delay. Complete structured summaries
may exceed per-field concision targets when they fit the hard 4,000-character
total budget; no decisions are truncated to satisfy those targets.

The 2026-10-22 follow-up addresses a real response that denied image access despite
receiving a capture. Visual inference now uses a compact, explicit image-capability
prompt instead of the entire conversation/persona prompt and old assistant claims.
Only bounded earlier user questions provide reference context. A synthetic player
image with a visible invented title was read correctly in **5.14 seconds**, including
first-use model verification. This is a synthetic timing, not a real-desktop latency
guarantee. Capture, image preparation, verification, and inference have separate
privacy-safe latency markers.

The current desktop is Hyprland on Wayland. Installed `grim` supports capturing
the cursor with `-c`; compositor metadata can supply output layout and pointer
position. Start with this backend behind a platform interface. The implementation
must establish the correct desktop environment for the automatic chat service
and report unavailable/locked sessions or denied capture cleanly. It must not
show an old image as if a new capture succeeded.

Mouse metadata and image coordinates must use a common frame. Carry output
offsets, scale, crop transforms, focus, and capture time alongside the image. If
pointer metadata is sampled separately from the pixels, account for movement
instead of claiming an exact target from stale coordinates.

## Implemented stages

1. **Capture contract and temporary cache.** A capture contains monitor
   images/layout, cursor information, timestamps, and owner/reset identity, with
   all-monitor capture, transformed cursor positions, bounded storage, expiry,
   cancellation, and audio-independent access from separate processes.
2. **Image messages and model routing.** Typed attachments and explicit vision
   configuration/capability checks preserve legacy private-history restrictions.
   Availability probing uses synthetic pixels before personal screen inference.
3. **Shared conversation flow.** Voice, desktop text, and paired Telegram requests
   use the same resolver, with fresh contextual follow-ups, previous
   image comparisons, targeted crops, timestamped observations, and expired-image
   responses without creating durable personal facts automatically.
4. **Explicit screenshot delivery.** Telegram image output enforces
   cancellation/owner/reset checks before delivery. The live checks below remain
   necessary before marking desktop acceptance complete.

## Acceptance checks

Use synthetic screenshots and mocked providers for automated regression coverage.
Live desktop acceptance follows explicit user requests and records results without
committing captures or other personal data.

- A requested capture covers every active monitor. Pointer position and target
  remain correct for negative output coordinates, mixed scales, and cropped views.
- “Look at my screen” without a specific question produces a brief observation
  and useful question. Unrelated conversation and instructions visible inside an
  image never trigger capture, tool execution, or memory writes.
- “What about now?” and contextual next-step requests obtain new pixels. Explicit
  questions about a previous image reuse it rather than silently taking a new one.
- Previous/current comparison identifies a visible change. If an image has expired
  or a detail is unreadable, Ciel explains the gap rather than inventing it.
- Only two captures are retained; derived crops share their parent's expiry.
  Expiry, restart, reset, and unpairing leave no accessible old image. Text
  observations retain their times and do not pretend to be current visual access.
- Voice-to-Telegram and Telegram-to-voice follow-ups share the authorized owner's
  context. Telegram can request a capture while the voice frontend is closed,
  without loading STT/TTS or requiring microphone access.
- Telegram replies are text-only by default; an explicit screenshot request sends
  the intended capture. Cancellation, reset, and unpairing suppress stale work
  before provider requests, history writes, and message/image delivery.
- A missing or incompatible model, provider failure, locked desktop, and failed
  capture produce actionable responses without silently changing providers,
  exposing protected history, or presenting an old image as a successful capture.
- Stop/cancel remains responsive during capture, cropping, inference, and delivery.
  Legacy private context remains protected across all image routes.

Completion requires a real request covering all monitors and the pointer, a fresh
follow-up/comparison, and a paired voice/chat handoff with the chosen vision model.
Automated checks alone do not establish that desktop acceptance.

Validation on 2026-10-22: **2,943 automated tests passed**, with five hardware/model
integration tests deselected. Ruff, documentation links, and wheel build passed.
Synthetic coverage includes separate-process sharing/crash cleanup, reset during
inference or error handling, expiry during comparison, crop geometry, request
negations, and explicit Telegram image delivery. Target-desktop metadata checks
confirmed unlocked-session status, two active monitors, cursor position support,
and background session discovery without inherited display variables. These
checks captured no pixels. Actual
all-monitor image quality, pointer targeting, and live paired Telegram handoff
remain acceptance work.

Incoming Telegram image uploads, documents, continuous observation, and clicking
or typing remain separate roadmap work. See the [roadmap](roadmap.md).
