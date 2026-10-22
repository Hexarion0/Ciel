# Screen awareness: agreed design and implementation plan

Status: **design agreed, implementation pending**. Updated: **2026-10-22**.

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

Current integration gaps:

- `providers/base.py` has text-only messages. Add typed image attachments and
  provider-specific serialization without turning images into transcript text.
- `providers/manager.py` can fall back across text models. Image requests need
  capability-filtered routing and the existing protected-history routing rules.
  Do not silently send an image to an incompatible model or another service.
- `controller.py`, `runtime.py`, and `chat/session.py` need a shared screen-request
  path. Keep application admission, owner checks, cancellation, and memory policy
  outside model-generated decisions.
- Telegram transport currently handles text. Add explicit outgoing screenshot
  delivery without bundling incoming image/document uploads into this milestone.
- Voice and Telegram run in separate processes. Provide one owner-scoped,
  audio-independent capture/cache interface that works with the logged-in desktop
  session and has explicit lifetime, expiry, and reset behavior.

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

## Implementation order

1. **Capture contract and temporary cache.** Define a capture containing monitor
   images/layout, cursor information, timestamps, and owner/reset identity. Add
   all-monitor capture, transformed cursor positions, bounded storage, expiry,
   cancellation, and audio-independent access from separate processes.
2. **Image messages and model routing.** Add attachment serialization and explicit
   vision configuration/capability checks. Preserve legacy private-history
   restrictions and failure behavior. Verify a configured model with a synthetic
   image; do not use a real screen for availability probing.
3. **Shared conversation flow.** Connect voice, desktop text, and paired Telegram
   requests to the same resolver. Support fresh contextual follow-ups, previous
   image comparisons, targeted crops, timestamped observations, and expired-image
   responses without creating durable personal facts automatically.
4. **Explicit screenshot delivery and acceptance.** Add Telegram image output,
   enforce cancellation/owner/reset checks before delivery, document setup and
   limitations, then perform the live checks below before marking the feature done.

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

Incoming Telegram image uploads, documents, continuous observation, and clicking
or typing remain separate roadmap work. See the [roadmap](roadmap.md).
