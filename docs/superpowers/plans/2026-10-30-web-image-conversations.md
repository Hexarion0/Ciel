# CIEL Web Image Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the paired owner upload images from PC or phone, discuss them with
CIEL, and read/hear the answer through the upgraded chat interface.

**Architecture:** Reuse the prior plans' authenticated device/request/selection
contracts and temporary media store. Validate and re-encode user images, then use
the selected verified VisionClient without granting permission to capture the PC.

**Tech Stack:** Existing Pillow/VisionClient, Python temporary media, browser File/Blob
and paste APIs, pytest and Node.
**Spec:** [Approved design](../specs/2026-10-30-web-interface-upgrade-design.md).

## Global constraints and files

Apply the [umbrella requirements](2026-10-30-web-interface-upgrade.md). Complete the
chat/history and local-phone request/media interfaces first. Images are temporary;
PDFs and durable raw-image storage remain outside this upgrade.

| File | Responsibility |
| --- | --- |
| `src/ciel/webapp/images.py` | Image validation, re-encoding, attachment scope and inference |
| `src/ciel/webapp/media.py`, `requests.py`, `http.py` | Reuse authenticated media handles and request admission |
| `src/ciel/runtime.py` | Explicit image-turn handling, delivered text/history and speech routing |
| `src/ciel/providers/vision.py` | Reuse selected model, verification, cancellation and local-only checks |
| `src/ciel/web_assets/attachments.js` | File/paste previews, upload/removal/retry and expiry |
| `src/ciel/web_assets/chat.js`, `chat.css`, `index.html` | Composer attachment state and message placeholders |
| `tests/test_web_images.py`, `tests/frontend/web_attachments.cjs` | Image and client regression boundaries |

## Review focus

Test decode bounds before expensive allocation, owner/switch/revocation during
inference, attachment expiry in history, instructions embedded in pixels, and
image requests crossing the local-only routing boundary.

## Task 1: Validated temporary image upload

**Interfaces:** `normalize_upload(data: bytes, content_type: str) -> bytes` accepts
JPEG/PNG/WebP, at most 10 MiB encoded. Use existing capture decoded-pixel/side bounds;
promote those constants to a shared helper only if needed. Re-encode to PNG and
discard unnecessary metadata. POST `/api/images` returns an opaque attachment handle;
each owner/request can hold at most two admitted images for 900 seconds.

- [ ] Add failing tests for supported formats, MIME spoofing, undecodable HEIC,
  oversized encoded/decoded images, image bombs, original-filename traversal,
  cross-owner/device access, two-image bounds, 900-second expiry and switch cleanup.
- [ ] Run `.venv/bin/pytest tests/test_web_images.py`; confirm the new behavior fails.
- [ ] Implement bounded image validation and scoped storage using `MediaStore`.
  Keep original filenames, image bytes, metadata and unspoken extracted text out of logs.
- [ ] Run image/media tests and existing capture-limit regressions.
- [ ] Review private storage permissions and authenticated image read/thumbnail routes.

## Task 2: Explicit image turns through CIEL's selected vision model

**Interfaces:** `UploadedImageService.answer(request: WebRequest, *,
cancel_event: threading.Event) -> LLMResponse | None` consumes admitted attachment
handles and original user wording. Resolve handles only in the exact current scope;
combine at most two images into one labeled image for the selected vision client.

- [ ] Add failing tests that image requests use the selected verified model, preserve
  local-only history, never invoke desktop capture, reject image-embedded commands,
  and cannot save facts without normal owner authorization. Expiry, Stop, device
  revocation and A→B→A during inference must suppress late output and history commits.
- [ ] Run the uploaded-image tests and confirm the new service behavior fails.
- [ ] Implement explicit image turns before generic text/tool routing. Reuse VisionClient
  policy, cancellation and supported crop behavior; distinguish uploaded images from
  requested PC captures. Archive only safe attachment metadata and delivered dialogue.
- [ ] Run image, screen, memory-authority and web speech-routing tests. Assert the
  answer is audible by default on the requesting device without extra PC playback.
- [ ] Review follow-ups and expired media: ask for reupload, never substitute a PC screenshot.

## Task 3: Image composer and complete cross-device finish

**Interfaces:** `AttachmentComposer.add(files)`, `remove(handle)`, `clear()` and
`admittedIds()` use authenticated upload results. Message attachment rendering uses
explicit available/expired states and never arbitrary source URLs.

- [ ] Add failing frontend tests for selecting/pasting images, previews, remove/retry,
  two-image/size errors, upload failure, preserving text drafts, expired history,
  selection changes during upload and revocation before Send.
- [ ] Run `node tests/frontend/web_attachments.cjs`; confirm missing behavior fails.
- [ ] Implement the attach control, preview/remove flow and safe history placeholders.
  Add no inert PDF button or fake upload success state. Keep phone touch targets and
  the animated orb visible while the keyboard or attachment preview is open.
- [ ] Run all web/frontend/image tests. Inspect complete desktop and phone layouts
  with real-shaped synthetic messages, images, code, sources, recording and playback.
- [ ] Complete the umbrella plan's full suite, Ruff, Python 3.10, wheel asset checks,
  and real local HTTPS/phone acceptance. Update README, CHANGELOG, roadmap and
  `docs/web-interface.md` with what actually works and separately recorded limits.
