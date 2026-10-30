# CIEL web interface upgrade

Date: 2026-10-30
Status: Written design for review; implementation has not started.

## Intent and agreed outcome

Make the web interface useful for sustained conversations on both the PC and a
phone connected to the same local network. Full text and voice belong to the same
conversation. The owner should be able to find an older answer, deliberately
continue that conversation, send an image, and hear CIEL on the device being used.

The interview established these decisions:

| Area | Agreed behavior |
| --- | --- |
| Main interface | Full chat with readable, persistent messages and voice available |
| History | Sidebar of earlier conversations; process startup still opens a fresh chat |
| Continuing history | Opening an archive is read-only browsing; "Continue this chat" activates it for text and voice |
| Devices | PC and phone, local network only; PC remains the running assistant host |
| Phone voice | Tap-to-talk with the phone microphone and speaker |
| Voice presentation | Small animated orb on PC; compact microphone controls on phone |
| Attachments | Images in this upgrade; PDFs deferred |
| Spoken replies | Every new reply is spoken by default for both typed and voice requests |
| Long answers | Full text stays visible; a shorter spoken version plays automatically; "Read full answer" is available |

Implementation defaults below are design choices, not additional user answers.

## Current constraints in the repository

`web.py` currently provides a loopback-only HTTP server, a bounded in-memory view
of the current conversation, and a text submission callback. The page hides older
messages and presents the latest answer in a temporary bubble. Its microphone
control toggles the PC listener; it does not record browser audio.

`runtime.py` owns conversation admission, cancellation, local actions, tools, and
PC speech. `memory/bridge.py` starts a fresh desktop archive on each launch and
projects the currently paired Telegram owner's matching archive into that launch.
It has no explicit browse/resume API. A history sidebar must use owner-admitted
archives rather than listing arbitrary SQLite sessions supplied by a client.

`TextToSpeech.synthesize_stream()` already exposes generated audio independently
of PortAudio playback. `SpeechToText` already handles local transcription. Use
those paths for phone voice while retaining configured model and voice choices.
Existing `VisionClient` validates image input and enforces protected local history.
It can support uploaded images through an explicit attachment route.

This is an architectural change. Deliver it through related stages with separate
acceptance checks, while retaining the complete PC/phone scope.

## Approach

Extend CIEL's Python service and existing HTML/CSS/JavaScript interface. Separate
HTTP transport, conversation selection, temporary media, and browser audio into
small modules instead of expanding the existing monolithic page or runtime.

Alternatives considered:

1. Redesign only the page. This improves reading but cannot deliver phone voice,
   owner-scoped history resumption, or image conversations.
2. Extend the current service with explicit session/media interfaces. Selected:
   it reuses CIEL's voice, memory, provider, and cancellation contracts.
3. Replace the application with a separate web framework/frontend stack. This adds
   a second application lifecycle and packaging burden without a requirement that
   justifies it for this upgrade.

Use same-origin JSON endpoints and the existing polling model initially. State
and message retrieval must be incremental and paginated; do not send a complete
archive or media payload on each poll. SSE/WebSocket migration is unnecessary for
the first complete version unless measured audio delivery requires it.

## Interface and visual contract

Retain the current dark surfaces, readable typography, purple/cyan accents, and
procedural CIEL orb. The primary object is the conversation. A single composer is
the main action; voice and attachments sit beside it. Avoid dashboard cards,
decorative metrics, and a large empty center dedicated to the orb.

### PC

- Left sidebar: New chat, chronological conversations, and the selected archive.
  Titles come from bounded first-message text; no background title-model requests.
- Main column: current/browsed conversation, clear active/browsing state, and a
  scrollable message timeline. Show the complete written answer, useful source
  links, code blocks with Copy, image thumbnails, and interruption/retry states.
- Bottom composer: multiline text, attach image, microphone, Send, and Stop while
  busy. Keep a draft while browsing history and during temporary disconnection.
- Compact voice area: reuse the animated orb with listening/thinking/speaking
  state. Keep speaker and microphone controls distinct. Screen-observation status
  and an explicit pause action refer to the PC desktop.
- Browsing an archive shows "Continue this chat". Starting a new chat or continuing
  an archive is a visible context change; merely reading an archive is not.

### Phone

- Use the full width for messages. A menu opens the conversation sidebar as a
  drawer, with focus management and an explicit close action.
- Keep the composer accessible above the browser keyboard using dynamic viewport
  sizing and safe-area insets. Messages wrap; code blocks scroll horizontally.
- Use a compact tap-to-talk control with visible recording duration, Send/Finish,
  and Cancel. Stop microphone tracks as soon as recording ends or is cancelled.
- Keep sound, playback stop, upload, connection state, and active-conversation
  identity usable without hover or slash-command knowledge.
- Target Android Chrome and iPhone Safari behavior. Actual device acceptance is
  required before claiming either phone's audio path works.

The timeline must preserve scroll position when the owner reads earlier messages.
Only follow new output when already near the bottom; otherwise show a New messages
action. Refresh and reconnect restore the active view without replaying old audio.

Render model/user text safely. Support useful Markdown structures without enabling
raw HTML or script execution. Local assets must ship in the wheel; use no CDN for
fonts, renderers, or scripts. If a Markdown library is selected during planning,
pin and package it with appropriate attribution and a safe rendering policy.

## Conversation selection and continuity

Introduce a conversation-session controller shared by desktop web and the runtime.
It exposes the active conversation, a monotonic active revision, admitted archive
catalogue/read operations, New chat, and Continue chat.

Each archive handle is server-issued and scoped to the current owner. Reuse the
same-owner archive admission rules already used by memory search. Add explicit
ownership metadata where required; do not infer authorization from a client-supplied
session string or enumerate all historical Telegram bindings. Paginate metadata and
turns separately. Browsing sends no archive content to an AI provider.

New chat and Continue chat are serialized with request cancellation. On a switch:

1. Cancel current generation, playback, pending microphone/browser uploads, and
   background initiative; invalidate the old active revision.
2. Flush the outgoing conversation's existing lifecycle checkpoint without copying
   its raw history into the selected conversation.
3. Select or create the admitted conversation and refresh its conversation manager,
   summary, dialogue/tool bindings, and currently authorized cross-channel projection.
4. Clear transient screen images, uploaded media, pending confirmations, interruption
   fragments, and unfinished task state. Keep saved personal facts and archives.
5. Publish the active selection/revision so PC, paired browser clients, and the
   currently paired Telegram worker agree on the current conversation.

Do not call history deletion/reset to implement switching. A switch A→B→A must
still reject late work started under the first activation of A. Preserve protected
local-history routing when continuing an old conversation.

All accepted requests carry a trusted owner/device identity, active conversation
handle/revision, request ID, cancellation state, and explicit input channel. These
fields come from validated server state. A browser cannot grant itself a different
owner or supply execution authority through attachment or model text.

## Local phone connection

Preserve default PC-only startup. Add an explicit local-network setup/start path
that reports the configured LAN address and pairing instructions on the PC. It
must never configure router forwarding, public tunnels, or internet hosting.

Phone microphone access requires a secure context. An HTTP URL using the PC's LAN
address does not get localhost's microphone exception. Provide a local HTTPS
configuration and guided phone trust setup for a certificate matching the selected
local host/IP. Private keys and pairing state live in private user data, not Git.
Keep trust installation an explicit setup step; do not tell users to bypass browser
security interstitials or claim microphone support over plain LAN HTTP.

Pair the phone using a short-lived, rate-limited code issued on the PC. Successful
pairing issues a device-specific authenticated session. Use secure, HttpOnly,
SameSite cookies for LAN HTTPS, server-side revocation, CSRF protection, strict
allowed Host/Origin checks, bounded request bodies, and authenticated media reads.
No credentials go in URLs, QR history links, frontend JavaScript, or logs. A QR may
contain the ordinary connection URL; pairing still requires the separate code.

One paired owner may use multiple devices. Pairing a device does not create a new
personal-memory owner. Revocation must invalidate queued requests and media access.
Retain the existing authorization rules for app actions and memory writes; add no
arbitrary remote file, shell, or desktop automation endpoint.

## Phone audio and spoken delivery

Record browser microphone audio only after a tap and browser permission. Use a
bounded PCM/WAV input contract so server transcription does not depend on optional
WebM/MP4 decoders. Browser audio code must handle actual device sample rates and
downsample to the STT input rate. Start with a 60-second recording limit and report
empty, too-long, denied, interrupted, and unsupported recordings clearly.

Send admitted audio to the existing STT worker through a structured request path.
Preserve confidence checks, original transcripts, local intent authorization, and
cancellation. Do not pretend uploaded voice is ordinary keyboard text or fabricate
a user request for internal playback events.

Route synthesized audio to the requesting device. Phone-originated replies must
not also play through the PC speakers. Desktop microphone replies retain PC output;
multiple open browser tabs must not each play the same reply. Associate audio
chunks/events with owner, device, request, conversation revision, and sentence.
Expire media promptly and make late/cancelled chunks inaccessible.

Every new delivered assistant reply is audible by default, including answers to
typed and image requests. Long answers use the existing spoken-excerpt behavior
while the complete written reply remains in the timeline. Keep a visible speaker
control for deliberate silence, with the user's chosen audible default preserved.

"Read full answer" reads the existing answer through the configured CIEL voice.
It does not make another model request, create another assistant answer, or read an
archive automatically. Stop/new input/context switch interrupts it immediately.
Apply the existing speech-cleaning rules to Markdown and code.

Browser autoplay restrictions can prevent sound until a user gesture. Use the
initial send/tap action to unlock audio where supported, observe playback failures,
and show an actionable Enable sound/Play control when blocked. Never show successful
playback or advance spoken captions solely because synthesis completed. Background
tabs, locked phones, and reconnection must not silently queue old replies for later.

## Image conversations

Accept JPEG, PNG, and WebP images selected or pasted by the owner. Start with at
most two images per message and a 10 MiB encoded limit per image, with decoded
pixel/dimension bounds. Show previews, remove/retry controls, upload progress, and
clear messages for unsupported formats such as an undecodable HEIC file.

Decode and re-encode supported images to a known safe representation, discard
unneeded metadata, and use opaque media identifiers. Do not use original filenames
as paths. Store images in an owner/conversation/revision-scoped temporary store;
reuse the existing two-image/15-minute retention policy where practical. Keep image
bytes and extracted unspoken content out of conversation SQL, summaries, saved
facts, logs, and version control.

Submit the owner's text and admitted image attachments to the configured, verified
image model with the existing local-only history rules. Uploaded images do not
authorize PC capture. A question about an expired attachment asks for reupload;
it never substitutes a new desktop screenshot. Image text cannot authorize actions
or save personal facts. Preserve source/time distinctions in visual follow-ups.

Archive the user message, safe attachment metadata, and the delivered answer. An
expired image shows an unavailable attachment placeholder in older conversations;
durable storage of uploaded image bytes is not part of this design.

## Implementation boundaries

Proposed modules are responsibilities, not a requirement to keep these exact names:

| Component | Responsibility |
| --- | --- |
| Web transport | Validated HTTP routes, HTTPS binding, pairing/session cookies, bounded request parsing |
| Conversation-session controller | Owner-admitted archive reads, explicit active selection/revision, switch cancellation and lifecycle |
| Temporary media store | Authenticated image/audio handles, size/lifetime bounds, invalidation and cleanup |
| Browser audio adapter | Recording, speech upload, CIEL audio delivery, playback acknowledgments and interruption |
| Runtime request adapter | Trusted structured request envelopes through existing conversation/tool/voice workers |
| Chat page modules | Timeline, safe formatting, history drawer, composer, attachments, voice state and responsive controls |

Keep the existing terminal and Telegram behavior working. Separate new transport
concerns from `runtime.py`; extract only the request/session seams needed by this
upgrade. No unrelated provider, memory-ranking, or speech-model changes.

## Delivery stages

1. Build the persistent chat timeline, safe formatting/copy actions, compact voice
   presentation, and complete empty/loading/error/interrupted/reconnecting states.
2. Add admitted history browsing and explicit Continue/New chat with shared revision
   and cancellation behavior. Preserve fresh conversation creation at process startup.
3. Add explicit local HTTPS configuration, phone pairing/revocation, and responsive
   phone text chat. Verify cross-device conversation identity and request isolation.
4. Add phone microphone upload, CIEL speech playback on the originating device,
   audible defaults, excerpt/full-reading controls, captions, and browser recovery.
5. Add temporary image upload/preview and image questions using the selected vision
   provider, including expired attachment behavior.
6. Run integrated desktop/mobile acceptance and update setup, privacy, and user docs.

These stages are the full upgrade's work breakdown. Completing the first page
redesign alone does not complete phone voice, history continuation, or images.

## Validation and finish criteria

- Test archive admission, pagination, ownership, fresh startup, explicit continuation,
  protected history, A→B→A switching, stale replies, current Telegram projection, and
  preservation of saved facts without arbitrary-session exposure.
- Test pairing expiry/rate limits/revocation, Origin/Host/CSRF enforcement, authenticated
  media reads, body/pixel/audio bounds, and cancellation before every durable commit.
- Test phone resampling/transcription, speech confidence, empty recordings, browser
  permissions, autoplay rejection, disconnects, late audio, multiple tabs/devices,
  full-reading replay, and independent PC microphone behavior.
- Test image decoding, supported/unsupported formats, expiry/reset/switch cleanup,
  image-model selection, local-only routing, and image instructions as untrusted data.
- Render actual UI at desktop, narrow desktop, and phone widths with a software
  keyboard, long code, long URLs, live output, expired images, and error states.
  Check contrast, focus, touch targets, scroll preservation, reduced motion, and
  persistent microphone/speaker/observation state. Fix observed layout breakage.
- Run focused regressions, the full pytest suite, Ruff, Python 3.10 compatibility,
  and a wheel-content check that includes packaged frontend/media helper assets.
- On the target PC and phone, pair over local HTTPS, send typed and spoken requests,
  hear the selected CIEL voice on the right device, interrupt playback, upload an
  image, continue an older conversation, and reconnect without replaying old audio.

Automated browser mocks do not establish microphone accuracy, usable phone audio,
certificate trust, or live two-device acceptance. Record those outcomes separately.

## Scope exclusions

Public internet access, native mobile apps, continuous phone listening, automatic
phone camera capture, PDFs, arbitrary computer control, multi-user accounts,
durable raw image storage, and a general provider/settings dashboard are excluded.
Current configuration remains the source for CIEL's persona, providers, and voice.

## Browser references

- [MDN: getUserMedia and secure-context requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- [MDN: autoplay restrictions and playback failure handling](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay)
