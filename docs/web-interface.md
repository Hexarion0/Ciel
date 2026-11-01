# Ciel web interface

Start `ciel web` and open `http://127.0.0.1:8765`. The default stays on this PC.
The interface uses locally packaged files with no CDN or external font requests.

The sidebar lists earlier conversations. Opening one only reads it. **Continue this
chat** activates it for new text and browser voice requests; **New conversation** starts
an empty chat and keeps saved facts. Each process launch also starts a fresh chat.
Changes in another browser preserve your draft and require choosing a chat before
sending it. Legacy channel archives that cannot safely be continued remain readable.

The animated orb stays in the sidebar on desktop and in the header on narrow screens.
Messages remain available, including paragraphs, lists, code with Copy, and validated
source links. Scroll up to read without new replies pulling the view down; **New
messages** returns to the bottom. Enter sends; Shift + Enter inserts a line break.

## This device's microphone and speaker

The browser tries to start continuous listening automatically. Allow microphone access;
if the browser blocks audio startup, choose **Enable microphone** once. After that,
speak naturally: local speech detection sends each utterance after about 0.9 seconds
of silence. Short pauses stay within the same utterance. Each clip is capped at
60 seconds. Quiet audio and brief clicks do not produce repeated uploads.

The microphone stream stays open between turns. Capture is ignored while Ciel is
processing or playing a reply, then resumes automatically with a short echo cooldown.
There is no wake phrase or per-turn click in the browser. **Pause mic** or the orb
releases this device's microphone; that choice is remembered across reloads. **Stop
reply** cancels a reply and discards partial speech while keeping listening enabled.
**Sound off** controls reply playback without turning off the microphone.

Switching tabs does not deliberately disable capture or playback. The browser or
phone OS can still suspend audio; Ciel tries to resume and shows **Enable microphone**
if another gesture is needed. Closing the page, losing the connection, or browsing
an inactive archive releases capture. Returning to the current chat or reconnecting
resumes enabled listening when the browser allows it. Ciel uses its local recognizer.
Web mode runs a request worker without opening the PC microphone or speaker. Browser
voice is a direct request, subject to the same confidence and command authorization checks.

New browser requests use the configured Ciel voice on the requesting device. They
do not also play through the PC speaker. If autoplay is blocked, select **Enable sound**. Completed
history is never automatically read aloud. Long answers keep the full text and use a
short spoken excerpt; **Read full answer** reads the stored answer without another
model request. Playback queues are temporary and expire; very old answers may need
another explicit request. Audio failure leaves the written reply available.
Browser speech and typing use the same written-answer depth. Speech recognition
still checks uncertainty before a recognized request is accepted.

## Local phone connection

For direct home-network access without certificates or pairing, set a specific private
`WEB_HOST` address and `WEB_INSECURE_LAN=true` in `.env`, then restart Ciel. Open
`http://<PC address>:8765` on the phone. Anyone on that network who can reach Ciel can
use chat and settings. Phone microphone recording is unavailable over HTTP; text,
images, and reply playback remain available. PC-local microphone recording still works
through `http://127.0.0.1:8765`. This mode is off by default. Set it back to `false`
to restore the HTTPS and pairing flow below.

Both devices must be on the same private home network. Open **Settings → Connect your
Android phone** on the PC-local page. The address field is filled from the PC's local
interfaces. **Enable phone access** creates a local certificate authority, signs a
server certificate for that address, and enables an authenticated HTTPS listener.
The private keys stay in the host's private data directory. System trust settings are
changed only by the owner on the device; Ciel never installs a trust certificate.

The same panel provides a plain HTTP Android setup page on the next port (normally
`http://192.168.1.100:8766`). That page serves public setup instructions and the public
CA certificate only. It exposes no chat, settings, credentials, memory, or assistant APIs.
Compare the certificate fingerprint with the one displayed on the PC before installing
it. On Android, install it through Settings → Security & privacy → More security settings
→ Encryption & credentials → Install a certificate → CA certificate. Menu labels can
vary; see [Android's certificate instructions](https://support.google.com/pixelphone/answer/2844832).

Then open the HTTPS app address (normally `https://192.168.1.100:8765`). On the PC,
choose **Create pairing code** and enter the code on the phone. Codes expire after five
minutes and work once; attempts are rate limited. Do not bypass certificate warnings.
Paired phones stay connected through restarts until 30 days pass without use; each use
restarts that window. Ciel keeps only hashes of device tokens, in the owner-only file
`data/web/paired-devices.json`. **Disconnect all phones** in the same Settings panel (PC
only) revokes every paired phone at once. The PC-local page stays available at
`http://127.0.0.1:8765`.

### Phone layout: voice and chat modes

On narrow screens (up to 759 px) Ciel uses a phone layout. A status pill under the title
always says what Ciel is doing: **Listening**, **Hearing you**, **Thinking** (with a seconds
counter, and a "taking longer than usual" hint after 20 seconds), **Speaking**, **Mic
paused**, **Not paired**, or **Offline**.

- **Voice mode** (the default) puts the orb in the middle. Tap the orb or the large mic
  button to pause or resume listening. Your last sentence and Ciel's reply appear below,
  and the bottom bar has **Type instead**, the mic, and **Stop**.
- **Chat mode** is a messaging layout with image, mic, and send buttons in the composer;
  a **Stop** button floats above it while Ciel is thinking or speaking.

Switch modes with the chat/voice button in the header. Each phone remembers its choice.
The header also has sound on/off and Settings. The desktop layout is unchanged.

### Install Ciel on the phone's home screen

Once the phone opens the HTTPS address without a certificate warning and is paired,
open Chrome's menu and choose **Install app** (or **Add to Home screen**). Ciel then opens
full-screen from its own icon, without the browser bar. If the PC is off or unreachable,
the app shows a short "Ciel's PC isn't reachable" screen with **Try again** instead of a
browser error. The service worker stores nothing: no chats, settings, or API responses
are cached on the phone. Installation needs the trusted HTTPS connection, so it is not
available in the plain-HTTP home-network mode.
Accidental HTTP visits to the HTTPS port redirect to the secure app before any private
API or device session is accessed. Plain HTTP submissions on that port are rejected.
Ordinary TLS disconnects and untrusted-certificate handshakes do not print tracebacks.

You can also run `ciel pair` in a second PC terminal while `ciel web` is running. It
shows the phone address, a new five-minute pairing code, and the first-time Android
certificate setup link and fingerprint. It prepares HTTPS automatically if there is
one private network address. With multiple interfaces, select one using
`ciel pair --host 192.168.1.100`. For a different web port, use `ciel pair --port 8767`.
This command contacts only the PC-local server and does not load audio or voice models.

Network preferences are saved locally, so later `ciel web` launches restore phone
access. Advanced `--web-host`, `--web-cert`, `--web-key`, and `--web-port` overrides
remain supported. No public bind, router forwarding, tunnel, or public hosting is set up.

## Manage Ciel in the browser

**Settings** controls the preferred name, primary provider, model routes, API keys,
reasoning, voice and recognition, image model, memory learning and recall, web search,
and personality preferences. Saved API keys are never returned to the browser. Empty
password fields keep existing keys; the explicit removal checkbox clears one.

Model and memory changes apply to new requests. Voice, recognition, and image-provider
changes show a restart notice; **Restart server** shuts down cleanly and reopens the
same hosted app, without needing terminal controls. Browsers reconnect; phones need a
new pairing code. Saved facts can be reviewed and individually forgotten after confirmation.

## Images

Choose **Image** or paste an image. Attach up to two JPEG, PNG, or WebP still images,
10 MB each. Wait for **Image ready**, add a question, and Send. A failed upload stays
visible so you can remove it and try again. Removing an uploaded image releases its
server allocation. PDFs and animated images are unsupported.

Images are decoded within pixel limits, resized, and re-encoded to strip metadata.
They remain in temporary memory for at most fifteen minutes. They use the selected,
verified image model and never silently fall back to a desktop capture. Protected
history requires a selected local image model. Instructions written inside an image
cannot authorize tools or saved facts. Archives retain the question, delivered answer,
and a temporary-image placeholder; raw uploads are not saved in conversation history.
Reattach an expired image for a new visual question.

## Verification status

Automated tests cover archive admission, pagination, activation revisions, media
scope/expiry/budgets, pairing/revocation, microphone resampling, input WAV validation,
image normalization, source safety, orb animation, and origin-only voice routing.
Desktop and 390-pixel phone layouts were inspected during implementation. The final
browser inspection was unavailable because local browser use was disabled in the app.
Actual phone HTTPS trust, permission prompts, audible model output, keyboard behavior,
and cross-device interruption still need acceptance on the owner's devices.

The reviewed implementation also passed a 105-test focused web/runtime group, a full
4,531-test suite, Ruff, Python 3.10 syntax parsing, a real local TLS handshake test,
and wheel packaging with all packaged browser assets. The independent review's six
issues were addressed with regression checks. These checks use synthetic data and
fixtures, and do not establish physical phone or speech-model acceptance.

The web-first follow-up removes host audio capture, adds graphical configuration and
restart, and automates local certificate preparation and Android onboarding. The host
interface is a complete browser app; native `ciel start` remains a separate voice mode.
The current implementation passed 58 focused connection and pairing checks, the full
suite (4,575 passed, 5 deselected),
Ruff, and Python 3.10 syntax parsing. Live checks verified PC-local access with host
capture disabled, the Android setup page and matching public certificate, a trusted
HTTPS connection on the private address, and rejection of unpaired settings access.
The transport follow-up also verified live HTTP-to-HTTPS redirects and rejection of
plain HTTP submissions on the secure port after a graceful server restart.
The built wheel includes all nine browser assets.
Physical Android certificate installation and microphone/playback acceptance remain
owner-device checks.

The continuous-listening update passed 103 focused web/audio/runtime regressions.
Synthetic PCM checks cover automatic speech boundaries, short utterances, repeated
turns on one stream, silence and click rejection, the 60-second bound, playback
suppression, background-tab operation, explicit pause, reconnect, denied permission,
blocked autoplay, and capture cancellation during a pending permission request.
These checks do not establish microphone sensitivity or acoustic echo performance
on the owner's physical devices.
