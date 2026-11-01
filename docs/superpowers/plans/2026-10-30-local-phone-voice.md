# CIEL Local Phone and Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the paired owner use text and tap-to-talk voice over local HTTPS,
hearing CIEL on the device where each request originated.

**Architecture:** Extend Plan 1's validated HTTP and active-selection interfaces
with device sessions, bounded temporary audio, and structured runtime requests.
Reuse local STT and configured CIEL TTS; keep browser audio separate from PC playback.

**Tech Stack:** Python ssl/secrets/SQLite, existing NumPy/soundfile/STT/TTS,
browser getUserMedia/AudioWorklet/Web Audio, pytest and Node.
**Spec:** [Approved design](../specs/2026-10-30-web-interface-upgrade-design.md).

## Global constraints and prerequisites

Apply the [umbrella requirements](2026-10-30-web-interface-upgrade.md).
Complete the chat/history plan's named interfaces first. No public tunnel or router
forwarding. Plain LAN HTTP is not a working phone microphone configuration.

## Files and interfaces

| File | Responsibility |
| --- | --- |
| `src/ciel/webapp/auth.py` | Expiring pairing, hashed device sessions, CSRF, revocation |
| `src/ciel/webapp/tls.py` | Validated configured certificate/key and private setup files |
| `src/ciel/webapp/media.py` | Bounded owner/device/request/revision-scoped temporary blobs |
| `src/ciel/webapp/requests.py` | Immutable, validated browser request envelopes |
| `src/ciel/webapp/audio.py` | PCM input validation and CIEL WAV chunk delivery |
| `src/ciel/webapp/http.py` | Authenticated LAN API/media routes and exact-origin checks |
| `src/ciel/audio/listener.py` | Serialize admitted structured inputs through the existing worker |
| `src/ciel/runtime.py` | Channel-aware admission, speech routing, cancellation and captions |
| `src/ciel/config.py`, `.env.example`, `src/ciel/__main__.py` | Explicit LAN/TLS configuration and launch flags |
| `src/ciel/web_assets/audio.js`, `pcm-worklet.js` | Browser capture/resampling/playback |
| `src/ciel/web_assets/chat.js`, `chat.css`, `index.html` | Pairing, recording, sound and replay controls |

`DeviceSession` contains `device_id`, `owner_key`, `expires_at`, and a revocation
generation. `WebRequest` contains `request_id`, `device_id`, `conversation_handle`,
`revision`, `input_kind` (`text` or `voice`), `text`, and `attachment_ids`.
The server assigns identities and activation data; clients cannot assert an owner.

`MediaStore.put(data: bytes, *, owner_key: str, device_id: str, request_id: str,
revision: int, content_type: str, ttl: float) -> str`; `get` requires the same scope.
`invalidate` accepts owner/device/request/revision filters. Never expose filesystem paths.
Set encoded-media limits to 10 MiB per item, 32 MiB per device, and 64 MiB per owner.
Enforce these limits before writes, including concurrent uploads. Expired/invalidated
items release their budget. Produce audio through bounded queues rather than storing
an entire long spoken answer ahead of playback.

## Review focus

Test stale device sessions, permission/autoplay denial, A→B→A during synthesis,
duplicate playback across tabs, and reconnect to completed output in their owning tasks.

## Task 1: Device pairing and authenticated sessions

**Files:** `webapp/auth.py`, `webapp/http.py`; `tests/test_web_auth.py`.

**Interfaces:** `DeviceAuth.issue_code() -> str`, `pair(code: str) -> DeviceSession`,
`authenticate(cookie: str) -> DeviceSession | None`, `revoke(device_id: str) -> None`.
Store token hashes, not bearer values. Pairing codes expire after 5 minutes and
permit at most 5 failed attempts per minute per source plus a server-wide cap.

- [ ] Add failing tests for code expiry/single use, rate limits, constant-time token
  comparison, wrong-origin POSTs, absent/wrong CSRF, cookie scope, and immediate
  revocation of queued requests and media. A pairing code alone must not read history.
- [ ] Run `.venv/bin/pytest tests/test_web_auth.py`; confirm the new interfaces fail.
- [ ] Implement private pairing/session persistence, cookie issuance, CSRF verification,
  explicit allowed origins/hosts, and revocation generations. Keep current loopback
  behavior available without making a public unauthenticated route. Keep CSP and
  microphone permissions same-origin; allow only required local worker/media/blob sources.
- [ ] Run auth and existing HTTP-origin tests; inspect logs for token/code leakage.
- [ ] Review device ownership against the active local owner before exposing archive APIs.

## Task 2: Explicit local HTTPS setup and launch

**Files:** `webapp/tls.py`, `config.py`, `.env.example`, `__main__.py`, `web.py`;
`tests/test_web_tls.py`, CLI/config tests, new `docs/web-interface.md`.

**Interfaces:** `WebBinding(host: str, port: int, certificate: Path | None,
private_key: Path | None)`; `validate_binding(binding: WebBinding) -> ssl.SSLContext | None`.
Add explicit `--web-host`, `--web-cert`, `--web-key` options; retain loopback defaults.

- [ ] Add failing tests that non-loopback startup requires usable matching TLS assets,
  unsafe host/origin values fail, keys have private permissions, missing/expired
  certificates produce actionable errors, and normal PC-only startup still works.
- [ ] Run `.venv/bin/pytest tests/test_web_tls.py tests/test_cli.py tests/test_config.py`.
- [ ] Add configuration/CLI validation, TLS wrapping, local URL reporting and guided
  certificate creation/trust instructions for the selected LAN host/IP. Validate any
  certificate-generation tool before invoking it; do not alter phone/OS trust automatically.
- [ ] Run the focused tests and a synthetic local HTTPS handshake with a test CA.
- [ ] Document real phone trust/pairing steps and how to return to PC-only operation.

## Task 3: Temporary media and structured request admission

**Files:** `webapp/media.py`, `webapp/requests.py`, `audio/listener.py`, `runtime.py`;
`tests/test_web_requests.py`, `tests/test_web_media.py`, listener/runtime tests.

**Interfaces:** `submit_request(request: WebRequest, audio: np.ndarray | None = None) -> bool`
on the listener; the old `submit_text(text)` remains compatible. `MediaStore` uses
private runtime storage, opaque random handles and bounded cleanup; audio expires
after 120 seconds unless request/session cancellation invalidates it sooner.

- [ ] Add failing tests for server-assigned identity, wrong-owner/revision/device
  rejection, voice confidence checks, single serialized output, media expiry,
  shutdown cleanup and byte budgets. Internal audio events cannot create user turns.
- [ ] Run the new request/media tests and confirm missing behavior fails.
- [ ] Implement immutable request envelopes and media storage. Adapt existing listener
  admission without labelling phone voice as keyboard input. Keep original transcribed
  wording as the authority for actions/memory and preserve current cancellation behavior.
- [ ] Run request/media, background-output, microphone-control, and runtime tests.
- [ ] Review all completion guards for device revocation and active-selection revision.

## Task 4: Phone audio input and local transcription

**Files:** `webapp/audio.py`, `webapp/http.py`, `audio/listener.py`, `runtime.py`;
`tests/test_web_audio_input.py`, runtime voice-admission tests.

**Interfaces:** `decode_voice_wav(data: bytes) -> np.ndarray` accepts mono PCM16 WAV,
16,000 Hz, at most 60 seconds/1,920,000 PCM bytes. Reject malformed headers, inflated
lengths, compressed codecs, multiple channels, NaN/invalid sample data, and empty input.
POST `/api/voice` admits the request against authenticated device and current revision.

- [ ] Add failing tests for exact 60-second bounds, malformed/empty audio, superseded
  recordings, low-confidence transcription, and A→B→A or device revocation during STT.
- [ ] Run `.venv/bin/pytest tests/test_web_audio_input.py` and confirm missing behavior fails.
- [ ] Decode bounded audio and route it through existing local STT and direct-address
  voice admission. Release raw recording buffers after transcription/cancellation.
- [ ] Run input, microphone, correction/privacy, and local-action authorization regressions.
- [ ] Verify no voice file or private transcript appears in ordinary transport logs.

## Task 5: Originating-device CIEL audio output

**Files:** `webapp/audio.py`, `webapp/media.py`, `runtime.py`, `web.py`;
`tests/test_web_audio_output.py`, typed-response and streaming tests.

**Interfaces:** `BrowserSpeech.speak(text: str, request: WebRequest, *, full: bool = False)
-> str` returns an authenticated playback-job handle. Jobs expose bounded WAV chunks,
sentence/caption timing and cancellation. Use `TextToSpeech.synthesize_stream()`;
this path must not call PortAudio playback. Acknowledgments are device/request scoped.

- [ ] Add failing tests that phone text/voice/image answers synthesize CIEL speech
  without PC playback, desktop microphone requests retain PC output, multiple tabs
  cannot consume a job twice, and Stop/revocation/switch invalidates late chunks.
  Assert an excerpt for long answers, full text on explicit replay, and no new LLM call.
- [ ] Run `.venv/bin/pytest tests/test_web_audio_output.py` and confirm missing behavior fails.
- [ ] Add speech routing, bounded chunk production, playback ownership, captions, and
  Read full answer using existing text-cleaning/excerpt rules. Keep provider-failure
  fallback before visible output and preserve original answer identity for replay.
- [ ] Run output, typed-response, streaming, interruption, and no-microphone tests.
- [ ] Verify a fresh page/reconnect receives history without jobs that already completed.

## Task 6: Browser recording and resampling

**Files:** `web_assets/audio.js`, `pcm-worklet.js`, `chat.js`, `index.html`, `chat.css`;
`tests/frontend/web_audio_input.cjs`, `tests/test_web_frontend.py`.

**Interfaces:** `BrowserRecorder.start()`, `finish() -> Promise<Blob>`, `cancel()`.
Keep actual device rate in the resampling state; output the server's PCM16/16kHz
contract. Do not load external encoders or depend on WebM/MP4 decoding.

- [ ] Add failing tests for 44.1/48kHz input duration/sample count, chunk-boundary
  resampling continuity, 60-second stop, recording cancellation, denied permission,
  missing secure context, navigation/disconnect, and stopped microphone tracks.
- [ ] Run `node tests/frontend/web_audio_input.cjs` and confirm missing behavior fails.
- [ ] Implement tap-to-talk capture with AudioWorklet, visible duration, Finish/Cancel,
  and upload progress. Keep text submission and playback Stop available on audio errors.
- [ ] Run frontend tests, then verify actual recordings on Android Chrome and iPhone Safari.
- [ ] Record permission and device behavior separately from synthetic signal tests.

## Task 7: Browser playback, audible defaults and recovery

**Files:** `web_assets/audio.js`, `chat.js`, `chat.css`, `index.html`;
`tests/frontend/web_audio_output.cjs`, `tests/test_web_frontend.py`.

**Interfaces:** `BrowserPlayer.play(jobHandle)`, `stop()`, `enableSound()`;
state includes `blocked`, `buffering`, `playing`, `stopped`, and `finished`.

- [ ] Add failing tests for autoplay rejection, enabled sound after a user gesture,
  sentence order, truthful captions, short/full replay, mute/Stop, multiple tabs,
  late chunks, hidden/locked device interruption, and reconnect without old playback.
- [ ] Run `node tests/frontend/web_audio_output.cjs` and confirm missing behavior fails.
- [ ] Play owner/device-scoped chunks, acknowledge actual playback, default sound to
  on for new replies, and show Enable sound/Play when blocked. Distinguish speaker
  mute from PC microphone mute; do not replay archive messages automatically.
- [ ] Run frontend and web output tests, then inspect real phone sound and interruption.
- [ ] Update phone setup/troubleshooting docs with observed outcomes and open limitations.
