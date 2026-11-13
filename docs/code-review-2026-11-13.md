# Code review — 2026-11-13

Every file under `src/ciel` (Python and web assets), `scripts/`, `setup.sh`, CI, and the
install docs was reviewed in eight slices. Each finding below was checked against the
code before it was fixed. Baseline: 4 failing tests on this machine (all TLS). After the
fixes and optimizations: **5391 passed**, Ruff clean, `bash -n` clean. Nothing here has been committed.

## Fixed (each with a regression test)

| Area | Problem | Files |
| --- | --- | --- |
| Phone HTTPS | Local server certificate had no key identifiers under OpenSSL 4, so strict clients (Python 3.13+) rejected it | `webapp/tls.py` |
| Persona | "Can you talk about the Roman Empire?", "Be honest, is this a good idea?" were saved as a permanent style and never answered | `persona.py` |
| Memory | "Forget it" with nothing in focus deleted any single fact containing the letters "it" (e.g. "favorite") | `memory/service.py` |
| Web voice | Replies lost audio after 8 replies within 2 minutes (finished jobs kept for replay filled the limit) | `webapp/speech.py` |
| Redaction | `sk-` etc. had no word boundary: URLs/words like `risk-management-guide` were dropped from search or shown as `[REDACTED_SECRET]`; "the bearer of bad news" was masked; keys inside logged exceptions/tracebacks and Telegram tokens in `/bot…` URLs were not | `logging.py` |
| Voice | A short "yes"/"no" (2 speech frames) was treated as a click and waited ~6 s for the timeout | `audio/recorder.py` |
| Config | Ollama's own `OLLAMA_HOST=127.0.0.1:11434` / `0.0.0.0` (no scheme) broke all local-model calls, including private turns | `config.py` |
| Exec safeguards | `echo hi;rm -rf ~`, `ls\|rm -rf ~`, `$(…)`, and `rm -rf *` / `.` / `..` from the home folder passed the destructive check; `du --files0-from=FILE` read files with no approval in safe mode | `shell_tool.py` |
| Memory | Duplicate-append registry grew one entry per note; past ~9,500 notes every memory read and write failed | `memory/workspace.py` |
| Privacy | Forgotten facts stayed in conversation titles (model's chat list, sidebar, header) | `memory/conversation_index.py`, `webapp/conversations.py` |
| Latency | `record_recall` scanned every reflection candidate each turn even when no daily note was hit (~185 ms measured) | `memory/lifecycle.py` |
| Turbo voice | Forced worker close waited 3 s before the fallback voice; one bad sentence disabled Turbo for the whole session | `audio/chatterbox_worker.py`, `audio/tts.py` |
| Speech | An unclosed `[` (e.g. `:[`) held back all speech until the reply ended | `audio/streaming.py`, `audio/speech_events.py` |
| Web reading | Wikipedia `…_(planet)` URLs cut; `&region=` became `®ion=`; "Thanks.What"/"Node.js" treated as websites (DNS lookup of typed text); `+0800` feed dates undated on 3.10; drip-fed pages ignored the deadline and held reader slots; ASP.NET pages (`<form>` body) empty; article headlines dropped; nested markup made parsing quadratic; `windows-1251` pages garbled; non-ASCII hostnames aborted the read loop; real articles mentioning CAPTCHA rejected | `internet.py` |
| Chat/Telegram | Turns cut at 60 s even in direct web mode or with long shell commands | `chat/session.py` |
| Runtime | `TypeError` on an unaddressed spoken "/local" in ambient mode; a speaker failure saved a second contradicting reply and escaped the turn | `runtime.py` |
| Web server | Shutdown hung forever after a failed second bind; non-ASCII CSRF header raised; open-LAN mode kept one session per cookieless request forever | `web.py`, `webapp/channels.py`, `webapp/auth.py` |
| Latency | Every turn's system prompt ran `nvidia-smi` (up to 1.5 s on a sleeping GPU); now cached 30 s (explicit hardware questions stay live) | `platform/system_info.py` |
| NIM | "It was a tough decision for the committee…" was truncated to "for the committee…" | `providers/nim.py` |
| Startup | `CIEL_LOG_LEVEL` / `CIEL_DEBUG` were ignored | `__main__.py` |
| Setup | `setup.sh` aborted after a successful wizard when `~/.local/bin/ciel` was unrelated | `setup.sh` |
| Docs | usage.md described the old web mode, claimed CUDA libs always install, and put ordinary settings in `.env`; README told users to copy `.env.example`, which silently overrides `config.json` | `README.md`, `docs/usage.md` |

## Needs your decision (not changed)

- **Paired phones can change `ollama_host` and API keys** through Settings (`webapp/admin.py:38`, `webapp/settings.py`). A phone could point local-only conversations at another host. Suggest: PC-only for those fields.
- **Pages over 256 KiB are rejected, not truncated** (`internet.py` `_MAX_BYTES`, pinned by `tests/test_internet.py`), and `Accept-Encoding: identity` makes most Wikipedia/news pages exceed it. Suggest: parse the first 256 KiB and accept gzip (the bounded decoder already exists).
## In files another session is editing (`agent.py`, `agent_tools.py`, `direct_web.py`)

- `memory_search` / `memory_get` / `conversation_read` results and non-safe-listed command output don't mark the turn untrusted, so with `exec_approval="never"` injected text could lead to a command without approval (`agent_tools.py` ~396-409, 578).
- A stale request raises `BackgroundDeferred` out of `MainAgent.send()` instead of returning canceled; chat then stores "I couldn't finish that request" (`agent.py` ~395-402).
- Tool exceptions and answer-filter failures are never logged; `AgentResult.error` is never read (`agent.py`).
- An approved command is SIGKILLed when the owner speaks again (turn superseded) (`agent_tools.py` ~584).
- Direct-mode `read_page` fetches any URL the model emits; injected pages could exfiltrate via query strings (`direct_web.py` ~165).
- `tools: []` with `tool_choice` is sent when no tools are registered (plausible; OpenAI-style APIs reject it).

## Other reported bugs, not yet fixed (verify first)

Audio: no microphone-stream watchdog (`listener.py` ~737, `is_streaming()` unused); `ChunkPlayback` stall check aborts audio still draining (`chunk_playback.py` 53/95/122); `train-wake` has no speech negatives so the model fires on any speech (`trainer.py` 127-139); STT fallback catches every exception and pins CPU (`stt.py` 458-464).
Memory: `memory_observations` keeps every owner turn forever (`lifecycle.py` ~385); a new `ConversationTitler` per turn (`manager.py` ~176); recall `limit` applied before the local-only filter (`context.py` 63-95).
Config/terminal: unused `split_env_file` would delete keys and bake in the environment (`config_store.py` ~256); `null` values in `config.json` are dropped and unknown keys ignored silently; setup saves device indices, not names (`setup_wizard.py` 195/207); terminal left in cbreak after SIGTERM (`terminal.py` 311-319); stale paste marker turns a typed `/command` into chat (`terminal_paste.py` 63-65).
Web: open-LAN tab keeps a stale CSRF token after a server restart (403 until reload, `audio.js` 52); no total request deadline or connection cap (`webapp/server.py`); `ValueError` from a conversation switch becomes 503 (`web.py` ~429); CSP allows `script-src 'unsafe-inline'` though no inline scripts exist.
Internet: `_URL` excludes `'` (`Ender's_Game` cut); `lookup()` swallows exceptions without logging; verified-mode URLs over 500 chars truncated; NAT64 addresses rejected.
Other: `controller.py:747` raises `ValidationError` for utterances over 400 chars (legacy path); a bare spoken "yes" in the non-ambient follow-up window can approve a held command (design).
CI/scripts: the wheel is installed `--no-deps` into the dev environment, so a missing runtime dependency can't fail CI (add a clean-venv install step); `benchmark_turn_detection.py` fails when run directly; TTS shootout guards hash only `.env`, not `config.json`.

## Second pass: `.env.example` and optimizations

- **`.env.example` is secrets-only.** Every other option is now a commented line, so a
  copied `.env` no longer overrides `config.json`. Template tests load both example files
  from a temp folder (they used to read the repo's real `config.json`), and a new test keeps
  the template secrets-only.
- **Provider connections are reused.** All providers and search clients share one keep-alive
  pool (`providers/streaming.py`): no TLS handshake per model round. A cancel while waiting
  still drops that connection, as before.
- **Web app uses HTTP/1.1 keep-alive** for GET polls, audio chunks, and assets; other
  methods close their connection. Audio requests wait up to 0.5 s for a nearly ready chunk
  instead of making the browser sleep 150 ms and ask again.
- **Screen questions:** cached captures are checked from the PNG header (0.0002 s instead of
  0.13 s per 4K image), and PNGs are encoded without `optimize=True`.
- **NIM fails over at once** when another configured provider follows it, instead of
  sleeping 0.5 s + 1 s on retries.
- **Search result pages are read together** (up to 4 at once) instead of one after another.
- **Memory:** forgotten-fact patterns and the `/local` scan are cached until any connection
  writes to the database (cross-process safe); one conversation titler is kept per router.

Not done, with reasons:
- Long-poll or server-sent events for `/api/state`: needs a frontend rewrite; keep-alive
  already removes most of the polling cost.
- Wake spotter during replies: in ambient mode its trigger marks a barge-in as
  `wake_verified`, which command authorization relies on.
- Overlapping Pocket sentence synthesis, and speculative STT: both change live audio timing
  and need microphone/speaker validation before shipping.
- Passing PIL images between screen steps instead of PNG bytes: a larger refactor.

## Feature ideas grounded in existing code

1. Provider circuit breaker and token/usage totals (roadmap item).
2. Wake-time prefetch: build the prompt, run recall, and warm the provider connection while the owner is still speaking.
3. Approve/deny buttons for held commands on web and Telegram (`Approvals` already holds command and expiry).
4. "Go on" after barge-in: `tts.stop()` already returns the unspoken text.
5. Telegram photo questions through `webapp/images.py` and the vision model.
6. QR-code phone pairing and a per-device revoke list in Settings.
7. `ciel config show --sources` / `migrate` to report `.env` shadowing, unknown keys, and loose file permissions.
8. Latency panel: keep recent `TurnTrace`s and show p50/p95 in `/status` or `ciel doctor`.
