# Repository review — 2026-10-10

## Repair follow-up

All eight reproduced bugs below have been repaired, with regression coverage in
`tests/test_repository_regressions.py`. Local routing now carries an explicit
provider restriction through streaming, summaries, persisted context, and ambient
intent classification, including after context trimming or summary failure.
The restriction lasts until the conversation is cleared. Forgetting uses exact
structured topics and retains old suppression patterns. Stream replacement releases devices without nested locking;
wake interruption requires detector evidence. Persona proposals expire and clear
on cancellation or unrelated requests. Custom wake models flow from environment
settings, and summary failures leave the processing cursor available for retry.

The additional deadline and configuration suggestions are also implemented:
STT startup readiness distinguishes failure, STT loading and Turbo response waits
have configurable deadlines, unsupported capture formats fail early, API key
entry is hidden, and the setup wizard sets `.env` permissions to `0600`. The new
CI workflow is configured to run tests (including frontend checks), Ruff,
dependency checks, shell syntax, wheel builds, and installed CLI checks on
Python 3.10 and 3.14. Hosted CI has not run; local validation used Python 3.14.7.

Final repair validation:

| Check | Result |
| --- | --- |
| Full default test suite | **744 passed, 3 integration tests deselected** |
| Ruff, dependency consistency, shell syntax, whitespace | Passed |
| Isolated wheel build | Passed |
| Wheel installed into a temporary target; CLI outside checkout | Help/version and packaged web asset passed, using existing dependencies |
| CI YAML | Parsed; triggers and Python matrix checked |

Real microphone/speaker and GPU integration acceptance remains unverified.

Runtime extraction, text-only startup, dependency locking/custom-voice wheel
packaging, and backup/export controls remain architectural follow-ups. The
sections below preserve the original review and its baseline evidence; source
line numbers refer to that baseline.

## Original review

Reviewed the source layout, voice loop, provider routing, memory, local actions,
desktop web backend/frontend, setup and launch scripts, packaging, tests, and
current documentation. Local data/model folders were inventoried by size;
private recordings, databases, API keys, and personal persona content were not
used as review inputs. This is a broad code review, not live audio acceptance or
a line-by-line certification of every file.

During the original review, application code was not changed. The existing edits to
`src/ciel/web_assets/index.html` and `tests/frontend/orb_animation.cjs` were
preserved. This report is the only repository file added by the review.

## Validation

| Check | Result |
| --- | --- |
| `.venv/bin/pytest` | **707 passed, 3 deselected** |
| `.venv/bin/ruff check src tests` | Passed |
| `.venv/bin/python -m pip check` | No broken requirements |
| `bash -n setup.sh scripts/install_ciel_command.sh scripts/launch_ciel_gpu.sh` | Passed |
| `git diff --check` | Passed for the existing working changes |
| CLI `--help` and `--version` | Passed; version 0.3.6 |
| Wheel build with declared isolated build backend | Passed; web page included |
| Additional isolated probes | Eight bugs reproduced using fake providers/audio and temporary memory/files |

The first test run had 11 web fixture errors because the sandbox prohibited
localhost sockets. The web tests and then the full suite passed outside that
restriction. Those initial errors were environmental, not application defects.
The first wheel attempt used `--no-build-isolation` and could not import the
locally absent Hatchling backend. The normal isolated build succeeded.

The three integration tests were excluded by the project's default selection.
Real microphone/speaker behavior, GPU inference, clean-machine installation,
and live provider availability were not verified. No cloud requests, real
recording, speech playback, or application launches were needed for the probes.

## Reproduced bugs, in suggested repair order

### 1. High: `/local` can send the conversation to a cloud provider

Sources: [router.py](../src/ciel/providers/router.py), line 126;
[manager.py](../src/ciel/providers/manager.py), lines 90 and 144.

The router chooses Ollama for `/local`, but the manager treats that choice as a
preference and appends the usual cloud fallback chain. If Ollama fails and a
cloud provider is configured, the same conversation is sent there.

**Reproduction:** submit `/local private example` to mocked providers with
Ollama unavailable and NIM working. Both `router.send()` and `router.stream()`
returned the NIM answer; NIM received `private example` in its messages.

**Suggested change:** carry an explicit permitted-provider policy through the
router and both manager paths. A `/local` request should report a local failure
without cloud fallback. Apply that policy to any summaries generated from a
conversation intended to stay local as well.

### 2. High: forgetting an absent topic can delete a different fact

Source: [service.py](../src/ciel/memory/service.py), line 269.

If a recognized fact key is absent, `_forget()` falls back to ranked search.
One candidate is accepted regardless of how weak or generic the overlap is.
That makes a shared word such as "favorite" enough to select an unrelated fact
for deletion.

**Reproduction:** save `remember my favorite game is CS2`, then request
`forget my favorite food`. The service says it forgot the fact and the saved
favorite-game record is gone.

**Suggested change:** when a request names a supported structured topic, require
that exact key. If it is absent, say so. For free-form notes, require sufficiently
specific evidence or ask the user to confirm the exact candidate before deletion.

### 3. Medium: saving a new value removes suppression of forgotten old values

Source: [store.py](../src/ciel/memory/store.py), `upsert_fact()`, line 269.

Saving a keyed fact deletes its entire entry from `forgotten_memories`, including
patterns belonging to older values. Archived conversations are intentionally
retained, so those old values can enter provider context again.

**Reproduction:** save CS2 as the favorite game, forget it, then save Minecraft
as the favorite game. Before the new save, `redact_forgotten()` masks the old
CS2 sentence. Afterward, it returns the original CS2 sentence unchanged.

**Suggested change:** retain suppression for previously forgotten values when
the same key receives a different value. Re-authorizing a new preference should
not automatically re-authorize all old values in archived text.

### 4. Medium: restarting an active microphone stream deadlocks

Source: [linux.py](../src/ciel/platform/linux.py), line 215.

`start_stream()` holds a non-reentrant `threading.Lock` and calls
`stop_stream()`, which acquires the same lock. The active-stream replacement
branch cannot proceed.

**Reproduction:** set a fake existing stream's `active` property to true and
call `start_stream()` in a thread. It stays blocked with the lock held, before
attempting any device access. The nested acquisition establishes the deadlock;
the short thread join only makes it observable.

**Suggested change:** use an internal stop helper that expects the caller to
hold the lock, or use a carefully reviewed reentrant lock. Check stream
replacement and recovery after a stream-start failure with mocked devices.

### 5. Medium: wake interruption mode also interrupts on loud noise

Source: [listener.py](../src/ciel/audio/listener.py), line 367.

When ambient listening is disabled, `barge_in_mode='wake'` still checks RMS
and stops output if it exceeds the default 0.030 threshold. This contradicts
the adjacent comment and the usage guidance recommending wake mode with
speakers. Speaker output or other sufficiently loud sound can take this branch.

**Reproduction:** with fake TTS reporting playback and the wake detector returning
no trigger, one constant audio frame with RMS 0.05 cancels the current reply,
calls TTS stop, and changes the listener to recording.

**Suggested change:** require a wake trigger for wake mode. If volume-based
interruption is wanted, expose it as a separate explicit setting and document
its behavior. The existing loud-audio interruption test currently preserves
this behavior, so its expectations should be reviewed alongside the change.

### 6. Medium: canceling a persona edit does not cancel the pending edit

Sources: [__main__.py](../src/ciel/__main__.py), lines 531 and 562;
[persona.py](../src/ciel/persona.py), line 55.

The stop/cancel path clears memory and action proposals but leaves
`persona_change_pending` true. In that mode, the parser accepts almost any
8–300 character follow-up as a style preference. The pending flag also has no
expiry, allowing an unrelated later request to become a persistent preference.

**Reproduction through the real CLI callbacks, with mocked audio:**

1. `Ciel, can you update your persona?`
2. `Ciel, cancel`
3. `Ciel, what's the time?`

The third turn writes `what's the time` into the managed persona block and
acknowledges a style update instead of answering the clock question.

**Suggested change:** clear the persona proposal on cancellation, farewell,
mode changes, and unrelated requests; give it an expiry. Reject ordinary
questions as style preferences even while a proposal is pending.

### 7. Medium: custom wake training is disconnected from normal configuration

Sources: [config.py](../src/ciel/config.py), lines 102 and 292;
[trainer.py](../src/ciel/audio/trainer.py), `train_custom_wakeword()`;
[wake.py](../src/ciel/audio/wake.py), `_resolve_model_paths()`.

`AudioConfig` defines `wake_models`, but environment-facing `Settings` does
not define it or forward it into the audio configuration. The CLI reads
`settings.audio.wake_models`, which consequently stays empty. The detector
does not automatically discover the model written by `train-wake`.

**Reproduction:** constructing `Settings(_env_file=None,
wake_models=['models/custom/hey_ciel.onnx'])` silently ignores the value;
`settings.audio.wake_models` is `[]`.

**Suggested change:** expose and forward a documented `WAKE_MODELS` setting,
and make training report how to activate the generated model. Test configuration
through the CLI detector constructor as well as the ONNX export itself. The
current "now calibrated" success message overstates the runtime result.

### 8. Medium: a failed summary permanently skips most of its batch

Source: [manager.py](../src/ciel/memory/manager.py), line 263.

When the provider fails, the fallback describes only the first three older
turns. The saved cursor nevertheless advances to the end of the entire batch,
which can contain 32 turns. The omitted turns remain in SQLite but are no
longer candidates for later summarization and eventually leave active context.

**Reproduction:** create 20 distinct turns with a two-turn recent window,
then make the summary provider fail. The fallback includes only details 0–2,
but saves turn ID 18 as processed. After provider recovery, summarization
returns `None`; detail 10 is absent from both summary and active context.

**Suggested change:** defer the summary on transient failure, or advance the
cursor only through turns actually represented by the fallback. Retain enough
unsummarized context for a bounded retry after recovery.

## Further improvements

These are follow-up engineering opportunities, distinct from the eight
reproduced bugs.

- **Add CI for the existing checks.** No tracked `.github` workflow was found.
  Run unit tests, Ruff, shell syntax, and a wheel/CLI smoke check on changes.
  Include Node so frontend tests cannot silently skip. Verify the declared
  Python 3.10 minimum in addition to the local Python 3.14 environment.
- **Extract the conversation runtime from the CLI.** `__main__.py` has 1,087
  lines and many closures sharing mutable lists. Move dialogue state,
  cancellation, proposals, and response execution into a runtime object used
  by voice, terminal, and web adapters. The persona cancellation bug shows why
  one explicit state lifecycle would help.
- **Let text chat survive audio startup failure.** Web and terminal modes
  currently initialize speech engines and start a microphone stream before
  accepting messages. Offer a text-only runtime path when input devices or
  native audio initialization fail.
- **Bound model startup and synthesis waits.** Turbo has a startup timeout,
  but its per-request response loop has no elapsed-time deadline while the
  process remains alive. STT transcription can also wait indefinitely for
  its loading event. Add deadlines and a recoverable degraded state. Clarify
  readiness: STT `wait_ready()` currently returns true when loading finishes
  with an error as well as when it succeeds.
- **Make installations reproducible.** Most core dependencies have open-ended
  minimum versions and there is no tracked lock/constraints file. Maintain a
  tested dependency set and a fresh-install check. The wheel includes the web
  page but lacks the custom Turbo worker/manifest, matching the documented
  source-checkout limitation; resolving that remains a packaging priority.
- **Improve configuration entry and validation.** The setup wizard uses echoed
  `input()` for API keys. Use hidden entry and restrictive permissions for new
  `.env` files. Validate unsupported sample rates and positive memory limits
  when loading settings, so users get a clear configuration error before
  microphone or database behavior fails.
- **Add deliberate data management.** The local `data/` folder is approximately
  8.8 GB, `models/` 182 MB, and `.venv/` 7.3 GB. Provide an inventory and explicit
  backup/export controls separating memory, private voice references, models,
  and caches. Do not automatically remove these folders: their contents may be
  needed for offline speech or recovery.

The package boundaries, allowlisted application launching, parameterized SQLite
operations, loopback HTTP checks, bounded queues, and existing test breadth
provide a useful base. The next pass should prioritize the local-only routing
and targeted-forgetting bugs, then cancellation and audio recovery, before
adding more capabilities.

## Probe artifacts

The temporary review harnesses are available locally at
`/tmp/ciel-audit-probes.py` and `/tmp/ciel-audit-persona-probe.py`.
The wheel is under `/tmp/ciel-audit-wheel/`. These are disposable review
artifacts and are not part of the tracked test suite.

Run the probes from the repository root:

```bash
.venv/bin/python /tmp/ciel-audit-probes.py
PYTHONPATH=.:src .venv/bin/python /tmp/ciel-audit-persona-probe.py
```

The audio restart probe uses a daemon thread to demonstrate the lock deadlock;
that thread ends with the probe process. All provider responses, application
audio, and saved facts in these probes are synthetic.
