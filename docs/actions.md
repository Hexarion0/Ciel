# Local actions

CIEL discovers modules inside the installed `ciel.actions` package. Each
module exports an `ACTION` instance implementing `Action`. Adding a trusted
module does not require editing the voice handler or registry. Restart after
adding one. This is Python code loaded with the application's privileges;
discovery is not a sandbox or an installer for third-party plugins.

The initial actions answer RAM usage, GPU temperature, and logical CPU core count,
and launch supported Linux apps. Full-sentence patterns run before the provider.
Polite requests such as “can you open Discord for me” also use the local handler.
Quoted questions and unrecognized requests remain conversational. Unsupported
`open ...` requests receive a supported-app list rather than being executed.

App references use up to three recent accepted user requests held in RAM for
at most 60 seconds. “Open it” can resolve a preceding exact app name, launch
request, or spelling correction. Unrelated turns stop resolution. Cancellation,
restart, and uncertain transcription discard this context. Assistant guesses and
old conversation archives are not used. A reference never grants permission to
launch: ambient side effects still require a confident direct address.

## Contract

Implement the interface in `src/ciel/actions/base.py`:

- Give the action a unique `name` and a short `description`.
- `match(text)` returns an argument dictionary for one complete command, or `None`.
- `match_with_context(text, recent_requests)` optionally resolves references; the
  default calls `match`. Context contains user text, never execution permission.
- `validate(arguments)` rejects missing, extra, and invalid fields with `ActionError`.
- `execute(arguments, context)` returns an honest user-facing result.
- Set `mutating = True` for side effects. Set `requires_confirmation = True` for
  destructive or otherwise confirmation-requiring operations.

All execution paths pass validation. Only confident, explicitly addressed requests
can perform side effects in ambient mode; guessed follow-ups can read telemetry.
A pending confirmation is tied to the exact validated arguments, expires after
60 seconds, and is consumed once. Cancellation, a different request, or an
unauthorized confirmation clears it. No destructive action ships in this version.

Execution has a cooperative three-second deadline. Call `context.remaining()`
before side effects and supply its remaining budget to blocking operations.
Check again before returning results from read operations. The registry does not
forcibly terminate Python code; action authors must honor this contract.
The GPU probe is limited to at most 1.5 seconds. Applications are started without
a shell and run independently; their lifetime is not limited to three seconds.
A daemon reaper waits for application exit without blocking the voice loop.

Registry errors produce a local explanation without dumping exception details
into speech. An unavailable metric is reported as unavailable, never as zero.
The app launcher resolves only a fixed allowlist with `shutil.which`, accepts no
user-supplied executable paths or arguments, and does not claim window readiness.
Once a launch has happened, later cancellation cannot undo it.

## Validation

```bash
pytest tests/test_actions.py tests/test_ambient.py tests/test_memory_service.py
ruff check src tests
```

Test new actions with temporary files and mocked external effects. Include invalid
arguments, ambiguous matches, denied authorization, cancellation, timeout, missing
dependencies, and permission errors. Manually verify real desktop behavior using
`docs/release-checks.md` before recording live acceptance.

Explicit, reliably transcribed requests for existing local handlers bypass the AI
controller. Other requests can reach the local-tool path through its decision.
Execution still parses the original
user request and applies the existing authorization checks; model-generated arguments
cannot create shell commands or approve memory writes. General function calling and
reminders remain future work.

## Public internet retrieval

The lightweight NIM controller selects search or public HTML/text page reading,
speaks a short acknowledgment, then retrieves evidence before reporting an answer.
It can refine a query or read a page when evidence is insufficient, within
`CONTROLLER_MAX_TOOL_STEPS`. This path runs separately from the trusted action
registry. It retrieves evidence and source links; it does not execute website
account actions, downloads, purchases, or browser automation. Tool output cannot
authorize a local action.

Current-information requests trigger searches by default. Explicit search and
public HTTP(S) links also request retrieval; research, comparison, and double-check
requests use larger bounded budgets. Relevant user memory and conversation context
can refine queries. `/local` keeps retrieval disabled for the session, including
follow-ups and restarts, until its history is cleared. `INTERNET_ENABLED=false`
disables retrieval globally. See [internet settings](../README.md#internet-access)
for limits and home-city setup.

Search or site failures must remain visible in the answer. Current facts require
retrieved evidence; general background after a failed lookup needs a clear label.
Source links appear in terminal and desktop text, while spoken replies stay concise.
PDF/media extraction and live voice/text acceptance remain pending.

```bash
pytest tests/test_controller.py tests/test_controller_runtime.py tests/test_internet.py
```
