# Conversation and search evaluation

The reading-continuation regression is covered in `tests/test_reading_followup.py`.
It exercises the supplied latest-episode → named-episode novel continuation across
desktop voice, desktop typing, and headless chat. The lookup receives the new reading
goal, not both full questions concatenated. Tests also cover unrelated subjects,
cancellation, raw transcript preservation, and avoiding redundant episode questions.
The final ordinary suite passed **4,067 tests**, with **5 hardware tests deselected**;
focused follow-up and related checks passed **87 tests** on 2026-10-28.
Two real public searches used the corrected goal but did not establish a supported
adaptation mapping. These samples verify query continuity, not a successful reading-point
answer. Diagnostics are under `data/diagnostics/reading-followup/`.

The follow-up latest-episode regression passed **4,051 ordinary tests**, with
**5 hardware tests deselected**, on 2026-10-28. See `tests/test_episode_search_repair.py`
and the [release checks](release-checks.md) for the spoken false-start, failure
explanation, schedule quotation, and live reproduction checks. The acceptance
results below record the preceding task/search upgrade.

The current-task and native search upgrades are exercised through the desktop
runtime and headless `ChatSession`. Tests use invented conversations and sources;
they do not read personal conversation archives or use microphone recordings.

## Behavior covered

| Requirement | Evidence |
| --- | --- |
| Corrections retain title, episode, season, and requested outcome across voice/text | `tests/test_conversation_search_replays.py`, `tests/test_search_correction_runtime.py` |
| Retry uses the full task; continue receives interruption context and retained evidence | `tests/test_task_context.py`, desktop/chat replay tests |
| Wait permits resumption; cancel/no-search do not revive old work | Desktop/chat control replays and `tests/test_conversation_controls.py` |
| New topics, reset, owner changes, and private context remain separate | Task-context tests, `tests/test_dialogue_task_context.py`, `tests/test_research_runtime_privacy.py` |
| Broad named searches receive an overview; compound questions keep all parts | Replay cases, model evaluation below, `tests/test_web_goal_check.py` |
| Partial findings name only missing details actually requested by the user | `tests/test_main_tools.py`, `tests/test_web_answer_gaps.py` |
| Repetition and tool limits reserve one final synthesis without more tool execution | `tests/test_unified_agent.py` and bounded-search replays |
| New sources survive a full source allowance; later lookup failure retains evidence | `tests/test_main_tools.py` |
| A shared native research deadline reserves answer/checking time and rejects late output | Slow-read and slow-answer tests in main-tools/unified-agent tests |
| Raw turns, full typed text, brief speech, and permission checks remain intact | Replay assertions and existing channel/privacy regression suites |

The task projection comes from bounded, current owner dialogue. It is prompt data,
not an extra model decision, durable memory, or permission to act. Trusted output
format instructions are sent separately from untrusted page bytes. Retrieved
claims still pass quotation, scope, source, and independent relevance checks.

Native research allows bounded read rounds and at most one final synthesis with
tools disabled. Its 30-second budget begins when the responding tool session is
created, including the initial model turn for an explicit search. Read work stops
with eight seconds reserved; answer generation leaves four seconds for checking.
Automatic lookups admitted later use that same original start time. Unrelated
ordinary conversation is not assigned this research deadline. The legacy controller
diagnostic path is not the default native tool path and retains its old bounds.

## Reproduce

```sh
pytest tests/test_conversation_search_replays.py tests/test_task_context.py \
  tests/test_dialogue_task_context.py tests/test_main_tools.py \
  tests/test_unified_agent.py tests/test_web_answer_gaps.py
python scripts/evaluate_conversation_search.py
python scripts/evaluate_conversation_search.py --live
```

The default script writes an offline case manifest. `--live` sends only eight
synthetic dialogues and reference excerpts to the configured model; actual web,
screen, audio, and personal-history access are not used. It checks the objective
at the real lookup boundary, useful supported output, citations, unchanged raw
turns, and the deadline. Each case permits six logical foreground model requests
and 30 seconds, with 240 seconds across the run. Provider fallback may add HTTP
attempts within a logical request. Reports include the synthetic model output for
diagnosis and are written under ignored `data/diagnostics/`.

These checks establish application behavior and a small hosted-model sample.
They do not establish microphone accuracy, live website compatibility, continuous
provider availability, or sustained desktop acceptance.

## Recorded acceptance — 2026-10-28

- Complete ordinary suite: **4,026 passed, 5 deselected**, 90.33 seconds.
- Focused checks listed above: **177 passed**, including **39 desktop/chat replays**.
- Hosted model gate: **8/8 passed**, 3.373–12.250 seconds, one fixture lookup and
  three logical model requests per case. Normal model: Super 3.5 VL 120B.
- A desktop playback interruption followed by typed continuation preserves retrieved
  evidence and citations without rerunning the completed search. The broader related
  interruption, memory, privacy, and ambient checks passed 210 tests.
- One real Parallel search returned a cited Python overview from official Python
  documentation in 7.043 seconds. Temporary paired-session smoke checks verified
  a Telegram repair is used in the next voice retry and unpairing removes access.
- Ruff, whitespace checks, isolated wheel build, and packaged import/behavior smoke
  checks passed. Results are in `data/diagnostics/task-search/pytest.log`,
  `replays.log`, `live-final.json`, `real-search.json`, `paired-smoke.log`,
  `build.log`, and `wheel-smoke.log`.

Earlier live probes exposed a format instruction buried inside a tool result:
the model's useful prose was correctly rejected by the strict report parser.
The current path supplies trusted report rules as system context after retrieval;
untrusted evidence remains tool data. Other probes encountered provider/checking
timeouts. The final gate passed without changing quote or relevance approval to
accept unchecked answers.
