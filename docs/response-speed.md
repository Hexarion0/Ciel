# Response speed pass — 2026-11-01

The owner's priority is a faster everyday assistant. This pass addresses measured
local overhead and delivery barriers. It does not measure complete microphone-to-
speaker latency on the owner's devices.

## Model reasoning

The local configuration forced reasoning on for every conversation. Four short
hosted requests used the same configured normal NIM model and an invented,
single-sentence desk-organizing question. Only timing and model metadata were saved;
the check sent no saved conversation, memory, recordings, or images.

| Mode | First visible text, two samples | Median |
| --- | --- | --- |
| Off | 0.9573 s, 0.6590 s | 0.8082 s |
| On | 1.2326 s, 1.5877 s | 1.4102 s |

This small sample measures first content, not answer quality, complete generation,
or speech. Hosted latency can vary. The saved local `NIM_REASONING` setting now
uses `auto`: ordinary replies skip reasoning, and complex requests retain it.
The configured models and voice remain selected. Restart a running CIEL process
to load the saved setting. The timing report is in the ignored
`data/diagnostics/response-speed/reasoning-timing.json`.

## Browser delivery

Browser text now uses the existing validated answer stream, so available text can
appear before the model finishes. One completed answer is archived; cancellation
marks the delivered partial answer as interrupted. Provider errors retain visible
text and explain the interrupted connection. Tool-answer validation still controls
which content the main agent releases. Streaming withholds possible credential
prefixes until they can be masked; interrupted display and archives discard a
withheld prefix. Successfully completed text releases ordinary trailing text.

Audio playback previously waited until one chunk finished before fetching and
decoding the next. Playback now prepares at most one upcoming chunk while the
current chunk plays. Tests hold playback open and establish that the next fetch
and decode overlap it. They also cover order, bounded prefetch, autoplay recovery,
network/decode failures, and discarding work after Stop. No physical phone or
speaker timing was measured.

## Curated memory lookup

Automatic recall restricts its search to `MEMORY.md` and `USER.md`, but the old
implementation discovered and reconciled the entire archive three times per
search. Exact-path searches now read and reconcile only the selected documents.
Explicit full-workspace searches retain complete reconciliation. Scoped updates
preserve unrelated index rows, and selected edits, deletions, provenance, and
local-only boundaries still receive fresh checks.

Five warm runs per synthetic workspace returned the same single result. The scoped
measurements below preload the complete archive index, so they include the cost of
working alongside all indexed notes:

| Extra archive notes | Previous median | Scoped median |
| --- | --- | --- |
| 0 | 1.616 ms | 1.908 ms |
| 100 | 18.748 ms | 1.736 ms |
| 1,000 | 168.496 ms | 1.899 ms |

These figures measure curated search only. They do not establish an improvement of
the same size for complete memory assembly or an everyday spoken turn.

## Remaining measurements

Browser speech still derives its brief excerpt from the completed written answer.
First-turn voice initialization, actual microphone recognition, hosted inference
under normal conversation context, public research, and audible device playback
need separate current measurements. Historical benchmarks are useful leads, but
are not current end-to-end acceptance results.

## Validation

- Full ordinary suite: **4,619 passed, 5 hardware/model integration tests deselected**
  in 124.56 seconds.
- Ruff passed for `src`, `tests`, and `scripts`; diff whitespace checks passed.
- Changed Python files parse as Python 3.10; browser audio JavaScript syntax passed.
- Independent review checked request ownership, cancellation, memory freshness,
  scoped index preservation, and streamed credential masking.
