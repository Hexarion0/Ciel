# Conversation latency findings

This is a historical benchmark report, not a list of current defaults. The
measured run used a 1.8-second endpoint; current code uses 0.7 seconds of silence
plus 0.2 seconds of grace. That later 0.9-second setting needs live acceptance
and does not inherit the no-cut or latency results below. Current configuration
is documented in the [README](../README.md).

## Requested larger models and reasoning — 2026-10-28

Normal conversation now requests `nvidia/nemotron-3.5-super-vl-120b-a12b`; complex work
requests `nvidia/nemotron-3-ultra-550b-a55b`. Ultra is also the explicit NIM backup.
The actual local configuration was updated as well as fresh-install defaults.

Synthetic live checks on 2026-10-27 found both identifiers in NVIDIA's catalogue.
Super inference returned 404, including through the streaming conversation path.
Ultra answered successfully, so the configured normal route fell back to Ultra
with a visible warning. Reasoning-off greeting, reasoning-on arithmetic, and an
automatic-reasoning native tool/result loop all returned final answers without
exposing internal reasoning. The tool loop used two main-model requests.
A 2026-10-28 refresh reproduced Super inference returning 404 and Ultra answering
successfully while the catalogue continued to list both models.

The replacement Super 3.5 VL endpoint passed direct synthetic inference on
2026-10-28 with reasoning off and on, native tool calls followed by tool results,
and streaming with the ordinary 400-token voice budget. No fallback was used.
The public request identifier is `nvidia/nemotron-3.5-super-vl-120b-a12b`; NVIDIA
reports backend alias `private/nvidia/nemotron-3.5-super-120b-a12b` in its response.
Both the saved local configuration and defaults now use the replacement.
Actual CIEL routing then completed a reasoning-off greeting, reasoning-on arithmetic,
and a two-round native tool/result reply on the replacement without falling back.

The reasoning preference is `auto` by default: off for ordinary replies, on for
complex work. `/reasoning on`, `/reasoning off`, and `/reasoning auto` change the
session preference. The NIM thinking allowance is at least 8,192 completion tokens;
hidden reasoning counts toward that limit. Voice delivery remains brief, while
typed chat supports detailed answers and complete code. Desktop typed replies also
speak a deterministic excerpt of at most 80 words after generation, with code and
the full answer retained in chat. That excerpt does not make another model request.

The complete ordinary regression run passed 3,821 tests, with five hardware cases
deselected. Packaging and lint checks passed. These are functional checks rather
than a controlled comparison of answer quality or end-to-end voice latency.
The subsequent model replacement and restored startup panel passed 312 focused
provider/routing checks and 109 terminal/startup/CLI checks.

## Main-model tool selection — 2026-10-27

The default foreground flow now calls the responding model directly. It can produce
an answer or native tool calls in that request. Tool results continue through the
main model; ordinary conversation no longer pays for a separate controller request.
Clear local commands and screen requests retain application fast paths. Ambiguous
screen follow-ups reach the main model without a separate screen classifier.
The retained controller benchmark below measures the old architecture, not the
new default path. `CONTROLLER_ENABLED=false` retains the legacy desktop intent flow.

Native adapters retain tool arguments outside text and speech. The application
validates declared schemas, current request ownership, tool permissions, and result
scope. Web reports are buffered until existing attribution and task relevance checks
pass. Relevance checks use normal conversation routing, not the decision model.
Tool-assisted requests can still require multiple inference requests.

Synthetic live checks against the configured NIM Lightning endpoint completed an
ordinary greeting in one main request and a tool/result/answer in two requests,
including streamed tool arguments. They used invented tool data, not saved history
or desktop pixels. The full ordinary regression suite passed 3,725 tests; five
hardware integration tests were deselected. No new microphone-to-audio latency
claim is inferred from these checks.

## Historical compact decisions and screen recall — 2026-10-27

The controller chooses an action before the reply model answers. A live synthetic
probe showed an ordinary conversational decision returning all 13 schema fields,
including unused query, goal, speech, and memory defaults. That response contained
220 characters and took 1.41 seconds. Two broader synthetic checks took 1.67 and
2.59 seconds to choose an action, independently of answer generation and speech.

The controller now requests only applicable fields. `answer` and `local` need
`intent`, `action`, and `confidence`; web, clarification, and memory actions add
their required arguments. The validator retains its existing defaults and strict
tool checks. Format repair uses the same compact contract and includes both
memory actions. Successful decision logs now include elapsed decision time.
Both prompt variants keep the actual current request in its own user message,
with earlier dialogue and routing context in a separately labeled system block.

An alternating before/after comparison used the same configured
`nvidia/nemotron-3.5-lightning-30b-a3b` route, two synthetic ordinary/contextual
cases, and two repeats per format. All eight choices returned the expected
`answer` action without a repair:

| Measurement | Previous full-field format | Compact format |
| --- | ---: | ---: |
| Median decision time | 2.186 s | 1.247 s |
| Observed decision range | 1.540–2.586 s | 0.955–1.689 s |
| Median returned JSON size | 297.5 characters | 56 characters |

This sample reduced median routing time by about 43%; it is not a guarantee of
future provider latency or complete speech-to-audio response time. Input
instructions grew slightly to explain conditional fields. The measured reduction
was in generated output; one contextual reply still emitted extra optional fields.
The existing eight-second shared decision/repair deadline remains unchanged.

The wider live checker passed 21 of 24 cases on its first compact-format sweep.
It exposed incorrect routing for a search-failure explanation, an ambiguous
selection among multiple choices, and a weather query missing its configured
city. Paired old/new probes reproduced the explanation failure in both formats;
the other two varied between runs. Clarifying the instructions and separating
the current message did not reliably eliminate these errors. These measurements
establish a formatting speedup, not complete routing reliability. The unresolved
cases remain in the checker rather than being reclassified as passes.

Controlled screen requests also reuse their already recalled context when building
the screen prompt. Regression tests measured two recall calls before the change
and one afterward for voice/keyboard input with ordinary/protected history.
They preserve the actual recalled preference, local-only marker, screen result,
and absence of a general controller call. This removes duplicate local work;
no separate wall-clock saving is claimed for that path.

The provider still contributes network and service wait, and an ambiguous active
screen follow-up can run a screen classifier before general routing. Those stages
remain separate. Use the synthetic checker to assess the currently configured
route without loading personal conversation data:

```bash
.venv/bin/python scripts/check_controller.py 'new task after pending request' 'number reaction'
```

## Controller follow-up — 2026-10-17

The conversation routes already selected Super, but the separate controller still
defaulted to Lightning. Six synthetic controller requests each reached the
eight-second application deadline without a valid model reply. The old checker
incorrectly marked two as passed after local safety rules supplied an answer;
diagnostics now require a valid model response as well as the expected final action.

A direct probe excluded CIEL's conversation and audio path: the authenticated model
catalog returned HTTP 200 in 0.46 seconds and listed Lightning, but a tiny Lightning
stream received no response headers before its 12-second read timeout. The same
synthetic request through the configured Super conversation model received headers
in 0.62 seconds, first content in 0.73 seconds, and completed in 0.77 seconds.
Changing only the controller model then passed the same six cases in 1.14–5.15
seconds each. These measurements locate this observed delay at the hosted model
endpoint; they cannot distinguish its internal queueing from model execution.
The corrected checker then passed all 21 checks: 20 valid controller decisions in
1.08–3.09 seconds each and one report validated against a synthetic page.

In that run, fresh controller configuration was changed to
`nvidia/nemotron-3-super-120b-a12b`, matching the measured conversation route. These
synthetic checks do not establish microphone or speaker quality or guarantee
future hosted-service latency.

## Endpoint availability follow-up — 2026-10-20

The configured Super model returned HTTP 404 during an actual greeting. The same
account's authenticated catalogue still listed it, so catalogue presence alone did
not establish inference availability. A direct synthetic request reproduced 404;
the old 90B vision backup exceeded a ten-second read timeout. This evidence does
not establish that either model is retired globally.

Lightning answered a tiny non-thinking JSON request in 0.45 seconds. Through the
actual CIEL controller, greeting, search selection, and contextual follow-up cases
returned the expected decisions in 1.34, 1.20, and 1.41 seconds. A normal streamed
greeting reply completed in 0.98 seconds. Only synthetic prompts were used.

Defaults now use `nvidia/nemotron-3.5-lightning-30b-a3b` for controller, ordinary and
complex replies, and the conversation backup. A matching primary/backup avoids a
redundant retry. Explicit model overrides remain available. The controller retains
its existing eight-second deadline and does not implicitly fall back to a large
model. These few successful checks repair this account's observed availability
problem; they do not replace the earlier benchmark or guarantee future latency.

## Result

CIEL was not waiting for the complete reply. It already requested an SSE stream,
read each event as it arrived, filtered hidden/code text incrementally, and sent complete
sentences into the existing bounded Chatterbox synthesis/playback pipeline. On the
original live requests, CIEL's “First reply text” measurement was the first usable
visible provider content, not the end of generation.

The observed wait is before NIM's first streamed content. The original 24–25 second
live events included 25.35s and 23.85s to first text. In isolated requests with the
same frozen CIEL persona, memories, recent turns, and spoken-input instructions,
TCP plus TLS took about 0.27–0.30s while NIM took 12.56–13.48s to send the first
content in the initial trials. Later trials of the same Lightning model ranged from
0.65s to 34.98s, with occasional read timeouts and one corrupted reply. This variation
is upstream of CIEL's first-token display: HTTP headers were sometimes withheld for
many seconds, and no chunks could be buffered or filtered locally before those headers.

Nemotron 3.5 thinking was already explicitly disabled in the request. In the captured
slow requests, the first model token and first answer content arrived together; no
reasoning text preceded them. Visible-text cleanup added less than a millisecond after
content. The prompt was around 14.4k characters / 3.1k NIM input tokens; local context
assembly took around 15–20ms. A reused HTTPS connection removed roughly 0.27s of setup,
not the multi-second service wait. The experiments cannot distinguish NVIDIA service
queueing from model prefill or other internal serving delay because those occur behind
the hosted endpoint's response-header boundary.

## Model comparison

The six comparison cases used byte-for-byte identical messages per case and two serial
repeats per model. The test preserved the current CIEL context snapshot and used
the same streaming payload shape and 400-token ceiling as the live path.

| Model | Cases | First content | Output quality check |
| --- | ---: | ---: | --- |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | 12 | 0.65–34.98s when content arrived; several stalled or failed | Mostly useful when fast; one malformed greeting and a difficult-case timeout |
| `nvidia/nemotron-3-super-120b-a12b` | 12 | 0.56–0.97s | Coherent greeting, caring everyday answer, follow-up, and distributed-queue explanation |

Super was consistently faster in this sample and kept enough detail for the difficult
question. Its answers are a little more compact. This is a small live service sample,
not a guarantee that NIM latency will always stay below one second. Two catalog-listed
12B alternatives returned HTTP 404, and the configured Llama 90B fallback timed out at
40s in a bounded comparison. Ollama on `localhost:11434` is not running. No Groq or
OpenRouter credentials are configured. NIM's model catalog alone was therefore not a
reliable indicator that a candidate could actually serve requests.

The selected change sets Nemotron 3 Super as both the conversational and complex NIM
route. A bounded direct test of the previous 550B complex model returned HTTP 200 with
an empty SSE stream; the configured 90B fallback had timed out at 40s in an earlier
direct test. Super completed that same question with a coherent 226-token answer in
2.75s.
Exact clock/date intents
already have local answer code, so these cases do not need an LLM after CIEL has
accepted the utterance. The benchmark includes counterfactual model responses for those
queries to compare hosted TTFT; the normal product flow answers them locally.

Streaming now gives NIM a 12s read-inactivity limit (while retaining the configured
overall limit for connect/write operations). A silent hosted stream can no longer hold
the voice reply open for the prior 60s default. Mid-stream read timeouts still cannot
splice a second model into speech after partial text; the provider manager tries the
remaining configured providers, and the current local Ollama endpoint is offline.

## Audio and speech measurements

The current STT model was measured as `small.en`, CUDA, `int8_float16`. Model loading
took 2.95s at cold start. Warm recognition of three clean 11–13s reference passages
took 0.52–0.66s, with exact transcripts and no retry. Four-second excerpts took
0.40–0.45s. This supports keeping the current recognition model and precision.

The endpoint setting is 1.0s VAD silence plus 0.8s pause grace. Replaying the three
human reference passages showed later speech after a 1.8s pause in two passages. At
0.6–1.0s, later words were also present after the cutoff across the passages. These are
not annotated user turns, but they do show a real false-cut risk. The default endpoint
was kept at 1.8s; the shorter settings did not pass the no-cut criterion. The standalone
replay and timings are in the ignored diagnostics directory.

The production stream already overlaps model generation, sentence synthesis, and audio
playback. It emits a sentence when punctuation is available (or at a 240-character
bound), so the first audio does not wait for the whole LLM answer. Existing Turbo
measurements put warm first audio around 0.73–1.56s after text becomes available. With
the measured conservative endpoint, STT, Super TTFT, short sentence boundary, and Turbo
startup, a warm direct-address turn is roughly 3.6–5.1s from final speech to first
audio; follow-up ambient speech can add a short gate request. The <3s target is not
supported by these measurements while keeping the current endpoint and voice startup.

## Instrumentation and foreground priority

Each completed microphone turn now logs privacy-safe monotonic boundaries for capture,
last VAD speech, endpoint, STT, speech gate, routing, prompt/context assembly, provider
request/HTTP/SSE/model/visible text, first speakable chunk, synthesis, playback, and turn
completion. Logs contain stage names, timing, model/provider identifiers, and prompt
size only; they do not include transcript, prompt text, audio, response content, or
headers. This will give the next real microphone session a direct before/after trace.

Background summaries now use cancelable provider streaming. A new foreground request
cancels an active summary, and summaries skipped or canceled this way remain pending
for a later attempt instead of saving partial content. Speech-gate work is also
cancelable on barge-in. No summary or gate mutex holds the conversation path while a
remote call is running.

The two original live traces did not include per-stage timing, so their 1.36s and 5.12s
gaps between first text and first audio cannot be split retrospectively into sentence,
synthesis, and playback time. The new turn trace records those stages for the next
live run. No Turbo inference settings or voice files were changed.

## Reproduction

From the repository root, use the already configured NIM account and keep the generated
diagnostics private (the prompt snapshot and responses live under ignored `data/`):

```bash
.venv/bin/python scripts/benchmark_conversation_latency.py \
  --cases time date greeting normal difficult followup \
  --models nvidia/nemotron-3.5-lightning-30b-a3b nvidia/nemotron-3-super-120b-a12b \
  --repeats 2 --reuse --label repeat
.venv/bin/python scripts/benchmark_speech_endpoint.py --stt
.venv/bin/python scripts/build_conversation_latency_report.py
```

The first run creates one read-only SQLite backup and freezes all six prompts in
`data/diagnostics/conversation-latency/contexts.json`; delete that directory to capture
a new context snapshot. Open the generated local `index.html` to compare response text
and first-content times. The endpoint script replays reference audio, not live user
speech, and reports later VAD activity after each hypothetical cutoff.
