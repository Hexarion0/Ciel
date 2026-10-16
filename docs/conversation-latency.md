# Conversation latency findings

This is a historical benchmark report, not a list of current defaults. The
measured run used a 1.8-second endpoint; current code uses 0.7 seconds of silence
plus 0.2 seconds of grace. That later 0.9-second setting needs live acceptance
and does not inherit the no-cut or latency results below. Current configuration
is documented in the [README](../README.md).

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
