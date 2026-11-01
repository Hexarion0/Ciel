# Speech performance comparison — 2026-11-01

The requested goal was lower delay, RAM and GPU use for the everyday assistant.
We measured Moonshine and Smart Turn locally before changing the speech pipeline.
The adopted change releases reference-only Chatterbox networks after voice setup.
Whisper and silence-based endpointing remain selected: the alternatives did not
pass all of the latency, resource and cutoff checks below.

## Adopted: release reference-only voice networks

The pinned Chatterbox Turbo worker prepares the primary voice and optional vocal
effect style once. Subsequent generation uses the prepared conditioning rather
than the voice encoder, S3 speech tokenizer or reference speaker encoder. The
worker now releases those three networks after preparation. S3Gen discovers its
device through its tokenizer's first parameter; a zero-element parameter retains
that device information without keeping the original tensors alive. A request to
encode another voice through this placeholder fails explicitly. This is specific
to the pinned, fixed-reference worker, not a general Chatterbox modification.

Two actual S3 decoder outputs, with identical tokens, conditioning and CPU RNG
seeds, were **bit-for-bit identical** before and after the change: 30,720 and
61,440 output samples, both with maximum absolute difference zero. The change
does not perform quantization or replace the voice. Timing differences between
these few decoder runs are not evidence of faster synthesis.

| Isolated measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| CPU decoder process private RAM | 1,589.25 MiB | 1,085.97 MiB | 503.28 MiB |
| CUDA allocation of the real reference networks | 516.50 MiB | 0 MiB | 516.50 MiB |

The CUDA probe moved only the three reference networks to GPU; the decoder stayed
on CPU. Their tensor accounting was 528,381,112 bytes. The CPU parity probe did not
load the T3 text-to-speech generator. These are component measurements, **not a
new before/after sample of the running CIEL desktop process**. Startup still loads
the networks before releasing them, so its peak requirement is not reduced by the
same amount. Restart CIEL to load the updated worker, then measure the whole app.
The worker's ready packet reports `reference_encoder_bytes_released`.
The CPU probe held encoder weights in system RAM; the normal GPU worker holds
them in VRAM. Its RAM saving should not be assumed to match the CPU probe's
503 MiB reduction.

Reports are in ignored `data/diagnostics/speech-performance/decoder-release.json`
and `reference-gpu-release.json`. Reproduce with the existing Turbo environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 data/voice/envs/clone/bin/python \
  scripts/benchmark_turbo_reference_memory.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 data/voice/envs/clone/bin/python \
  scripts/benchmark_turbo_reference_memory.py --reference-gpu-only \
  --output data/diagnostics/speech-performance/reference-gpu-release.json
```

Neither probe plays audio, records a microphone, downloads weights, or restarts
the existing assistant. The GPU probe requires at least 1 GiB free and tests only
the reference networks. Full GPU synthesis and audible acceptance after restart
remain separate device checks.

## Recognition: retain Whisper for now

The corpus contained the first 12 clean validation clips from
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
and the existing CIEL reference recording with its transcript: 13 recordings,
142.73 seconds total, two speakers. The public corpus revision was
`5be91486e11a2d616f4ec5db8d3fd248585ac07a`. The local manifest records audio hashes.
This is a small read-speech sample, not the owner's accent, microphone, ambient
noise or a representative assistant-command evaluation.

Whisper used CIEL's configured `small.en`, CUDA `int8_float16`, beam size 5 and
current confidence/retry handling on a GTX 1660 SUPER. Moonshine Voice 0.1.5 used
English Small Streaming, architecture 4, `quantized_26_08_21`, with updates every
0.5 seconds. Replay delivered 80 ms frames at real-time cadence. Backends ran in
separate processes; desktop background activity was not controlled.

| Measurement | Whisper | Moonshine streaming | Moonshine single-thread experiment |
| --- | ---: | ---: | ---: |
| Repeats per recording | 2 | 2 | 1 |
| Weighted word error rate | 7.19% | 5.39% | 5.39% |
| Median time after final input frame | 551 ms | 361 ms | 594 ms |
| Maximum time after final input frame | 1,065 ms | 1,943 ms | 2,484 ms |
| Process CPU seconds / audio second | 0.143 | 4.976 | 0.788 |
| Peak private process RAM | 808.49 MiB | 580.33 MiB | 591.65 MiB |

Whisper starts decoding after the complete recording; Moonshine's row measures
remaining work after replaying the audio incrementally, including ingestion lag.
Neither number includes endpoint silence, LLM work, TTS or speaker playback. CPU
time includes resource sampling and inference and is not an idle-CPU estimate.
The GPU probe observed 498 MiB for the isolated Whisper process. GPU querying was
unavailable inside the Moonshine test environment; its zero-valued raw GPU fields
are not measurements. Moonshine used its CPU runtime.

The single-thread experiment used Moonshine's
`MOONSHINE_ORT_SINGLE_THREAD=1` diagnostic setting, which upstream describes as a
ThreadSanitizer reliability escape hatch, not a supported production tuning API.
It reduced CPU work but worsened tail delay. The default streaming configuration
had better median delay and aggregate WER on this sample, but much higher CPU
work and worse maximum delay. It also does not currently provide CIEL's existing
Whisper confidence/retry contract. We therefore did not replace the recognizer or
enable live Moonshine transcription. Its partial transcripts were benchmark-only
and never entered the conversation or authorized an action.

The reusable offline comparator is `scripts/benchmark_speech_backends.py`.
Install optional dependencies with `pip install -e '.[voice,speech-benchmark]'`
in a separate environment. Model acquisition is explicit, through
[`get_model_for_language`](https://moonshine-voice.readthedocs.io/en/latest/using/)
with `ModelArch.SMALL_STREAMING` and a selected cache root. The benchmark itself
never downloads. Example commands after preparing a manifest and model directory:

```bash
python scripts/benchmark_speech_backends.py --backend whisper \
  --manifest data/diagnostics/speech-performance/corpus.json \
  --output data/diagnostics/speech-performance/whisper.json

python scripts/benchmark_speech_backends.py --backend moonshine-stream \
  --manifest data/diagnostics/speech-performance/corpus.json \
  --model-path data/diagnostics/speech-performance/models/download.moonshine.ai/model/small-streaming-en/quantized_26_08_21 \
  --output data/diagnostics/speech-performance/moonshine-stream.json
```

## Endpointing: Smart Turn did not pass the cutoff proxy

We tested Pipecat's
[`smart-turn-v3.2-cpu.onnx`](https://huggingface.co/pipecat-ai/smart-turn-v3)
locally with ONNX Runtime, CPU provider and one inference thread. The model was
8,679,182 bytes, SHA-256
`2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`.

Silero supplied the same speech/silence labels throughout each replay. We checked
completion after 320 ms of silence, using Smart Turn's last-eight-seconds,
left-padding and waveform-normalization contract. The existing faster-whisper
feature extractor avoided importing Transformers/PyTorch. Its features matched
the official Pipecat NumPy implementation within maximum absolute error
1.03e-6 on three synthetic lengths.

At the model's default probability threshold of 0.5, it marked 9 of 13 completed
clips as complete at that first candidate. It also introduced **6 potential
cutoffs before later recorded speech** that the existing 0.9-second silence rule
would have retained. Median feature extraction plus model inference was 86 ms,
in addition to the 320 ms silence. We did not enable earlier endpoints.

Read-speech pauses are only a proxy: later speech can begin a separate sentence
or turn. These results do not establish that Smart Turn is generally inaccurate;
they establish that this proposed early-endpoint configuration did not pass our
no-additional-cutoff check on the available recordings.

Reproduce using the local model and existing corpus:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. python \
  scripts/benchmark_turn_detection.py \
  --model data/diagnostics/speech-performance/models/smart-turn-v3.2-cpu.onnx \
  --manifest data/diagnostics/speech-performance/corpus.json \
  --output data/diagnostics/speech-performance/smart-turn.json
```

## Validation

Regression tests cover release of original tensors, preservation of device and
conditioning, rejecting re-encoding, and refusing unsupported/unprepared layouts
before mutation. Benchmark tests cover transcript edit distance, silent-reference
insertions, stereo resampling, invalid samples and the endpoint model's recent
audio/padding contract. Real component parity and CUDA accounting are recorded
separately above. No other project's assistant application code was copied into
CIEL; Moonshine and Smart Turn remained isolated evaluation dependencies.

- Focused audio, worker, cancellation, resource and benchmark checks: 156 passed.
- Full suite with local server/session access: 4,681 passed, 5 deselected, 1 failed.
  The remaining pre-existing failure is
  `tests/test_web_evidence_passages.py::test_many_sources_fit_the_native_tool_result_and_keep_late_relevant_evidence`:
  web-tool JSON is 74,833 characters against a 64,000-character limit. Speech files
  do not modify that path. A restricted initial run also blocked local sockets and
  session probes; the host-access rerun resolved those permission-only failures.
- Scoped Ruff, whitespace checks and Python 3.10 syntax parsing passed.

The unused experiment weights and temporary environment were removed afterward
(about 204 MiB on disk); reports, corpus and model hashes remain under the ignored
diagnostics directory. Model-based reproduction requires fetching those weights
again explicitly.

## Post-restart desktop snapshot

The updated voice worker was confirmed running after the source update. The idle
whole-application sample used proportional RAM (PSS), including the small background
service, and GPU allocations reported for the app and voice worker.

| Measurement | Previous live check | Updated live check |
| --- | ---: | ---: |
| CIEL GPU allocation | 3,360 MiB (3.28 GiB) | 3,000 MiB (2.93 GiB) |
| CIEL proportional RAM | 3,549.6 MiB (3.47 GiB) | 3,578.0 MiB (3.49 GiB) |

Observed GPU use was 360 MiB lower; RAM was effectively unchanged. This is not a
controlled workload comparison, so the live difference need not equal the isolated
reference-network saving. The three-second idle CPU sample totaled 2.6% of one core.
The device reported 1,808 MiB free GPU memory. Measurements and restart evidence
are saved in ignored `data/diagnostics/speech-performance/live-after-restart.json`.
