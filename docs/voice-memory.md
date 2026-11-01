# Voice memory optimization — 2026-11-01

## Latest pass: release fixed-reference encoders

After both reference styles are prepared, the worker now releases the three
networks used only for reference encoding. The actual CPU decoder produced two
bitwise-identical outputs before and after release. Isolated probes reclaimed
503.28 MiB private CPU RAM and 516.50 MiB CUDA allocation. These are component
measurements; restart CIEL before measuring the whole desktop application.
Startup's peak loading requirement remains separate from resident memory.

See [speech performance measurements](speech-performance.md) for the exact scope,
reproduction commands, and Moonshine/Smart Turn evaluation results.

## Earlier pass: inference mode and allocator cleanup

The resource review found approximately 2.5 GiB private RAM and 3.4 GiB GPU memory
in the running Chatterbox Turbo worker. This process had loaded the earlier code;
the complete application needs a restart before measuring the changes below.

## Changes

- Prepare the normal voice reference and optional vocal-effect reference inside
  `torch.inference_mode()`. The reference speaker encoder previously retained an
  autograd graph even though the worker performs inference only.
- Collect unreachable objects and release unused CUDA allocator blocks after
  startup. On Linux with glibc, also return free CPU heap pages with `malloc_trim`.
- Repeat that cleanup once after 30 seconds without a synthesis request. A bounded
  input queue handles buffered requests; cleanup never runs concurrently with
  synthesis. Another request starts a new idle period. Maintenance failures do not
  disable a working voice.
- Retain model weights, prepared references, sampling settings, model precision,
  and the selected voice. No models are downloaded or substituted.

PyTorch distinguishes memory occupied by live tensors from reserved allocator
blocks. `empty_cache()` releases unused blocks; it does not shrink live model
weights. See [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

## Measured reference-encoder comparison

An isolated process loaded the actual local speaker-encoder weights and used ten
seconds of synthetic audio. No private recordings were used. The GPU probe capped
its allocator at 10% of the 6 GiB card and left the running assistant untouched.

| Measurement | Gradient tracking | Inference mode |
| --- | ---: | ---: |
| CUDA allocated after reference encoding | 307.71 MiB | 35.32 MiB |
| Saved training activations, CPU probe | 267.67 MiB | 0 MiB |
| Private RAM of isolated CPU probe | 563.59 MiB | 559.12 MiB |
| Reference embedding equality | Baseline | Bitwise identical on CPU and GPU |

The CUDA allocation reduction is **272.39 MiB for this component**. These are not
measurements of the whole voice worker or entire CIEL application. The CPU process
reduction was modest in this probe; total RAM reclaimed from checkpoint loading
and idle allocator cleanup remains to be measured after restarting CIEL. Do not
interpret these numbers as a guarantee of a particular total RAM or VRAM saving.

## Validation

The focused audio, worker-resource, vocal-cue, streaming, cancellation, and
repository-regression group passed 134 tests, with one model/hardware case
deselected. Dedicated regressions cover reference inference mode, optional effect
failure, cache accounting, cleanup failure, buffered requests, cleanup while idle,
and resuming requests after cleanup. Physical playback after a restart and a new
whole-application memory sample remain device acceptance checks.
