# Ciel XTTS Kaggle pilot

Private notebook: https://www.kaggle.com/code/hexarion0/ciel-xtts-kaggle-pilot

This automates the reviewed `xtts_ciel.ipynb` using the official Kaggle CLI/API.
Only experimental training scripts are changed. Production Ciel TTS, `.env`, accepted
Chatterbox settings and the original notebook are untouched. Credentials are never bundled.

This is an archived experiment. The 2026-10-17 cleanup removed its rejected local
XTTS environment and model downloads. Running it again requires deliberately
recreating those dependencies and the original sealed private transfer inputs.

## Setup and commands

Install the official `kaggle==2.2.4` CLI in a separate local environment:
`uv venv --python python3 .venv-kaggle`, then
`uv pip install --python .venv-kaggle/bin/python kaggle==2.2.4`.
The CLI reads the user's `~/.kaggle/access_token` directly. Keep this file mode `600`.
Pass your notebook account with `--owner` or set `KAGGLE_USERNAME`; the runner has
no default personal account. The reviewed notebook and dataset above belong to
the original pilot; another account also needs access to its training inputs.

From Ciel's root:

```bash
export KAGGLE_USERNAME=your-kaggle-account
python scripts/kaggle_pilot.py prepare --kaggle .venv-kaggle/bin/kaggle
python scripts/kaggle_pilot.py run --kaggle .venv-kaggle/bin/kaggle
```

`run` verifies the authenticated owner, refuses duplicate active runs, preserves existing
notebook source/logs, builds official metadata, pushes/runs, reads server settings back,
polls, downloads all outputs/logs, verifies/imports the inference archive, checks the
checkpoint locally on CPU and builds Ciel's existing rating-gated blind comparison.
Generated notebook and default runs live under the gitignored directory
`data/voice/finetuning/20261014-01/cloud/kaggle-pilot/`.
The optional `scripts/kaggle_forward_probe.py` also requires `--owner` or
`KAGGLE_USERNAME` and accepts `--kaggle` and `--output`. Its default diagnostic
output now lives in this private directory's `runs/forward-diagnostic-TAG/` folder.

For an already submitted run, use `monitor`, never an unnecessary new push:

```bash
python scripts/kaggle_pilot.py monitor \
  --kaggle .venv-kaggle/bin/kaggle \
  --run-directory data/voice/finetuning/20261014-01/cloud/kaggle-pilot/runs/pilot-05
```

`fetch` retries download; `finish` retries local import/check/comparison. Neither submits.
Inspect `run-state.json`, `cli.log`, `kaggle-execution.log`, `outputs/pilot-training.log`
and `failure-diagnosis.json`. An ambiguous or failed submission cannot be blindly repeated
in the same run directory. A reviewed retry uses a new run directory and the same private
notebook. The active session is polled every 45 seconds with `kaggle kernels status`.
Live logs are available through `kaggle kernels logs OWNER/SLUG --follow`.

Output download uses the authenticated official Kaggle SDK's output listing and streams its
supplied download URLs in 4 MiB chunks. CLI 2.2.4's `kernels output` buffers whole files in RAM,
which is unsuitable for these multi-gigabyte checkpoints on the local 16 GB machine.
The streaming helper preserves every output path, checks HTTP completion, commits downloads
atomically, enforces a local 3 GiB disk reserve, and keeps signed URLs out of logs.
Progress is in `download.log`; `streamed-download-inventory.json` records downloaded hashes.
The API currently selects latest output, matching the official CLI; the exported archive and
resume checkpoint are checked against their committed SHA-256 values before acceptance.

## Runtime and storage

- Private, internet-enabled `NvidiaTeslaT4` notebook attached to
  `hexarion0/ciel-xtts-training`. Source mount is
  `/kaggle/input/datasets/hexarion0/ciel-xtts-training`.
- Supports the observed expanded dataset folders and sealed ZIPs; every source file is
  checked against the original sealed transfer hashes. All 408 clips, split metadata,
  tokenizer IDs, reference and seven accepted Chatterbox baseline WAVs are validated.
- Retained outputs go to `/kaggle/working`; code/assets/venv go to `/tmp`.
  `MPLBACKEND=Agg`; exactly one GPU is exposed.
- Exact Python 3.12.14 venv is safely created or reused. Changed/incomplete environments
  abort; lock identity, Torch/torchaudio pins and `uv pip check` are verified.
- `TOTAL_EPOCHS=2` plus baseline-only epoch 0. Native AdamW, accumulation, batch and
  precision stay unchanged. No local training or OOM retries.
- `CIEL_KAGGLE_PILOT=1` explicitly replaces the 60 GiB preflight with capacity checks.
  Other cloud runs keep their original storage requirement.
- Retain one latest full model/Adam/scheduler/scaler/config/identity checkpoint. A second
  file exists transiently during replacement. The old state is deleted only after the
  successor is written, fsynced, hashed and its latest pointer committed. Same-step epoch
  saves use a new filename, protecting the previous pointer during a failed commit.
- Best validation is a compact inference-only checkpoint in a separate directory; it is
  explicitly marked non-resumable. The latest full checkpoint is the strict resume source.
  No redundant native best-model copy. Baseline epoch 0 cannot become a trained export.
- Preserve all WAVs/manifests, logs, metrics and final inference ZIP. Old sample manifests
  retain provenance hashes when their periodic full checkpoint is pruned.
- Enforce a 3 GiB disk floor throughout setup and training, including subprocess watchdogs.
  Checkpoint/export writes reserve their full estimated size plus the floor. Insufficient
  space stops the run while retaining the previous committed full state.
- Kaggle working disk is ephemeral during execution. Successful output persistence and
  download provide post-run recovery; abrupt VM loss can lose in-session progress.
  A new CLI push does not implicitly reattach prior outputs. Cross-session resume needs
  an explicitly attached/copied complete prior workspace and strict `--resume` validation.
  Compact inference archives cannot resume training.

## Diagnosed setup failures

Attempt 1 stopped before model construction: CUDA usable memory was below the inherited
15 GiB physical-memory threshold. The T4 has 15360 MiB physical memory, but CUDA reports
14911.6875 MiB after 448.3125 MiB driver reservation. The explicit T4 pilot now allows at
most 512 MiB reservation, requires physical capacity of 15 GiB, CUDA usable capacity of
at least 14.5 GiB and at least 12 GiB CUDA-free. Other GPUs retain the 15 GiB threshold.
Both memory readings are logged. Actual OOMs stop without batch/optimizer substitutions.

Attempt 2 passed memory/dataset checks, then rejected `Infinity` in native Coqui config's
unlimited `max_audio_len` and `max_text_len`. Resume identity now represents those native
sentinels explicitly in strict JSON. Model config and non-finite-loss safeguards are
unchanged. Neither failed attempt performed optimizer updates. All outputs/logs are retained
under `/home/hexarion/test/ciel-kaggle-pilot/runs/pilot-01` and `pilot-02`.

Attempt 3 completed baseline evaluation and seven XTTS baseline WAVs, then safely stopped
on the first training batch: text, mel and total loss were NaN at global step 0. No optimizer
updates, trained checkpoint or final inference export were produced. Its complete logs,
metrics and baseline WAVs are preserved under `runs/pilot-03` in the automation workspace.
The forward-only diagnostic `hexarion0/ciel-xtts-kaggle-diagnostics` reproduced the problem:
all inputs were finite; the first non-finite activation was the conditioning Perceiver's
first feed-forward block in FP16. Identical FP32 passes were finite; conditioning masking
did not change the result. Diagnostic runs perform no backward or optimizer updates and
verify that model parameters remain unchanged.

`CIEL_XTTS_FP32_PERCEIVER=1` selects an explicit FP32 autocast region for that block while
retaining native FP16 AMP elsewhere, FP32 parameter identities, AdamW, scheduler and scaler.
This policy is recorded in configuration and the full resume identity; a differently
configured checkpoint cannot silently resume. This fix is tested separately before training.
Non-finite forward losses now stop before backward and record exact loss names and step.
The T4 forward-only test confirmed finite training/evaluation losses with the FP32 region:
training total loss 5.05053 versus the all-FP32 reference 5.05091 on the same batch.
All-FP16 controls still produced NaN. Parameter tensors remained unchanged and no optimizer
updates occurred during the diagnostic. The Kaggle pilot explicitly enables the tested policy.

Attempt 4 then completed 31 finite training batches and stopped before backward on batch 32
with mel-only NaN. No optimizer update had occurred (accumulation is 252). The native prompt
loss mask used bounds in 256-sample Perceiver units against 1024-sample audio-code targets,
which can ignore every mel target in a short clip and produce an undefined mean loss.
`CIEL_XTTS_MASK_UNITS_FIX=1` corrects only those two loss bounds, preserving the native
conditioning mask and every model/optimizer state key. The reviewed upstream function's
SHA-256 must match before this runtime patch applies; its policy enters strict resume identity.
The exact native forward reproduced this failure locally with finite logits and became
finite after the two-bound fix. The separate T4 validation then visited all **369 training
clips**, with training forward/masking enabled: every loss was finite, no backward or optimizer
calls occurred, and parameter versions were unchanged. Evidence is in the automation workspace
under `runs/forward-diagnostic-03/outputs/ciel-xtts-training/forward-diagnostic.json`.
The next pilot uses the validated precision and masking fixes, still for two TOTAL epochs.

## Local artifacts and listening

The downloader keeps the full Kaggle output tree, including the one latest full checkpoint.
`pilot-complete.json` identifies the final inference ZIP/hash. Import checks every archived
file; a separate CPU `torch.load(weights_only=True, mmap=True)` check verifies trained
GPT tensor keys, epoch/step and absence of optimizer state. This is a structural check,
not a local speech/performance benchmark or voice-quality acceptance.

`blind-comparison/page/index.html` contains seven independently randomized comparisons
using real downloaded XTTS epoch WAVs and unchanged accepted Chatterbox baseline WAVs,
with matching text/reference and validated audio hashes. Ratings and preference must be
entered before identities reveal. These are listening samples; the manifests identify
their synthesis protocols and no latency/RTF comparison is inferred from them.
Use `tts-cloud-preflight.md` for optional matched local inference and sustained benchmarks.
Production remains unchanged pending a separate quality decision.

## Completed pilot and four-way progression listening

The downloaded `runs/pilot-05` summary records a successful two-epoch Tesla T4 pilot:
**471.97 seconds (7 min 52 s)** in the measured training session, excluding environment setup
and download time. There were 738 microbatches and **four AdamW updates**. Peak allocated VRAM
was 9,144.83 MiB; peak reserved VRAM was 10,902 MiB. This is cloud training evidence, not a
GTX 1660 SUPER runtime benchmark or a local training ETA.

| Check | Base XTTS | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Validation mel cross-entropy | 4.4763 | 4.2939 | 4.1808 |

The epoch-2 inference export and its full optimizer/scheduler/scaler resume state are verified.
Epoch-1 WAVs and checkpoint provenance remain available; its model weights were pruned by the
explicit Kaggle storage policy. Falling validation loss does not establish more natural speech.
Four optimizer updates are a pilot, not enough to declare adaptation succeeds or fails.

The original two-way `blind-comparison/page/` remains intact. A separate progression page compares
the accepted Chatterbox audio, native base XTTS and both epochs using all seven identical texts.
XTTS baseline/epoch inference protocols, source reference hashes and all WAV hashes are checked.
Each row has independently shuffled A–D letters and requires ratings before revealing identities.
These are downloaded validation WAVs; no new GPU synthesis is needed to audition them.

Reproduce the progression page from the repository root:

```bash
.venv/bin/python -m scripts.kaggle_pilot_compare data/voice/finetuning/20261014-01/cloud/kaggle-pilot/runs/pilot-05/imported-experiment data/voice/finetuning/20261014-01/cloud/kaggle-pilot/runs/pilot-05/progression-comparison --include-progression
.venv/bin/python -m http.server 8771 --bind 127.0.0.1 --directory data/voice/finetuning/20261014-01/cloud/kaggle-pilot/runs/pilot-05/progression-comparison/page
```

Open `http://127.0.0.1:8771/`. Start with the conversational row, then check warm, soft,
technical and longer speech. Download listening notes before moving browsers. The server exposes
only the page/reference/blind audio folder, without model weights, source material or credentials.
`progression-comparison/verification.json` records the HTTP/audio checks and current pending
listening decision. Human listening and matched local inference benchmarks still determine whether
the trained voice is worth further training or approaches the accepted Chatterbox voice.
