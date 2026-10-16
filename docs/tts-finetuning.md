# Ciel speaker adaptation experiment

**Archived experiment:** on 2026-10-16 the user rejected the XTTS pilot voice and
requested deletion of local training data/checkpoints and the dedicated cloud
preflight environment. Paths below describe the former experiment and are no
longer populated. Downloaded source speech and all cloning references were kept.
The next evaluation is [Chatterbox Nano on CPU](chatterbox-nano-audition.md).
Do not restart training automatically.

This experiment is **not a completed migration or a trained model**. Chatterbox Turbo remains the
accepted quality baseline. No runtime provider, voice reference, Pocket/Piper implementation, or
production setting is changed by these tools. Fine-tuning may improve XTTS identity/prosody; it does
not shrink the architecture or establish lower GPU power. Listening and sustained measurements
must decide that.

Current verified export: **408 clips, 32.56 minutes**, 369 train / 39 validation. The user
confirmed audio quality, every displayed transcript and speaker identity in this session. Three
clips were purged at split boundaries. A non-spoken caption marker and repeated whitespace were
removed without changing spoken words; the native XTTS loader checks the complete export.

Local artifacts: `data/voice/finetuning/20261014-01/`. Recordings, models, review decisions and
checkpoints are private ignored data, not repository assets.

## Dataset and review

The existing four permitted Clem Whispers archives contain 90.78 minutes in total. Preparation
created 411 non-overlapping **candidates**, 32.79 minutes. They are not certified usable speech.
The original archives and all existing references remain preserved. The superseded caption-timing
draft is retained under `dataset/`; use `dataset-v2/` for review. Its next-word boundary padding was
corrected before any review decisions or training export.

| Source ID | Candidate clips | Candidate minutes | Particular concern |
| --- | ---: | ---: | --- |
| 3mZ2ZbciDPM | 58 | 4.63 | Promotional opening excluded through 95 s |
| iP0AY_BnlHE | 127 | 10.06 | Thunderstorm ambience; inspect every retained clip |
| cv0ovU6kb-s | 136 | 11.44 | Soft delivery and clip boundaries require listening |
| lSaNBdJbhds | 90 | 6.66 | Promotional opening excluded through 118 s |

Provenance in `dataset-v2/dataset.json` includes URLs, precise timestamps, source/raw/prepared hashes,
caption proposals, measured durations, preprocessing and pending review state. Shared channel
identity is evidence, not verification that every clip has one matching speaker.

Preparation uses original-rate unprocessed slices under `raw/` and 22,050 Hz mono PCM16 under
`clips/`. It applies a single polyphase resample, quiet-edge trim with 100 ms padding, and bounded
linear RMS leveling toward -23 dBFS (at most ±3 dB, peak <=0.95). No denoise, pitch/formant changes,
EQ, separation, compression, or generated replacement speech. Prepared clips are 3–11.6 seconds.
Caption timing is approximate: truncated words and misplaced utterance boundaries must be rejected.

Independent cached Whisper medium.en runs on CPU INT8. It proposes transcripts and flags caption
disagreements; agreement is not a listening check. Never train directly from the automatic text.

From the repository root:

```bash
.venv/bin/python scripts/ciel_tts_finetune.py review data/voice/finetuning/20261014-01/dataset-v2
.venv/bin/python -m http.server 8769 --bind 127.0.0.1 --directory data/voice/finetuning/20261014-01/dataset-v2
```

Open `http://127.0.0.1:8769/review.html`. Compare with the unchanged baseline, listen to both raw and
prepared audio, correct the transcript, and explicitly check transcript accuracy, same speaker/no
overlap, and acoustic quality before approving. Reject uncertain clips. There is no bulk approval.
Editing a transcript clears its verification check. Decisions are stored locally in that browser;
download `ciel-dataset-reviews.json` before changing browsers or moving the page.

Import the downloaded decisions (adjust the path to the actual file):

```bash
.venv/bin/python scripts/ciel_tts_finetune.py import-reviews data/voice/finetuning/20261014-01/dataset-v2 ~/Downloads/ciel-dataset-reviews.json
.venv/bin/python scripts/ciel_tts_finetune.py export data/voice/finetuning/20261014-01/dataset-v2 data/voice/finetuning/20261014-01/verified-v2
```

Exports require at least 30 **verified** minutes after leakage purging. Split seed 73 holds out
roughly 10% by 120-second source/time blocks, removes training clips within 12 seconds of validation
clips, and removes exact normalized validation-text duplicates from training. The all-candidate
proposal is 369 train / 39 validation / 3 purged, 32.56 retained minutes; actual verified splits
will differ. No train/validation CSV is written until listening decisions have been imported.
If rejection takes the dataset below 30 minutes, obtain more clean permitted material or revisit
omitted boundaries; do not duplicate speech or lower the quality threshold to meet the target.

Export includes native Coqui pipe CSVs, WAVs, reviewed transcripts, provenance and an unchanged
evaluation reference. Audio hashes, sample rate, mono format, duration and CSV text are rechecked
before training. Splits are not random neighboring utterances from the same scene.

## XTTS feasibility and installation

The pinned workflow is maintained **coqui-tts 0.27.5**, coqui-tts-trainer 0.3.3, PyTorch/torchaudio
2.6.0+cu124, Transformers 4.57.6, NumPy 1.26.4. The real APIs are `GPTArgs`, `GPTTrainerConfig`,
`GPTTrainer.init_from_config`, `load_tts_samples`, and `Trainer`; see the
[versioned upstream recipe](https://github.com/idiap/coqui-ai-TTS/blob/v0.27.5/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)
and [supported XTTS documentation](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#training).

All 441,018,563 GPT parameters are trainable. FP32 weights + gradients + two Adam moments alone
require **6,729 MiB**, exceeding the card's 6,144 MiB before DVAE, other weights, activations,
CUDA context or desktop/STT allocations. FP16 autocast does not remove those FP32 states.
The training guard rejects this card before importing Torch or loading a model. No local OOM
training attempt, swap workaround, 8-bit optimizer, LoRA or partial unsupported adaptation was used.

Use a private 16 GB GPU or larger; 24 GB preferred. This is a conservative preflight threshold,
not a promise that every cloud GPU/configuration fits. Stop on an actual memory failure.
Keep at least 60 GiB free for full Adam checkpoints and inference/download copies; training checks
this before loading a model.

On a Python 3.12 cloud machine, install into an isolated environment:

```bash
python3.12 -m venv .training
.training/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.training/bin/python -m pip install -r scripts/voice_requirements/xtts_finetune.txt
.training/bin/python scripts/ciel_tts_finetune.py xtts-assets pretrained
.training/bin/python scripts/ciel_tts_finetune.py train-xtts verified pretrained pilot --epochs 2
```

Model revision is `coqui/XTTS-v2@6c2b0d75eae4b7047358e3b6bd9325f857d43f77`, identical to the
zero-shot baseline. The separate training inventory adds real `dvae.pth` and `mel_stats.pth`
without changing the existing runtime model directory. Retain its Coqui model license; the user's
previous personal non-commercial acceptance is recorded in the existing shootout research folder.

Settings: one Ciel speaker, English, pretrained GPT adaptation only; frozen DVAE, HiFiGAN/vocoder
and speaker encoder; batch 1, eval batch 1, accumulation 252; AdamW LR 5e-6, betas 0.9/0.96,
epsilon 1e-8, weight decay 0.01; native FP16 AMP with FP32 parameters; 22.05 kHz input/DVAE,
24 kHz output; up to 11.6 s utterances. Native text token length is checked before training.

The first bounded pilot evaluates baseline epoch 0 then trains two epochs. It saves every optimizer boundary and trained
epoch (retains five periodic checkpoints plus best validation checkpoint), loss metrics, time,
peak VRAM, and native TensorBoard test audio. There may be only a handful of optimizer updates
with this effective batch: two epochs cannot establish whether full adaptation succeeds or fails.
Audition before extending toward 5/10 epochs. Validation loss selects samples to audition, not a
voice-quality winner. Non-finite losses and sustained validation worsening while train loss falls
stop the experiment. Create `WORKSPACE/STOP` to save and stop at the next optimizer boundary after a listening rejection.

The cloud preflight now supports **full-state continuation**, replacing the earlier weights-only
warm-start recipe. `--resume` restores the committed latest state; `--restore checkpoint.pth`
requires the same strict full model/optimizer/scheduler/scaler/recipe state. Interrupted epochs
replay; completed epochs advance. Do not use old weights-only instructions or transfer-v2 bundles.

Use [the final cloud preflight guide](tts-cloud-preflight.md) and
`data/voice/finetuning/20261014-01/cloud/xtts_ciel.ipynb`. Upload the two files under
`cloud/transfer-final/`. The notebook installs its own pinned Python environment, enforces
15 GiB total VRAM, saves to persistent storage, exports validation WAVs and retains recoverable
full optimizer checkpoints. It includes the accepted Chatterbox listening baseline.

## Checkpoint listening and runtime measurements

Run each engine/checkpoint in a separate process, using the existing shootout environment. The
Chatterbox adapter calls the same persistent Turbo worker with unchanged inference settings.

```bash
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate chatterbox data/voice/finetuning/20261014-01/eval/chatterbox
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate xtts data/voice/finetuning/20261014-01/eval/xtts-zero
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate styletts2 data/voice/finetuning/20261014-01/eval/style-zero --device cpu
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate xtts data/voice/finetuning/20261014-01/eval/xtts-epoch2 --checkpoint /absolute/path/to/checkpoint.pth
.venv/bin/python scripts/ciel_tts_finetune.py listening data/voice/finetuning/20261014-01/listening --results data/voice/finetuning/20261014-01/eval/chatterbox/result.json data/voice/finetuning/20261014-01/eval/xtts-zero/result.json data/voice/finetuning/20261014-01/eval/style-zero/result.json data/voice/finetuning/20261014-01/eval/xtts-epoch2/result.json
```

The seven texts match this experiment, including the shorter technical sentence. Do not reuse the
old shootout's longer technical recording as though it were identical. Every trial uses the same
baseline speaker WAV and seed schedule. Checkpoint hash participates in the cached conditioning
key, so fine-tuned conditioning cannot silently reuse a zero-shot state.

The page randomizes each sentence independently and persists assignments. It requires all nine
ratings (or an explicit Unsure) and a preference before identities can be revealed. Ratings and
choice before reveal are retained separately from subsequent edits. With no checkpoints, the
current page explicitly says there are no fine-tuning results; it does not manufacture samples.

Each evaluation has three warm iterations, cold startup, first usable streamed audio, total time,
RTF, process-tree CPU/RAM, engine VRAM and whole-device GPU utilization/power/temperature/fan.
For listening finalists, use a fresh output and add `--sustained-seconds 300` or `600` to the same
evaluate command. This cycles the seven lines with paced idle time and retains a continuous
telemetry trace including idle periods. It simulates synthesis load, not a live STT/LLM conversation.
Fan speed is controller percent, not measured RPM or audible comfort. Do not attribute unrelated
desktop/STT GPU power to TTS or infer fan comfort from a single short run.

## StyleTTS2: deferred until XTTS evaluation

Local full fine-tuning is not a sensible 6 GB experiment. The official workflow includes diffusion
and SLM/adversarial training; upstream specifically warns that the joint stage can exceed VRAM and
that omitting it can reduce quality. No stage has been silently skipped to make this card fit.
See the [official fine-tuning instructions at the pinned revision](https://github.com/yl4579/StyleTTS2/tree/5cedc71c333f8d8b8551ca59378bdcc7af4c9529#finetuning).

If XTTS listening results justify a second cloud experiment:

1. Use the official repository revision `5cedc71c333f8d8b8551ca59378bdcc7af4c9529` and a separate
   environment, starting from the official fine-tuning Colab at that revision. Provision a larger
   GPU for all stages (24 GB+ as a candidate, 40 GB preferred); measure actual headroom rather than
   claiming that a specific memory capacity is sufficient.
2. Use the same verified clip IDs/split. Prepare 24 kHz mono directly from the preserved original
   slices, applying the recorded trim/gain once. Do not upsample the 22.05 kHz training export when
   the original is available. Convert verified text using the pretrained model's espeak-ng English
   phoneme frontend; rows are `filename.wav|phonemes|0`. Preserve plain verified text separately.
3. Load the pinned LibriTTS checkpoint used in the zero-shot study
   (`yl4579/StyleTTS2-LibriTTS@3aa7ba7f8f275ec13dce21682a61494c35089e2a`), plus the official ASR,
   JDC, PLBERT and OOD text assets. Keep `multispeaker: true`: it selects the compatible pretrained
   denoiser architecture, even though this adaptation dataset has one speaker.
4. Start from the pinned `Configs/config_ft.yml`, set private dataset paths and a new log directory,
   retain native losses/diffusion/joint stages and initially `batch_size: 1`, `max_len: 400`.
   Preserve `second_stage_load_pretrained: true`, `load_only_params: true`; do not disable SLM.
   Use the official command:

   ```bash
   accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path Configs/ciel_ft.yml
   ```

5. Save/audition staged checkpoints including the joint stage. Stop on OOM, NaNs, speaker drift,
   worse speech or overfitting. The dependency environment and full training stages still require
   cloud validation; this deferred recipe is not a tested StyleTTS2 fine-tuning result.

Fine-tuning does not establish that StyleTTS2's older-GPU high-frequency numerical artifacts have
gone away. A useful checkpoint would still require GTX 1660 SUPER and CPU listening tests.

## Verification

```bash
.venv/bin/python -m pytest tests/test_tts_finetune.py tests/test_tts_shootout.py
node tests/frontend/tts_finetune.cjs
.venv/bin/ruff check src tests scripts/tts_finetune scripts/ciel_tts_finetune.py scripts/tts_shootout/providers.py
.venv/bin/python -m pytest
```

Ordinary tests use generated fixtures and mocks, not real models. The separate real CPU API smoke
loaded the actual pretrained GPT/DVAE and created its native optimizer without stepping it; only
GPT parameters were covered and CUDA remained uninitialized. Cloud training, checkpoint quality,
sustained finalist benchmarks and visual browser QA remain unverified until their prerequisites
are available. See the local experiment report for the exact current status and suite failures.
