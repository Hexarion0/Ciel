# Ciel XTTS cloud preflight

This prepares an experimental cloud run. No fine-tuned voice has passed listening acceptance.
Chatterbox Turbo remains the accepted voice; these commands do not activate a provider or edit `.env`.
No further XTTS training should run on the 6 GB GTX 1660 SUPER.

## Transfer and clean runtime

Open `data/voice/finetuning/20261014-01/cloud/xtts_ciel.ipynb` in your private cloud notebook.
Use Linux x86_64 and a CUDA GPU: **16 GB acceptable, 24 GB preferred**. The early `nvidia-smi`
guard aborts below **15 GiB total / 12 GiB free**, before importing Torch; CUDA device 0 is checked
again before model construction. Exactly one GPU is exposed. No OOM retries, optimizer substitutions,
automatic batch reductions or swap workarounds are used. The guard is a minimum, not a guarantee
that every supported GPU/runtime has sufficient headroom.

Upload these two files when prompted (the notebook embeds their SHA-256 checksums):

- `data/voice/finetuning/20261014-01/cloud/transfer-final/ciel-finetune-code.zip`
- `data/voice/finetuning/20261014-01/cloud/transfer-final/ciel-verified-dataset.zip`

The code ZIP includes all modules used by the notebook, the full dependency lock and this guide.
Dataset ZIP includes the 408 verified WAVs, both metadata files, review/provenance records,
the unchanged speaker reference and seven Chatterbox baseline WAVs. It excludes original videos,
credentials, unrelated project components and downloaded model weights. Keep both ZIPs private.
Preparation/transcription/review CLI commands are local repository tools; they are not needed or
invoked in the cloud training workflow.

The notebook bootstraps `uv==0.12.19`, CPython **3.12.14**, then an isolated environment with
PyTorch/torchaudio **2.6.0+cu124** and `scripts/voice_requirements/xtts_cloud.lock.txt`.
Every runtime dependency is pinned, including Coqui TTS **0.27.5**, Trainer **0.3.3**,
Transformers **4.57.6** and NumPy **1.26.4**. `uv pip check` must succeed.
It downloads seven base assets from `coqui/XTTS-v2` at revision
`6c2b0d75eae4b7047358e3b6bd9325f857d43f77` and hashes them. Internet access is needed for setup.
The user's existing personal, non-commercial Coqui model-license acceptance applies.

Colab prompts you to mount private Drive. Other cloud notebooks must set
`CIEL_TRAINING_STORAGE` to a persistent volume before running the first cell. Use **60 GiB or more
free persistent space**, including actual Drive quota. Do not use an ephemeral VM filesystem for
the only checkpoint copy. The notebook never creates a paid instance or uploads data by itself.

## Dataset sanity gate

The real validated dataset is 22,050 Hz mono PCM16:

| Measure | Value |
| --- | ---: |
| All clips | 408 |
| Train clips / duration | 369 / 29.2233 min |
| Validation clips / duration | 39 / 3.3383 min |
| Total duration | 32.5616 min |
| Shortest / longest | 3.00 / 11.14 s |
| Average duration | 4.7885 s |
| Missing files | 0 |
| Duplicate normalized transcripts | 0 |
| Maximum native text tokens | 109 (limit 200) |

The notebook repeats checks after unpacking: explicit human verification, WAV hashes/headers,
paths contained inside the export, unique IDs/paths, valid split names, exact CSV/provenance text,
nonempty transcripts, no train/validation text overlap, durations and native tokenizer IDs/limits.
Missing files, modified audio or invalid metadata abort rather than silently dropping samples.
The unchanged reference and all seven Chatterbox baseline texts/hashes must also match.

## Native recipe, listening and stopping

This uses the [versioned supported GPTTrainer recipe](https://github.com/idiap/coqui-ai-TTS/blob/v0.27.5/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)
and [native Trainer 0.3.3 checkpoint state](https://github.com/idiap/coqui-ai-Trainer/blob/v0.3.3/trainer/trainer.py).
Pretrained GPT adaptation only: 441,018,563 GPT parameters trainable, DVAE/vocoder/speaker encoder
frozen; AdamW, learning rate 5e-6, betas 0.9/0.96, epsilon 1e-8, decay 0.01, batch 1,
gradient accumulation 252. Effective batch 252 follows the supported recipe.
Native mixed precision uses FP16 AMP with FP32 weights/Adam state. No experimental optimizer.
The initial run requests **two training epochs**, plus a baseline-only epoch 0.
This yields very few optimizer updates; a pilot alone cannot prove adaptation succeeds or fails.
The native MultiStepLR schedule, all hyperparameters and dependency versions are recorded.

The same seven fixed lines in `scripts/tts_finetune/common.py` generate at baseline and every
epoch using native `GPTTrainer.test_run`, conditioning length 3 seconds and seed 73. These include
the six requested lines and a four-sentence conversational paragraph. Validation generation restores
the training RNG state afterward. WAVs are available outside TensorBoard, with sentence text,
step, reference hash and checkpoint path/hash in each manifest.
Six Chatterbox baseline WAVs reuse the accepted shootout exactly; the shorter technical sentence
was generated once through the unchanged current Turbo worker. No Chatterbox cloud install needed.
The cloud base-XTTS sample uses the same validation routine as cloud fine-tuned samples;
the later local streaming benchmark uses the same existing inference settings for both XTTS models.
Do not mix these two inference protocols in a performance comparison.

Non-finite losses stop immediately. Three consecutively worse validation checks with more than
5% overall worsening stop automatically, even if train loss still looks good. For worse naturalness,
speaker identity, pronunciation, artifacts or memorization, create `ciel-xtts-training/STOP` in the
persistent storage browser (or a second cell). The native callbacks stop at the next optimizer
boundary and save first. This may take time during accumulation. After reviewing why, archive STOP and `stop-reason.json` before permitting continuation.
Native interrupt-time saving is disabled because it can capture unfinished accumulation.
Cell termination can be abrupt; persistent committed checkpoints provide recovery.
Audition before increasing `TOTAL_EPOCHS` to 5 or 10. Loss alone is not voice-quality acceptance.

## Resume and artifact layout

```text
ciel-xtts-training/
  config/                 requested native config, identity, dependencies, sanity report
  logs/                   TensorBoard train/eval curves and test audio
  checkpoints/
    native/<run>/         native full .pth files and config
    latest/checkpoint.json
    best/checkpoint.json
    step-XXXXXXXX/checkpoint.json
  samples/
    baseline/chatterbox/  accepted baseline WAVs and provenance
    baseline/xtts-zero-shot/
    step-XXXXXXXX/        seven WAVs and checkpoint manifest
  epoch-metrics.json
  training-summary.json
```

Full checkpoints contain native model, optimizer, scheduler, AMP scaler, config, step and epoch,
plus dataset/recipe/dependency identity and best-so-far loss. Saves are atomic and checksum-committed
after every optimizer update (batch 252 and the final batch of the 369-clip epoch), then after
validation. The latest five periodic files plus native best validation files are retained.
Pointer JSONs avoid extra copies of multi-GiB tensors. Older sample manifests retain checkpoint
hashes for provenance even when their periodic model has been pruned; consult the retained-file index.

On reconnection rerun upload/setup with the same persistent storage. `--resume` chooses and verifies
the committed latest full checkpoint; incompatible/missing state aborts rather than partially loading
weights or resetting Adam. The strict restore hook follows the native single-optimizer state workflow
while removing Trainer 0.3.3's silent incompatible-optimizer/partial-weight fallbacks.
The original config's recipe, dataset and dependencies must match; only paths and the reviewed total
epoch limit may differ. **Interrupted epochs replay from the start with restored optimizer state**;
completed epochs advance. This does not reproduce the exact previous dataloader/RNG position.
`--epochs 5` means five TOTAL training epochs, not five additional ones.

The final cell exports a separate folder/archive such as `ciel-xtts-final-pilot-01.zip`.
It includes one best-validation inference checkpoint without Adam state, base inference config,
vocab, speaker file, unchanged reference, license, dependencies, logs, all WAV samples/manifests,
summary and these integration instructions. Frozen native model tensors remain unchanged.
No redundant base/download caches, raw dataset or other full optimizer checkpoints enter this ZIP.

Persistent storage is the main recovery copy. Set `MAKE_RECOVERY_ZIP=True` before destroying a
cloud instance/volume if you want a portable recovery archive. It contains latest and best full
states once each, plus config/logs/samples. Unpack into a new persistent workspace, use the same
dataset/code ZIPs and pinned assets, then run `train-xtts VERIFIED PRETRAINED WORKSPACE --epochs 2
--resume` in the notebook environment. Compact inference exports cannot resume training.

## Exact local import, speech, benchmark and blind comparison

Download the final ZIP privately to `~/Downloads/ciel-xtts-final-pilot-01.zip`. From the Ciel root:

```bash
.venv/bin/python scripts/ciel_tts_finetune.py import-xtts ~/Downloads/ciel-xtts-final-pilot-01.zip data/voice/experiments/ciel-xtts-pilot
```

This verifies every archived file and registers `experiment.json` in that private directory.
It does not register a production provider, edit `.env`, replace references or activate anything.
Use the existing Python 3.12 shootout environment to generate seven test WAVs and benchmark with
three warm iterations per sentence. Each command uses a separate process and a new output directory:

```bash
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate xtts data/voice/finetuning/20261014-01/eval-cloud/xtts-pilot --experiment data/voice/experiments/ciel-xtts-pilot/experiment.json --iterations 3
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate xtts data/voice/finetuning/20261014-01/eval-cloud/xtts-zero --iterations 3
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate chatterbox data/voice/finetuning/20261014-01/eval-cloud/chatterbox --iterations 3
.venv/bin/python scripts/ciel_tts_finetune.py listening data/voice/finetuning/20261014-01/listening-cloud --results data/voice/finetuning/20261014-01/eval-cloud/xtts-pilot/result.json data/voice/finetuning/20261014-01/eval-cloud/xtts-zero/result.json data/voice/finetuning/20261014-01/eval-cloud/chatterbox/result.json
.venv/bin/python -m http.server 8770 --bind 127.0.0.1 --directory data/voice/finetuning/20261014-01/listening-cloud
```

Open `http://127.0.0.1:8770/`. Letters randomize separately per sentence; ratings and preference are
required before engine identities reveal. Compare naturalness, robotic quality, identity, warmth,
prosody, pronunciation, expression and conversational feel. Existing XTTS conditioning cache keys
include the checkpoint and asset inventory hashes. The model is loaded once per evaluation process.
The imported model folder supplies its own config/vocab/speaker assets.

If an earlier trained checkpoint also deserves listening, export/import it separately; never label
weights as trained based only on a filename. Keep full checkpoints in the private cloud workspace.

For listening finalists, run the existing five-minute sustained inference benchmark (ten minutes
can use `600`). This is local **inference**, not training:

```bash
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate xtts data/voice/finetuning/20261014-01/eval-cloud/xtts-pilot-sustained --experiment data/voice/experiments/ciel-xtts-pilot/experiment.json --iterations 3 --sustained-seconds 300
data/voice/envs/shootout/bin/python scripts/ciel_tts_finetune.py evaluate chatterbox data/voice/finetuning/20261014-01/eval-cloud/chatterbox-sustained --iterations 3 --sustained-seconds 300
```

`result.json` reports cold startup, warm first usable streamed audio, synthesis time, RTF,
CPU/RAM, VRAM, GPU utilization/power/temperature/fan and a continuous sustained trace. Fan is controller
percent, not RPM or a listening comfort measurement. Whole-device power may include desktop/STT;
engine VRAM/CPU/RAM use the process tree. The older shootout had a telemetry driver mismatch;
do not reuse its missing power/fan data as a valid comparison. No finalist runtime measurements exist
until a trained checkpoint is downloaded and these commands actually run.

Chatterbox remains the accepted choice unless fine-tuned XTTS passes blind listening **and**
sustained GTX 1660 SUPER inference measurements demonstrate a worthwhile comfort tradeoff.
