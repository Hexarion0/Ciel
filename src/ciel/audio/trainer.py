"""Voice sample recorder and custom wake word model trainer for personalized voice tuning."""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ciel.logging import get_logger
from ciel.paths import resolve_asset_path
from ciel.platform import get_audio_backend

logger = get_logger("audio.trainer")


def record_voice_samples(
    output_dir: str | Path = "data/wake_samples",
    count: int = 8,
    phrase: str = "Hey Ciel",
) -> list[Path]:
    """Interactively record personal voice samples to train a custom wake word model."""
    target_dir = resolve_asset_path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    backend = get_audio_backend()

    saved_files: list[Path] = []

    print("\n==================================================")
    print("  CIEL - Personalized Wake Word Voice Training")
    print("==================================================")
    print(f"Target Phrase: '{phrase}'")
    print("Privacy: All audio is saved locally in data/ (never uploaded).")
    print(f"We will record {count} short samples of your voice.\n")

    prompts = [
        "Normal speaking tone and volume",
        "Normal speaking tone and volume",
        "Fast / casual speaking tone",
        "Fast / casual speaking tone",
        "Quieter / soft tone",
        "Quieter / soft tone",
        "Slightly further away from mic",
        "Clear and distinct volume",
    ]

    for i in range(1, count + 1):
        guidance = prompts[(i - 1) % len(prompts)]
        input(f"[{i}/{count}] Ready ({guidance}). Press [Enter] and speak '{phrase}'...")

        print(f"🎙️ Recording sample {i}/{count} (2.0 seconds)...")
        audio_data = backend.record(duration=2.0, sample_rate=16000, channels=1)

        analysis = backend.analyze_audio(audio_data)
        if analysis["is_silent"]:
            print("⚠️ Warning: Sample appears quiet/silent. Check your microphone.")

        file_path = target_dir / f"sample_{i:02d}.wav"
        backend.save_wav(audio_data, file_path, sample_rate=16000)
        saved_files.append(file_path)
        print(f" Saved sample to {file_path.name}\n")
        time.sleep(0.3)

    print(f"✨ Successfully recorded {len(saved_files)} voice samples in '{target_dir}'.")
    print("Run `python -m ciel train-wake` to generate your custom ONNX wake model.")
    return saved_files


def train_custom_wakeword(
    samples_dir: str | Path = "data/wake_samples",
    output_model_path: str | Path = "models/custom/hey_ciel.onnx",
    phrase_name: str = "hey_ciel",
) -> Path:
    """Train and export an openWakeWord-compatible ONNX model from recorded voice samples."""
    try:
        import onnx
        import openwakeword.utils
        from onnx import TensorProto, helper
        from sklearn.linear_model import LogisticRegression
    except ImportError as err:
        raise RuntimeError(
            'Wake training dependencies are missing. Install with: pip install -e ".[train]"'
        ) from err

    samples_path = resolve_asset_path(samples_dir)
    wav_files = list(samples_path.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(
            f"No voice samples found in '{samples_path}'. "
            "Please run `python -m ciel record-samples` first."
        )

    logger.info("Loading %d voice samples for training...", len(wav_files))
    af = openwakeword.utils.AudioFeatures()

    positive_clips: list[np.ndarray] = []
    for f in wav_files:
        audio, sr = sf.read(f)
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Normalize to 16-bit PCM scale
        if np.issubdtype(audio.dtype, np.floating):
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            pcm16 = audio.astype(np.int16)

        # Ensure clip is at least 2 seconds (32,000 samples)
        if len(pcm16) < 32000:
            pcm16 = np.pad(pcm16, (0, 32000 - len(pcm16)))
        else:
            pcm16 = pcm16[:32000]

        # Augment with variations (shifts, noise)
        positive_clips.append(pcm16)
        # Shifted variations
        positive_clips.append(np.roll(pcm16, 1600))
        positive_clips.append(np.roll(pcm16, -1600))
        # Volume variations
        positive_clips.append((pcm16 * 0.7).astype(np.int16))
        positive_clips.append((pcm16 * 1.2).clip(-32767, 32767).astype(np.int16))

    positive_batch = np.array(positive_clips, dtype=np.int16)
    logger.info("Extracting embeddings for %d positive samples...", len(positive_batch))
    pos_embeddings = af.embed_clips(positive_batch)  # shape: (N, 16, 96)

    # Generate synthetic negative samples (silence, gaussian noise, random frequencies)
    negative_clips: list[np.ndarray] = []
    for _ in range(len(positive_clips) * 3):
        noise_type = np.random.choice(["silence", "white", "sine"])
        if noise_type == "silence":
            clip = np.zeros(32000, dtype=np.int16)
        elif noise_type == "white":
            clip = (np.random.randn(32000) * 800).clip(-32767, 32767).astype(np.int16)
        else:
            freq = np.random.uniform(100, 3000)
            t = np.linspace(0, 2.0, 32000, endpoint=False)
            clip = (1500 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
        negative_clips.append(clip)

    neg_batch = np.array(negative_clips, dtype=np.int16)
    logger.info("Extracting embeddings for %d negative samples...", len(neg_batch))
    neg_embeddings = af.embed_clips(neg_batch)

    # Flatten embeddings from (N, 16, 96) to (N, 1536) for linear classification
    X_pos = pos_embeddings.reshape(len(pos_embeddings), -1)
    y_pos = np.ones(len(X_pos), dtype=np.float32)

    X_neg = neg_embeddings.reshape(len(neg_embeddings), -1)
    y_neg = np.zeros(len(X_neg), dtype=np.float32)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])

    logger.info("Training binary classifier on %d feature vectors...", len(X))
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X, y)
    score = clf.score(X, y)
    logger.info("Classifier training accuracy: %.2f%%", score * 100)

    # Extract weights W and bias B
    W = clf.coef_.T.astype(np.float32)  # shape (1536, 1)
    B = clf.intercept_.astype(np.float32)  # shape (1,)

    # Build ONNX graph
    output_target = resolve_asset_path(output_model_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)

    X_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 16, 96])
    Y_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    flat_shape = helper.make_tensor("flat_shape", TensorProto.INT64, [2], [-1, 1536])
    flatten_node = helper.make_node("Reshape", ["input", "flat_shape"], ["flattened"])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, [1536, 1], W.flatten())
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [1], B.flatten())

    gemm_node = helper.make_node("Gemm", ["flattened", "W", "B"], ["linear"], alpha=1.0, beta=1.0)
    sigmoid_node = helper.make_node("Sigmoid", ["linear"], ["output"])

    graph = helper.make_graph(
        [flatten_node, gemm_node, sigmoid_node],
        phrase_name,
        [X_tensor],
        [Y_tensor],
        [flat_shape, W_init, B_init],
    )

    onnx_model = helper.make_model(
        graph,
        producer_name="ciel_voice_tuner",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_target))

    logger.info("Personalized wake word model saved to: %s", output_target)
    print(f"\n🎉 Successfully created personalized wake word model: '{output_target}'")
    print("To activate it, set this JSON list in .env, then restart CIEL:")
    print(f"WAKE_MODELS={json.dumps([str(output_target)])}")
    return output_target
