"""Tests for wake word sample training and ONNX model export."""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

from ciel.audio.trainer import train_custom_wakeword


@pytest.mark.integration
def test_train_custom_wakeword_synthetic():
    """Verify training and ONNX export from synthetic audio samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples_dir = Path(tmpdir) / "samples"
        samples_dir.mkdir()
        model_output = Path(tmpdir) / "custom_wake.onnx"

        # Generate 3 synthetic 2.0s audio sample WAVs
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        for i in range(3):
            freq = 400 + i * 50
            sine_sample = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sf.write(samples_dir / f"sample_{i:02d}.wav", sine_sample, 16000, subtype="PCM_16")

        # Train model
        exported_path = train_custom_wakeword(
            samples_dir=samples_dir,
            output_model_path=model_output,
            phrase_name="test_phrase",
        )
        assert exported_path.exists()

        # Load with ONNX Runtime and verify inference
        sess = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        inp = np.random.randn(1, 16, 96).astype(np.float32)
        out = sess.run(None, {"input": inp})

        assert out[0].shape == (1, 1)
        assert 0.0 <= float(out[0][0][0]) <= 1.0
