"""Tests for audio backend interface and Linux implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ciel.platform import get_audio_backend
from ciel.platform.base import AudioBackend, AudioDeviceInfo
from ciel.platform.linux import LinuxAudioBackend


def test_audio_device_info():
    """Verify AudioDeviceInfo properties."""
    dev_in = AudioDeviceInfo(
        index=0,
        name="Test Mic",
        hostapi="ALSA",
        max_input_channels=2,
        max_output_channels=0,
        default_samplerate=44100.0,
        is_default_input=True,
    )
    assert dev_in.is_input is True
    assert dev_in.is_output is False

    dev_out = AudioDeviceInfo(
        index=1,
        name="Test Speaker",
        hostapi="ALSA",
        max_input_channels=0,
        max_output_channels=2,
        default_samplerate=48000.0,
        is_default_output=True,
    )
    assert dev_out.is_input is False
    assert dev_out.is_output is True


def test_audio_analysis():
    """Verify audio analysis for silence and clipping."""
    # Test silence
    silent_audio = np.zeros((16000, 1), dtype=np.float32)
    analysis = AudioBackend.analyze_audio(silent_audio)
    assert analysis["is_silent"] is True
    assert analysis["is_clipped"] is False
    assert analysis["peak"] == 0.0

    # Test clipping
    clipped_audio = np.ones((16000, 1), dtype=np.float32)
    analysis_clipped = AudioBackend.analyze_audio(clipped_audio)
    assert analysis_clipped["is_clipped"] is True
    assert analysis_clipped["peak"] == 1.0


def test_save_wav():
    """Verify writing audio numpy arrays to standard WAV files."""
    backend = LinuxAudioBackend()
    sample_rate = 16000
    duration = 0.5
    # Generate 440 Hz sine wave tone
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "test_output.wav"
        saved = backend.save_wav(sine_wave, target_path, sample_rate=sample_rate)
        assert saved.exists()

        # Read back and verify properties
        data, sr = sf.read(saved)
        assert sr == sample_rate
        assert len(data) == len(sine_wave)


@pytest.mark.integration
def test_backend_device_listing():
    """Verify that backend can discover host audio devices without errors."""
    backend = get_audio_backend()
    devices = backend.list_devices()
    assert isinstance(devices, list)

    input_devs = backend.list_input_devices()
    output_devs = backend.list_output_devices()
    assert isinstance(input_devs, list)
    assert isinstance(output_devs, list)


def test_resolve_devices_by_numeric_string_and_name(monkeypatch):
    backend = LinuxAudioBackend()
    devices = [
        AudioDeviceInfo(
            index=7,
            name="USB Headset",
            hostapi="test",
            max_input_channels=1,
            max_output_channels=2,
            default_samplerate=16000,
        )
    ]
    monkeypatch.setattr(backend, "list_input_devices", lambda: devices)
    monkeypatch.setattr(backend, "list_output_devices", lambda: devices)
    for is_input in (True, False):
        assert backend.resolve_device("3", is_input=is_input) == 3
        assert backend.resolve_device("Headset", is_input=is_input) == 7
