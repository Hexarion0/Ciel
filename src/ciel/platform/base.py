"""Abstract base interface and data structures for audio backends."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AudioDeviceInfo:
    """Represents a discovered hardware or virtual audio device."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_input(self) -> bool:
        """Return True if device supports audio input (recording)."""
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """Return True if device supports audio output (playback)."""
        return self.max_output_channels > 0


class AudioBackend(ABC):
    """Abstract interface for operating system audio backends."""

    @abstractmethod
    def list_devices(self) -> list[AudioDeviceInfo]:
        """List all available audio input and output devices."""
        pass

    def list_input_devices(self) -> list[AudioDeviceInfo]:
        """List devices that support audio input."""
        return [dev for dev in self.list_devices() if dev.is_input]

    def list_output_devices(self) -> list[AudioDeviceInfo]:
        """List devices that support audio output."""
        return [dev for dev in self.list_devices() if dev.is_output]

    @abstractmethod
    def get_default_input_device(self) -> AudioDeviceInfo | None:
        """Return the default microphone/input device, if any."""
        pass

    @abstractmethod
    def get_default_output_device(self) -> AudioDeviceInfo | None:
        """Return the default speaker/output device, if any."""
        pass

    @abstractmethod
    def resolve_device(self, device: int | str | None, is_input: bool = True) -> int | None:
        """Resolve a device index or substring name into a concrete device index."""
        pass

    @abstractmethod
    def record(
        self,
        duration: float,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> np.ndarray:
        """Record audio for a fixed duration and return as a float32 or int16 numpy array."""
        pass

    @abstractmethod
    def save_wav(
        self,
        audio_data: np.ndarray,
        file_path: str | Path,
        sample_rate: int = 16000,
    ) -> Path:
        """Save a numpy audio array to a standard 16-bit PCM WAV file."""
        pass

    @abstractmethod
    def start_stream(
        self,
        callback: Callable[[np.ndarray, int, Any, Any], None],
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        blocksize: int = 1024,
    ) -> None:
        """Start a continuous non-blocking input audio stream."""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop and close the active audio stream."""
        pass

    @abstractmethod
    def is_streaming(self) -> bool:
        """Check if an audio stream is currently active."""
        pass

    @staticmethod
    def analyze_audio(audio_data: np.ndarray) -> dict[str, float | bool]:
        """Analyze audio chunk/array for clipping, silence, and volume levels."""
        if audio_data.size == 0:
            return {
                "rms": 0.0,
                "peak": 0.0,
                "is_silent": True,
                "is_clipped": False,
            }

        # Convert to float for normalized calculations
        if np.issubdtype(audio_data.dtype, np.floating):
            samples = audio_data
        else:
            samples = audio_data.astype(np.float32) / 32768.0

        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples**2)))
        is_silent = rms < 0.005
        is_clipped = peak >= 0.99

        return {
            "rms": rms,
            "peak": peak,
            "is_silent": is_silent,
            "is_clipped": is_clipped,
        }
