"""Linux audio backend implementation using sounddevice (PipeWire / PulseAudio / ALSA)."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ciel.audio.native import sd
from ciel.config import normalize_audio_device
from ciel.logging import get_logger
from ciel.platform.base import AudioBackend, AudioDeviceInfo, AudioInputError

logger = get_logger("platform.linux")


class LinuxAudioBackend(AudioBackend):
    """Audio backend tailored for Linux desktop environments."""

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._stream_lock = threading.Lock()

    def list_devices(self) -> list[AudioDeviceInfo]:
        """Query and return all audio devices recognized by PortAudio / ALSA / PipeWire."""
        devices: list[AudioDeviceInfo] = []
        try:
            raw_devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_input, default_output = sd.default.device

            for idx, dev in enumerate(raw_devices):
                api_name = (
                    hostapis[dev["hostapi"]]["name"]
                    if dev["hostapi"] < len(hostapis)
                    else "Unknown"
                )
                devices.append(
                    AudioDeviceInfo(
                        index=idx,
                        name=dev["name"],
                        hostapi=api_name,
                        max_input_channels=dev["max_input_channels"],
                        max_output_channels=dev["max_output_channels"],
                        default_samplerate=dev["default_samplerate"],
                        is_default_input=(idx == default_input),
                        is_default_output=(idx == default_output),
                    )
                )
        except Exception as err:
            logger.error("Failed to query audio devices: %s", err)
        return devices

    def get_default_input_device(self) -> AudioDeviceInfo | None:
        """Find the system default input device."""
        devices = self.list_devices()
        for dev in devices:
            if dev.is_default_input and dev.is_input:
                return dev
        inputs = [d for d in devices if d.is_input]
        return inputs[0] if inputs else None

    def get_default_output_device(self) -> AudioDeviceInfo | None:
        """Find the system default output device."""
        devices = self.list_devices()
        for dev in devices:
            if dev.is_default_output and dev.is_output:
                return dev
        outputs = [d for d in devices if d.is_output]
        return outputs[0] if outputs else None

    def resolve_device(self, device: int | str | None, is_input: bool = True) -> int | None:
        """Resolve a device specification (int index, string query, or None) to an index."""
        device = normalize_audio_device(device)
        if device is None:
            default_dev = (
                self.get_default_input_device() if is_input else self.get_default_output_device()
            )
            return default_dev.index if default_dev else None

        if isinstance(device, int):
            return device

        # String matching by name substring (case-insensitive)
        candidates = self.list_input_devices() if is_input else self.list_output_devices()
        query = device.lower()
        for dev in candidates:
            if query in dev.name.lower():
                return dev.index

        logger.warning(
            "Could not find %s audio device matching '%s', falling back to default.",
            "input" if is_input else "output",
            device,
        )
        default_dev = (
            self.get_default_input_device() if is_input else self.get_default_output_device()
        )
        return default_dev.index if default_dev else None

    def record(
        self,
        duration: float,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> np.ndarray:
        """Record audio from the microphone for a fixed duration."""
        resolved_device = self.resolve_device(device, is_input=True)
        num_frames = int(duration * sample_rate)

        logger.debug(
            "Recording %.1fs (rate=%d, channels=%d, device=%s)",
            duration,
            sample_rate,
            channels,
            resolved_device,
        )

        try:
            recording = sd.rec(
                frames=num_frames,
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                device=resolved_device,
                blocking=True,
            )
        except sd.PortAudioError as err:
            logger.warning(
                "Recording failed at %d Hz on device %s: %s. Attempting fallback sample rate.",
                sample_rate,
                resolved_device,
                err,
            )
            # Try recording at 48000 Hz or device default and downsample
            recording = self._record_with_resample(
                duration=duration,
                target_sample_rate=sample_rate,
                channels=channels,
                device=resolved_device,
            )

        analysis = self.analyze_audio(recording)
        if analysis["is_silent"]:
            logger.warning(
                "Recorded audio appears to be silent (RMS: %.5f)",
                analysis["rms"],
            )
        if analysis["is_clipped"]:
            logger.warning(
                "Recorded audio peak indicates clipping (Peak: %.3f)",
                analysis["peak"],
            )

        return recording

    def _record_with_resample(
        self,
        duration: float,
        target_sample_rate: int,
        channels: int,
        device: int | None,
    ) -> np.ndarray:
        """Fallback recorder for devices requiring native 44.1k/48k sample rates."""
        native_rate = 48000
        num_frames = int(duration * native_rate)
        raw_audio = sd.rec(
            frames=num_frames,
            samplerate=native_rate,
            channels=channels,
            dtype="float32",
            device=device,
            blocking=True,
        )
        # Linear downsampling from native_rate to target_sample_rate
        target_frames = int(duration * target_sample_rate)
        orig_indices = np.linspace(0, num_frames - 1, num_frames)
        new_indices = np.linspace(0, num_frames - 1, target_frames)

        if channels == 1:
            resampled = np.interp(new_indices, orig_indices, raw_audio.squeeze()).reshape(-1, 1)
        else:
            resampled = np.zeros((target_frames, channels), dtype=np.float32)
            for ch in range(channels):
                resampled[:, ch] = np.interp(new_indices, orig_indices, raw_audio[:, ch])
        return resampled.astype(np.float32)

    def save_wav(
        self,
        audio_data: np.ndarray,
        file_path: str | Path,
        sample_rate: int = 16000,
    ) -> Path:
        """Write numpy array audio into standard 16-bit PCM WAV file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Normalize and convert to 16-bit PCM format
        sf.write(target, audio_data, sample_rate, subtype="PCM_16")
        logger.debug("Saved audio WAV file to: %s", target)
        return target

    def start_stream(
        self,
        callback: Callable[[np.ndarray, int, Any, Any], None],
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | str | None = None,
        blocksize: int = 1024,
    ) -> None:
        """Start non-blocking continuous input stream."""
        with self._stream_lock:
            if self._stream is not None:
                logger.info("Replacing existing audio stream.")
                self._stop_stream_locked()

            resolved_device = self.resolve_device(device, is_input=True)

            def _internal_callback(
                indata: np.ndarray, frames: int, time_info: Any, status: Any
            ) -> None:
                if status:
                    logger.debug("Stream status notice: %s", status)
                callback(indata.copy(), frames, time_info, status)

            try:
                self._stream = sd.InputStream(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=resolved_device,
                    blocksize=blocksize,
                    callback=_internal_callback,
                )
                self._stream.start()
            except sd.PortAudioError as error:
                self._stop_stream_locked()
                raise AudioInputError("The selected microphone could not be opened.") from error
            except BaseException:
                self._stop_stream_locked()
                raise
            logger.info(
                "Audio stream started on device %s at %d Hz.",
                resolved_device,
                sample_rate,
            )

    def stop_stream(self) -> None:
        """Stop and close active audio stream."""
        with self._stream_lock:
            self._stop_stream_locked()

    def _stop_stream_locked(self) -> None:
        """Close the current stream while the caller owns the lifecycle lock."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                try:
                    if stream.active:
                        stream.stop()
                finally:
                    stream.close()
            except Exception as err:
                logger.error("Error closing audio stream: %s", err)
            logger.info("Audio stream stopped.")

    def is_streaming(self) -> bool:
        """Check if stream is active."""
        with self._stream_lock:
            return self._stream is not None and self._stream.active
