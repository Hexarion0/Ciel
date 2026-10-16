"""Platform audio backend factory and platform detection."""

import sys

from ciel.platform.base import AudioBackend, AudioDeviceInfo
from ciel.platform.linux import LinuxAudioBackend
from ciel.platform.system_info import (
    GpuInfo,
    SystemSnapshot,
    generate_system_prompt,
    get_system_snapshot,
)


def get_audio_backend() -> AudioBackend:
    """Instantiate and return the appropriate audio backend for the host operating system."""
    if sys.platform.startswith("linux"):
        return LinuxAudioBackend()
    elif sys.platform.startswith("win"):
        # Windows backend will be added in Milestone 0.8 / future updates
        # sounddevice is cross-platform and handles WASAPI on Windows too
        return LinuxAudioBackend()
    else:
        return LinuxAudioBackend()


__all__ = [
    "AudioBackend",
    "AudioDeviceInfo",
    "LinuxAudioBackend",
    "get_audio_backend",
    "GpuInfo",
    "SystemSnapshot",
    "get_system_snapshot",
    "generate_system_prompt",
]
