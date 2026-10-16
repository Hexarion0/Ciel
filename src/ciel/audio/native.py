"""Defer PortAudio initialization until a device is actually used."""

from importlib import import_module


class _SoundDevice:
    def __getattr__(self, name: str):
        return getattr(import_module("sounddevice"), name)


sd = _SoundDevice()
