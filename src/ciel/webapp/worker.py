"""Capture-free adapters for reusing the serialized assistant request worker."""


class HostedAudioBackend:
    """A hosted web session owns no PC audio devices."""

    def stop_stream(self) -> None:
        """There is no server microphone stream to stop."""


class BrowserWakeDetector:
    """Browser requests carry their own address signal and never run wake detection."""

    def reset(self, *, set_cooldown: bool = True) -> None:
        """Keep the shared worker's cancellation lifecycle without loading wake models."""
