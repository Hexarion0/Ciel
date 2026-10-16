"""Small streaming speech detector for ambient capture and resumed speech."""

import numpy as np


class SpeechActivity:
    """Consume arbitrary mono frames using the installed local Silero VAD model."""

    def __init__(self) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self.model = SileroVoiceActivityDetector()
        self.pending = np.empty(0, dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> bool:
        frame = audio.reshape(-1)
        if np.issubdtype(frame.dtype, np.integer):
            frame = frame.astype(np.float32) / 32768.0
        self.pending = np.concatenate((self.pending, frame.astype(np.float32)))
        count = self.model.chunk_samples()
        speech = False
        while self.pending.size >= count:
            speech |= float(self.model.process_array(self.pending[:count])) >= 0.5
            self.pending = self.pending[count:]
        return speech

    def reset(self) -> None:
        """Clear recurrent state between listening sessions."""
        self.pending = np.empty(0, dtype=np.float32)
        self.model.reset()
