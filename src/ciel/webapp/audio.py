"""Bounded PCM voice input and speech output without PC audio playback."""

import io
import wave

import numpy as np


def decode_voice_wav(data: bytes) -> np.ndarray:
    """Accept at most sixty seconds of mono 16kHz PCM16 WAV."""
    if not 44 <= len(data) <= 1_920_256:
        raise ValueError("Record between a moment and 60 seconds of speech.")
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            frames = reader.getnframes()
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16000
                or reader.getcomptype() != "NONE"
                or not 0 < frames <= 960000
            ):
                raise ValueError("Use mono 16kHz PCM16 WAV audio.")
            pcm = reader.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("The recording is incomplete. Try again.")
    except (wave.Error, EOFError, OSError) as error:
        raise ValueError("The recording is not a valid WAV file.") from error
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def encode_audio_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a bounded synthesized audio chunk for native browser playback."""
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return output.getvalue()
