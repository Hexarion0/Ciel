"""Temporary media cannot cross device, request, or conversation boundaries."""

import io
import wave

import numpy as np
import pytest


def test_media_scope_expiry_and_budget():
    from ciel.webapp.media import MediaStore

    now = [0.0]
    store = MediaStore(clock=lambda: now[0], item_limit=8, device_limit=12, owner_limit=16)
    scope = dict(owner_key="owner", device_id="phone", request_id="request", revision=1)
    handle = store.put(b"12345678", **scope, content_type="image/png", ttl=10)
    assert store.get(handle, **scope) == b"12345678"
    with pytest.raises(ValueError):
        store.get(handle, **{**scope, "device_id": "other"})
    with pytest.raises(ValueError):
        store.put(b"12345", **scope, content_type="image/png", ttl=10)
    now[0] = 11
    with pytest.raises(ValueError):
        store.get(handle, **scope)
    assert store.put(b"12345", **scope, content_type="image/png", ttl=10)
    store.invalidate(device_id="phone")
    assert not store._items


def test_voice_wav_is_bounded_mono_pcm16_at_16khz():
    from ciel.webapp.audio import decode_voice_wav

    def wav(rate=16000, channels=1, samples=160):
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(b"\0\0" * samples * channels)
        return output.getvalue()

    audio = decode_voice_wav(wav())
    assert audio.shape == (160,) and audio.dtype == np.float32
    for data in [b"bad", wav(rate=48000), wav(channels=2), wav(samples=0), wav(samples=960001)]:
        with pytest.raises(ValueError):
            decode_voice_wav(data)
