"""Uploaded images are bounded, normalized, and never confused with desktop captures."""

import io

import pytest
from PIL import Image


def test_image_normalization_discards_metadata_and_rejects_mime_spoofing():
    from ciel.webapp.images import normalize_upload

    source = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(source, format="JPEG", comment=b"private")
    data = normalize_upload(source.getvalue(), "image/jpeg")
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG" and image.size == (32, 24)
        assert "comment" not in image.info
    for raw, kind in [
        (source.getvalue(), "image/png"),
        (b"bad", "image/webp"),
        (b"x" * (10 * 1024**2 + 1), "image/png"),
    ]:
        with pytest.raises(ValueError):
            normalize_upload(raw, kind)


def test_image_expiry_during_inference_suppresses_the_late_answer():
    import threading

    from ciel.providers.base import LLMResponse
    from ciel.webapp.images import UploadedImageService
    from ciel.webapp.media import MediaStore
    from ciel.webapp.requests import WebRequest

    now = [0.0]
    media = MediaStore(clock=lambda: now[0])
    source = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(source, format="PNG")
    scope = dict(owner_key="owner", device_id="phone", request_id="request", revision=1)
    handle = media.put(source.getvalue(), **scope, content_type="image/png", ttl=900)
    request = WebRequest("request", "phone", "chat", 1, "text", "What is here?", (handle,))

    class Vision:
        def send(self, messages, **kwargs):
            assert len(messages[-1].images) == 1
            now[0] = 901
            return LLMResponse("Late answer", "selected", "vision")

    service = UploadedImageService(Vision(), media)
    assert service.answer(request, cancel_event=threading.Event()) is None
