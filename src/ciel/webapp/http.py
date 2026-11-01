"""Explicit packaged assets; request paths never become filesystem paths."""

from pathlib import Path

_ASSETS = {
    "/assets/chat.css": ("chat.css", "text/css; charset=utf-8"),
    "/assets/settings.js": ("settings.js", "text/javascript; charset=utf-8"),
    "/assets/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
    "/assets/orb.js": ("orb.js", "text/javascript; charset=utf-8"),
    "/assets/render.js": ("render.js", "text/javascript; charset=utf-8"),
    "/assets/audio.js": ("audio.js", "text/javascript; charset=utf-8"),
    "/assets/attachments.js": ("attachments.js", "text/javascript; charset=utf-8"),
    "/assets/pcm-worklet.js": ("pcm-worklet.js", "text/javascript; charset=utf-8"),
}


def load_web_asset(path: str) -> tuple[bytes, str] | None:
    """Read only a named application asset from the installed package."""
    asset = _ASSETS.get(path)
    if asset is None:
        return None
    name, content_type = asset
    return (Path(__file__).parent.parent / "web_assets" / name).read_bytes(), content_type
