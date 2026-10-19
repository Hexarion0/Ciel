"""Resolve voice selections without importing speech engines or their dependencies."""


def resolve_tts_engine(voice_name: str, engine: str = "auto") -> str:
    """Normalize an explicit engine or retain Piper for legacy automatic selection."""
    selected = engine.lower()
    if selected == "auto":
        return "piper"
    if selected in {"chatterbox", "hq"}:
        selected = "chatterbox_turbo"
    if selected not in {"piper", "pocket", "chatterbox_turbo"}:
        raise ValueError(f"Unknown TTS engine: {engine}")
    return selected
