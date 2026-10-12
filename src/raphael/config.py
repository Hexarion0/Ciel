"""Centralized configuration and secrets management for RAPHAEL."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_audio_device(value: int | str | None) -> int | str | None:
    """Normalize environment indices while retaining device name queries."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.isdecimal():
            return int(value)
    return value


class AppConfig(BaseModel):
    """General application configuration."""

    env: str = Field(
        default="development",
        description="Application environment (development, production, test)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    debug: bool = Field(default=False, description="Enable debug mode")


class MemoryConfig(BaseModel):
    """Memory and persistence configuration."""

    db_path: str = Field(
        default="data/raphael.db",
        description="Path to SQLite memory database file",
    )
    max_short_term_turns: int = Field(
        default=10,
        ge=1,
        description="Maximum turns of conversation to keep in short-term context",
    )


class InternetConfig(BaseModel):
    """Free public web search and page-reading limits."""

    enabled: bool = Field(default=True, description="Enable public web retrieval")
    automatic: bool = Field(default=True, description="Search requests needing current facts")
    home_city: str = Field(
        default="", max_length=200, description="Optional default city and country",
    )
    timeout_seconds: float = Field(default=8.0, ge=0.1, le=60)
    research_timeout_seconds: float = Field(default=20.0, ge=0.1, le=120)
    max_results: int = Field(default=5, ge=1, le=10)
    research_max_results: int = Field(default=8, ge=1, le=20)
    max_pages: int = Field(default=2, ge=0, le=5)
    research_max_pages: int = Field(default=4, ge=0, le=10)

    @field_validator("home_city", mode="before")
    @classmethod
    def strip_home_city(cls, value: str) -> str:
        """Normalize an optional city without inferring the user's location."""
        return value.strip() if isinstance(value, str) else value


class ProviderConfig(BaseModel):
    """AI Provider API keys and endpoints."""

    nim_api_key: SecretStr | None = Field(
        default=None,
        description="NVIDIA NIM API key",
    )
    nim_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="Default NVIDIA NIM model name",
    )
    nim_complex_model: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="NVIDIA NIM model for complex reasoning and coding tasks",
    )
    nim_fallback_model: str = Field(
        default="meta/llama-3.2-90b-vision-instruct",
        description="NVIDIA NIM fallback model for failover / rate-limiting",
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama local server endpoint",
    )


class AudioConfig(BaseModel):
    """Audio, wake word, and speech-to-text settings."""

    wake_word: str = Field(default="hey raphael", description="Wake word trigger phrase")
    wake_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Wake word detection threshold (0.0 - 1.0)",
    )
    wake_cooldown: float = Field(
        default=2.0,
        description="Cooldown seconds after wake trigger",
    )
    wake_stt_model: str = Field(default="base.en")
    wake_min_rms: float = Field(default=0.006, gt=0.0, le=0.05)
    wake_window_seconds: float = Field(default=3.0, ge=1.5, le=5.0)
    ambient_listening: bool = Field(default=False)
    show_transcripts: bool = Field(default=False)
    barge_in_mode: Literal["speech", "wake"] = Field(default="wake")
    barge_in_speech_seconds: float = Field(default=0.24, ge=0.16, le=1.0)
    ambient_followup_seconds: float = Field(default=300.0, ge=2, le=600)
    ambient_followup_policy: Literal["conversation", "strict"] = Field(default="conversation")
    wake_models: list[str] = Field(
        default_factory=list,
        description="openWakeWord model names/paths (empty = Whisper-only keyword spotter)",
    )
    stt_model: str = Field(
        default="base.en",
        description="faster-whisper model size (e.g. tiny.en, base.en, small.en)",
    )
    stt_device: str = Field(
        default="auto",
        description="Inference device for whisper (auto, cuda, cpu)",
    )
    stt_compute_type: str = Field(
        default="default",
        description="Compute precision (default, int8, float16, float32)",
    )
    stt_language: str = Field(
        default="en",
        description="Primary language code for STT transcription",
    )
    stt_startup_timeout_seconds: float = Field(default=120.0, ge=0.1, le=300)
    stt_beam_size: int = Field(default=3, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default=None)
    stt_retry_min_free_mb: int = Field(default=2048, ge=0, le=65536)
    utterance_silence_seconds: float = Field(default=0.7, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.2, ge=0, le=3)
    tts_engine: str = Field(
        default="auto",
        description=(
            "TTS engine: auto, piper, or chatterbox_turbo"
        ),
    )
    tts_voice: str = Field(
        default="en_US-amy-medium",
        description="TTS profile or Piper voice name",
    )
    tts_fallback_voice: str = Field(
        default="en_US-raphael-medium",
        description="Local Piper voice used when Chatterbox Turbo is unavailable",
    )
    tts_voice_profiles: str = Field(default="voice_profiles")
    tts_chatterbox_model: str = Field(default="data/voice/models/chatterbox-turbo")
    tts_chatterbox_python: str = Field(default="data/voice/envs/clone/bin/python")
    tts_min_free_vram_mb: int = Field(default=3000, ge=0, le=6144)
    tts_audio_queue_size: int = Field(default=2, ge=1, le=4)
    tts_expressiveness: Literal["expressive", "natural", "off"] = Field(
        default="expressive", description="Vocal event frequency; not a model mood control",
    )
    tts_synthesis_timeout_seconds: float = Field(default=60.0, ge=0.1, le=300)
    tts_playback_latency: float = Field(
        default=0.12, gt=0, le=1,
        description="Output buffer latency in seconds to tolerate audio scheduling delays",
    )
    tts_speed: float = Field(
        default=1.0,
        gt=0.0, le=3.0,
        description="Speech synthesis speed multiplier (1.0 = normal)",
    )
    tts_enabled: bool = Field(
        default=True,
        description="Enable speech synthesis voice output",
    )
    tts_streaming: bool = Field(default=True)
    show_ai_transcripts: bool = Field(default=False)
    sample_rate: int = Field(default=16000, ge=16000, le=16000, description="16 kHz audio")
    channels: int = Field(default=1, ge=1, le=1, description="Mono audio")
    input_device: int | str | None = Field(
        default=None,
        description="Microphone device index or substring name",
    )
    output_device: int | str | None = Field(
        default=None,
        description="Speaker device index or substring name",
    )

    @field_validator("input_device", "output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Accept indices and names for direct audio configuration."""
        return normalize_audio_device(value)


class Settings(BaseSettings):
    """Centralized settings for RAPHAEL loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General app settings
    raphael_env: str = Field(default="development")
    raphael_log_level: str = Field(default="INFO")
    raphael_debug: bool = Field(default=False)
    raphael_preferred_name: str = Field(default="")
    raphael_persona_file: str = Field(
        default="persona.txt",
        description="Optional UTF-8 personality preferences file; empty disables it",
    )

    # Provider settings
    nim_api_key: SecretStr | None = Field(default=None)
    nim_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b")
    nim_complex_model: str = Field(default="nvidia/nemotron-3-super-120b-a12b")
    nim_fallback_model: str = Field(default="meta/llama-3.2-90b-vision-instruct")
    openrouter_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)
    ollama_host: str = Field(default="http://localhost:11434")

    # Memory & Persistence settings
    memory_db_path: str = Field(default="data/raphael.db")
    memory_max_short_term_turns: int = Field(default=10, ge=1)

    # Public internet retrieval settings
    internet_enabled: bool = Field(default=True)
    internet_automatic: bool = Field(default=True)
    internet_home_city: str = Field(default="", max_length=200)
    internet_timeout_seconds: float = Field(default=8.0, ge=0.1, le=60)
    internet_research_timeout_seconds: float = Field(default=20.0, ge=0.1, le=120)
    internet_max_results: int = Field(default=5, ge=1, le=10)
    internet_research_max_results: int = Field(default=8, ge=1, le=20)
    internet_max_pages: int = Field(default=2, ge=0, le=5)
    internet_research_max_pages: int = Field(default=4, ge=0, le=10)

    # Audio & Voice settings
    wake_word: str = Field(default="hey raphael")
    wake_models: list[str] = Field(default_factory=list)
    wake_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    wake_cooldown: float = Field(default=2.0)
    wake_stt_model: str = Field(default="base.en")
    wake_min_rms: float = Field(default=0.006, gt=0.0, le=0.05)
    wake_window_seconds: float = Field(default=3.0, ge=1.5, le=5.0)
    ambient_listening: bool = Field(default=False)
    show_transcripts: bool = Field(default=False)
    barge_in_mode: Literal["speech", "wake"] = Field(default="wake")
    barge_in_speech_seconds: float = Field(default=0.24, ge=0.16, le=1.0)
    ambient_followup_seconds: float = Field(default=300.0, ge=2, le=600)
    ambient_followup_policy: Literal["conversation", "strict"] = Field(default="conversation")
    stt_model: str = Field(default="base.en")
    stt_device: str = Field(default="cpu")
    stt_compute_type: str = Field(default="int8")
    stt_language: str = Field(default="en")
    stt_startup_timeout_seconds: float = Field(default=120.0, ge=0.1, le=300)
    stt_beam_size: int = Field(default=3, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default=None)
    stt_retry_min_free_mb: int = Field(default=2048, ge=0, le=65536)
    utterance_silence_seconds: float = Field(default=0.7, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.2, ge=0, le=3)
    tts_engine: str = Field(default="auto")
    tts_voice: str = Field(default="en_US-amy-medium")
    tts_fallback_voice: str = Field(default="en_US-raphael-medium")
    tts_voice_profiles: str = Field(default="voice_profiles")
    tts_chatterbox_model: str = Field(default="data/voice/models/chatterbox-turbo")
    tts_chatterbox_python: str = Field(default="data/voice/envs/clone/bin/python")
    tts_min_free_vram_mb: int = Field(default=3000, ge=0, le=6144)
    tts_audio_queue_size: int = Field(default=2, ge=1, le=4)
    tts_expressiveness: Literal["expressive", "natural", "off"] = Field(default="expressive")
    tts_synthesis_timeout_seconds: float = Field(default=60.0, ge=0.1, le=300)
    tts_playback_latency: float = Field(default=0.12, gt=0, le=1)
    tts_speed: float = Field(default=1.0, gt=0.0, le=3.0)
    tts_enabled: bool = Field(default=True)
    tts_streaming: bool = Field(default=True)
    show_ai_transcripts: bool = Field(default=False)
    audio_sample_rate: int = Field(default=16000, ge=16000, le=16000)
    audio_channels: int = Field(default=1, ge=1, le=1)
    audio_input_device: int | str | None = Field(default=None)
    audio_output_device: int | str | None = Field(default=None)

    @field_validator("internet_home_city", mode="before")
    @classmethod
    def strip_home_city(cls, value: str) -> str:
        """Trim the configured default location before validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("audio_input_device", "audio_output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Parse device selections loaded from environment variables."""
        return normalize_audio_device(value)

    @property
    def app(self) -> AppConfig:
        """Structured application configuration."""
        return AppConfig(
            env=self.raphael_env,
            log_level=self.raphael_log_level.upper(),
            debug=self.raphael_debug or (self.raphael_log_level.upper() == "DEBUG"),
        )

    @property
    def memory(self) -> MemoryConfig:
        """Structured memory and persistence configuration."""
        return MemoryConfig(
            db_path=self.memory_db_path,
            max_short_term_turns=self.memory_max_short_term_turns,
        )

    @property
    def internet(self) -> InternetConfig:
        """Structured public web retrieval configuration."""
        return InternetConfig(
            enabled=self.internet_enabled,
            automatic=self.internet_automatic,
            home_city=self.internet_home_city,
            timeout_seconds=self.internet_timeout_seconds,
            research_timeout_seconds=self.internet_research_timeout_seconds,
            max_results=self.internet_max_results,
            research_max_results=self.internet_research_max_results,
            max_pages=self.internet_max_pages,
            research_max_pages=self.internet_research_max_pages,
        )

    @property
    def providers(self) -> ProviderConfig:
        """Structured provider configuration."""
        return ProviderConfig(
            nim_api_key=self.nim_api_key,
            nim_model=self.nim_model,
            nim_complex_model=self.nim_complex_model,
            nim_fallback_model=self.nim_fallback_model,
            openrouter_api_key=self.openrouter_api_key,
            groq_api_key=self.groq_api_key,
            ollama_host=self.ollama_host,
        )

    @property
    def audio(self) -> AudioConfig:
        """Structured audio configuration."""
        return AudioConfig(
            wake_word=self.wake_word,
            wake_models=self.wake_models,
            wake_threshold=self.wake_threshold,
            wake_cooldown=self.wake_cooldown,
            wake_stt_model=self.wake_stt_model,
            wake_min_rms=self.wake_min_rms,
            wake_window_seconds=self.wake_window_seconds,
            ambient_listening=self.ambient_listening,
            show_transcripts=self.show_transcripts,
            barge_in_mode=self.barge_in_mode,
            barge_in_speech_seconds=self.barge_in_speech_seconds,
            ambient_followup_seconds=self.ambient_followup_seconds,
            ambient_followup_policy=self.ambient_followup_policy,
            stt_model=self.stt_model,
            stt_device=self.stt_device,
            stt_compute_type=self.stt_compute_type,
            stt_language=self.stt_language,
            stt_startup_timeout_seconds=self.stt_startup_timeout_seconds,
            stt_beam_size=self.stt_beam_size,
            stt_min_confidence=self.stt_min_confidence,
            stt_retry_confidence=self.stt_retry_confidence,
            stt_retry_model=self.stt_retry_model,
            stt_retry_min_free_mb=self.stt_retry_min_free_mb,
            utterance_silence_seconds=self.utterance_silence_seconds,
            utterance_pause_grace_seconds=self.utterance_pause_grace_seconds,
            tts_engine=self.tts_engine,
            tts_voice=self.tts_voice,
            tts_fallback_voice=self.tts_fallback_voice,
            tts_voice_profiles=self.tts_voice_profiles,
            tts_chatterbox_model=self.tts_chatterbox_model,
            tts_chatterbox_python=self.tts_chatterbox_python,
            tts_min_free_vram_mb=self.tts_min_free_vram_mb,
            tts_audio_queue_size=self.tts_audio_queue_size,
            tts_expressiveness=self.tts_expressiveness,
            tts_synthesis_timeout_seconds=self.tts_synthesis_timeout_seconds,
            tts_playback_latency=self.tts_playback_latency,
            tts_speed=self.tts_speed,
            tts_enabled=self.tts_enabled,
            tts_streaming=self.tts_streaming,
            show_ai_transcripts=self.show_ai_transcripts,
            sample_rate=self.audio_sample_rate,
            channels=self.audio_channels,
            input_device=self.audio_input_device,
            output_device=self.audio_output_device,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
