"""Centralized configuration and secrets management for CIEL."""

from functools import lru_cache
from typing import Literal

from dotenv.main import resolve_variables
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_NIM_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_NIM_COMPLEX_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_NIM_FALLBACK_MODEL = DEFAULT_NIM_COMPLEX_MODEL
DEFAULT_CONTROLLER_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_SCREEN_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def _register_search_secret(value: SecretStr | None) -> SecretStr | None:
    """Mask configured search credentials in shared logs, retrieval, and saved evidence."""
    if value is not None:
        from ciel.logging import SecretMaskingFilter

        SecretMaskingFilter.register_secrets(value.get_secret_value())
    return value


def resolve_setup_values(configuration: dict[str, str]) -> dict[str, str]:
    """Expand pending dotenv references without changing their saved literal values."""
    return {
        key: value if value is not None else ""
        for key, value in resolve_variables(configuration.items(), override=True).items()
    }


def _preserve_explicit_voice_selection(values: object) -> object:
    """Keep legacy Piper choices while fresh installations default to Pocket."""
    if not isinstance(values, dict):
        return values
    values = values.copy()
    if "tts_voice" in values and "tts_engine" not in values:
        values["tts_engine"] = "chatterbox_turbo" if values["tts_voice"] == "ciel" else "auto"
    if "tts_voice" not in values and values.get("tts_engine") in {"auto", "piper"}:
        values["tts_voice"] = "en_US-amy-medium"
    return values


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
        default="data/ciel.db",
        description="Path to SQLite memory database file",
    )
    max_short_term_turns: int = Field(
        default=10,
        ge=1,
        description="Maximum turns of conversation to keep in short-term context",
    )
    learning_mode: Literal["confirm", "automatic", "explicit"] = "confirm"
    confirm_corrections: bool = True
    recall_enabled: bool = True
    recall_limit: int = Field(default=4, ge=1, le=50)
    recall_max_chars: int = Field(default=6000, ge=256, le=100000)
    max_context_chars: int = Field(default=24000, ge=256, le=200000)
    semantic_enabled: bool = False
    semantic_model: str = Field(default="", max_length=256)
    semantic_host: str = "http://127.0.0.1:11434"
    semantic_timeout_seconds: float = Field(default=1.5, ge=0.1, le=10)
    workspace_enabled: bool = True
    workspace_path: str = Field(default="", max_length=4096)
    workspace_bootstrap_max_chars: int = Field(default=8000, ge=256, le=100000)
    daily_capture_enabled: bool = True
    flush_enabled: bool = True
    consolidation_enabled: bool = True
    consolidation_time: str = Field(default="03:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    search_max_results: int = Field(default=6, ge=1, le=50)
    search_min_score: float = Field(default=0.35, ge=0, le=1)


class ControllerConfig(BaseModel):
    """Compatibility settings for main tools, progress, and legacy diagnostics."""

    enabled: bool = True
    model: str = DEFAULT_CONTROLLER_MODEL
    timeout_seconds: float = Field(default=8.0, ge=0.1, le=30)
    max_tool_steps: int = Field(default=3, ge=1, le=5)


class InternetConfig(BaseModel):
    """Selected public search provider, credentials, and page-reading limits."""

    enabled: bool = Field(default=True, description="Enable public web retrieval")
    automatic: bool = Field(default=True, description="Search requests needing current facts")
    provider: Literal["parallel", "ddgs", "brave", "tavily"] = Field(default="parallel")
    brave_api_key: SecretStr | None = Field(default=None)
    tavily_api_key: SecretStr | None = Field(default=None)
    home_city: str = Field(
        default="", max_length=200, description="Optional default city and country",
    )
    timeout_seconds: float = Field(default=8.0, ge=0.1, le=60)
    research_timeout_seconds: float = Field(default=20.0, ge=0.1, le=120)
    max_results: int = Field(default=5, ge=1, le=10)
    research_max_results: int = Field(default=8, ge=1, le=20)
    max_pages: int = Field(default=2, ge=0, le=5)
    research_max_pages: int = Field(default=4, ge=0, le=10)

    @field_validator("brave_api_key", "tavily_api_key")
    @classmethod
    def register_search_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Register validated keys without reading any configuration file again."""
        return _register_search_secret(value)

    @field_validator("home_city", mode="before")
    @classmethod
    def strip_home_city(cls, value: str) -> str:
        """Normalize an optional city without inferring the user's location."""
        return value.strip() if isinstance(value, str) else value


class ProviderConfig(BaseModel):
    """AI Provider API keys and endpoints."""

    primary_provider: Literal["auto", "nim", "groq", "openrouter", "ollama"] = "auto"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "openrouter/free"
    ollama_model: str = "llama3.2"
    nim_api_key: SecretStr | None = Field(
        default=None,
        description="NVIDIA NIM API key",
    )
    nim_model: str = Field(
        default=DEFAULT_NIM_MODEL,
        description="Default NVIDIA NIM model name",
    )
    nim_complex_model: str = Field(
        default=DEFAULT_NIM_COMPLEX_MODEL,
        description="NVIDIA NIM model for complex reasoning and coding tasks",
    )
    nim_fallback_model: str = Field(
        default=DEFAULT_NIM_FALLBACK_MODEL,
        description="NVIDIA NIM fallback model for failover / rate-limiting",
    )
    nim_reasoning: Literal["auto", "on", "off"] = Field(
        default="auto", description="Reasoning for conversation: auto, on, or off",
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


class ChatConfig(BaseModel):
    """Optional messaging channels, independent of desktop audio."""

    services: list[Literal["telegram"]] = Field(default_factory=list)
    telegram_bot_token: SecretStr | None = None
    state_path: str = Field(default="data/chat.db", min_length=1)


class AudioConfig(BaseModel):
    """Audio, wake word, and speech-to-text settings."""

    wake_word: str = Field(default="hey ciel", description="Wake word trigger phrase")
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
    ambient_listening: bool = Field(default=True)
    show_transcripts: bool = Field(default=False)
    barge_in_mode: Literal["speech", "wake"] = Field(default="speech")
    barge_in_speech_seconds: float = Field(default=0.24, ge=0.16, le=1.0)
    ambient_followup_seconds: float = Field(default=300.0, ge=2, le=600)
    ambient_followup_policy: Literal["conversation", "strict", "always"] = Field(
        default="conversation",
    )
    wake_models: list[str] = Field(
        default_factory=list,
        description="openWakeWord model names/paths (empty = Whisper-only keyword spotter)",
    )
    stt_model: str = Field(
        default="small.en",
        description="faster-whisper model size (e.g. tiny.en, base.en, small.en)",
    )
    stt_device: str = Field(
        default="cuda",
        description="Inference device for whisper (auto, cuda, cpu)",
    )
    stt_compute_type: str = Field(
        default="int8_float16",
        description="Compute precision (default, int8, int8_float16, float16, float32)",
    )
    stt_language: str = Field(
        default="en",
        description="Primary language code for STT transcription",
    )
    stt_startup_timeout_seconds: float = Field(default=120.0, ge=0.1, le=300)
    stt_beam_size: int = Field(default=5, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default="medium.en")
    stt_retry_min_free_mb: int = Field(default=2048, ge=0, le=65536)
    utterance_silence_seconds: float = Field(default=0.7, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.2, ge=0, le=3)
    tts_engine: Literal["auto", "piper", "pocket", "chatterbox", "chatterbox_turbo", "hq"] = Field(
        default="pocket",
        description=(
            "TTS engine: auto (Piper), piper, pocket, or chatterbox_turbo (hq)"
        ),
    )
    tts_voice: str = Field(
        default="ciel",
        description="TTS profile or Piper voice name",
    )
    tts_fallback_voice: str = Field(
        default="en_US-ciel-medium",
        description="Local Piper voice used when Pocket or Chatterbox is unavailable",
    )
    tts_pocket_device: Literal["cpu"] = Field(default="cpu")
    tts_pocket_quantization: Literal["none", "int8"] = Field(default="none")
    tts_pocket_streaming: bool = Field(default=True)
    tts_pocket_gain_db: float = Field(default=0.0, ge=-60, le=0, allow_inf_nan=False)
    tts_pocket_references: str = Field(default="data/voices/ciel")
    tts_pocket_neutral_reference: str | None = Field(default=None)
    tts_pocket_language: str = Field(default="english_2026-09", min_length=1)
    tts_pocket_builtin_voice: str = Field(default="alba", min_length=1)
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
    show_ai_transcripts: bool = Field(default=True)
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

    @model_validator(mode="before")
    @classmethod
    def preserve_explicit_voice_selection(cls, values: object) -> object:
        """Apply new voice defaults without changing existing Piper selections."""
        return _preserve_explicit_voice_selection(values)

    @field_validator("input_device", "output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Accept indices and names for direct audio configuration."""
        return normalize_audio_device(value)


class Settings(BaseSettings):
    """Centralized settings for CIEL loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General app settings
    ciel_env: str = Field(default="development")
    ciel_log_level: str = Field(default="INFO")
    ciel_debug: bool = Field(default=False)
    ciel_preferred_name: str = Field(default="")
    ciel_persona_file: str = Field(
        default="persona.txt",
        description="Optional UTF-8 personality preferences file; empty disables it",
    )

    # Hosted web access; an exact private IP requires a local HTTPS certificate.
    web_host: str = '127.0.0.1'
    web_insecure_lan: bool = False
    web_cert: str = ''
    web_key: str = ''

    primary_provider: Literal["auto", "nim", "groq", "openrouter", "ollama"] = "auto"
    groq_model: str = "llama-3.1-8b-instant"
    openrouter_model: str = "openrouter/free"
    ollama_model: str = "llama3.2"

    # Provider settings
    nim_api_key: SecretStr | None = Field(default=None)
    nim_model: str = Field(default=DEFAULT_NIM_MODEL)
    nim_complex_model: str = Field(default=DEFAULT_NIM_COMPLEX_MODEL)
    nim_fallback_model: str = Field(default=DEFAULT_NIM_FALLBACK_MODEL)
    nim_reasoning: Literal["auto", "on", "off"] = "auto"
    openrouter_api_key: SecretStr | None = Field(default=None)
    groq_api_key: SecretStr | None = Field(default=None)
    ollama_host: str = Field(default="http://localhost:11434")

    # Screenshots use only this explicit image connection, never text-model fallback.
    screen_enabled: bool = Field(default=True)
    screen_provider: Literal["nim", "groq", "openrouter", "ollama"] = "nim"
    screen_model: str = Field(default=DEFAULT_SCREEN_MODEL, min_length=1, max_length=256)
    screen_timeout_seconds: float = Field(default=60.0, ge=1, le=120)

    # Messaging is disabled until a service is selected and paired locally.
    chat_services: str = ""
    telegram_bot_token: SecretStr | None = None
    chat_state_path: str = Field(default="data/chat.db", min_length=1)

    # Memory & Persistence settings
    memory_db_path: str = Field(default="data/ciel.db")
    memory_max_short_term_turns: int = Field(default=10, ge=1)
    memory_learning_mode: Literal["confirm", "automatic", "explicit"] = "confirm"
    memory_confirm_corrections: bool = True
    memory_recall_enabled: bool = True
    memory_recall_limit: int = Field(default=4, ge=1, le=50)
    memory_recall_max_chars: int = Field(default=6000, ge=256, le=100000)
    memory_max_context_chars: int = Field(default=24000, ge=256, le=200000)
    memory_semantic_enabled: bool = False
    memory_semantic_model: str = Field(default="", max_length=256)
    memory_semantic_host: str = "http://127.0.0.1:11434"
    memory_semantic_timeout_seconds: float = Field(default=1.5, ge=0.1, le=10)
    memory_workspace_enabled: bool = True
    memory_workspace_path: str = Field(default="", max_length=4096)
    memory_workspace_bootstrap_max_chars: int = Field(default=8000, ge=256, le=100000)
    memory_daily_capture_enabled: bool = True
    memory_flush_enabled: bool = True
    memory_consolidation_enabled: bool = True
    memory_consolidation_time: str = Field(
        default="03:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
    )
    memory_search_max_results: int = Field(default=6, ge=1, le=50)
    memory_search_min_score: float = Field(default=0.35, ge=0, le=1)

    # Compatibility names: default foreground tool selection belongs to the main model.
    controller_enabled: bool = Field(default=True)
    controller_model: str = Field(default=DEFAULT_CONTROLLER_MODEL)
    controller_timeout_seconds: float = Field(default=8.0, ge=0.1, le=30)
    controller_max_tool_steps: int = Field(default=3, ge=1, le=5)

    # Public internet retrieval settings
    internet_enabled: bool = Field(default=True)
    internet_automatic: bool = Field(default=True)
    internet_provider: Literal["parallel", "ddgs", "brave", "tavily"] = Field(default="parallel")
    brave_search_api_key: SecretStr | None = Field(default=None)
    tavily_api_key: SecretStr | None = Field(default=None)
    internet_home_city: str = Field(default="", max_length=200)
    internet_timeout_seconds: float = Field(default=8.0, ge=0.1, le=60)
    internet_research_timeout_seconds: float = Field(default=20.0, ge=0.1, le=120)
    internet_max_results: int = Field(default=5, ge=1, le=10)
    internet_research_max_results: int = Field(default=8, ge=1, le=20)
    internet_max_pages: int = Field(default=2, ge=0, le=5)
    internet_research_max_pages: int = Field(default=4, ge=0, le=10)

    # Audio & Voice settings
    wake_word: str = Field(default="hey ciel")
    wake_models: list[str] = Field(default_factory=list)
    wake_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    wake_cooldown: float = Field(default=2.0)
    wake_stt_model: str = Field(default="base.en")
    wake_min_rms: float = Field(default=0.006, gt=0.0, le=0.05)
    wake_window_seconds: float = Field(default=3.0, ge=1.5, le=5.0)
    ambient_listening: bool = Field(default=True)
    show_transcripts: bool = Field(default=False)
    barge_in_mode: Literal["speech", "wake"] = Field(default="speech")
    barge_in_speech_seconds: float = Field(default=0.24, ge=0.16, le=1.0)
    ambient_followup_seconds: float = Field(default=300.0, ge=2, le=600)
    ambient_followup_policy: Literal["conversation", "strict", "always"] = Field(
        default="conversation",
    )
    stt_model: str = Field(default="small.en")
    stt_device: str = Field(default="cuda")
    stt_compute_type: str = Field(default="int8_float16")
    stt_language: str = Field(default="en")
    stt_startup_timeout_seconds: float = Field(default=120.0, ge=0.1, le=300)
    stt_beam_size: int = Field(default=5, ge=1, le=10)
    stt_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    stt_retry_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    stt_retry_model: str | None = Field(default="medium.en")
    stt_retry_min_free_mb: int = Field(default=2048, ge=0, le=65536)
    utterance_silence_seconds: float = Field(default=0.7, ge=0.3, le=5.0)
    utterance_pause_grace_seconds: float = Field(default=0.2, ge=0, le=3)
    tts_engine: Literal["auto", "piper", "pocket", "chatterbox", "chatterbox_turbo", "hq"] = Field(
        default="pocket",
    )
    tts_voice: str = Field(default="ciel")
    tts_fallback_voice: str = Field(default="en_US-ciel-medium")
    tts_pocket_device: Literal["cpu"] = Field(default="cpu")
    tts_pocket_quantization: Literal["none", "int8"] = Field(default="none")
    tts_pocket_streaming: bool = Field(default=True)
    tts_pocket_gain_db: float = Field(default=0.0, ge=-60, le=0, allow_inf_nan=False)
    tts_pocket_references: str = Field(default="data/voices/ciel")
    tts_pocket_neutral_reference: str | None = Field(default=None)
    tts_pocket_language: str = Field(default="english_2026-09", min_length=1)
    tts_pocket_builtin_voice: str = Field(default="alba", min_length=1)
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
    show_ai_transcripts: bool = Field(default=True)
    audio_sample_rate: int = Field(default=16000, ge=16000, le=16000)
    audio_channels: int = Field(default=1, ge=1, le=1)
    audio_input_device: int | str | None = Field(default=None)
    audio_output_device: int | str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def preserve_explicit_voice_selection(cls, values: object) -> object:
        """Apply new voice defaults without changing existing Piper selections."""
        return _preserve_explicit_voice_selection(values)

    @field_validator("internet_home_city", mode="before")
    @classmethod
    def strip_home_city(cls, value: str) -> str:
        """Trim the configured default location before validation."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("brave_search_api_key", "tavily_api_key")
    @classmethod
    def register_search_key(cls, value: SecretStr | None) -> SecretStr | None:
        """Mask flat environment credentials before structured configuration is accessed."""
        return _register_search_secret(value)

    @field_validator("audio_input_device", "audio_output_device", mode="before")
    @classmethod
    def parse_device(cls, value: int | str | None) -> int | str | None:
        """Parse device selections loaded from environment variables."""
        return normalize_audio_device(value)

    @field_validator("chat_services", mode="before")
    @classmethod
    def parse_chat_services(cls, value: str) -> str:
        """Accept comma-separated implemented services without enabling future placeholders."""
        if not isinstance(value, str):
            raise ValueError("CHAT_SERVICES must be a comma-separated string")
        names = list(dict.fromkeys(
            part.strip().lower() for part in value.split(",") if part.strip()
        ))
        if any(name != "telegram" for name in names):
            raise ValueError("CHAT_SERVICES supports telegram; leave empty to disable chat")
        return ",".join(names)

    @property
    def app(self) -> AppConfig:
        """Structured application configuration."""
        return AppConfig(
            env=self.ciel_env,
            log_level=self.ciel_log_level.upper(),
            debug=self.ciel_debug or (self.ciel_log_level.upper() == "DEBUG"),
        )

    @property
    def memory(self) -> MemoryConfig:
        """Structured memory and persistence configuration."""
        return MemoryConfig(
            db_path=self.memory_db_path,
            max_short_term_turns=self.memory_max_short_term_turns,
            learning_mode=self.memory_learning_mode,
            confirm_corrections=self.memory_confirm_corrections,
            recall_enabled=self.memory_recall_enabled,
            recall_limit=self.memory_recall_limit,
            recall_max_chars=self.memory_recall_max_chars,
            max_context_chars=self.memory_max_context_chars,
            semantic_enabled=self.memory_semantic_enabled,
            semantic_model=self.memory_semantic_model,
            semantic_host=self.memory_semantic_host,
            semantic_timeout_seconds=self.memory_semantic_timeout_seconds,
            workspace_enabled=self.memory_workspace_enabled,
            workspace_path=self.memory_workspace_path,
            workspace_bootstrap_max_chars=self.memory_workspace_bootstrap_max_chars,
            daily_capture_enabled=self.memory_daily_capture_enabled,
            flush_enabled=self.memory_flush_enabled,
            consolidation_enabled=self.memory_consolidation_enabled,
            consolidation_time=self.memory_consolidation_time,
            search_max_results=self.memory_search_max_results,
            search_min_score=self.memory_search_min_score,
        )

    @property
    def controller(self) -> ControllerConfig:
        """Compatibility tool-loop and legacy diagnostic configuration."""
        return ControllerConfig(
            enabled=self.controller_enabled, model=self.controller_model,
            timeout_seconds=self.controller_timeout_seconds,
            max_tool_steps=self.controller_max_tool_steps,
        )

    @property
    def internet(self) -> InternetConfig:
        """Structured public web retrieval configuration."""
        return InternetConfig(
            enabled=self.internet_enabled,
            automatic=self.internet_automatic,
            provider=self.internet_provider,
            brave_api_key=self.brave_search_api_key,
            tavily_api_key=self.tavily_api_key,
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
            primary_provider=self.primary_provider, groq_model=self.groq_model,
            openrouter_model=self.openrouter_model, ollama_model=self.ollama_model,
            nim_api_key=self.nim_api_key,
            nim_model=self.nim_model,
            nim_complex_model=self.nim_complex_model,
            nim_fallback_model=self.nim_fallback_model,
            nim_reasoning=self.nim_reasoning,
            openrouter_api_key=self.openrouter_api_key,
            groq_api_key=self.groq_api_key,
            ollama_host=self.ollama_host,
        )

    @property
    def chat(self) -> ChatConfig:
        """Structured optional chat configuration; secrets remain masked."""
        return ChatConfig(
            services=self.chat_services.split(",") if self.chat_services else [],
            telegram_bot_token=self.telegram_bot_token,
            state_path=self.chat_state_path,
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
            tts_pocket_device=self.tts_pocket_device,
            tts_pocket_quantization=self.tts_pocket_quantization,
            tts_pocket_streaming=self.tts_pocket_streaming,
            tts_pocket_gain_db=self.tts_pocket_gain_db,
            tts_pocket_references=self.tts_pocket_references,
            tts_pocket_neutral_reference=self.tts_pocket_neutral_reference,
            tts_pocket_language=self.tts_pocket_language,
            tts_pocket_builtin_voice=self.tts_pocket_builtin_voice,
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
