"""Interactive setup and configuration wizard for CIEL."""

import json
import os
from getpass import getpass
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import dotenv_values, set_key, unset_key

from ciel.audio.tts import TextToSpeech, resolve_tts_engine
from ciel.config import AudioConfig, ProviderConfig, get_settings, normalize_audio_device
from ciel.logging import get_logger
from ciel.paths import resolve_asset_path, voice_profile_path
from ciel.platform import get_audio_backend
from ciel.terminal_ui import ConsoleUI

logger = get_logger("setup")


def custom_voice_asset_problems(configuration: dict[str, str]) -> list[str]:
    """Check local custom-voice prerequisites without loading models or downloading files."""
    defaults = AudioConfig()

    problems = []
    python = resolve_asset_path(
        configuration.get("TTS_CHATTERBOX_PYTHON", defaults.tts_chatterbox_python),
        preserve_symlinks=True,
    )
    if not python.is_file():
        problems.append(f"Turbo Python environment: {python}")
    manifest = voice_profile_path(
        configuration["TTS_VOICE"],
        configuration.get("TTS_VOICE_PROFILES", defaults.tts_voice_profiles),
    )
    try:
        profile = json.loads(manifest.read_text(encoding="utf-8"))
        if profile["engine"] != "chatterbox_turbo":
            raise ValueError("Expected a Chatterbox Turbo profile")
        for key in ("reference_audio", "reference_transcript"):
            path = resolve_asset_path(profile[key])
            if not path.is_file() or not path.stat().st_size:
                problems.append(f"{key}: {path}")
    except (OSError, ValueError, KeyError, TypeError):
        problems.append(f"Missing or invalid voice profile: {manifest}")
    model = resolve_asset_path(
        configuration.get("TTS_CHATTERBOX_MODEL", defaults.tts_chatterbox_model),
    )
    for name in (
        "inventory.json", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors",
        "ve.safetensors", "added_tokens.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "vocab.json",
    ):
        path = model / name
        if not path.is_file() or not path.stat().st_size:
            problems.append(f"Turbo model file: {path}")
    return problems


def _initial_configuration(current_env: dict[str, str]) -> dict[str, str]:
    """Fill missing settings while preserving explicit and legacy voice selections."""
    audio = AudioConfig()
    existing_engine = current_env.get("TTS_ENGINE")
    current_voice = current_env.get(
        "TTS_VOICE",
        "en_US-amy-medium" if existing_engine in {"auto", "piper"} else audio.tts_voice,
    )
    current_engine = existing_engine or (
        ("chatterbox_turbo" if current_voice == "ciel" else resolve_tts_engine(current_voice))
        if "TTS_VOICE" in current_env else audio.tts_engine
    )
    defaults = {
        "CIEL_ENV": "development",
        "CIEL_LOG_LEVEL": "INFO",
        "NIM_MODEL": ProviderConfig().nim_model,
        "OLLAMA_HOST": "http://localhost:11434",
        "WAKE_WORD": "hey ciel",
        "WAKE_THRESHOLD": "0.5",
        "WAKE_COOLDOWN": "2.0",
        "STT_MODEL": audio.stt_model,
        "STT_DEVICE": audio.stt_device,
        "STT_COMPUTE_TYPE": audio.stt_compute_type,
        "STT_LANGUAGE": audio.stt_language,
        "STT_BEAM_SIZE": str(audio.stt_beam_size),
        "STT_RETRY_MODEL": audio.stt_retry_model or "",
        "STT_RETRY_MIN_FREE_MB": str(audio.stt_retry_min_free_mb),
        "AMBIENT_LISTENING": str(audio.ambient_listening).lower(),
        "SHOW_AI_TRANSCRIPTS": str(audio.show_ai_transcripts).lower(),
        "BARGE_IN_MODE": audio.barge_in_mode,
        "TTS_VOICE": current_voice,
        "TTS_ENGINE": current_engine,
        "TTS_SPEED": "1.0",
        "TTS_ENABLED": "true",
    }
    for key, value in defaults.items():
        current_env.setdefault(key, value)
    return current_env


def _save_configuration(env_path: Path, configuration: dict[str, str]) -> None:
    """Keep unchanged comments, key casing, and literal values when saving settings."""
    original_env = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
    original_keys = {key.upper(): key for key in original_env}
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=env_path.parent, prefix=".ciel-setup-", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(env_path.read_text(encoding="utf-8") if env_path.exists()
                            else "# CIEL Configuration File\n")
        for key in original_keys.keys() - configuration.keys():
            if key in {"AUDIO_INPUT_DEVICE", "AUDIO_OUTPUT_DEVICE"}:
                unset_key(temporary_path, original_keys[key])
        for key, value in configuration.items():
            original_key = original_keys.get(key, key)
            if original_env.get(original_key) != value:
                set_key(temporary_path, original_key, value, quote_mode="always")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, env_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    get_settings.cache_clear()


def _configure_keys(ui: ConsoleUI, configuration: dict[str, str], *, primary_only: bool) -> None:
    """Request secrets without displaying any portion of previously saved keys."""
    ui.step("AI connection")
    ui.note("Keys are optional. Without a cloud key, local tools remain available.")
    providers = [("NIM_API_KEY", "NVIDIA NIM")]
    if not primary_only:
        providers.extend([("GROQ_API_KEY", "Groq"), ("OPENROUTER_API_KEY", "OpenRouter")])
    for key, name in providers:
        status = ("configured; Enter to keep" if configuration.get(key)
                  else "optional; Enter to skip")
        answer = getpass(f"{name} API key [{status}]: ").strip()
        if answer:
            configuration[key] = answer


def _select_device(ui: ConsoleUI, configuration: dict[str, str], *, output: bool) -> None:
    """Select a listed device or retain a custom index or name."""
    key = "AUDIO_OUTPUT_DEVICE" if output else "AUDIO_INPUT_DEVICE"
    title = "Speakers" if output else "Microphone"
    ui.step(title)
    choices = [
        ("keep", f"Keep current ({configuration.get(key) or 'system default'})"),
        ("default", "Use system default"),
    ]
    try:
        for device in get_audio_backend().list_devices():
            channels = device.max_output_channels if output else device.max_input_channels
            if channels > 0:
                choices.append((f"device:{device.index}", device.name))
    except Exception:
        ui.note("Audio devices could not be listed. You can still enter a device name or index.")
    choices.append(("custom", "Enter a device name or index"))
    selected = ui.choose(title, choices, default="keep")
    if selected == "default":
        configuration.pop(key, None)
    elif selected == "custom":
        value = ui.prompt("Device name or index (Enter to keep)").strip()
        if value:
            configuration[key] = value
    elif selected.startswith("device:"):
        configuration[key] = selected.split(":", 1)[1]


def _enabled(configuration: dict[str, str], key: str) -> bool:
    """Interpret the boolean spellings accepted by environment configuration."""
    return configuration.get(key, "").strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _voice_label(configuration: dict[str, str]) -> str:
    """Use recognizable names while retaining arbitrary custom voice selections."""
    voice, engine = configuration["TTS_VOICE"], configuration["TTS_ENGINE"]
    if voice == "ciel":
        if engine in {"chatterbox", "chatterbox_turbo", "hq"}:
            return "Ciel Turbo"
        if engine == "pocket":
            return "Ciel Pocket"
    if voice == "en_US-amy-medium":
        return "Amy · Piper"
    return voice


def _customize(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Walk through one decision at a time while retaining existing choices on Enter."""
    _configure_keys(ui, configuration, primary_only=False)
    ui.step("Voice")
    ui.note("Pocket runs on CPU; Turbo requires its separate model and private reference.")
    voice = ui.choose("Choose a voice", [
        ("keep", f"Keep current ({_voice_label(configuration)})"),
        ("pocket", "Ciel Pocket · CPU"),
        ("amy", "Amy · Piper (CPU)"),
        ("turbo", "Ciel Turbo · NVIDIA GPU"),
    ], default="keep")
    if voice != "keep":
        configuration["TTS_VOICE"] = "en_US-amy-medium" if voice == "amy" else "ciel"
        configuration["TTS_ENGINE"] = {"amy": "piper", "turbo": "chatterbox_turbo"}.get(
            voice, "pocket",
        )
    _select_device(ui, configuration, output=False)
    _select_device(ui, configuration, output=True)
    ui.step("Conversation")
    listening = ui.choose("Listening", [
        ("ambient", "Ambient conversation"), ("wake", "Wake word only"),
    ], default="ambient" if _enabled(configuration, "AMBIENT_LISTENING") else "wake")
    if (listening == "ambient") != _enabled(configuration, "AMBIENT_LISTENING"):
        configuration["AMBIENT_LISTENING"] = "true" if listening == "ambient" else "false"
    captions = ui.choose("AI reply captions", [("on", "Show captions"), ("off", "Hide captions")],
                         default="on" if _enabled(configuration, "SHOW_AI_TRANSCRIPTS") else "off")
    if (captions == "on") != _enabled(configuration, "SHOW_AI_TRANSCRIPTS"):
        configuration["SHOW_AI_TRANSCRIPTS"] = "true" if captions == "on" else "false"


def _review_configuration(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Show the choices that will be saved without claiming an untested connection."""
    providers = [name for key, name in (
        ("NIM_API_KEY", "NVIDIA NIM"), ("GROQ_API_KEY", "Groq"),
        ("OPENROUTER_API_KEY", "OpenRouter"),
    ) if configuration.get(key)]
    ui.step("Review your setup")
    ui.summary([
        ("AI connection", f"{', '.join(providers)} · key configured" if providers
         else "No cloud key; local tools available"),
        ("Voice", _voice_label(configuration) if _enabled(configuration, "TTS_ENABLED")
         else "Voice output off"),
        ("Recognition", f"{configuration['STT_MODEL']} · {configuration['STT_DEVICE'].upper()}"),
        ("Listening", "Ambient conversation" if _enabled(configuration, "AMBIENT_LISTENING")
         else "Wake word only"),
        ("Live captions", "On" if _enabled(configuration, "SHOW_AI_TRANSCRIPTS") else "Off"),
        ("Microphone", configuration.get("AUDIO_INPUT_DEVICE") or "System default"),
        ("Speakers", configuration.get("AUDIO_OUTPUT_DEVICE") or "System default"),
    ])
    if providers:
        ui.note("The AI connection hasn't been tested yet.")


def _test_voice(ui: ConsoleUI, configuration: dict[str, str]) -> bool:
    """Attempt real playback and report failure or missing assets honestly."""
    if not _enabled(configuration, "TTS_ENABLED"):
        ui.note("Voice output is disabled. No sample was played.")
        return False
    chosen_voice = configuration["TTS_VOICE"]
    engine = configuration["TTS_ENGINE"]
    tts = None
    try:
        custom_voice = resolve_tts_engine(chosen_voice, engine) == "chatterbox_turbo"
        problems = custom_voice_asset_problems(configuration) if custom_voice else []
        if problems:
            ui.note("Custom voice is not ready. Missing or invalid local assets:")
            for problem in problems:
                ui.note(problem)
            ui.note("Your choice is saved. Restore the assets or rerun setup and choose Amy.")
            ui.note("See docs/chatterbox-turbo-integration.md for the requirements.")
            return False
        ui.note(f"Playing a short sample with {chosen_voice}.")
        tts = TextToSpeech(
            voice_name=chosen_voice, engine=engine, enabled=True,
            output_device=normalize_audio_device(configuration.get("AUDIO_OUTPUT_DEVICE")),
        )
        success = tts.speak("Hi, I'm Ciel. Let's check that you can hear me.", block=True)
        if success:
            ui.note("Voice playback completed. Please check what you heard.")
            if custom_voice or engine == "pocket":
                ui.note(f"Playback engine: {tts.last_engine}. Check any fallback warnings.")
            return True
        ui.note("Audio test failed: speech playback did not succeed.")
    except Exception as error:
        ui.note(
            f"Audio test failed ({type(error).__name__}). Check your voice and speaker settings.",
        )
    finally:
        if tts is not None:
            try:
                tts.close()
            except Exception as error:
                ui.note(f"Voice cleanup did not finish ({type(error).__name__}).")
    return False


def run_setup_wizard() -> None:
    """Review configuration before saving, then let the user verify voice playback."""
    ui = ConsoleUI()
    saved = False
    saving = False
    try:
        ui.banner("Let's make CIEL yours.")
        ui.note("Review everything before saving. Press Ctrl+C to leave setup.")
        env_path = Path(".env")
        original_env = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
        configuration = _initial_configuration({
            key.upper(): value for key, value in original_env.items() if value is not None
        })
        mode = ui.choose("Get started", [
            ("quick", "Quick start"), ("customize", "Customize"),
        ], default="quick")
        if mode == "customize":
            _customize(ui, configuration)
        elif not any(configuration.get(key) for key in (
            "NIM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
        )):
            _configure_keys(ui, configuration, primary_only=True)
        while True:
            _review_configuration(ui, configuration)
            action = ui.choose("Ready to save?", [
                ("save", "Save and test"), ("customize", "Customize"), ("cancel", "Cancel"),
            ], default="save")
            if action == "cancel":
                ui.finish("Setup cancelled. No changes were saved.")
                return
            if action == "save":
                break
            _customize(ui, configuration)
        saving = True
        _save_configuration(env_path, configuration)
        saved = True
        saving = False
        ui.step("Check your voice")
        success = _test_voice(ui, configuration)
        while True:
            action = ui.choose("Did it sound right?", [
                ("yes", "Yes"), ("again", "Play again"),
                ("output", "Change output device"), ("finish", "Finish without audio"),
            ], default="yes" if success else "finish")
            if action == "yes":
                if success:
                    ui.note("Voice confirmed.")
                else:
                    ui.note("Playback was not successful. Audio still needs checking.")
                break
            if action == "finish":
                ui.note("Audio is not confirmed. You can rerun setup whenever you're ready.")
                break
            if action == "output":
                _select_device(ui, configuration, output=True)
                _save_configuration(env_path, configuration)
            success = _test_voice(ui, configuration)
        ui.finish("Configuration saved. Run: ciel")
    except (KeyboardInterrupt, EOFError):
        if saved:
            ui.finish("Setup stopped. Configuration is saved. Run: ciel")
        elif saving:
            ui.finish("Setup stopped while saving. Run setup again to review .env.")
        else:
            ui.finish("Setup cancelled. No changes were saved.")
