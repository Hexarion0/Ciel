"""Interactive setup and configuration wizard for CIEL."""

import json
import os
from getpass import getpass
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import dotenv_values, set_key, unset_key

from ciel.audio.engines import resolve_tts_engine
from ciel.chat.setup import apply_chat_autostart, chat_review_rows, configure_chat
from ciel.config import (
    AudioConfig,
    ProviderConfig,
    get_settings,
    normalize_audio_device,
    resolve_setup_values,
)
from ciel.logging import get_logger
from ciel.paths import resolve_asset_path, voice_profile_path
from ciel.platform import get_audio_backend
from ciel.terminal_ui import ConsoleUI

logger = get_logger("setup")

_API_PROVIDERS = (
    ("nim", "NVIDIA NIM", "NIM_API_KEY"),
    ("groq", "Groq", "GROQ_API_KEY"),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY"),
)
_SEARCH_PROVIDERS = (
    ("ddgs", "Free search · no account or key", None),
    ("brave", "Brave Search", "BRAVE_SEARCH_API_KEY"),
    ("tavily", "Tavily", "TAVILY_API_KEY"),
)


def TextToSpeech(*args, **kwargs):
    """Load voice dependencies only after the user has selected and installed them."""
    from ciel.audio.tts import TextToSpeech as Speech

    return Speech(*args, **kwargs)


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
        "CIEL_PREFERRED_NAME": "",
        "INTERNET_HOME_CITY": "",
        "INTERNET_PROVIDER": "ddgs",
        "MEMORY_LEARNING_MODE": "confirm",
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
        "CHAT_SERVICES": "",
        "CHAT_STATE_PATH": "data/chat.db",
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


def _configure_keys(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Choose a provider before requesting its secret, preserving all other saved keys."""
    ui.step("AI connection")
    ui.note("Keys are optional. Without a cloud key, local tools remain available.")
    ui.note("Choose whose API key to add or update. "
            "With multiple providers, Ciel routes requests automatically.")
    while True:
        configured = any(configuration.get(key) for _value, _name, key in _API_PROVIDERS)
        choices = [
            (value, f"{name} · key configured" if configuration.get(key) else name)
            for value, name, key in _API_PROVIDERS
        ]
        choices.append((
            "done", "Keep saved providers and continue" if configured else "Skip for now",
        ))
        selected = ui.choose(
            "API key provider", choices, default="done" if configured else "nim",
        )
        if selected == "done":
            return
        _value, name, key = next(item for item in _API_PROVIDERS if item[0] == selected)
        status = ("configured; Enter to keep" if configuration.get(key)
                  else "optional; Enter to skip")
        answer = getpass(f"{name} API key [{status}]: ").strip()
        if answer:
            configuration[key] = answer
        if ui.choose("Add another provider?", [
            ("no", "Continue setup"), ("yes", "Add or update another provider"),
        ], default="no") == "no":
            return


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


def _configure_search(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Select one search service before asking for its key; keep other saved keys."""
    ui.step("Web search")
    ui.note("Free search needs no account. Optional services use their own API key and quota.")
    ui.note("Ciel uses only your selected search provider, even if other keys are saved.")
    while True:
        current = configuration.get("INTERNET_PROVIDER", "ddgs")
        choices = [
            (value, f"{name} · key configured" if key and configuration.get(key) else name)
            for value, name, key in _SEARCH_PROVIDERS
        ]
        selected = ui.choose(
            "Search provider", choices,
            default=current if current in {value for value, _label in choices} else "ddgs",
        )
        _value, name, key = next(row for row in _SEARCH_PROVIDERS if row[0] == selected)
        if key is None:
            configuration["INTERNET_PROVIDER"] = selected
            return
        status = "configured; Enter to keep" if configuration.get(key, "").strip() else "required"
        answer = getpass(f"{name} API key [{status}]: ").strip()
        if answer:
            configuration[key] = answer
        if configuration.get(key, "").strip():
            configuration["INTERNET_PROVIDER"] = selected
            return
        ui.note(f"{name} needs an API key. Enter a key or choose Free search.")


def _search_label(configuration: dict[str, str]) -> str:
    """Describe the selected search service without exposing credentials."""
    selected = configuration.get("INTERNET_PROVIDER", "ddgs")
    row = next((row for row in _SEARCH_PROVIDERS if row[0] == selected), None)
    if row is None:
        return f"{selected} · unsupported; choose a provider in setup"
    _value, name, key = row
    if key is None:
        return name
    return f"{name} · " + (
        "key configured; connection untested" if configuration.get(key, "").strip()
        else "key needed"
    )


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


def _personalize(ui: ConsoleUI, configuration: dict[str, str]) -> None:
    """Ask about the user while retaining explicit values until they choose a change."""
    ui.step("About you")
    ui.note("Press Enter to keep each answer. Type - to clear a name or city.")
    for key, question, fallback, limit in (
        ("CIEL_PREFERRED_NAME", "What should Ciel call you?", "computer username", 100),
        ("INTERNET_HOME_CITY", "What city and country should Ciel use for local searches?",
         "ask when needed", 200),
    ):
        current = configuration.get(key, "")
        while True:
            answer = ui.prompt(f"{question} [{current or fallback}]").strip()
            if not answer:
                break
            if answer == "-":
                configuration[key] = ""
                break
            if len(answer) > limit:
                ui.note(f"Please use {limit} characters or fewer.")
                continue
            if not answer.isprintable():
                ui.note("Please enter text without control characters.")
                continue
            configuration[key] = answer
            break
    ui.note("Memory choices affect saved facts. Conversation history is kept separately.")
    choices = [
        ("confirm", "Ask before saving supported personal details"),
        ("explicit", "Save only when I say remember"),
        ("automatic", "Save supported clear personal statements automatically"),
    ]
    current_mode = configuration.get("MEMORY_LEARNING_MODE", "confirm")
    learning = ui.choose(
        "Remembering things", choices,
        default=current_mode if current_mode in {value for value, _label in choices} else "confirm",
    )
    if learning != current_mode:
        configuration["MEMORY_LEARNING_MODE"] = learning


def _configure_persona(
    ui: ConsoleUI, configuration: dict[str, str], pending: str | None,
    original_names: dict[str, str] | None = None,
) -> str | None:
    """Offer an interview without changing the existing profile until setup is saved."""
    choices = [
        ("keep", "Keep the reviewed draft" if pending is not None else "Keep current personality"),
        ("interview", "Build a custom personality with AI"),
    ]
    if pending is not None:
        choices.append(("discard", "Discard the draft and keep the saved personality"))
    selected = ui.choose("Personality", choices, default="keep")
    if selected == "discard":
        return None
    if selected == "interview":
        from ciel.setup_persona import run_persona_interview

        names = original_names or {}
        draft = run_persona_interview(ui, {
            names.get(key, key): value for key, value in configuration.items()
        })
        if draft is None and pending is not None:
            ui.note("Your previously reviewed draft is still selected. "
                    "Choose Discard the draft to keep the saved personality instead.")
        return draft if draft is not None else pending
    return pending


def _customize(
    ui: ConsoleUI, configuration: dict[str, str], pending_persona: str | None = None,
    original_names: dict[str, str] | None = None,
) -> str | None:
    """Walk through one decision at a time while retaining existing choices on Enter."""
    _personalize(ui, configuration)
    _configure_keys(ui, configuration)
    _configure_search(ui, configuration)
    configure_chat(ui, configuration)
    pending_persona = _configure_persona(ui, configuration, pending_persona, original_names)
    ui.step("Voice")
    ui.note("Pocket runs on CPU; Turbo requires its separate model and private reference.")
    ui.note("Pocket includes CPU PyTorch. Amy needs a smaller voice download.")
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
    return pending_persona


def _review_configuration(
    ui: ConsoleUI, configuration: dict[str, str], pending_persona: str | None = None,
) -> None:
    """Show the choices that will be saved without claiming an untested connection."""
    providers = [name for _value, name, key in _API_PROVIDERS if configuration.get(key)]
    ui.step("Review your setup")
    ui.summary([
        ("Call you", configuration.get("CIEL_PREFERRED_NAME") or "Computer username"),
        ("Default city", configuration.get("INTERNET_HOME_CITY") or "Ask when needed"),
        ("Memory learning", {
            "confirm": "Ask before saving", "explicit": "Remember commands only",
            "automatic": "Save supported statements automatically",
        }.get(configuration.get("MEMORY_LEARNING_MODE", "confirm"), "Ask before saving")),
        ("Personality", "Reviewed draft; saved after confirmation" if pending_persona is not None
         else "Keep current personality"),
        ("AI connection", f"{', '.join(providers)} · key configured" if providers
         else "No cloud key; local tools available"),
        ("Web search", _search_label(configuration)),
        ("Voice", _voice_label(configuration) if _enabled(configuration, "TTS_ENABLED")
         else "Voice output off"),
        ("Recognition", f"{configuration['STT_MODEL']} · {configuration['STT_DEVICE'].upper()}"),
        ("Listening", "Ambient conversation" if _enabled(configuration, "AMBIENT_LISTENING")
         else "Wake word only"),
        ("Live captions", "On" if _enabled(configuration, "SHOW_AI_TRANSCRIPTS") else "Off"),
        ("Microphone", configuration.get("AUDIO_INPUT_DEVICE") or "System default"),
        ("Speakers", configuration.get("AUDIO_OUTPUT_DEVICE") or "System default"),
        *chat_review_rows(configuration),
    ])
    if providers and pending_persona is not None:
        ui.note("The personality interview received an AI response. "
                "Other provider connections were not tested.")
    elif providers:
        ui.note("The AI connection hasn't been tested yet.")
    if pending_persona is not None:
        ui.note("Personality draft:")
        ui.note(pending_persona)


def _configure_acceleration(
    ui: ConsoleUI, configuration: dict[str, str],
) -> None:
    """Let the user choose GPU downloads explicitly while preserving saved STT settings."""
    ui.step("Speech recognition")
    ui.note("CPU needs a smaller download. NVIDIA acceleration needs large GPU libraries.")
    saved_device = configuration["STT_DEVICE"]
    choices = [("cpu", "CPU"), ("cuda", "NVIDIA GPU")]
    default = saved_device
    if default not in {"cpu", "cuda"}:
        choices.insert(0, ("keep", f"Keep current ({saved_device})"))
        default = "keep"
    selected = ui.choose("Run speech recognition on", choices, default=default)
    if selected != "keep" and selected != saved_device:
        configuration["STT_DEVICE"] = selected
        configuration["STT_COMPUTE_TYPE"] = "int8" if selected == "cpu" else "int8_float16"


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


def run_setup_wizard(*, install_components: bool = False) -> bool:
    """Review configuration before saving, then let the user verify voice playback."""
    ui = ConsoleUI()
    saved = False
    saving = False
    try:
        ui.banner("Let's make CIEL yours.")
        ui.note("Review everything before saving. Press Ctrl+C to leave setup.")
        if install_components:
            ui.note("Choose first. Selected components download only after you save.")
        env_path = Path(".env")
        original_env = dotenv_values(env_path, interpolate=False) if env_path.is_file() else {}
        original_names = {key.upper(): key for key in original_env}
        configuration = _initial_configuration({
            key.upper(): value for key, value in original_env.items() if value is not None
        })
        if install_components and "STT_DEVICE" not in original_names:
            configuration["STT_DEVICE"] = "cpu"
            configuration["STT_COMPUTE_TYPE"] = "int8"
        mode = ui.choose("Get started", [
            ("quick", "Quick start"), ("customize", "Customize"),
        ], default="quick")
        pending_persona = None
        if mode == "customize":
            pending_persona = _customize(ui, configuration, original_names=original_names)
        else:
            _personalize(ui, configuration)
            _configure_keys(ui, configuration)
            _configure_search(ui, configuration)
            configure_chat(ui, configuration)
            pending_persona = _configure_persona(ui, configuration, pending_persona, original_names)
        if install_components:
            _configure_acceleration(ui, configuration)
        while True:
            _review_configuration(ui, configuration, pending_persona)
            if install_components:
                ui.note("Saving installs the selected speech recognition and voice components. "
                        "Existing downloads are reused. "
                        "Turbo's private assets are checked locally.")
                if configuration["TTS_ENGINE"] == "pocket":
                    ui.note("The Pocket voice includes CPU PyTorch. "
                            "Choose Customize and Amy for a smaller voice download.")
            action = ui.choose("Ready to save?", [
                ("save", "Save, install, and test" if install_components else "Save and test"),
                ("customize", "Customize"), ("cancel", "Cancel"),
            ], default="save")
            if action == "cancel":
                ui.finish("Setup cancelled. No changes were saved.")
                return False
            if action == "save":
                break
            pending_persona = _customize(ui, configuration, pending_persona, original_names)
            if install_components:
                _configure_acceleration(ui, configuration)
        saving = True
        if pending_persona is not None:
            configuration.setdefault("CIEL_PERSONA_FILE", "persona.txt")
        _save_configuration(env_path, configuration)
        saved = True
        saving = False
        chat_ready = apply_chat_autostart(ui, configuration, project_dir=env_path.parent.resolve())
        if pending_persona is not None:
            from ciel.persona import update_setup_persona_file

            try:
                effective = resolve_setup_values({
                    original_names.get(key, key): value for key, value in configuration.items()
                })
                configured_persona = {key.upper(): value for key, value in effective.items()}.get(
                    "CIEL_PERSONA_FILE", "persona.txt",
                )
                persona_path = Path(
                    configured_persona if configured_persona.strip() else "persona.txt",
                ).expanduser()
                persona_saved = (
                    persona_path.resolve() != env_path.resolve()
                    and update_setup_persona_file(str(persona_path), pending_persona)
                )
            except (OSError, ValueError, RuntimeError):
                persona_saved = False
            if not persona_saved:
                ui.note("Settings were saved, but the custom personality could not be saved. "
                        "Your existing personality file was kept. Rerun setup to try again.")
            elif configured_persona.strip():
                ui.note("Custom personality saved. Your own file notes and style edits are kept.")
            else:
                activated_configuration = {
                    **configuration, "CIEL_PERSONA_FILE": str(persona_path),
                }
                try:
                    _save_configuration(env_path, activated_configuration)
                except (OSError, UnicodeError):
                    ui.note("The custom personality file was saved, "
                            "but it could not be enabled. Your previous setting was kept. "
                            "Rerun setup to enable it.")
                else:
                    configuration = activated_configuration
                    ui.note("Custom personality saved. "
                            "Your own file notes and style edits are kept.")
        if install_components:
            from ciel.installation import prepare_components

            ui.step("Prepare your selected components")
            try:
                prepare_components(get_settings().audio, ui.note)
            except Exception as error:
                ui.note(f"Preparation did not finish ({type(error).__name__}). "
                        "Your settings are saved. Check the download output and available disk "
                        "space, then run ciel setup again to retry or change your choices.")
                ui.finish("Setup needs another step before CIEL can start.")
                return False
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
        ui.finish("Configuration saved. Run: ciel" if chat_ready else
                  "Configuration saved. Automatic chat needs attention; "
                  "Ciel will retry when opened. Run: ciel")
        return chat_ready
    except (KeyboardInterrupt, EOFError):
        if saved:
            ui.finish("Setup stopped. Configuration is saved. "
                      + ("Rerun ciel setup to finish preparation." if install_components
                         else "Run: ciel"))
        elif saving:
            ui.finish("Setup stopped while saving. Run setup again to review .env.")
        else:
            ui.finish("Setup cancelled. No changes were saved.")
        return False
