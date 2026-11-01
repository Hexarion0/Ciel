"""Read-only command tools that do not start a desktop listening session."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ciel import __version__

if TYPE_CHECKING:
    from ciel.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    """Describe each tool independently of application dependencies."""
    parser = argparse.ArgumentParser(prog="ciel", description="CIEL setup and inspection tools")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "status": "Show configured settings; running sessions and connectivity are not checked.",
        "doctor": "Check local prerequisites without downloads, inference, or network requests.",
        "devices": "List microphones and speakers without opening an audio stream.",
        "config": "Show effective settings with API keys hidden. Use 'ciel setup' to change them.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--json", action="store_true", help="Print machine-readable JSON")
        if name == "devices":
            direction = command.add_mutually_exclusive_group()
            direction.add_argument("--input", action="store_true", help="Show microphones only")
            direction.add_argument("--output", action="store_true", help="Show speakers only")
        elif name == "config":
            command.add_argument("action", nargs="?", choices=["show"], default="show")
    return parser


def _error(message: str, *, json_output: bool) -> int:
    """Report a safe failure while preserving parseable JSON output."""
    if json_output:
        print(json.dumps({"error": message}))
    else:
        print(f"CIEL: {message}", file=sys.stderr)
    return 1


def _configuration(settings: Settings) -> dict[str, Any]:
    """Expose effective environment names, replacing every secret with its state."""
    from pydantic import SecretStr

    values = settings.model_dump(mode="json")
    configuration = {}
    for name, serialized in values.items():
        value = getattr(settings, name)
        if isinstance(value, SecretStr) or name.endswith("_api_key"):
            serialized = "configured" if value else "unconfigured"
        configuration[name.upper()] = serialized
    return configuration


def _status(settings: Settings) -> dict[str, Any]:
    """Summarize configuration without probing hardware or contacting providers."""
    from ciel.paths import runtime_data_root

    return {
        "version": __version__,
        "mode": "ambient" if settings.ambient_listening else "wake-word",
        "wake_word": settings.wake_word,
        "voice": {
            "enabled": settings.tts_enabled,
            "engine": settings.tts_engine,
            "profile": settings.tts_voice,
        },
        "recognition": {
            "model": settings.stt_model,
            "device": settings.stt_device,
            "compute_type": settings.stt_compute_type,
        },
        "captions": settings.show_ai_transcripts,
        "providers": {
            "nim": {"configured": bool(settings.nim_api_key), "model": settings.nim_model},
            "groq": {"configured": bool(settings.groq_api_key)},
            "openrouter": {"configured": bool(settings.openrouter_api_key)},
            "ollama": {"host": settings.ollama_host, "connectivity": "not checked"},
        },
        "paths": {
            "config_file": str(Path(".env").resolve()),
            "config_file_exists": Path(".env").is_file(),
            "data_root": str(runtime_data_root()),
            "memory_database": (
                ":memory:" if settings.memory_db_path == ":memory:"
                else str(Path(settings.memory_db_path).resolve())
            ),
        },
    }


def _print_status(status: dict[str, Any]) -> None:
    """Print a compact report with an explicit configuration-only scope."""
    from ciel.terminal_ui import ConsoleUI

    ui = ConsoleUI()
    voice, recognition = status["voice"], status["recognition"]
    ui.banner("Configured setup", version=status["version"])
    ui.panel([
        ("Listening", status["mode"]),
        ("Wake phrase", status["wake_word"]),
        ("Voice", f"{voice['profile']} ({voice['engine']})"),
        ("Voice output", "Enabled" if voice["enabled"] else "Disabled"),
        ("Recognition", f"{recognition['model']} on {recognition['device']} "
         f"({recognition['compute_type']})"),
        ("AI captions", "On" if status["captions"] else "Off"),
    ])
    providers = []
    for name, provider in status["providers"].items():
        if name == "ollama":
            providers.append(("Ollama", f"{provider['host']} (connectivity not checked)"))
        else:
            state = "configured" if provider["configured"] else "unconfigured"
            providers.append((name.upper(), state))
    ui.commands([("Connections", providers), ("Local files", [
        (name.replace("_", " ").capitalize(), str(value))
        for name, value in status["paths"].items() if name != "config_file_exists"
    ])])
    if not status["paths"]["config_file_exists"]:
        ui.note("No .env file here; environment settings and defaults still apply.")
    ui.note("Running sessions, models, and provider health are not checked. Try 'ciel doctor'.")


def _devices(args: argparse.Namespace) -> int:
    """Query device metadata without loading settings, models, or opening streams."""
    try:
        from ciel.device_probe import list_devices

        devices = list_devices()
    except TimeoutError:
        return _error(
            "Audio device discovery timed out. Check your desktop audio service and try again.",
            json_output=args.json,
        )
    except Exception:
        return _error(
            "Audio devices could not be listed. Check PortAudio and your desktop audio service.",
            json_output=args.json,
        )
    if args.input:
        devices = [device for device in devices if device.is_input]
    elif args.output:
        devices = [device for device in devices if device.is_output]
    if args.json:
        print(json.dumps([asdict(device) for device in devices], indent=2, ensure_ascii=False))
    elif devices:
        print("Audio devices (indexes can be selected in 'ciel setup'):")
        for device in devices:
            roles = []
            if device.is_input:
                roles.append("input" + (" default" if device.is_default_input else ""))
            if device.is_output:
                roles.append("output" + (" default" if device.is_default_output else ""))
            print(f"  {device.index}: {device.name} [{', '.join(roles)}] "
                  f"({device.hostapi}, {device.default_samplerate:g} Hz)")
    else:
        direction = "input " if args.input else "output " if args.output else ""
        print(f"No {direction}audio devices found. Check your desktop audio service.")
    return 0 if devices else 1


def main(argv: list[str] | None = None) -> int:
    """Run a local inspection command with safe errors and optional JSON output."""
    args = _build_parser().parse_args(argv)
    if args.command == "devices":
        return _devices(args)

    from pydantic import ValidationError

    from ciel.config import get_settings

    try:
        settings = get_settings()
    except (ValidationError, ValueError, OSError):
        return _error(
            "Configuration could not be loaded. Check .env and environment settings; "
            "run 'ciel setup' to reconfigure.",
            json_output=args.json,
        )
    if args.command == "status":
        status = _status(settings)
        if args.json:
            print(json.dumps(status, indent=2, ensure_ascii=False))
        else:
            _print_status(status)
        return 0
    if args.command == "config":
        configuration = _configuration(settings)
        if args.json:
            print(json.dumps(configuration, indent=2, ensure_ascii=False))
        else:
            print("Effective configuration (API keys hidden; edit with 'ciel setup'):")
            for name, value in configuration.items():
                print(f"{name}={json.dumps(value, ensure_ascii=False)}")
        return 0

    from ciel.diagnostics import run_checks

    checks = run_checks(settings)
    counts = {state: sum(check.status == state for check in checks)
              for state in ("ok", "warning", "error")}
    if args.json:
        print(json.dumps({"checks": [asdict(check) for check in checks], "summary": counts},
                         indent=2, ensure_ascii=False))
    else:
        print("CIEL doctor - local prerequisites (models and connectivity are not tested)")
        for check in checks:
            print(f"[{check.status.upper()}] {check.name}: {check.detail}")
            if check.hint:
                label = "Note" if check.status == "ok" else "Fix"
                print(f"  {label}: {check.hint}")
        print(f"{counts['ok']} passed, {counts['warning']} warnings, {counts['error']} errors")
    return 1 if counts["error"] else 0
