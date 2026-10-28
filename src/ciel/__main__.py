"""Entry point for running CIEL via `python -m ciel`."""

import argparse
import sys
import time
from pathlib import Path

from ciel import __version__

TOOLKIT_COMMANDS = {"status", "doctor", "devices", "config"}
COMMANDS = [
    "run", "start", "listen", "web", "setup", "record-samples", "train-wake", "memory",
    "status", "doctor", "devices", "config", "chat", "help",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build launch help without importing audio or model dependencies."""
    parser = argparse.ArgumentParser(
        description="CIEL Desktop AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick commands:\n"
            "  ciel start [dev]    Start voice conversation\n"
            "  ciel web            Open the desktop web interface\n"
            "  ciel setup          Configure voices, providers, and audio\n"
            "  ciel status         Show the configured setup\n"
            "  ciel doctor         Check local prerequisites and suggest fixes\n"
            "  ciel devices        List microphones and speakers\n"
            "  ciel config         Show effective settings with API keys hidden\n"
            "  ciel memory --help  Inspect and manage saved memory\n"
            "  ciel chat --help    Configure automatically started messaging\n"
            "  ciel help COMMAND   Show help for a command"
        ),
    )
    parser.add_argument("--version", action="version", version=f"CIEL {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=COMMANDS,
        help="Command to run",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["dev"],
        help="Use verbose debug logging (for example: ciel start dev)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Start the continuous wake-word listener loop with STT transcription",
    )
    parser.add_argument(
        "--ambient",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Continuously transcribe speech and reply only when clearly addressed",
    )
    parser.add_argument(
        "--show-transcripts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log raw STT candidates, including rejected ambient speech, for troubleshooting",
    )
    parser.add_argument(
        "--show-ai-transcripts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reveal AI captions as the voice speaks",
    )
    parser.add_argument(
        "--text-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept keyboard messages and /mute, /stop, /exit (default: on in a terminal)",
    )
    parser.add_argument(
        "--web-port", type=int, default=8765,
        help="PC-local web interface port (default: 8765)",
    )
    parser.add_argument(
        "--open-browser", action=argparse.BooleanOptionalAction, default=True,
        help="Open the browser when starting 'ciel web'",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of voice samples to record for training (default: 8)",
    )
    parser.add_argument(
        "--phrase",
        type=str,
        default="Hey Ciel",
        help="Target wake phrase to train (default: 'Hey Ciel')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch local tools or initialize and launch the desktop assistant."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    parser = _build_parser()
    if arguments[:1] == ["help"]:
        if len(arguments) > 2:
            parser.error("usage: ciel help [command]")
        if len(arguments) == 1:
            parser.print_help()
            return 0
        if arguments[1] not in COMMANDS:
            parser.error(f"unknown command: {arguments[1]}")
        return main([arguments[1], "--help"]) if arguments[1] != "help" else main(["--help"])
    if arguments[:1] == ["memory"]:
        from ciel.memory.cli import main as memory_main

        return memory_main(arguments[1:])
    if arguments[:1] == ["chat"]:
        from ciel.chat.cli import main as chat_main

        return chat_main(arguments[1:])
    if arguments and arguments[0] in TOOLKIT_COMMANDS:
        from ciel.cli import main as toolkit_main

        return toolkit_main(arguments)
    args = parser.parse_args(arguments)
    if args.command in TOOLKIT_COMMANDS | {"memory", "chat", "help"}:
        parser.error(f"use 'ciel {args.command}' without launch options before the command")
    if not 1 <= args.web_port <= 65535:
        parser.error("--web-port must be between 1 and 65535")
    if args.mode == "dev" and args.command not in {"start", "listen", "web"}:
        parser.error("the 'dev' mode can only be used with 'start', 'listen', or 'web'")

    # If user ran `python -m ciel setup`
    if args.command == "setup":
        from ciel.setup_wizard import run_setup_wizard

        return 0 if run_setup_wizard(install_components=True) else 1

    # If user ran `python -m ciel record-samples`
    if args.command == "record-samples":
        from ciel.audio.trainer import record_voice_samples

        record_voice_samples(count=args.count, phrase=args.phrase)
        return 0

    # If user ran `python -m ciel train-wake`
    if args.command == "train-wake":
        from ciel.audio.trainer import train_custom_wakeword

        try:
            train_custom_wakeword()
        except Exception as err:
            print(f"\n❌ Training failed: {err}")
            return 1
        return 0

    from ciel.config import get_settings
    from ciel.logging import setup_logging

    settings = get_settings()
    from ciel.chat.autostart import reconcile_chat_startup

    chat_startup = reconcile_chat_startup(settings.chat, project_dir=Path.cwd())

    from ciel.platform import get_audio_backend
    from ciel.platform.base import AudioInputError

    dev_mode = args.mode == "dev"
    logger = setup_logging("DEBUG" if dev_mode else "INFO", concise=not dev_mode)

    if not dev_mode:
        from ciel.terminal_ui import ConsoleUI

        console = ConsoleUI()
        if args.command != "web":
            console.clear()
        console.banner("Your voice. Your companion.", version=__version__)
        console.note("Preparing voice and recognition. First startup can take a moment.")

    if not chat_startup.success:
        print(chat_startup.message)

    logger.info("==========================================")
    logger.info("   CIEL - Desktop AI Assistant v%s  ", __version__)
    logger.info("==========================================")
    logger.info(
        "Environment: %s | Log Level: %s",
        settings.app.env,
        "DEBUG" if dev_mode else "INFO",
    )

    # Initialize and report audio backend status
    audio_backend = get_audio_backend()
    default_input = audio_backend.get_default_input_device()
    default_output = audio_backend.get_default_output_device()
    input_name = default_input.name if default_input else "None found"
    output_name = default_output.name if default_output else "None found"

    logger.info(
        "Audio Backend: Input='%s' | Output='%s' | Rate=%d Hz",
        input_name,
        output_name,
        settings.audio.sample_rate,
    )
    logger.info(
        "Wake Word: '%s' | STT Model: '%s' (Device: %s)",
        settings.audio.wake_word,
        settings.audio.stt_model,
        settings.audio.stt_device,
    )

    # Report provider configuration status safely (boolean status only, no keys printed)
    nim_status = "configured" if settings.providers.nim_api_key else "not set"
    openrouter_status = "configured" if settings.providers.openrouter_api_key else "not set"
    groq_status = "configured" if settings.providers.groq_api_key else "not set"
    logger.info(
        "AI Providers: NIM (%s), OpenRouter (%s), Groq (%s), Ollama (%s)",
        nim_status,
        openrouter_status,
        groq_status,
        settings.providers.ollama_host,
    )

    from ciel.runtime import run_desktop

    try:
        return run_desktop(args, settings, logger, audio_backend, clock=time)
    except AudioInputError:
        logger.debug("Microphone startup failed", exc_info=True)
        print(
            "Ciel couldn't open the microphone. Connect or enable an input device in your "
            "system sound settings, then try again.\n"
            "Use 'ciel devices --input' to list microphones or 'ciel setup' to select one.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[CIEL] Shutting down cleanly...")
        sys.exit(0)
