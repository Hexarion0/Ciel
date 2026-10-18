"""Entry point for running CIEL via `python -m ciel`."""

import argparse
import sys
import time

from ciel import __version__


def main() -> int:
    """Initialize core settings, start logging, and launch CIEL."""
    if sys.argv[1:2] == ["memory"]:
        from ciel.memory.cli import main as memory_main

        return memory_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description="CIEL Desktop AI Assistant")
    parser.add_argument("--version", action="version", version=f"CIEL {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "start",
            "listen",
            "web",
            "setup",
            "record-samples",
            "train-wake",
            "memory",
        ],
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
    args = parser.parse_args()
    if not 1 <= args.web_port <= 65535:
        parser.error("--web-port must be between 1 and 65535")
    if args.mode == "dev" and args.command not in {"start", "listen", "web"}:
        parser.error("the 'dev' mode can only be used with 'start' or 'listen'")

    # If user ran `python -m ciel setup`
    if args.command == "setup":
        from ciel.setup_wizard import run_setup_wizard

        run_setup_wizard()
        return 0

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
    from ciel.platform import get_audio_backend

    settings = get_settings()
    dev_mode = args.mode == "dev"
    logger = setup_logging("DEBUG" if dev_mode else "INFO", concise=not dev_mode)

    if not dev_mode:
        from ciel.terminal_ui import ConsoleUI

        console = ConsoleUI()
        console.banner("Your voice. Your companion.", version=__version__)
        console.note("Preparing voice and recognition. First startup can take a moment.")

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

    return run_desktop(args, settings, logger, audio_backend, clock=time)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[CIEL] Shutting down cleanly...")
        sys.exit(0)
