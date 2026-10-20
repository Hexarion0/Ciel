"""Configure, pair, and run messaging without initializing desktop audio."""

import argparse
import signal
import sqlite3
import threading
from pathlib import Path

from ciel.chat.state import ChatAlreadyRunning, ChatState, polling_lock
from ciel.chat.telegram import TelegramClient, TelegramError, TelegramService
from ciel.config import get_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ciel chat", description="Optional Ciel messaging, independent of voice.",
        epilog=(
            "Start with ciel chat setup, then ciel chat pair telegram. "
            "The PC must stay on and connected; autostart runs chat after login. "
            "Requests older than five minutes are skipped to avoid executing stale commands."
        ),
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="Run the configured owner-only chat services")
    commands.add_parser("setup", help="Choose messaging services and configure their credentials")
    for name, description in (
        ("pair", "Pair your Telegram account using an expiring local code"),
        ("status", "Show local configuration and pairing status without connecting"),
        ("unpair", "Remove local owner bindings; stop the service first"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("service", nargs="?", choices=["telegram"], default="telegram")
    autostart = commands.add_parser("autostart", help="Manage chat startup after login on Linux")
    autostart.add_argument("action", choices=["enable", "disable", "status"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run optional channels without importing an audio backend or loading voice models."""
    args = _parser().parse_args(argv)
    if args.command == "setup":
        from ciel.chat.setup import run_chat_setup

        return 0 if run_chat_setup() else 1
    if args.command == "autostart":
        from ciel.chat.autostart import main as autostart_main

        return autostart_main(args.action)
    settings = get_settings()
    configuration = settings.chat
    if args.command == "status":
        print("Chat services: " + (", ".join(configuration.services) or "none selected"))
        token_status = "configured" if configuration.telegram_bot_token else "not set"
        print("Telegram token: " + token_status)
        if Path(configuration.state_path).is_file():
            state = ChatState(configuration.state_path)
            try:
                bindings = state.bindings()
            finally:
                state.close()
            for bot_id, owner_id in bindings:
                print(f"Local pairing: bot {bot_id}, owner {owner_id}")
            if not bindings:
                print("Local pairing: not paired")
        else:
            print("Local pairing: not paired")
        print("The PC chat service must be running. Requests older than five minutes are skipped.")
        return 0
    if args.command != "unpair" and (
        "telegram" not in configuration.services or not configuration.telegram_bot_token
        or not configuration.telegram_bot_token.get_secret_value().strip()
    ):
        print("Select Telegram and set its bot token first: ciel chat setup")
        return 1
    stop = threading.Event()
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(signum, lambda *_: stop.set())
    state: ChatState | None = None
    client: TelegramClient | None = None
    session = None
    try:
        with polling_lock(configuration.state_path):
            state = ChatState(configuration.state_path)
            if args.command == "unpair":
                state.unpair()
                print("Local Telegram pairing removed. Pair again before starting chat.")
                return 0
            client = TelegramClient(configuration.telegram_bot_token.get_secret_value())
            identity = client.identity()
            client.check_polling()
            bot_id = identity["id"]

            def reply(text: str, cancel: threading.Event) -> str:
                nonlocal session
                if session is None:
                    from ciel.chat.session import ChatSession
                    from ciel.persona import PERSONA_CONTEXT_VERSION

                    owner = state.owner(bot_id)
                    session = ChatSession(
                        settings, channel="telegram",
                        session_id=f"telegram:{bot_id}:{owner}:{PERSONA_CONTEXT_VERSION}",
                    )
                return session.reply(text, cancel_event=cancel)

            service = TelegramService(client, state, bot_id, reply, stop=stop)
            if args.command == "pair":
                def show_code(code: str) -> None:
                    username = identity.get("username", "your bot")
                    print(f"Open @{username} in Telegram, then send this within five minutes:")
                    print(f"/start {code}")
                    print("Keep this code private; it pairs the sender as Ciel's owner.")

                owner = service.pair(show_code)
                if owner is None:
                    print("Pairing stopped or expired. Run ciel chat pair telegram to try again.")
                    return 0 if stop.is_set() else 1
                print("Telegram paired. Start chatting with ciel chat run.")
                return 0
            print("Ciel Telegram chat is running. Press Ctrl-C to stop.")
            print("Requests older than five minutes are skipped. "
                  "Interrupted requests are not replayed.")
            service.run()
            return 0
    except (TelegramError, ChatAlreadyRunning) as error:
        print(str(error))
        return 1
    except (OSError, sqlite3.Error):
        print("Ciel could not open its chat state. Check CHAT_STATE_PATH and local permissions.")
        return 1
    except KeyboardInterrupt:
        stop.set()
        return 0
    finally:
        stop.set()
        if session is not None:
            session.close()
        if client is not None:
            client.close()
        if state is not None:
            state.close()
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
