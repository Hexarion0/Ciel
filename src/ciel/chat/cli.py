"""Configure and pair automatically managed messaging without desktop audio."""

import argparse
import signal
import sqlite3
import sys
import threading
from pathlib import Path

from ciel.chat.state import ChatAlreadyRunning, ChatState, polling_lock
from ciel.chat.telegram import TelegramClient, TelegramError, TelegramService
from ciel.config import Settings, get_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ciel chat", description="Ciel messaging starts automatically after setup.",
        epilog=(
            "Start with ciel chat setup, then ciel chat pair telegram. "
            "Chat starts automatically now and at login; keep this PC on and connected. "
            "Requests older than five minutes are skipped to avoid executing stale commands."
        ),
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("setup", help="Choose messaging services and configure their credentials")
    for name, description in (
        ("pair", "Pair your Telegram account using an expiring local code"),
        ("status", "Show local configuration, automatic startup, and pairing status"),
        ("unpair", "Remove local owner bindings; automatically pause chat while changing them"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("service", nargs="?", choices=["telegram"], default="telegram")
    return parser


def _configured(settings: Settings) -> bool:
    configuration = settings.chat
    return bool("telegram" in configuration.services and configuration.telegram_bot_token
                and configuration.telegram_bot_token.get_secret_value().strip())


def _status(settings: Settings) -> int:
    """Inspect configuration and local service state without contacting Telegram."""
    from ciel.chat.autostart import autostart_status

    configuration = settings.chat
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
    print("Chat startup: " + autostart_status())
    print("This PC must be on and connected. Requests older than five minutes are skipped.")
    return 0


def _serve(settings: Settings, stop: threading.Event, *, pair: bool = False) -> int:
    """Own polling exclusively; release the lock before the automatic worker resumes."""
    configuration = settings.chat
    state: ChatState | None = None
    client: TelegramClient | None = None
    session = None
    try:
        with polling_lock(configuration.state_path):
            state = ChatState(configuration.state_path)
            if not pair and not state.bindings():
                # An unpaired worker must not consume /start codes or repeatedly contact Telegram.
                stop.wait(5)
                return 0
            client = TelegramClient(configuration.telegram_bot_token.get_secret_value())
            attach_stop = getattr(client, "set_stop_event", None)
            if callable(attach_stop):
                attach_stop(stop)
            identity = client.identity()
            client.check_polling()
            bot_id = identity["id"]
            if not pair and state.owner(bot_id) is None:
                stop.wait(5)
                return 0

            def reply(text: str, cancel: threading.Event) -> str:
                nonlocal session
                if session is None:
                    from ciel.chat.session import ChatSession
                    from ciel.memory.bridge import telegram_conversation_id

                    owner = state.owner(bot_id)
                    if owner is None:
                        return ""
                    session = ChatSession(settings, channel="telegram",
                                          session_id=telegram_conversation_id(bot_id, owner))
                return session.reply(text, cancel_event=cancel)

            service = TelegramService(client, state, bot_id, reply, stop=stop)
            if pair:
                def show_code(code: str) -> None:
                    username = identity.get("username", "your bot")
                    print(f"Open @{username} in Telegram, then send this within five minutes:")
                    print(f"/start {code}")
                    print("Keep this code private; it pairs the sender as Ciel's owner.")

                owner = service.pair(show_code)
                if owner is None:
                    print("Pairing stopped or expired. Run ciel chat pair telegram to try again.")
                    return 0 if stop.is_set() else 1
                print("Telegram paired. Chat will resume automatically.")
                return 0
            service.run()
            return 0
    finally:
        if session is not None:
            session.close()
        if client is not None:
            client.close()
        if state is not None:
            state.close()


def _worker(settings: Settings, stop: threading.Event) -> int:
    """Run only as the managed service, retrying transient connection failures indefinitely."""
    retry = 1
    while not stop.is_set():
        try:
            _serve(settings, stop)
            retry = 1
        except TelegramError as error:
            if stop.is_set():
                return 0
            print(str(error), flush=True)
            if not error.transient:
                return 1
            print("Automatic chat will retry the connection.", flush=True)
            if stop.wait(error.retry_after or min(2 ** retry, 30)):
                break
            retry += 1
        except ChatAlreadyRunning:
            print("A chat poller already owns this configuration; retrying automatically.",
                  flush=True)
            if stop.wait(5):
                break
        except (OSError, sqlite3.Error):
            print("Automatic chat could not open its state; check CHAT_STATE_PATH and permissions.",
                  flush=True)
            return 1
    return 0


def _change_pairing(command: str, settings: Settings, stop: threading.Event) -> int:
    """Temporarily stop the worker, modify the owner, then resume automatic operation."""
    from ciel.chat.autostart import pause_for_pairing, reconcile_chat_startup

    result = 1
    paused = False
    try:
        with pause_for_pairing() as operation:
            if not operation.success:
                print(operation.message)
                return 1
            paused = True
            if command == "unpair":
                with polling_lock(settings.chat.state_path):
                    state = ChatState(settings.chat.state_path)
                    try:
                        state.unpair()
                    finally:
                        state.close()
                print("Local Telegram pairing removed. Chat will wait for a new pairing.")
                result = 0
            else:
                result = _serve(settings, stop, pair=True)
    except (TelegramError, ChatAlreadyRunning) as error:
        if stop.is_set():
            result = 0
        else:
            print(str(error))
    except (OSError, sqlite3.Error):
        print("Ciel could not open its chat state. Check CHAT_STATE_PATH and local permissions.")
    except KeyboardInterrupt:
        stop.set()
        result = 0
    finally:
        # Both the polling lock and pairing guard must be released before systemd can start.
        if paused:
            # Another setup can save new selections while pairing holds the service guard.
            # Resume from the saved configuration, rather than stale tokens or service choices.
            try:
                clear_cache = getattr(get_settings, "cache_clear", None)
                if clear_cache is not None:
                    clear_cache()
                resumed = reconcile_chat_startup(get_settings().chat, project_dir=Path.cwd())
                if not resumed.success:
                    print(resumed.message)
                    result = 1
            except (OSError, ValueError):
                print("Pairing finished, but the current chat settings could not be loaded. "
                      "Review ciel chat setup so automatic startup can resume.")
                result = 1
    return result


def main(argv: list[str] | None = None) -> int:
    """Configure chat publicly; reserve the hidden worker entry for the automatic service."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    worker = arguments == ["_worker"]
    parser = _parser()
    args = None if worker else parser.parse_args(arguments)
    if not worker and args.command is None:
        parser.print_help()
        return 0
    if not worker and args.command == "setup":
        from ciel.chat.setup import run_chat_setup

        return 0 if run_chat_setup() else 1
    settings = get_settings()
    if not worker and args.command == "status":
        return _status(settings)
    if (worker or args.command != "unpair") and not _configured(settings):
        print("Select Telegram and set its bot token first: ciel chat setup")
        return 1
    stop = threading.Event()
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        return _worker(settings, stop) if worker else _change_pairing(args.command, settings, stop)
    except KeyboardInterrupt:
        stop.set()
        return 0
    finally:
        stop.set()
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
