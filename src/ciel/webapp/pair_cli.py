"""Prepare phone access and display a short-lived code from the hosted web server."""

import argparse
import ipaddress
import re
import sys

import httpx


def private_host(value: str) -> str:
    """Accept a specific private IPv4 address for the phone-facing listener."""
    try:
        address = ipaddress.ip_address(value)
        if (address.version != 4 or not address.is_private or address.is_loopback
                or address.is_unspecified or address.is_multicast):
            raise ValueError()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Choose the PC's private network IP address.") from error
    return str(address)


def make_client(port: int) -> httpx.Client:
    """Contact only the PC-local server, independent of proxy environment settings."""
    return httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=35)


def main(argv: list[str] | None = None) -> int:
    """Create a phone pairing code without initializing audio or loading models."""
    parser = argparse.ArgumentParser(
        prog="ciel pair", description="Connect your phone to the running Ciel web server.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Web server port (default: 8765)")
    parser.add_argument("--host", type=private_host, help="PC's private IP for initial phone setup")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        with make_client(args.port) as client:
            response = client.get("/api/network")
            response.raise_for_status()
            network = response.json()
            if network.get("insecure_lan"):
                print(f"Direct home-network access is enabled: {network['url']}")
                print("Pairing is disabled in this mode.")
                return 0
            if not network.get("enabled") or args.host:
                addresses = network.get("addresses", [])
                host = args.host or (addresses[0] if len(addresses) == 1 else None)
                if not host:
                    print("Choose the PC's network address with ciel pair --host <IP>.",
                          file=sys.stderr)
                    return 1
                host = private_host(host)
                response = client.post("/api/network/setup", json={"host": host})
                response.raise_for_status()
                network = response.json()
            response = client.post("/api/pair-code", json={})
            response.raise_for_status()
            code = response.json()["code"]
            if not isinstance(code, str) or not re.fullmatch(r"\d{8}", code):
                raise ValueError("Invalid code")
            print(f"Open on your phone: {network['url']}")
            print(f"Pairing code: {code} (expires in 5 minutes; use once)")
            if network.get("setup_url"):
                print(f"First connection? Install the local certificate: {network['setup_url']}")
            fingerprint = network.get("fingerprint", "")
            if re.fullmatch(r"[a-f0-9]{64}", fingerprint):
                print(f"Certificate SHA-256: {fingerprint}")
            print("Keep ciel web running. Run ciel pair again if the code expires.")
            return 0
    except httpx.ConnectError:
        command = "ciel web" if args.port == 8765 else f"ciel web --web-port {args.port}"
        print(f"Ciel web is not reachable. Start {command}, then run ciel pair again.",
              file=sys.stderr)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        print("Pairing could not be prepared. Check the running Ciel web server and its "
              "phone connection in Settings.", file=sys.stderr)
    return 1
