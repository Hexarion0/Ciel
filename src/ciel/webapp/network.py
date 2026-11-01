"""Private Wi-Fi connection setup and public certificate material for owned devices."""

import hashlib
import ipaddress
import json
import ssl
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from ciel.webapp.preferences import save_preferences
from ciel.webapp.tls import prepare_local_tls


def local_addresses() -> list[str]:
    """Discover private interface addresses without contacting an external server."""
    try:
        result = subprocess.run(
            ["ip", "-j", "-4", "address", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return sorted(
            {
                entry["local"]
                for device in json.loads(result.stdout)
                for entry in device.get("addr_info", [])
                if ipaddress.ip_address(entry["local"]).is_private
                and not ipaddress.ip_address(entry["local"]).is_loopback
            }
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return []


class LocalConnection:
    """Keep local trust setup explicit; only the public CA can be downloaded."""

    def __init__(self, settings, ui, directory: Path, env_path: Path, port: int) -> None:
        self.settings, self.ui = settings, ui
        self.directory, self.env_path, self.port = directory, env_path, port
        self.setup_server = None

    def snapshot(self) -> dict:
        """Return connection instructions, with no private key or private filesystem paths."""
        addresses = local_addresses()
        host = (
            self.settings.web_host
            if self.settings.web_host != "127.0.0.1"
            else (addresses[0] if addresses else "")
        )
        if self.ui.phone_url:
            host = urlsplit(self.ui.phone_url).hostname or host
        plain = getattr(self.ui, "insecure_lan", False)
        ca = self.directory / "ciel-ca.crt"
        fingerprint = (
            hashlib.sha256(ssl.PEM_cert_to_DER_cert(ca.read_text())).hexdigest()
            if ca.is_file() and not plain
            else ""
        )
        return {
            "addresses": addresses,
            "host": host,
            "enabled": bool(self.ui.phone_url),
            "insecure_lan": plain,
            "url": self.ui.phone_url,
            "fingerprint": fingerprint,
            "certificate_available": ca.is_file() and not plain,
            "setup_url": self.setup_server.url if self.setup_server else "",
        }

    def enable(self, host: str) -> dict:
        """Create local trust material, bind the private interface, then save the setup."""
        address = ipaddress.ip_address(host)
        if (
            address.version != 4
            or not address.is_private
            or address.is_loopback
            or address.is_unspecified
            or address.is_multicast
        ):
            raise ValueError("Choose the PC private Wi-Fi or Ethernet address.")
        bundle = prepare_local_tls(str(address), self.directory)
        self.ui.enable_lan(str(address), bundle.certificate, bundle.key)
        changes = {
            "web_host": str(address),
            "web_cert": str(bundle.certificate),
            "web_key": str(bundle.key),
        }
        save_preferences(self.env_path, {key.upper(): value for key, value in changes.items()})
        for name, value in changes.items():
            setattr(self.settings, name, value)
        self.start_public_setup()
        return self.snapshot()

    def start_public_setup(self) -> None:
        if (self.setup_server is not None or not self.ui.phone_url
                or getattr(self.ui, "insecure_lan", False)):
            return
        status = self.snapshot()
        if status["certificate_available"]:
            setup_port = self.port + 1 if self.port < 65535 else self.port - 1
            try:
                self.setup_server = AndroidSetupServer(
                    status["host"],
                    setup_port,
                    self.directory / "ciel-ca.crt",
                    self.ui.phone_url,
                    status["fingerprint"],
                )
            except OSError:
                pass  # Secure access and the PC certificate download remain available.

    def close(self) -> None:
        if self.setup_server is not None:
            self.setup_server.close()

    def certificate(self) -> bytes:
        """Return only the public CA certificate; private keys never have HTTP routes."""
        return (self.directory / "ciel-ca.crt").read_bytes()


class AndroidSetupServer:
    """Serve public setup instructions and the CA only, never the assistant APIs."""

    def __init__(self, host: str, port: int, ca: Path, secure_url: str, fingerprint: str) -> None:
        import html
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        document = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>Connect to Ciel</title><style>body{color:#f1ede8;background:#1c1b20;"
            "font:16px/1.7 system-ui;max-width:620px;padding:28px;margin:auto}"
            "a{color:#b8a0d9}li{margin:18px 0}code{overflow-wrap:anywhere}</style>"
            "<h1>Bring Ciel to your phone.</h1><ol><li>Download the "
            '<a href="/ciel-ca.crt" download>local Ciel certificate</a>.</li>'
            "<li>Before installing, compare this certificate fingerprint with the one "
            "in Settings on your PC:<br><code>" + html.escape(fingerprint) + "</code></li>"
            "<li>On Android, open Settings → Security &amp; privacy → More security "
            "settings → Encryption &amp; credentials → Install a certificate → "
            "CA certificate. Select the downloaded file.</li>"
            '<li>Open <a href="' + html.escape(secure_url) + '">secure Ciel</a> '
            "and enter the pairing code shown on your PC.</li></ol>"
            "<p>This page provides setup files only. Chat and voice use the secure app.</p>"
            "</html>"
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.headers.get("Host") != f"{host}:{self.server.server_address[1]}":
                    self.send_error(403)
                    return
                if self.path == "/":
                    body, content_type = document, "text/html; charset=utf-8"
                elif self.path == "/ciel-ca.crt":
                    body, content_type = ca.read_bytes(), "application/x-x509-ca-cert"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="ciel-android-setup", daemon=True
        )
        self.thread.start()
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
