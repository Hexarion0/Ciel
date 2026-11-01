"""Explicit private-network HTTPS configuration; never modifies OS trust settings."""

import hashlib
import ipaddress
import os
import ssl
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


def server_tls(
    host: str, certificate: str | None, key: str | None, *, insecure_lan: bool = False,
) -> ssl.SSLContext | None:
    """Require HTTPS unless plain private-network access is explicitly enabled."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("Use the PC local IP address for --web-host.") from error
    if address.is_loopback and not certificate and not key:
        return None
    if not address.is_private or address.is_unspecified or address.is_multicast:
        raise ValueError("Choose a specific private Wi-Fi IP address for --web-host.")
    if insecure_lan:
        return None
    if not certificate or not key:
        raise ValueError("Phone access needs --web-cert and --web-key for local HTTPS.")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate, key)
    return context


_SETUP_LOCK = threading.Lock()


@dataclass(frozen=True)
class LocalTLS:
    """Public trust material and private server key locations."""

    certificate: Path
    key: Path
    ca_certificate: Path
    fingerprint: str


def prepare_local_tls(host: str, directory: Path) -> LocalTLS:
    """Create a private local CA and a matching server leaf; never install trust."""
    address = ipaddress.ip_address(host)
    if not address.is_private or address.is_unspecified or address.is_multicast:
        raise ValueError("Choose a private local network address.")
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    ca, ca_key = directory / "ciel-ca.crt", directory / "ciel-ca-key.pem"
    generation = uuid4().hex
    certificate = directory / f"server-{generation}.pem"
    key = directory / f"server-{generation}-key.pem"

    def openssl(*arguments: str) -> None:
        try:
            subprocess.run(["openssl", *arguments], check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError(
                "Local HTTPS setup failed. Check that OpenSSL is installed."
            ) from error

    with _SETUP_LOCK:
        if ca.is_file() != ca_key.is_file():
            raise ValueError(
                "The local certificate authority is incomplete. Restore its saved files."
            )
        if not ca.is_file():
            openssl(
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "3650",
                "-keyout",
                str(ca_key),
                "-out",
                str(ca),
                "-subj",
                "/CN=Ciel local device CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext",
                "keyUsage=critical,keyCertSign,cRLSign",
            )
        ca_key.chmod(0o600)
        # A private directory and process umask protect key material while OpenSSL writes it.
        old_mask = os.umask(0o077)
        try:
            csr = directory / "server.csr"
            extensions = directory / "server.ext"
            extensions.write_text(
                "basicConstraints=critical,CA:FALSE\n"
                "keyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth\n"
                f"subjectAltName=IP:{address},IP:127.0.0.1,DNS:localhost\n",
            )
            openssl(
                "req",
                "-new",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(csr),
                "-subj",
                "/CN=Ciel local web",
            )
            openssl(
                "x509",
                "-req",
                "-in",
                str(csr),
                "-CA",
                str(ca),
                "-CAkey",
                str(ca_key),
                "-CAcreateserial",
                "-out",
                str(certificate),
                "-days",
                "365",
                "-sha256",
                "-extfile",
                str(extensions),
            )
            key.chmod(0o600)
            csr.unlink(missing_ok=True)
            extensions.unlink(missing_ok=True)
        finally:
            os.umask(old_mask)
        fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(ca.read_text())).hexdigest()
        return LocalTLS(certificate, key, ca, fingerprint)
