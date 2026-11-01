"""Threaded web transport with quiet handling of ordinary client disconnects."""

import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import MSG_PEEK, socket


class BrowserHTTPServer(ThreadingHTTPServer):
    """Accept raw connections so each worker can distinguish HTTP from TLS."""

    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *,
        tls_context: ssl.SSLContext | None = None,
    ) -> None:
        self.tls_context = tls_context
        super().__init__(address, handler)

    def finish_request(self, request: socket, client_address: tuple[str, int]) -> None:
        """Negotiate TLS in the worker and close the wrapped socket after its handler."""
        request.settimeout(3)
        if self.tls_context is not None and request.recv(1, MSG_PEEK) == b'\x16':
            secure = self.tls_context.wrap_socket(
                request, server_side=True, do_handshake_on_connect=False,
            )
            try:
                secure.do_handshake()
                super().finish_request(secure, client_address)
            finally:
                self.shutdown_request(secure)
        else:
            super().finish_request(request, client_address)

    def handle_error(self, request: socket, client_address: tuple[str, int]) -> None:
        """Keep network/handshake failures quiet while reporting application errors."""
        if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)
