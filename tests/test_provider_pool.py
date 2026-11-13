"""Provider calls share keep-alive connections, and cancellation still ends a blocked wait."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ciel.providers.streaming import pooled_client, streaming_client


@pytest.fixture
def server():
    connections, release = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            connections.append(self.client_address)

        def do_GET(self):
            if self.path == "/slow":
                release.wait(5)
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", connections, release
    release.set()
    httpd.shutdown()
    httpd.server_close()


def test_consecutive_provider_requests_reuse_one_connection(server):
    url, connections, _release = server
    for _ in range(3):
        with pooled_client(5.0) as client:
            assert client.get(url + "/fast").text == "ok"
    assert len(connections) == 1


def test_a_canceled_request_leaves_the_shared_pool_usable(server):
    url, connections, release = server
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    threading.Timer(0.6, release.set).start()
    with streaming_client(10.0, cancel) as client:
        client.get(url + "/slow")  # Its error is swallowed because the reply was canceled.
    started = time.monotonic()
    with pooled_client(5.0) as client:
        assert client.get(url + "/fast").text == "ok"
    assert time.monotonic() - started < 2.0
    assert len(connections) == 2  # The canceled connection was dropped, not reused.
