"""Provider HTTP clients on one keep-alive pool, with cancellation of blocked requests."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

_POOL_LOCK = threading.Lock()
_POOL: httpx.HTTPTransport | None = None


def _shared_pool() -> httpx.HTTPTransport:
    """One connection pool for every provider, so each model round skips a TLS handshake."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = httpx.HTTPTransport(
                limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=60.0),
            )
        return _POOL


def _reset_pool(pool: httpx.HTTPTransport) -> None:
    """Replace the pool, closing its connections to abort a request still awaiting headers."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is pool:
            _POOL = None
    try:
        pool.close()
    except Exception:
        pass


class _RequestTransport(httpx.BaseTransport):
    """One client's view of the shared pool.

    Closing it after its requests finish only closes its own responses, keeping the pooled
    connection warm. Closing it while a request still waits for a response is a
    cancellation: the shared pool is replaced so the blocked read ends at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._responses: list[httpx.Response] = []
        self._pending: set[httpx.HTTPTransport] = set()
        self._closed = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        pool = _shared_pool()
        with self._lock:
            if self._closed:
                raise httpx.ReadError("The request was canceled", request=request)
            self._pending.add(pool)
        try:
            response = pool.handle_request(request)
        finally:
            with self._lock:
                self._pending.discard(pool)
        with self._lock:
            closed = self._closed
            if not closed:
                self._responses.append(response)
        if closed:
            response.close()
            raise httpx.ReadError("The request was canceled", request=request)
        return response

    def close(self) -> None:
        with self._lock:
            self._closed = True
            responses, self._responses = self._responses, []
            pending = set(self._pending)
        for response in responses:
            try:
                response.close()
            except Exception:
                pass
        for pool in pending:
            _reset_pool(pool)


def pooled_client(timeout: float | httpx.Timeout, **options: object) -> httpx.Client:
    """An HTTP client on the shared keep-alive pool; closing it never drops other requests'
    connections unless it is canceling its own blocked request.

    Each client still starts without cookies; only TCP/TLS connections are shared.
    """
    return httpx.Client(timeout=timeout, transport=_RequestTransport(), **options)


@contextmanager
def streaming_client(
    timeout: float | httpx.Timeout, cancel_event: threading.Event | None = None,
) -> Iterator:
    """Close the HTTP client when its owning reply is canceled."""
    finished = threading.Event()
    with pooled_client(timeout) as client:
        watcher = None
        if cancel_event is not None:
            def cancel() -> None:
                while not finished.wait(0.02):
                    if cancel_event.is_set():
                        try:
                            client.close()
                        except Exception:
                            # The owner still closes it when the read unwinds.
                            pass
                        return

            watcher = threading.Thread(target=cancel, name="ciel-stream-cancel", daemon=True)
            watcher.start()
        try:
            yield client
        except Exception:
            if cancel_event is None or not cancel_event.is_set():
                raise
        finally:
            finished.set()
            if watcher is not None:
                watcher.join(timeout=0.1)
