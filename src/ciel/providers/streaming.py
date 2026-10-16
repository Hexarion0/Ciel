"""Cancel blocked HTTP streams without passing local controls to provider APIs."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import httpx


@contextmanager
def streaming_client(
    timeout: float | httpx.Timeout, cancel_event: threading.Event | None = None,
) -> Iterator:
    """Close the HTTP client when its owning reply is canceled."""
    finished = threading.Event()
    with httpx.Client(timeout=timeout) as client:
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
