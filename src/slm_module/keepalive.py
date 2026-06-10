from __future__ import annotations

import threading
from typing import Callable


class SLMKeepAlive:
    """Background heartbeat that keeps the SLM link alive.

    Calls `ping` every `interval_seconds` on a daemon thread; its return
    value is ignored (the GUI passes a callable that re-sends the current
    pattern over DVI). A failing ping is reported through `on_status` and
    retried on the next tick.
    """

    def __init__(
        self,
        ping: Callable[[], object],
        interval_seconds: float = 15.0,
        on_status: Callable[[bool, str], None] | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._ping = ping
        self._interval = float(interval_seconds)
        self._on_status = on_status
        self._stop = threading.Event()
        self._suspended = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run, name="slm-keepalive", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def suspend(self) -> None:
        """Skip heartbeats (e.g. while a scan owns the device)."""
        self._suspended.set()
        self._wake.set()

    def resume(self) -> None:
        self._suspended.clear()
        self._wake.set()

    def set_interval(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("interval must be positive")
        self._interval = float(seconds)
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self._suspended.is_set():
                continue
            try:
                self._ping()
            except Exception as exc:  # keep ticking; next ping retries
                self._report(False, str(exc))
            else:
                self._report(True, "ok")

    def _report(self, ok: bool, message: str) -> None:
        if self._on_status is not None:
            try:
                self._on_status(ok, message)
            except Exception:
                pass
