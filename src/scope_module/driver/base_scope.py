from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import numpy as np


class ScopeError(RuntimeError):
    """Base error for oscilloscope driver failures."""


class ScopeConnectionError(ScopeError):
    """Raised when the transport is not connected or drops unexpectedly."""


class ScopeTimeoutError(ScopeError):
    """Raised when the scope does not respond within the timeout."""


class BaseScope(ABC):
    """Abstract base for oscilloscopes.

    Mirrors BaseOSA: concrete drivers implement the transport hooks
    (_open_transport, _close_transport, _write, _query, _query_binary) plus the
    vendor command set, while the connection lifecycle and the thread-safe
    public wrappers live here so every scope -- regardless of transport (VISA,
    raw socket, USB) -- shares one contract.

    A single re-entrant lock serializes every transaction. A request/response
    instrument link has no thread affinity; it only requires that a command and
    its reply are never interleaved with another thread's. The extra
    _query_binary hook exists because waveform downloads return an IEEE-488.2
    binary block rather than a single ASCII line.
    """

    name: str = "scope"

    def __init__(self) -> None:
        self._io_lock = threading.RLock()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- transport hooks implemented by concrete drivers -----------------
    @abstractmethod
    def _open_transport(self) -> None:
        """Open the link and perform any handshake; raise on failure."""

    @abstractmethod
    def _close_transport(self) -> None:
        """Release the underlying transport."""

    @abstractmethod
    def _write(self, command: str) -> None:
        """Send one command, no reply expected. Assumes the io lock is held."""

    @abstractmethod
    def _query(self, command: str) -> str:
        """Send one command and return its reply. Assumes the io lock is held."""

    @abstractmethod
    def _query_binary(self, command: str, datatype: str) -> np.ndarray:
        """Send one command and return its binary block as a float array.

        ``datatype`` is a struct format code (e.g. 'f' for REAL,32 or 'h' for
        INT,16). Assumes the io lock is held.
        """

    # --- public, thread-safe API -----------------------------------------
    def connect(self) -> None:
        with self._io_lock:
            if self._connected:
                return
            self._open_transport()
            self._connected = True

    def disconnect(self) -> None:
        with self._io_lock:
            try:
                if self._connected:
                    self._close_transport()
            finally:
                self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ScopeConnectionError(
                f"{self.name} is not connected; call connect() first"
            )

    def write(self, command: str) -> None:
        """Send a command that produces no reply."""
        with self._io_lock:
            self._ensure_connected()
            self._write(command)

    def query(self, command: str) -> str:
        """Send a command and return its single-line reply."""
        with self._io_lock:
            self._ensure_connected()
            return self._query(command)

    def query_binary(self, command: str, datatype: str = "f") -> np.ndarray:
        """Send a command and return its binary block as a float array."""
        with self._io_lock:
            self._ensure_connected()
            return self._query_binary(command, datatype)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
