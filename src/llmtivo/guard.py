"""The network guard — making `REPLAY` mean replay for calls that never reach the seam.

[Recorder][llmtivo.recorder.Recorder] raises on a miss, which covers every client LLMTivo was
pointed at and nothing else. A model call that goes AROUND the seam — a provider SDK used directly,
an embedding client nobody patched, an HTTP call inside an agent's tool — reaches the real network
from a suite reporting itself as replaying. It is billed, it is non-deterministic, and it looks
exactly like a pass. That is the same silence this library exists to break, one level up.

    from llmtivo.guard import blocked_network

    with blocked_network():
        run_the_suite()

Borrowed from pytest-recording, including the shape of the fix: only `connect` and `connect_ex` are
patched, never `socket.socket` itself. A test suite legitimately talks to local services — a
database, a fixture server — and blocking those turns a guard against surprise SPENDING into a guard
against testing. Loopback is allowed by default; anything else needs naming.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

#: Addresses a test suite may always reach: its own machine. Everything else is a remote it has to
#: ask for by name.
_ALWAYS_ALLOWED = ("127.0.0.1", "::1", "localhost")

_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex


class NetworkBlocked(RuntimeError):
    """Raised when guarded code attempts an outbound connection."""


def _host_of(address: Any, family: int) -> str:
    if family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple) and address:
        return str(address[0])
    if isinstance(address, (bytes, bytearray)):
        return address.decode(errors="replace")
    return str(address)


def _is_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    return host in allowed


def _guard(original: Any, allowed: tuple[str, ...]) -> Any:
    def network_guard(self: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = _host_of(address, self.family)
        if _is_allowed(host, allowed):
            return original(self, address, *args, **kwargs)
        # naming the host IS the diagnosis: it says which un-recorded client tried to call out
        raise NetworkBlocked(
            f"network is blocked — {host} was contacted while replaying. Some call is not going "
            f"through LLMTivo, so it would hit the real API and be billed. Record it, patch its "
            f"client, or pass allowed_hosts=['{host}'] if it is a local service."
        )

    return network_guard


@contextmanager
def blocked_network(allowed_hosts: Sequence[str] | None = None) -> Iterator[None]:
    """Refuse outbound connections for the duration of the block.

    Loopback is always permitted; `allowed_hosts` adds to it. Restored unconditionally, so a failure
    inside the block never leaves the network disabled for the rest of the run.
    """
    allowed = _ALWAYS_ALLOWED + tuple(allowed_hosts or ())
    socket.socket.connect = _guard(_original_connect, allowed)  # type: ignore[method-assign]
    socket.socket.connect_ex = _guard(_original_connect_ex, allowed)  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = _original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = _original_connect_ex  # type: ignore[method-assign]
