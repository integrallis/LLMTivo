"""The network guard — replay must mean replay, including for calls that bypass the seam.

`REPLAY` raises on a miss AT THE SEAM. That covers every client LLMTivo was pointed at, and nothing
else. A model call that goes around the seam — a provider SDK used directly, an embedding client
nobody patched, an HTTP call inside a tool — reaches the real network from a suite that reports
itself as replaying. It is billed, it is non-deterministic, and it looks exactly like a pass.

Borrowed from pytest-recording, which blocks the socket for the same reason.
"""

from __future__ import annotations

import re
import socket

import pytest

from llmtivo.guard import NetworkBlocked, blocked_network


def connect_to(host: str = "203.0.113.1", port: int = 80) -> None:
    """Attempt a real outbound connection. 203.0.113.0/24 is TEST-NET-3 (RFC 5737) and routes
    nowhere, so an unguarded call fails on timeout rather than reaching a live host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.05)
    try:
        s.connect((host, port))
    finally:
        s.close()


def test_an_outbound_connection_is_refused():
    with pytest.raises(NetworkBlocked, match="network is blocked"), blocked_network():
        connect_to()


def test_the_block_names_the_host_it_stopped():
    """A bare 'network is disabled' sends you hunting. The host is the whole diagnosis."""
    with pytest.raises(NetworkBlocked, match=re.escape("203.0.113.1")), blocked_network():
        connect_to()


def test_localhost_is_reachable_by_default():
    """A test suite legitimately talks to local services — a database, a fixture server. Blocking
    those turns a guard against surprise SPENDING into a guard against testing."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        with blocked_network():
            connect_to("127.0.0.1", port)  # must not raise
    finally:
        server.close()


def test_an_allowed_host_passes_through():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        with blocked_network(allowed_hosts=["127.0.0.1"]):
            connect_to("127.0.0.1", port)
    finally:
        server.close()


def test_the_socket_is_restored_even_when_the_block_raises():
    """A guard that leaks past its own block would disable the network for the rest of the run."""
    before = socket.socket.connect
    with pytest.raises(ValueError, match="boom"), blocked_network():
        raise ValueError("boom")
    assert socket.socket.connect is before


def test_connect_ex_is_guarded_too():
    """`connect_ex` returns an error code instead of raising, so a client using it would slip past
    a guard that only covered `connect`."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.05)
    try:
        with pytest.raises(NetworkBlocked), blocked_network():
            s.connect_ex(("203.0.113.1", 80))
    finally:
        s.close()
