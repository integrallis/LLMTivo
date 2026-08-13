"""The pytest plugin — a per-test recorder, and a mode you set once for the whole run.

    # conftest.py
    pytest_plugins = ["llmtivo.plugin"]

    def test_the_build(llmtivo):
        with llmtivo.patch(ChatAnthropic, "invoke"):
            assert build() == expected

    $ pytest                        # replay (the default — CI-safe, costs nothing)
    $ pytest --llmtivo=record-new    # record only the tests with no tape yet
    $ pytest --llmtivo=record        # re-record everything. costs money.

## The default is REPLAY, deliberately

The convenient default would be `replay-or-record`: silently call the real model whenever a tape is
missing. That is how a suite starts billing an API without anyone noticing, and how a "green" run
stops being reproducible. Replay refuses instead, and names the test to re-record. Spending money is
something you ask for.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from llmtivo.guard import blocked_network
from llmtivo.intercept import RequestBuilder, default_request, patched, patched_all
from llmtivo.modes import Mode
from llmtivo.recorder import Recorder
from llmtivo.store import CassetteStore, FileStore

_MODE_KEY = pytest.StashKey[Mode]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmtivo", "record and replay real LLM calls")
    group.addoption(
        "--llmtivo",
        action="store",
        default="replay",
        choices=[m.value.replace("_", "-") for m in Mode],
        help="replay (default, free) | record-new | record | replay-or-record | off",
    )
    group.addoption(
        "--llmtivo-allowed-hosts",
        action="store",
        default="",
        help="comma-separated hosts reachable while replaying (loopback always is)",
    )
    group.addoption(
        "--llmtivo-dir",
        action="store",
        default="tests/cassettes",
        help="where cassettes live (default: tests/cassettes)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_MODE_KEY] = Mode(str(config.getoption("--llmtivo")).replace("-", "_"))
    config.addinivalue_line("markers", "llmtivo(mode): override the LLMTivo mode for one test")


class LLMTivo:
    """The per-test handle the `llmtivo` fixture yields."""

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    @property
    def mode(self) -> Mode:
        return self.recorder.mode

    @property
    def stats(self) -> Any:
        return self.recorder.stats

    def patch(
        self,
        target: type,
        method: str = "invoke",
        *,
        build_request: RequestBuilder = default_request,
    ) -> AbstractContextManager[Recorder]:
        """Intercept one method for the duration of a `with` block."""
        return patched(target, method, self.recorder, build_request=build_request)

    def patch_all(
        self, targets: list[tuple[type, str]], *, build_request: RequestBuilder = default_request
    ) -> AbstractContextManager[Recorder]:
        """Intercept several methods onto ONE tape, preserving the real call order."""
        return patched_all(targets, self.recorder, build_request=build_request)

    def call(self, request: dict[str, Any], perform: Any) -> Any:
        """Route one call by hand, when patching is not the right tool."""
        return self.recorder.call(request, perform)


#: Env vars whose VALUES must never reach a committed cassette. Matched by suffix, so a provider
#: variable this list has never heard of is still covered.
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def _env_secrets() -> list[str]:
    """The credential values present in this environment.

    Key-NAME filtering catches `api_key=...`; it does nothing for a key interpolated into a prompt
    or a URL, which is just text. Redacting the actual values covers both, and needs no
    configuration to be right on the first run — the run that records the tape you commit.
    """
    return [v for k, v in os.environ.items() if k.upper().endswith(_SECRET_SUFFIXES) and v]


def _allowed_hosts(config: pytest.Config) -> list[str]:
    raw = str(config.getoption("--llmtivo-allowed-hosts") or "")
    return [h.strip() for h in raw.split(",") if h.strip()]


@pytest.fixture(scope="session")
def llmtivo_store(pytestconfig: pytest.Config) -> CassetteStore:
    """The cassette store for the session. Override to plug in a database backend."""
    return FileStore(Path(str(pytestconfig.getoption("--llmtivo-dir"))))


@pytest.fixture
def llmtivo(request: pytest.FixtureRequest, llmtivo_store: CassetteStore) -> Iterator[LLMTivo]:
    """A recorder scoped to THIS test, keyed by its nodeid."""
    mode = request.config.stash[_MODE_KEY]
    if (marker := request.node.get_closest_marker("llmtivo")) and marker.args:
        mode = Mode(str(marker.args[0]).replace("-", "_"))

    drifts: list[tuple[str, str, str]] = []
    recorder = Recorder(
        llmtivo_store,
        request.node.nodeid,
        mode=mode,
        on_drift=lambda *a: drifts.append(a),
        secrets=_env_secrets(),
    )
    handle = LLMTivo(recorder)

    # REPLAY promises the network is never touched. The recorder can only keep that promise for
    # calls that reach it; anything going around the seam — an unpatched client, an HTTP call inside
    # a tool — would be billed by a suite reporting itself as replaying. Derived from the mode
    # rather than a flag, because an opt-in guard is one nobody sets.
    guard = (
        blocked_network(_allowed_hosts(request.config)) if mode.strict else contextlib.nullcontext()
    )
    with guard:
        yield handle

    # A test that FAILED while recording leaves a partial tape that `exists()` reports as real, so
    # the next record-new run resolves to replay and misses. Throw it away instead.
    report = getattr(request.node, "rep_call", None)
    if report is not None and report.failed and recorder.stats.recorded:
        recorder.discard()
    else:
        recorder.finish()

    if recorder.stats.unplayed:
        request.node.add_report_section(
            "teardown",
            "llmtivo",
            f"{recorder.stats.unplayed} recorded interaction(s) were never reached — the tape is "
            f"longer than the code; re-record this test",
        )
    if drifts:
        # a re-recorded call is a real event: it cost money and the tape changed under you
        request.node.add_report_section(
            "teardown",
            "llmtivo",
            f"{len(drifts)} stale interaction(s) re-recorded — commit the cassette",
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator[None, Any, None]:
    """Expose the call-phase result to the fixture, so a failed recording can be discarded."""
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """One line saying what the run cost — silence here is how a billing suite hides."""
    mode = config.stash.get(_MODE_KEY, None)
    if mode is None or mode is Mode.OFF:
        return
    terminalreporter.write_sep(
        "-", f"llmtivo: mode={mode.value} dir={config.getoption('--llmtivo-dir')}"
    )
