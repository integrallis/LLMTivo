"""The pytest plugin, exercised through pytest itself (`pytester`)."""

from __future__ import annotations

import pytest  # noqa: TC002 - the Pytester fixture is a runtime object

pytest_plugins = ["pytester"]

CONFTEST = ""  # the pytest11 entry point registers the plugin; naming it again conflicts

RECORDS_THEN_REPLAYS = """
class FakeChat:
    calls = 0
    def __init__(self): self.model = "sonnet"
    def invoke(self, messages, **kw):
        FakeChat.calls += 1
        return "live:" + messages[0]["content"]

def test_one(llmtivo):
    with llmtivo.patch(FakeChat, "invoke"):
        assert FakeChat().invoke([{"role": "user", "content": "hi"}]) == "live:hi"
"""


def test_replay_is_the_default_mode(pytester: pytest.Pytester) -> None:
    """The convenient default would silently bill an API on every missing tape. This one refuses."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        test_m="def test_mode(llmtivo):\n    assert llmtivo.mode.value == 'replay'\n"
    )
    pytester.runpytest("-q").assert_outcomes(passed=1)


def test_record_new_then_replay_across_runs(pytester: pytest.Pytester) -> None:
    """The everyday loop: record once, then every later run is free and offline."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_m=RECORDS_THEN_REPLAYS)

    first = pytester.runpytest("-q", "--llmtivo=record-new")
    first.assert_outcomes(passed=1)
    assert list(pytester.path.glob("tests/cassettes/*.jsonl.zst")), "a tape was written"

    second = pytester.runpytest("-q", "--llmtivo=replay")
    second.assert_outcomes(passed=1)


def test_replay_fails_loudly_when_a_tape_is_missing(pytester: pytest.Pytester) -> None:
    """CI's contract: a test that quietly started calling a paid model is a defect, so a miss is a
    failure naming the test to re-record — never a silent fallback to the network."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_m=RECORDS_THEN_REPLAYS)
    result = pytester.runpytest("-q", "--llmtivo=replay")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*no recording*"])


def test_a_marker_overrides_the_mode_for_one_test(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        test_m="""
        import pytest

        @pytest.mark.llmtivo("off")
        def test_off(llmtivo):
            assert llmtivo.mode.value == "off"

        def test_default(llmtivo):
            assert llmtivo.mode.value == "replay"
        """
    )
    pytester.runpytest("-q").assert_outcomes(passed=2)


def test_each_test_gets_its_own_tape(pytester: pytest.Pytester) -> None:
    """Cassettes are per-test so re-recording one never risks another."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        test_m="""
        class FakeChat:
            def __init__(self): self.model = "s"
            def invoke(self, messages, **kw): return "x"

        def test_a(llmtivo):
            with llmtivo.patch(FakeChat, "invoke"):
                FakeChat().invoke([{"role": "user", "content": "a"}])

        def test_b(llmtivo):
            with llmtivo.patch(FakeChat, "invoke"):
                FakeChat().invoke([{"role": "user", "content": "b"}])
        """
    )
    pytester.runpytest("-q", "--llmtivo=record-new").assert_outcomes(passed=2)
    assert len(list(pytester.path.glob("tests/cassettes/*.jsonl.zst"))) == 2


def test_the_cassette_directory_is_configurable(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_m=RECORDS_THEN_REPLAYS)
    pytester.runpytest("-q", "--llmtivo=record-new", "--llmtivo-dir=tapes").assert_outcomes(
        passed=1
    )
    assert list(pytester.path.glob("tapes/*.jsonl.zst"))


def test_a_failed_recording_is_discarded_so_the_next_run_re_records(
    pytester: pytest.Pytester,
) -> None:
    """A test that FAILS while recording leaves a partial tape that `exists()` reports as real, so
    the next record-new run resolves to REPLAY and misses. The tape must not survive the failure.

    This is plugin wiring, not Recorder behaviour: `Recorder.discard` existed and was unit-tested
    while the fixture never called it, which is indistinguishable from not having built it.
    """
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        test_f="""
class FakeChat:
    def __init__(self): self.model = "m"
    def invoke(self, messages, **kw): return "recorded"

def test_fails_after_calling_the_model(llmtivo):
    with llmtivo.patch(FakeChat, "invoke"):
        FakeChat().invoke([{"role": "user", "content": "hi"}])
    assert False, "the test fails AFTER the model answered"
"""
    )
    pytester.runpytest("--llmtivo=record-new").assert_outcomes(failed=1)

    tapes = (
        list((pytester.path / "tests" / "cassettes").glob("*"))
        if (pytester.path / "tests" / "cassettes").exists()
        else []
    )
    assert not tapes, f"a failed recording left a tape behind: {tapes}"


def test_an_unplayed_tape_is_reported(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two recorded calls, one made on replay. Silence about the other looks like a clean match."""
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        test_u="""
import os

class FakeChat:
    def __init__(self): self.model = "m"
    def invoke(self, messages, **kw): return "r"

def test_calls(llmtivo):
    n = 2 if os.environ.get("BOTH") else 1
    with llmtivo.patch(FakeChat, "invoke"):
        for i in range(n):
            FakeChat().invoke([{"role": "user", "content": "q"}])
"""
    )
    monkeypatch.setenv("BOTH", "1")
    pytester.runpytest("--llmtivo=record").assert_outcomes(passed=1)
    monkeypatch.delenv("BOTH")
    result = pytester.runpytest("--llmtivo=replay", "-rA")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*never reached*"])


def test_the_package_version_matches_pyproject():
    """Two places hold the version, so they can disagree — and did.

    `pyproject.toml` said 0.1.5 while `llmtivo.__version__` still said 0.1.4, the release workflow's
    own check caught it (`AssertionError: 0.1.4 != 0.1.5`) and the PyPI publish never ran. That check
    lives in CI, minutes away and after a tag has been cut; this one is a second away and runs on
    every commit."""
    import tomllib
    from pathlib import Path

    import llmtivo

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert llmtivo.__version__ == declared, (
        f"llmtivo.__version__ is {llmtivo.__version__} and pyproject.toml says {declared} — "
        f"bump both, or the release fails after the tag is cut"
    )
