"""The recording modes, and what each one promises.

The mode answers one question: **when the tape does not have what this call needs, what happens?**
Every failure mode of a record/replay system lives in that answer, so it is spelled out here rather
than left to a boolean.
"""

from __future__ import annotations

from enum import StrEnum


class Mode(StrEnum):
    """How LLMTivo behaves for a given test."""

    #: Re-record from scratch: drop the cassette, call the real model, keep every response.
    #: Costs money. The only mode that reaches the network when a cassette already exists.
    RECORD = "record"

    #: Record ONLY tests with no cassette yet; replay the rest. The everyday way to add a test
    #: without re-billing the suite.
    RECORD_NEW = "record_new"

    #: Replay only. A missing interaction is an ERROR, never a silent call to the real model.
    #: This is what CI runs: a test that quietly started calling a paid API is a defect, and a
    #: fallback that hides it is worse than a failure.
    REPLAY = "replay"

    #: Replay when recorded, record when not. Convenient locally; NOT for CI, because it will
    #: happily spend money to paper over a cassette that should have been committed.
    REPLAY_OR_RECORD = "replay_or_record"

    #: Passthrough — no interception at all. Real calls, real cost, nothing written.
    OFF = "off"

    @property
    def may_record(self) -> bool:
        return self in (Mode.RECORD, Mode.RECORD_NEW, Mode.REPLAY_OR_RECORD)

    @property
    def may_replay(self) -> bool:
        return self in (Mode.RECORD_NEW, Mode.REPLAY, Mode.REPLAY_OR_RECORD)

    @property
    def strict(self) -> bool:
        """Whether a cassette miss must fail rather than reach the network."""
        return self is Mode.REPLAY


class CassetteMiss(RuntimeError):
    """No recorded interaction for a call, in a mode that forbids reaching the real model."""


class FingerprintDrift(RuntimeError):
    """A recorded interaction exists at this ordinal, but it answers a DIFFERENT question.

    Order addresses an interaction; the fingerprint validates it. When they disagree the tape is
    stale — the model answered a prompt this code no longer sends, so replaying it would assert
    downstream behaviour against a response the current code could never elicit.

    In `REPLAY` this raises: CI must not pass on a stale tape. In a recordable mode the stale
    interaction and everything after it are dropped and re-recorded, because in an agentic loop
    call N+1's prompt contains call N's response — once N is re-answered, the tail is a reply to an
    abandoned branch.

    Cosmetic edits do not trigger this: the fingerprint normalises whitespace, so re-wrapping a
    Jinja template is not a new question.
    """


def resolve(mode: Mode, has_cassette: bool) -> Mode:
    """The EFFECTIVE mode for one test, given whether it already has a cassette.

    `RECORD_NEW` is the only mode that depends on the tape's existence, and collapsing it here
    keeps that branch out of the interception path.
    """
    if mode is Mode.RECORD_NEW:
        return Mode.REPLAY if has_cassette else Mode.RECORD
    return mode
