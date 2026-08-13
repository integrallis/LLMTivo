"""LLMTivo — record real model calls once, replay them forever, on the filesystem.

A test suite that talks to a language model has two bad options: call the real thing (slow, costly,
non-deterministic) or hand-write a fake (free, fast, and proof of nothing). LLMTivo is the third:
record the REAL responses once, commit the tape, and replay it on every run after.

    from llmtivo import Recorder, FileStore, Mode

    rec = Recorder(FileStore("tests/cassettes"), test_id, mode=Mode.REPLAY)
    response = rec.call(request, perform=lambda: real_client.invoke(request))

Design in one line each — the reasoning lives in each module:

  * [llmtivo.store]     the filesystem is the DEFAULT; zstd-9 JSONL, measured, lossless, appendable.
                       A database is a pluggable backend, never a prerequisite.
  * [llmtivo.cassette]  one tape per test, keyed by call ORDER so prompt edits do not invalidate it.
  * [llmtivo.keys]      a fingerprint of each request, so order-keying cannot drift silently.
  * [llmtivo.modes]     what happens on a cassette miss — the question every failure mode reduces to.
  * [llmtivo.recorder]  the state machine, testable without patching anything.
  * [llmtivo.intercept] the seam that puts a recorder in front of a real client, narrowly.
  * [llmtivo.plugin]    the pytest integration: `--llmtivo=record-new`, and an `llmtivo` fixture.
"""

from __future__ import annotations

from llmtivo.cassette import Cassette, Interaction
from llmtivo.intercept import patched, patched_all
from llmtivo.keys import fingerprint
from llmtivo.modes import CassetteMiss, FingerprintDrift, Mode
from llmtivo.recorder import Recorder, Stats
from llmtivo.store import CassetteStore, FileStore, MemoryStore

__version__ = "0.1.1"

__all__ = [
    "Cassette",
    "CassetteMiss",
    "CassetteStore",
    "FileStore",
    "FingerprintDrift",
    "Interaction",
    "MemoryStore",
    "Mode",
    "Recorder",
    "Stats",
    "fingerprint",
    "patched",
    "patched_all",
]
