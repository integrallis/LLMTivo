"""Concurrent model calls — the case ordinal keying does not survive unaided.

Real pipelines fan out. KMPilot's builder runs a `ThreadPoolExecutor` over stories and another over
best-of-N candidates; a LangGraph node can `asyncio.gather`. Two things break:

  * **The counter races.** `self._ordinal += 1` is read-modify-write, so two threads can take the
    same ordinal and one recording overwrites the other.
  * **Arrival order is scheduling order.** Call A may be ordinal 3 while recording and ordinal 5
    while replaying, purely because a thread was scheduled differently. Every response after that
    point is then served to the wrong caller — and because each still *has* a recording, nothing
    looks wrong until the assertions fail somewhere unrelated.

Order still ADDRESSES an interaction; the fingerprint now also RECOVERS one that arrived out of
order. A genuinely changed prompt matches nothing and still drifts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from llmtivo import MemoryStore, Mode, Recorder
from llmtivo.modes import FingerprintDrift


def ask(rec: Recorder, prompt: str) -> str:
    return str(
        rec.call(
            {"model": "m", "messages": [{"role": "user", "content": prompt}]},
            lambda: f"answer to {prompt}",
        )
    )


def test_concurrent_recording_loses_no_interaction():
    """Sixteen threads, sixteen interactions, sixteen distinct ordinals."""
    rec = Recorder(MemoryStore(), "conc::record", mode=Mode.RECORD)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: ask(rec, f"prompt {i}"), range(16)))

    tape = rec.cassette.load()
    assert len(tape) == 16, f"lost interactions to the race: {len(tape)}"
    assert sorted(i.ordinal for i in tape) == list(range(1, 17)), "duplicate or skipped ordinals"


def test_a_tape_replays_when_the_calls_arrive_in_a_DIFFERENT_order():
    """Recorded 1..8 sequentially, replayed in reverse. Every caller must get ITS answer."""
    store = MemoryStore()
    rec = Recorder(store, "conc::reorder", mode=Mode.RECORD)
    for i in range(8):
        ask(rec, f"prompt {i}")

    replay = Recorder(store, "conc::reorder", mode=Mode.REPLAY)
    got = {i: ask(replay, f"prompt {i}") for i in reversed(range(8))}
    assert all(got[i] == f"answer to prompt {i}" for i in range(8)), got


def test_concurrent_replay_serves_each_thread_its_own_answer():
    store = MemoryStore()
    rec = Recorder(store, "conc::threads", mode=Mode.RECORD)
    for i in range(12):
        ask(rec, f"prompt {i}")

    replay = Recorder(store, "conc::threads", mode=Mode.REPLAY)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda i: (i, ask(replay, f"prompt {i}")), range(12)))
    for i, answer in results:
        assert answer == f"answer to prompt {i}", f"thread {i} got {answer!r}"


def test_the_same_prompt_twice_still_yields_its_two_DIFFERENT_answers():
    """Recovering by fingerprint must not collapse into content-addressing.

    best-of-N sends an identical prompt twice at a higher temperature precisely to get diverse
    drafts. Both recordings have the SAME fingerprint, so each must be consumed once."""
    store = MemoryStore()
    rec = Recorder(store, "conc::twins", mode=Mode.RECORD)
    drafts = iter(["draft A", "draft B"])
    req = {"model": "m", "messages": [{"role": "user", "content": "same"}]}
    for _ in range(2):
        rec.call(req, lambda: next(drafts))

    replay = Recorder(store, "conc::twins", mode=Mode.REPLAY)
    assert [replay.call(req, lambda: "live"), replay.call(req, lambda: "live")] == [
        "draft A",
        "draft B",
    ]


def test_a_genuinely_changed_prompt_still_drifts():
    """Out-of-order recovery must not become 'anything goes'. A prompt that appears NOWHERE on the
    tape is a changed question, and replay must refuse it."""
    store = MemoryStore()
    rec = Recorder(store, "conc::drift", mode=Mode.RECORD)
    ask(rec, "the original question")

    replay = Recorder(store, "conc::drift", mode=Mode.REPLAY)
    with pytest.raises(FingerprintDrift):
        ask(replay, "an entirely different question")
