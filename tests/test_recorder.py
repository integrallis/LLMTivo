"""The record/replay state machine — what happens on a hit, a miss, and drift."""

from __future__ import annotations

import pytest

from llmtivo import Cassette, CassetteMiss, FingerprintDrift, MemoryStore, Mode, Recorder

TEST_ID = "tests/test_x.py::test_y"


def _req(content: str = "hello", model: str = "sonnet") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def _perform(value: str = "the answer"):
    """A stand-in for the real model call that COUNTS how often it was reached."""
    calls: list[int] = []

    def perform():
        calls.append(1)
        return value

    perform.calls = calls  # type: ignore[attr-defined]
    return perform


def test_records_then_replays_without_touching_the_model_again():
    store = MemoryStore()
    live = _perform("recorded answer")

    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    assert rec.call(_req(), live) == "recorded answer"
    assert len(live.calls) == 1

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    assert replay.call(_req(), live) == "recorded answer"
    assert len(live.calls) == 1, "replay must not reach the model"
    assert replay.stats.replayed == 1 and replay.stats.recorded == 0


def test_replay_mode_refuses_to_call_the_model_on_a_miss():
    """CI's contract. A test that quietly started calling a paid API is a defect; falling back to
    the network would hide exactly that."""
    live = _perform()
    rec = Recorder(MemoryStore(), TEST_ID, mode=Mode.REPLAY)
    with pytest.raises(CassetteMiss, match="no recording"):
        rec.call(_req(), live)
    assert live.calls == [], "a miss must not reach the model in replay mode"


def test_record_new_records_only_the_untaped_test():
    store = MemoryStore()
    live = _perform("first")
    Recorder(store, TEST_ID, mode=Mode.RECORD_NEW).call(_req(), live)
    assert len(live.calls) == 1

    again = Recorder(store, TEST_ID, mode=Mode.RECORD_NEW)
    assert again.mode is Mode.REPLAY, "a taped test resolves to replay"
    assert again.call(_req(), live) == "first"
    assert len(live.calls) == 1, "an existing cassette is never re-billed"


def test_a_cosmetic_prompt_edit_still_replays():
    """The resilience order-keying buys: re-wrapping or re-indenting a template is not a different
    question, and must not bill a re-record. The fingerprint normalises whitespace to make that so."""
    store = MemoryStore()
    Recorder(store, TEST_ID, mode=Mode.RECORD).call(
        _req("Generate the domain model."), _perform("A")
    )

    live = _perform("SHOULD NOT BE CALLED")
    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    assert replay.call(_req("Generate   the domain\n  model."), live) == "A"
    assert live.calls == []
    assert replay.stats.drifted == 0


def test_a_semantic_prompt_edit_makes_the_tape_stale_and_replay_refuses_it():
    """The correctness order-keying alone cannot give. A response recorded under the OLD prompt is
    not an answer to the NEW one, so replaying it would assert downstream behaviour against
    something the current code could never elicit. CI must fail, not pass."""
    store = MemoryStore()
    Recorder(store, TEST_ID, mode=Mode.RECORD).call(
        _req("Generate the DOMAIN model."), _perform("A")
    )

    live = _perform("SHOULD NOT BE CALLED")
    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    with pytest.raises(FingerprintDrift, match="no longer an answer"):
        replay.call(_req("Generate the SCREEN model."), live)
    assert live.calls == []


def test_a_stale_call_invalidates_its_TAIL_too():
    """In an agentic loop call N+1's prompt contains call N's response, so re-answering N makes
    every later recording a reply to an abandoned branch. Keeping them would replay a conversation
    that never happens."""
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    rec.call(_req("step one"), _perform("A"))
    rec.call(_req("step two"), _perform("B"))
    rec.call(_req("step three"), _perform("C"))
    assert len(rec.cassette) == 3

    live = _perform("re-recorded")
    edited = Recorder(store, TEST_ID, mode=Mode.REPLAY_OR_RECORD)
    assert edited.call(_req("step one CHANGED"), live) == "re-recorded"
    assert edited.stats.drifted == 1
    # the tail is gone: calls 2 and 3 answered a branch this run abandoned
    assert [i.ordinal for i in Cassette(store, TEST_ID).load()] == [1]


def test_a_recordable_mode_re_records_the_stale_call_instead_of_failing():
    """Locally you want the tape repaired, not a stack trace — but only a mode that is ALLOWED to
    reach the model does it."""
    store = MemoryStore()
    Recorder(store, TEST_ID, mode=Mode.RECORD).call(_req("as recorded"), _perform("old"))

    seen: list[tuple[str, str, str]] = []
    live = _perform("fresh")
    fixed = Recorder(store, TEST_ID, mode=Mode.REPLAY_OR_RECORD, on_drift=lambda *a: seen.append(a))
    assert fixed.call(_req("edited"), live) == "fresh"
    assert len(live.calls) == 1 and len(seen) == 1


def test_call_order_is_what_addresses_an_interaction():
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    rec.call(_req("one"), _perform("first"))
    rec.call(_req("two"), _perform("second"))

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    assert replay.call(_req("one"), _perform("x")) == "first"
    assert replay.call(_req("two"), _perform("x")) == "second"


def test_re_recording_truncates_rather_than_half_overwriting():
    store = MemoryStore()
    first = Recorder(store, TEST_ID, mode=Mode.RECORD)
    first.call(_req("a"), _perform("A"))
    first.call(_req("b"), _perform("B"))

    second = Recorder(store, TEST_ID, mode=Mode.RECORD)
    second.call(_req("a"), _perform("A2"))

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    assert replay.call(_req("a"), _perform("x")) == "A2"
    with pytest.raises(CassetteMiss):
        replay.call(_req("b"), _perform("x"))  # the stale second interaction is GONE


def test_off_mode_is_pure_passthrough():
    store = MemoryStore()
    live = _perform("live")
    rec = Recorder(store, TEST_ID, mode=Mode.OFF)
    assert rec.call(_req(), live) == "live"
    assert len(live.calls) == 1
    assert store.names() == [], "OFF writes nothing"


def test_credentials_never_reach_the_tape():
    """Cassettes are committed. A secret that lands in one is leaked by git history forever, so the
    filter runs on the way IN rather than being a review responsibility."""
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    rec.call(
        {**_req(), "api_key": "sk-live-SECRET", "headers": {"Authorization": "Bearer SECRET"}},
        _perform("ok"),
    )
    tape = "".join(store.read_lines(rec.cassette.name))
    assert "SECRET" not in tape
    assert "hello" in tape, "the actual request is still recorded"


def test_finish_compacts_a_recorded_tape_without_changing_what_it_holds(tmp_path):
    """Appending costs compression: each interaction is its own zstd frame, so a real 26-response
    tape stored 17.0 KiB per-frame vs 11.3 KiB as one. `finish()` buys that back AFTER the tape is
    known complete, so durability during recording is never traded away."""
    from llmtivo import FileStore

    store = FileStore(tmp_path / "c")
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    body = "class Foo { fun bar() = 42 }\n" * 60
    for i in range(12):
        rec.call(_req(f"gen {i}"), _perform(body))

    before = store.size_bytes(rec.cassette.name)
    lines_before = store.read_lines(rec.cassette.name)
    rec.finish()
    after = store.size_bytes(rec.cassette.name)

    assert store.read_lines(rec.cassette.name) == lines_before, "compaction is lossless"
    assert after < before, f"expected compaction to shrink the tape, {before} -> {after}"

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    assert replay.call(_req("gen 0"), _perform("unused")) == body


def test_finish_leaves_a_replay_only_run_alone(tmp_path):
    """Nothing was recorded, so there is nothing to compact — and rewriting someone else's tape on
    a read-only run would be a surprise."""
    from llmtivo import FileStore

    store = FileStore(tmp_path / "c")
    Recorder(store, TEST_ID, mode=Mode.RECORD).call(_req(), _perform("A"))
    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    replay.call(_req(), _perform("x"))
    size = store.size_bytes(replay.cassette.name)
    replay.finish()
    assert store.size_bytes(replay.cassette.name) == size


# ── gaps found by auditing vcrpy, which has had years to meet these ───────────────────────────────


def test_a_tape_longer_than_the_run_is_reported():
    """vcrpy tracks `all_played`; this had no equivalent, so interactions the run STOPPED making sat
    on the tape forever, silently. That is the mirror of a cassette miss: the tape and the code have
    diverged, and by this library's own rule that must not pass quietly."""
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    for i in range(5):
        rec.call(_req(f"call {i}"), _perform(f"r{i}"))
    rec.finish()

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    for i in range(2):
        replay.call(_req(f"call {i}"), _perform("unused"))

    assert replay.unplayed == 3, "three recorded interactions were never reached"
    assert replay.all_played is False


def test_a_fully_used_tape_reports_nothing_unplayed():
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    rec.call(_req("one"), _perform("A"))
    rec.finish()

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    replay.call(_req("one"), _perform("unused"))
    assert replay.all_played is True and replay.unplayed == 0


def test_a_recording_that_never_finished_does_not_masquerade_as_a_complete_tape():
    """vcrpy's `record_on_exception`. Recording APPENDS per call, so a run that dies at call 3 of 10
    leaves a 2-interaction tape that `exists()` happily reports — and the next RECORD_NEW run
    resolves to REPLAY and misses. `discard()` lets a failed recording leave nothing behind."""
    store = MemoryStore()
    rec = Recorder(store, TEST_ID, mode=Mode.RECORD)
    rec.call(_req("one"), _perform("A"))
    rec.call(_req("two"), _perform("B"))
    assert rec.cassette.exists()

    rec.discard()
    assert not rec.cassette.exists(), "a half-recorded tape is worse than none"
    assert Recorder(store, TEST_ID, mode=Mode.RECORD_NEW).mode is Mode.RECORD


def test_discarding_a_replay_only_run_leaves_the_tape_alone():
    """Nothing was recorded, so there is nothing of ours to throw away — destroying someone else's
    committed tape because a test raised would be a catastrophe, not a cleanup."""
    store = MemoryStore()
    Recorder(store, TEST_ID, mode=Mode.RECORD).call(_req("one"), _perform("A"))

    replay = Recorder(store, TEST_ID, mode=Mode.REPLAY)
    replay.call(_req("one"), _perform("unused"))
    replay.discard()
    assert replay.cassette.exists(), "a replay run never destroys the tape"


def test_a_key_embedded_in_a_PROMPT_is_redacted():
    """Key-name filtering only catches a secret that arrives under a name it recognises.

    A credential interpolated into a prompt, a system message or a URL is just text in `messages`,
    and sails onto a tape that gets committed. Betamax substitutes by VALUE for this reason.
    """
    store = MemoryStore()
    rec = Recorder(store, "leak::prompt", mode=Mode.RECORD, secrets=["sk-live-abcdef123"])
    rec.call(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "call the API with sk-live-abcdef123 please"}],
        },
        lambda: {"ok": True},
    )
    written = str(rec.cassette.load()[0].request)
    assert "sk-live-abcdef123" not in written, "the key reached the tape"
    assert "<REDACTED>" in written, "and its place is marked, so the tape stays readable"


def test_redaction_reaches_nested_values():
    store = MemoryStore()
    rec = Recorder(store, "leak::nested", mode=Mode.RECORD, secrets=["tok-987"])
    rec.call(
        {"model": "m", "messages": [{"role": "system", "content": ["a", "Bearer tok-987"]}]},
        lambda: {"ok": True},
    )
    assert "tok-987" not in str(rec.cassette.load()[0].request)


def test_an_empty_secret_is_ignored():
    """An unset env var read as `""` would otherwise redact every character of every request."""
    store = MemoryStore()
    rec = Recorder(store, "leak::empty", mode=Mode.RECORD, secrets=["", None])
    rec.call({"model": "m", "messages": [{"role": "user", "content": "hello"}]}, lambda: {"ok": 1})
    assert "hello" in str(rec.cassette.load()[0].request)


def test_a_chat_fingerprint_is_unchanged_by_the_tool_fields():
    """Tool name and args enter the digest only when present. A chat call must hash exactly as it
    did before tools existed as a concept — otherwise adding the feature silently invalidates every
    cassette already committed and bills a re-record for prompts that never changed."""
    from llmtivo.keys import fingerprint

    chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert fingerprint(chat) == "ca84ac18369fb031", "the chat digest moved"


def test_a_tool_call_is_addressed_by_its_name_and_arguments():
    from llmtivo.keys import fingerprint

    a = fingerprint({"tool": "search", "args": {"q": "kotlin"}})
    b = fingerprint({"tool": "search", "args": {"q": "swift"}})
    c = fingerprint({"tool": "fetch", "args": {"q": "kotlin"}})
    assert a != b, "different arguments are a different question"
    assert a != c, "a different tool is a different question"
    assert a == fingerprint({"tool": "search", "args": {"q": "kotlin"}}), "and it is stable"
