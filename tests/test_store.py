"""The filesystem store — the default backend, and the format claims made about it."""

from __future__ import annotations

import json

import pytest

from llmtivo.store import FileStore, MemoryStore


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    """Both stores must satisfy the same contract — that is what makes the backend pluggable."""
    return FileStore(tmp_path / "cassettes") if request.param == "file" else MemoryStore()


def test_append_then_read_round_trips_in_order(store):
    for i in range(1, 4):
        store.append_line("t", json.dumps({"ordinal": i}))
    assert [json.loads(x)["ordinal"] for x in store.read_lines("t")] == [1, 2, 3]


def test_absent_cassette_reads_empty_rather_than_raising(store):
    assert store.exists("nope") is False
    assert store.read_lines("nope") == []
    assert store.size_bytes("nope") == 0


def test_delete_is_idempotent(store):
    store.append_line("t", "{}")
    store.delete("t")
    store.delete("t")
    assert store.exists("t") is False


def test_names_lists_what_was_written(store):
    store.append_line("b", "{}")
    store.append_line("a", "{}")
    assert store.names() == ["a", "b"]


def test_each_append_is_its_own_frame_so_a_crash_keeps_what_landed(tmp_path):
    """Recording appends one frame per interaction. A run that dies mid-test must leave every
    earlier interaction readable — which a single-document format cannot promise."""
    store = FileStore(tmp_path / "c")
    store.append_line("t", json.dumps({"ordinal": 1}))
    store.append_line("t", json.dumps({"ordinal": 2}))
    path = store.root / "t.jsonl.zst"

    partial = path.read_bytes()
    store.append_line("t", json.dumps({"ordinal": 3}))
    path.write_bytes(partial)  # simulate the third append never completing

    assert [json.loads(x)["ordinal"] for x in store.read_lines("t")] == [1, 2]


def test_the_stored_bytes_are_compressed_and_lossless(tmp_path):
    """The format claim: smaller on disk, byte-identical coming back."""
    store = FileStore(tmp_path / "c")
    lines = [
        json.dumps({"ordinal": i, "response": "class Foo { fun bar() = 42 }\n" * 40})
        for i in range(1, 21)
    ]
    for line in lines:
        store.append_line("t", line)

    raw = sum(len(x) + 1 for x in lines)
    stored = store.size_bytes("t")
    assert store.read_lines("t") == lines, "lossless: every byte comes back"
    assert stored < raw / 3, f"expected real compression, got {raw} -> {stored}"


def test_a_cassette_is_readable_with_ordinary_zstd_tooling(tmp_path):
    """Nobody should need this library to read a tape it wrote — `zstdcat` must work."""
    import zstandard as zstd

    store = FileStore(tmp_path / "c")
    store.append_line("t", json.dumps({"hello": "world"}))
    with (store.root / "t.jsonl.zst").open("rb") as fh:
        decoded = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True).read()
    assert json.loads(decoded.decode())["hello"] == "world"
