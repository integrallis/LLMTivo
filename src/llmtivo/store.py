"""Cassette storage — a pluggable backend, with the filesystem as the default.

THE DEFAULT IS THE FILESYSTEM. A cassette is an artifact of the repo: reviewable in a diff, carried
by the same git history as the test that produced it, and needing nothing booted. A database (or any
other service) is a legitimate BACKEND for teams that want one, but it is never a prerequisite for
running the tests.

## The wire format, and why

`<name>.jsonl.zst` — JSON Lines, one interaction per line, compressed with zstandard. Every recorded
byte is kept: the format is LOSSLESS, so a replayed response is exactly what the model returned.

Measured on a real corpus of model output (26 interactions, 65 KiB of generated Kotlin + prompts):

    codec              size      ratio   compress   decompress
    gzip -6           12.3 KiB    5.3x      1.6 ms      0.2 ms
    gzip -9           12.2 KiB    5.4x      2.6 ms      0.2 ms
    zstd -3           13.2 KiB    5.0x      0.1 ms     <0.1 ms
    zstd -9           11.9 KiB    5.5x      0.6 ms     <0.1 ms   <- chosen
    zstd -19          11.4 KiB    5.7x     15.5 ms     <0.1 ms
    msgpack + zstd-3  13.3 KiB    4.9x

zstd -9 beats gzip -9 on BOTH axes — smaller output, ~4x faster to write, ~5x faster to read — so
there is no compression/performance trade to make here. Level 19 buys 4% more for 25x the write
cost, which a test suite writing thousands of interactions would feel. msgpack was rejected: the
payload is text (prompts and generated source), so binary framing saves ~5% before compression and
the compressor erases even that, while costing the ability to read a cassette with `zstdcat`.

JSONL rather than one document because recording APPENDS: each interaction is its own zstd frame, so
a run that dies halfway leaves every earlier interaction intact and replayable. zstd decodes
concatenated frames as a single stream, so appending stays a plain `open(..., "ab")`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import zstandard as zstd

#: The compression level. See the table above — the knee of the curve for this workload.
LEVEL = 9


@runtime_checkable
class CassetteStore(Protocol):
    """Where cassettes live. Implement this to back cassettes with a database or object store."""

    def exists(self, name: str) -> bool:
        """Whether a cassette has been recorded under `name`."""
        ...

    def read_lines(self, name: str) -> list[str]:
        """Every recorded line, in recorded order. Empty when the cassette does not exist."""
        ...

    def append_line(self, name: str, line: str) -> None:
        """Append ONE line, durably — a crash must not lose the lines already written."""
        ...

    def delete(self, name: str) -> None:
        """Remove the cassette. Idempotent."""
        ...

    def names(self) -> list[str]:
        """Every cassette name in the store, sorted."""
        ...

    def size_bytes(self, name: str) -> int:
        """Stored size, for reporting. 0 when absent."""
        ...

    def compact(self, name: str) -> None:
        """Rewrite the cassette as compactly as the backend can, preserving every line.

        Called when a recording finishes CLEANLY. Appending gives crash-safety at a real cost:
        each interaction is its own zstd frame and compresses independently, so a tape of 26 real
        model responses stored 17.0 KiB per-frame against 11.3 KiB as one frame — 34% wasted. This
        buys that back without giving up the durable append, because it only runs once the tape is
        known complete. A no-op is a valid implementation.
        """
        ...


class FileStore:
    """The default store: one `<name>.jsonl.zst` per cassette under `root`."""

    SUFFIX = ".jsonl.zst"

    def __init__(self, root: Path | str, level: int = LEVEL) -> None:
        self.root = Path(root)
        self.level = level

    def _path(self, name: str) -> Path:
        return self.root / f"{name}{self.SUFFIX}"

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def read_lines(self, name: str) -> list[str]:
        path = self._path(name)
        if not path.is_file():
            return []
        dctx = zstd.ZstdDecompressor()
        with path.open("rb") as fh:
            # read_across_frames: an appended cassette is N concatenated frames, one per interaction
            raw = dctx.stream_reader(fh, read_across_frames=True).read()
        return [line for line in raw.decode("utf-8").splitlines() if line.strip()]

    def append_line(self, name: str, line: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        frame = zstd.ZstdCompressor(level=self.level).compress((line + "\n").encode("utf-8"))
        with self._path(name).open("ab") as fh:
            fh.write(frame)

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name[: -len(self.SUFFIX)] for p in self.root.glob(f"*{self.SUFFIX}"))

    def size_bytes(self, name: str) -> int:
        path = self._path(name)
        return path.stat().st_size if path.is_file() else 0

    def compact(self, name: str) -> None:
        """Rewrite N appended frames as ONE, recovering the ratio the append cost us."""
        lines = self.read_lines(name)
        if len(lines) < 2:
            return
        blob = ("\n".join(lines) + "\n").encode("utf-8")
        frame = zstd.ZstdCompressor(level=self.level).compress(blob)
        tmp = self._path(name).with_suffix(".tmp")
        tmp.write_bytes(frame)
        tmp.replace(self._path(name))  # atomic: a crash mid-compact leaves the appended tape intact


class MemoryStore:
    """An in-process store — for LLMTivo's own tests, and for a throwaway session."""

    def __init__(self) -> None:
        self._data: dict[str, list[str]] = {}

    def exists(self, name: str) -> bool:
        return name in self._data

    def read_lines(self, name: str) -> list[str]:
        return list(self._data.get(name, []))

    def append_line(self, name: str, line: str) -> None:
        self._data.setdefault(name, []).append(line)

    def delete(self, name: str) -> None:
        self._data.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._data)

    def size_bytes(self, name: str) -> int:
        return sum(len(line) + 1 for line in self._data.get(name, []))

    def compact(self, name: str) -> None:
        """Nothing to compact in memory — the protocol allows a no-op."""
