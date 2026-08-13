"""A cassette — the recorded model calls for ONE test, over a pluggable [store][llmtivo.store].

## Keying, and the drift problem

An interaction is addressed by **call ORDER within its test**: `ordinal` 1, 2, 3… That choice is
deliberate and it is the opposite of what an HTTP-level recorder does.

Keying on a hash of the request is exact, and useless in practice: editing one word of a prompt
invalidates every cassette that prompt appears in, and prompts get edited constantly. Ordinal keying
survives prompt edits entirely — which is the whole point of recording model output rather than
mocking it.

The cost is that ordinal keying cannot, by itself, notice that the code changed underneath the
recording: reorder two calls and each silently replays the other's response. So every interaction
also stores a `fingerprint` of the request it was recorded for. Playback compares, and a mismatch is
reported rather than swallowed — see [llmtivo.keys.fingerprint]. Order gives resilience; the
fingerprint gives honesty. Neither alone is enough.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llmtivo.store import CassetteStore  # noqa: TC001 - runtime protocol, not just a hint

#: Bumped when the on-disk record shape changes incompatibly.
SCHEMA = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def test_id_to_name(test_id: str) -> str:
    """A pytest nodeid → one flat, filesystem-safe cassette name.

    Flat rather than nested: `tests/unit/test_x.py::TestY::test_z` becomes a single name, so a
    cassette directory can be listed and diffed at a glance instead of walked.
    """
    return _UNSAFE.sub("_", test_id).strip("_")


@dataclass(frozen=True)
class Interaction:
    """One recorded model call."""

    ordinal: int
    fingerprint: str
    request: dict[str, Any]
    response: Any
    model: str = ""
    provider: str = ""
    recorded_at: str = ""
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True, default=str)

    @staticmethod
    def from_line(line: str) -> Interaction:
        data = json.loads(line)
        known = set(Interaction.__dataclass_fields__)
        return Interaction(**{k: v for k, v in data.items() if k in known})


class Cassette:
    """The recorded interactions for one test."""

    def __init__(self, store: CassetteStore, test_id: str) -> None:
        self.store = store
        self.test_id = test_id
        self.name = test_id_to_name(test_id)
        self._loaded: list[Interaction] | None = None

    # ---- reading -------------------------------------------------------------------------

    def exists(self) -> bool:
        return self.store.exists(self.name)

    def load(self) -> list[Interaction]:
        """Every interaction in recorded order (cached — a cassette is immutable while replaying)."""
        if self._loaded is None:
            self._loaded = [Interaction.from_line(x) for x in self.store.read_lines(self.name)]
        return self._loaded

    def get(self, ordinal: int) -> Interaction | None:
        """The interaction recorded as call number `ordinal` (1-based), or None if not recorded."""
        return next((i for i in self.load() if i.ordinal == ordinal), None)

    def __len__(self) -> int:
        return len(self.load())

    # ---- writing -------------------------------------------------------------------------

    def append(self, interaction: Interaction) -> None:
        """Append one interaction, durably — a crashed recording keeps everything already written."""
        self.store.append_line(self.name, interaction.to_line())
        if self._loaded is not None:
            self._loaded.append(interaction)

    def truncate_from(self, ordinal: int) -> None:
        """Drop the interaction at `ordinal` AND every one after it.

        A stale interaction poisons its tail. In an agentic loop call N+1's prompt CONTAINS call N's
        response, so once N is re-answered the conversation diverges and every later recording is a
        reply to a question that will no longer be asked. Keeping them would replay answers from an
        abandoned branch."""
        kept = [i for i in self.load() if i.ordinal < ordinal]
        self.store.delete(self.name)
        self._loaded = []
        for interaction in kept:
            self.append(interaction)

    def truncate(self) -> None:
        """Drop the cassette, so a re-record starts clean rather than half-overwritten."""
        self.store.delete(self.name)
        self._loaded = None

    # ---- inspection ----------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        items = self.load()
        return {
            "test_id": self.test_id,
            "name": self.name,
            "interactions": len(items),
            "bytes": self.store.size_bytes(self.name),
            "models": sorted({i.model for i in items if i.model}),
        }
