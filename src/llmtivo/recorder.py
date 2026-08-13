"""The recorder — one test's tape, and the decision of where each call's answer comes from.

This is the whole state machine, deliberately kept away from the interception machinery so it can be
tested without patching anything: give it a request and a callable that would perform the real call,
and it returns the response plus a record of where that response came from.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from llmtivo.cassette import Cassette, Interaction
from llmtivo.keys import fingerprint
from llmtivo.modes import CassetteMiss, FingerprintDrift, Mode, resolve
from llmtivo.store import CassetteStore  # noqa: TC001 - runtime protocol, not just a hint

#: Distinguishes "no recording" from a recorded response that happens to be None.
_MISS = object()


@dataclass(frozen=True)
class _Decision:
    """Where one call's answer comes from: a replayed value, or _MISS meaning perform it."""

    ordinal: int
    fingerprint: str
    replay: Any


@dataclass
class Stats:
    """What happened during a test — asserted on in CI, printed in the summary."""

    replayed: int = 0
    recorded: int = 0
    drifted: int = 0
    #: Recorded interactions the run never reached — the tape is longer than the code.
    unplayed: int = 0

    @property
    def total(self) -> int:
        return self.replayed + self.recorded


class Codec:
    """How a response becomes JSON and comes back as the SAME TYPE it was.

    A cassette is JSON, and `json.dumps(default=str)` turns any unknown object into its `repr`.
    Without a codec a replayed `AIMessage` arrives as a STRING, and the application's next line —
    `response.content`, `response.tool_calls` — raises `AttributeError`. Framework adapters supply
    one; a client whose responses are already plain dicts needs none.
    """

    def __init__(
        self,
        encode: Callable[[Any], Any] = lambda v: v,
        decode: Callable[[Any], Any] = lambda v: v,
    ) -> None:
        self.encode = encode
        self.decode = decode


#: The no-op codec: responses that are already JSON-safe go on the tape as they are.
IDENTITY = Codec()


class Recorder:
    """Serves one test's model calls from tape, from the network, or refuses."""

    def __init__(
        self,
        store: CassetteStore,
        test_id: str,
        mode: Mode = Mode.RECORD_NEW,
        *,
        on_drift: Callable[[str, str, str], None] | None = None,
        secrets: Sequence[str | None] = (),
    ) -> None:
        # empty entries are dropped: an unset env var read as "" would otherwise match between
        # every character of every request and redact the whole tape
        self.secrets = tuple(s for s in secrets if s)
        self.cassette = Cassette(store, test_id)
        self.requested_mode = mode
        self.mode = resolve(mode, self.cassette.exists())
        self.on_drift = on_drift
        self.stats = Stats()
        self._ordinal = 0
        self._lock = threading.Lock()
        #: Ordinals already served this run — a recording answers exactly one call.
        self._served: set[int] = set()
        if self.mode is Mode.RECORD:
            # a re-record starts from nothing: half-overwritten tape is worse than no tape
            self.cassette.truncate()

    def _decide(self, request: dict[str, Any]) -> _Decision:
        """Where this call's answer comes from — the shared decision for both sync and async.

        Kept in ONE place on purpose: two copies of this branching would drift, and the async copy
        is the one nobody exercises until it matters.

        Held under a lock because real pipelines fan out — a thread pool over stories, an
        `asyncio.gather` over candidates. `self._ordinal += 1` is read-modify-write, so two workers
        could otherwise take the same ordinal and one recording would overwrite the other."""
        with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
        fp = fingerprint(request)

        if self.mode.may_replay:
            with self._lock:
                recorded = self._claim(ordinal, fp)
            if recorded is not None:
                self.stats.replayed += 1
                return _Decision(ordinal, fp, recorded.response)

            # Nothing on the tape answers this question, at this position or anywhere else.
            stale = self.cassette.get(ordinal)
            if stale is not None:
                # STALE. The tape answers a question this code no longer asks, so it is not evidence
                # of anything — order ADDRESSES an interaction, the fingerprint VALIDATES it.
                self.stats.drifted += 1
                detail = (
                    f"{self.cassette.test_id} call #{ordinal}: recorded for request "
                    f"{stale.fingerprint}, now asked {fp} — the prompt changed, so the recorded "
                    f"response is no longer an answer to it. Re-record this test."
                )
                if self.mode.strict:
                    raise FingerprintDrift(detail)
                if self.on_drift:
                    self.on_drift(self.cassette.test_id, stale.fingerprint, fp)
                self.cassette.truncate_from(ordinal)  # the tail replies to an abandoned branch

            if self.mode.strict:
                raise CassetteMiss(
                    f"{self.cassette.test_id} call #{ordinal} has no recording "
                    f"({len(self.cassette)} on tape). Re-record this test; replay mode will not "
                    f"call a paid model to cover a missing cassette."
                )

        if not self.mode.may_record:
            raise CassetteMiss(
                f"{self.cassette.test_id} call #{ordinal}: mode {self.mode} cannot record"
            )
        return _Decision(ordinal, fp, _MISS)

    def _claim(self, ordinal: int, fp: str) -> Interaction | None:
        """The recording that answers this question, consumed so nothing serves it twice.

        The one at `ordinal` wins when its fingerprint matches — order is still the primary
        address, which is what keeps two identical prompts (best-of-N) mapped to their two
        DIFFERENT responses. Otherwise any UNCONSUMED interaction with the same fingerprint is
        taken, which is how a concurrent run recovers: a call that was ordinal 3 while recording
        can be ordinal 5 while replaying purely because a thread was scheduled differently, and
        serving position 5's answer to it would be wrong in a way nothing downstream could detect.

        Caller holds the lock.
        """
        at = self.cassette.get(ordinal)
        if at is not None and at.fingerprint == fp and ordinal not in self._served:
            self._served.add(ordinal)
            return at
        for other in self.cassette.load():
            if other.fingerprint == fp and other.ordinal not in self._served:
                self._served.add(other.ordinal)
                return other
        return None

    def _record(
        self, ordinal: int, fp: str, request: dict[str, Any], response: Any, latency_ms: float
    ) -> None:
        with self._lock:
            self._append(ordinal, fp, request, response, latency_ms)

    def _append(
        self, ordinal: int, fp: str, request: dict[str, Any], response: Any, latency_ms: float
    ) -> None:
        self.cassette.append(
            Interaction(
                ordinal=ordinal,
                fingerprint=fp,
                request=_scrub(request, self.secrets),
                response=response,
                model=str(request.get("model", "")),
                recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                latency_ms=round(latency_ms, 1),
            )
        )
        self.stats.recorded += 1

    def call(
        self,
        request: dict[str, Any],
        perform: Callable[[], Any],
        *,
        codec: Codec = IDENTITY,
    ) -> Any:
        """Answer one model call. `perform` runs ONLY when the mode permits the network."""
        if self.mode is Mode.OFF:
            return perform()
        decision = self._decide(request)
        if decision.replay is not _MISS:
            return codec.decode(decision.replay)
        started = time.perf_counter()
        response = perform()
        self._record(
            decision.ordinal,
            decision.fingerprint,
            request,
            codec.encode(response),
            (time.perf_counter() - started) * 1000,
        )
        return response

    async def acall(
        self,
        request: dict[str, Any],
        perform: Callable[[], Any],
        *,
        codec: Codec = IDENTITY,
    ) -> Any:
        """The async twin of [call][llmtivo.recorder.Recorder.call].

        A coroutine cannot be recorded — it has to be AWAITED first, and the awaited value is what
        goes on the tape. Routing an async client through the sync path records the coroutine object
        itself, which fails to serialise and leaves the call never awaited."""
        if self.mode is Mode.OFF:
            return await perform()

        decision = self._decide(request)
        if decision.replay is not _MISS:
            return codec.decode(decision.replay)

        started = time.perf_counter()
        response = await perform()
        self._record(
            decision.ordinal,
            decision.fingerprint,
            request,
            codec.encode(response),
            (time.perf_counter() - started) * 1000,
        )
        return response

    def stream(
        self,
        request: dict[str, Any],
        perform: Callable[[], Iterable[Any]],
        *,
        codec: Codec = IDENTITY,
    ) -> Iterator[Any]:
        """Answer one STREAMED model call, chunk by chunk.

        A streaming entry point returns a generator, and a generator is not a response — recording
        its return value puts a generator OBJECT on the tape, and replay hands back something that
        yields nothing. The chunks are what happened, so the chunks are what gets recorded.

        Chunks are yielded THROUGH as they arrive rather than collected first, so a streamed test
        still observes streaming (a token-at-a-time UI, an early `break`) instead of one late burst.
        A consumer that abandons the stream early records only what it actually consumed: the tape
        is an account of the run, not of what the provider was prepared to send.
        """
        if self.mode is Mode.OFF:
            yield from perform()
            return

        decision = self._decide(request)
        if decision.replay is not _MISS:
            for chunk in decision.replay:
                yield codec.decode(chunk)
            return

        started = time.perf_counter()
        chunks: list[Any] = []
        try:
            for chunk in perform():
                chunks.append(codec.encode(chunk))
                yield chunk
        finally:
            # in `finally` so an abandoned or failed stream still records the part that DID happen
            self._record(
                decision.ordinal,
                decision.fingerprint,
                request,
                chunks,
                (time.perf_counter() - started) * 1000,
            )

    async def astream(
        self,
        request: dict[str, Any],
        perform: Callable[[], AsyncIterator[Any]],
        *,
        codec: Codec = IDENTITY,
    ) -> AsyncIterator[Any]:
        """The async twin of [stream][llmtivo.recorder.Recorder.stream]."""
        if self.mode is Mode.OFF:
            async for chunk in perform():
                yield chunk
            return

        decision = self._decide(request)
        if decision.replay is not _MISS:
            for chunk in decision.replay:
                yield codec.decode(chunk)
            return

        started = time.perf_counter()
        chunks: list[Any] = []
        try:
            async for chunk in perform():
                chunks.append(codec.encode(chunk))
                yield chunk
        finally:
            self._record(
                decision.ordinal,
                decision.fingerprint,
                request,
                chunks,
                (time.perf_counter() - started) * 1000,
            )

    @property
    def unplayed(self) -> int:
        """Recorded interactions the run never reached — live, not only after `finish()`.

        Every ordinal past the last one served is an answer the code stopped asking for."""
        if not self.mode.may_replay:
            return 0
        # counted by what was CONSUMED, not by how far the ordinal counter got: an out-of-order
        # run can serve interaction 8 on its first call, and it has still been played
        return max(0, len(self.cassette) - len(self._served))

    @property
    def all_played(self) -> bool:
        """Whether the run reached every interaction on the tape.

        The mirror of a cassette miss. A miss says the code asks something the tape lacks; an
        UNPLAYED interaction says the tape holds an answer the code stopped asking for. Both mean
        tape and code have diverged, and this library's whole position is that divergence must not
        pass quietly. Read after the run — `finish()` records the count on [Stats][llmtivo.recorder.Stats].
        """
        return self.unplayed == 0

    def discard(self) -> None:
        """Throw away a tape THIS recorder was recording — for a run that failed partway.

        Recording appends per call, so a run that dies at call 3 of 10 leaves a short tape that
        `exists()` reports as real: the next RECORD_NEW resolves to REPLAY and misses. A recording
        that never finished should leave nothing behind.

        Only touches a tape this recorder WROTE. A replay-only run destroying a committed cassette
        because its test raised would be a catastrophe dressed as cleanup."""
        if self.stats.recorded:
            self.cassette.truncate()

    def finish(self) -> None:
        """Call when the test ends CLEANLY — compacts a freshly recorded tape.

        Only after a clean finish: compacting a tape whose recording died halfway would rewrite a
        partial cassette as if it were complete."""
        self.stats.unplayed = self.unplayed  # stamped for the run report
        if self.stats.recorded:
            self.cassette.store.compact(self.cassette.name)


#: Request keys never written to a cassette — cassettes are committed to git.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "token",
        "access_token",
        "secret",
        "password",
        "headers",
        "extra_headers",
        "client",
        "http_client",
    }
)


#: Stands in for a redacted secret, so the tape still reads as a request rather than losing a field.
REDACTED = "<REDACTED>"


def _scrub(request: dict[str, Any], secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Drop credentials and unserialisable transport objects before anything is written.

    A cassette is a committed artifact. Anything secret that reaches it is leaked permanently by
    git history, so the filter runs on the way IN rather than being someone's review responsibility.

    TWO filters, because one is not enough. Dropping known KEYS catches a credential passed as
    `api_key=...`. It does nothing for one interpolated into a prompt, a system message or a URL,
    which is just text in `messages` — so known secret VALUES are substituted wherever they appear,
    however deeply nested. Betamax calls these placeholders and it is right to have them.
    """
    out: dict[str, Any] = {}
    for k, v in request.items():
        if k.lower() in _SECRET_KEYS:
            continue
        try:
            out[k] = v if isinstance(v, (str, int, float, bool, type(None))) else _plain(v)
        except Exception:  # an unserialisable extra is never worth failing a record
            out[k] = repr(v)
    return _redact(out, secrets) if secrets else out


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    """Replace every occurrence of each secret, at any depth."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    return value


def _plain(value: Any) -> Any:
    """A JSON-safe view of a nested value, preserving message shape."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "content") and hasattr(value, "type"):  # a LangChain message
        return {"role": str(value.type), "content": str(value.content)}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)
