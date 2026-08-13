"""The interception seam — the patch must be narrow, restored, and honest about double-patching."""

from __future__ import annotations

import pytest

from llmtivo import MemoryStore, Mode, Recorder
from llmtivo.intercept import patched, patched_all


class FakeChat:
    """Stands in for a real client. `calls` counts what actually reached it."""

    calls = 0

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    def invoke(self, messages, **kwargs):
        FakeChat.calls += 1
        return f"live:{messages[0]['content']}"


class FakeEmbed:
    calls = 0

    def __init__(self) -> None:
        self.model = "embed-3"

    def embed(self, messages, **kwargs):
        FakeEmbed.calls += 1
        return [0.1, 0.2]


@pytest.fixture(autouse=True)
def _reset():
    FakeChat.calls = FakeEmbed.calls = 0


def _msgs(text: str):
    return [{"role": "user", "content": text}]


def test_records_through_the_patch_then_replays_without_reaching_the_client():
    store = MemoryStore()
    with patched(FakeChat, "invoke", Recorder(store, "t::a", mode=Mode.RECORD)):
        assert FakeChat().invoke(_msgs("hi")) == "live:hi"
    assert FakeChat.calls == 1

    with patched(FakeChat, "invoke", Recorder(store, "t::a", mode=Mode.REPLAY)):
        assert FakeChat().invoke(_msgs("hi")) == "live:hi"
    assert FakeChat.calls == 1, "replay must not reach the client"


def test_the_original_method_is_restored_even_when_the_test_raises():
    """A failing test must not leak a patched class into the next one."""
    before = FakeChat.invoke
    with pytest.raises(ValueError, match="boom"):
        with patched(FakeChat, "invoke", Recorder(MemoryStore(), "t::b", mode=Mode.RECORD)):
            raise ValueError("boom")
    assert FakeChat.invoke is before


def test_a_double_patch_is_refused_rather_than_silently_nested():
    """Nesting would record every call twice and quietly corrupt the tape."""
    rec = Recorder(MemoryStore(), "t::c", mode=Mode.RECORD)
    with patched(FakeChat, "invoke", rec):
        with pytest.raises(RuntimeError, match="already intercepted"):
            with patched(FakeChat, "invoke", rec):
                pass


def test_patch_all_puts_several_clients_on_one_tape_in_real_call_order():
    """A run that talks to a chat model and an embedder records ONE faithful account of the run,
    not two per-client fragments."""
    store = MemoryStore()
    targets = [(FakeChat, "invoke"), (FakeEmbed, "embed")]
    with patched_all(targets, Recorder(store, "t::d", mode=Mode.RECORD)):
        FakeChat().invoke(_msgs("first"))
        FakeEmbed().embed(_msgs("second"))
        FakeChat().invoke(_msgs("third"))

    with patched_all(targets, Recorder(store, "t::d", mode=Mode.REPLAY)):
        assert FakeChat().invoke(_msgs("first")) == "live:first"
        assert FakeEmbed().embed(_msgs("second")) == [0.1, 0.2]
        assert FakeChat().invoke(_msgs("third")) == "live:third"
    assert FakeChat.calls == 2 and FakeEmbed.calls == 1, "replay reached neither client again"


def test_patch_all_restores_every_target():
    chat, embed = FakeChat.invoke, FakeEmbed.embed
    with pytest.raises(ValueError, match="boom"):
        with patched_all(
            [(FakeChat, "invoke"), (FakeEmbed, "embed")],
            Recorder(MemoryStore(), "t::e", mode=Mode.RECORD),
        ):
            raise ValueError("boom")
    assert FakeChat.invoke is chat and FakeEmbed.embed is embed


def test_the_model_name_is_captured_from_the_client():
    store = MemoryStore()
    rec = Recorder(store, "t::f", mode=Mode.RECORD)
    with patched(FakeChat, "invoke", rec):
        FakeChat(model="claude-opus-4-8").invoke(_msgs("hi"))
    assert rec.cassette.load()[0].model == "claude-opus-4-8"


# ── async: the half my sync-only wrapper could not touch ──────────────────────────────────────────


class FakeAsyncChat:
    """`acompletion` / `ainvoke` are coroutine functions. Wrapping one synchronously puts the
    COROUTINE OBJECT on the tape — it fails to serialise and the call is never awaited."""

    calls = 0

    def __init__(self, model: str = "sonnet") -> None:
        self.model = model

    async def ainvoke(self, messages, **kwargs):
        FakeAsyncChat.calls += 1
        return "live:" + messages[0]["content"]


def test_an_async_method_records_the_AWAITED_value_not_the_coroutine():
    import asyncio

    FakeAsyncChat.calls = 0
    store = MemoryStore()
    rec = Recorder(store, "t::async", mode=Mode.RECORD)
    with patched(FakeAsyncChat, "ainvoke", rec):
        out = asyncio.run(FakeAsyncChat().ainvoke(_msgs("hi")))

    assert out == "live:hi"
    assert rec.cassette.load()[0].response == "live:hi", "the awaited value is what lands on tape"


def test_an_async_method_replays_without_reaching_the_client():
    import asyncio

    FakeAsyncChat.calls = 0
    store = MemoryStore()
    with patched(FakeAsyncChat, "ainvoke", Recorder(store, "t::async2", mode=Mode.RECORD)):
        asyncio.run(FakeAsyncChat().ainvoke(_msgs("hi")))
    assert FakeAsyncChat.calls == 1

    with patched(FakeAsyncChat, "ainvoke", Recorder(store, "t::async2", mode=Mode.REPLAY)):
        again = asyncio.run(FakeAsyncChat().ainvoke(_msgs("hi")))
    assert again == "live:hi"
    assert FakeAsyncChat.calls == 1, "replay must not await the real client"


def test_sync_and_async_share_one_ordinal_sequence():
    """An agent that mixes blocking and awaited calls records ONE account of the run."""
    import asyncio

    store = MemoryStore()
    rec = Recorder(store, "t::mixed", mode=Mode.RECORD)
    with patched_all([(FakeChat, "invoke"), (FakeAsyncChat, "ainvoke")], rec):
        FakeChat().invoke(_msgs("one"))
        asyncio.run(FakeAsyncChat().ainvoke(_msgs("two")))
        FakeChat().invoke(_msgs("three"))
    assert [i.ordinal for i in rec.cassette.load()] == [1, 2, 3]
    assert rec.cassette.load()[1].response == "live:two"
