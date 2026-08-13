"""The LiteLLM seam — one integration standing in for a per-provider adapter set."""

from __future__ import annotations

import sys
import types

import pytest

from llmtivo import MemoryStore, Mode, Recorder
from llmtivo.litellm import litellm_request, patched_litellm


@pytest.fixture
def fake_litellm(monkeypatch):
    """A stand-in for the real `litellm` module, injected for the duration of a test."""
    calls: list[dict] = []
    mod = types.ModuleType("litellm")

    def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "live-" + kwargs["messages"][0]["content"]}}]}

    def embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    mod.completion = completion
    mod.embedding = embedding
    mod.calls = calls
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


def test_the_request_builder_reads_the_model_from_the_call_not_the_instance():
    """LiteLLM is keyword-driven: `completion(model=..., messages=[...])`. A chat CLIENT keeps the
    model on the instance; LiteLLM puts it in the call, and reading the wrong one records `model=""`
    on every interaction."""
    req = litellm_request(
        None,
        (),
        {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        },
    )
    assert req["model"] == "anthropic/claude-sonnet-4-6"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in req, "sampling detail is not part of the question asked"


def test_the_request_builder_falls_back_to_the_instance_for_chat_litellm():
    class ChatLike:
        model = "groq/llama-3"

    req = litellm_request(ChatLike(), ([{"role": "user", "content": "hi"}],), {})
    assert req["model"] == "groq/llama-3"
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_records_a_litellm_call_then_replays_it(fake_litellm):
    import litellm

    store = MemoryStore()
    with patched_litellm(Recorder(store, "t::a", mode=Mode.RECORD)):
        out = litellm.completion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert out["choices"][0]["message"]["content"] == "live-hi"
    assert len(fake_litellm.calls) == 1

    with patched_litellm(Recorder(store, "t::a", mode=Mode.REPLAY)):
        again = litellm.completion(
            model="openai/gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
    assert again == out
    assert len(fake_litellm.calls) == 1, "replay must not reach litellm"


def test_chat_and_embedding_share_one_tape_in_real_call_order(fake_litellm):
    """One ordinal sequence across entry points, so the tape is an account of the RUN."""
    import litellm

    store = MemoryStore()
    with patched_litellm(Recorder(store, "t::b", mode=Mode.RECORD)) as rec:
        litellm.completion(model="m", messages=[{"role": "user", "content": "first"}])
        litellm.embedding(model="e", input="second")
        litellm.completion(model="m", messages=[{"role": "user", "content": "third"}])
    assert [i.ordinal for i in rec.cassette.load()] == [1, 2, 3]
    assert rec.cassette.load()[1].model == "e"


def test_the_same_prompt_twice_replays_two_DIFFERENT_responses(fake_litellm):
    """The case content-addressed matching cannot express, and the reason ordinal keying was chosen.

    best-of-N sends an IDENTICAL prompt twice at a higher temperature precisely to get diverse
    drafts. A recorder that matches on request content serves the first response both times and
    silently defeats it."""
    import litellm

    store = MemoryStore()
    responses = iter(["draft A", "draft B"])
    fake_litellm.completion = lambda **kw: {"draft": next(responses)}

    with patched_litellm(Recorder(store, "t::c", mode=Mode.RECORD)):
        litellm.completion(model="m", messages=[{"role": "user", "content": "same prompt"}])
        litellm.completion(model="m", messages=[{"role": "user", "content": "same prompt"}])

    with patched_litellm(Recorder(store, "t::c", mode=Mode.REPLAY)):
        first = litellm.completion(model="m", messages=[{"role": "user", "content": "same prompt"}])
        second = litellm.completion(
            model="m", messages=[{"role": "user", "content": "same prompt"}]
        )
    assert first["draft"] == "draft A"
    assert second["draft"] == "draft B", "both drafts are reachable — order, not content, addresses"


def test_a_missing_litellm_fails_loudly_rather_than_recording_nothing(monkeypatch):
    """A silent no-op would look exactly like a test whose calls WERE recorded — the precise
    failure this library exists to prevent."""
    monkeypatch.setitem(sys.modules, "litellm", None)
    monkeypatch.setitem(sys.modules, "langchain_litellm", None)
    with pytest.raises(RuntimeError, match="neither litellm nor langchain_litellm"):
        with patched_litellm(Recorder(MemoryStore(), "t::d", mode=Mode.RECORD)):
            pass


def test_the_original_functions_are_restored(fake_litellm):
    import litellm

    before = litellm.completion
    with pytest.raises(ValueError, match="boom"):
        with patched_litellm(Recorder(MemoryStore(), "t::e", mode=Mode.RECORD)):
            raise ValueError("boom")
    assert litellm.completion is before
