"""Embeddings — model calls that are not chat models, and escape a chat-model seam.

`OpenAIEmbeddings` is an `Embeddings`, not a `BaseChatModel`, so patching chat models leaves it
reaching the network on every replay: billed, non-deterministic, and invisible. Retrieval-augmented
pipelines embed constantly — a lessons index, a spec matcher, any vector recall — so this is not an
edge case, it is most of the calls in a RAG-shaped run.

Found by the network guard on a real pipeline, which is what the guard is for: the chat calls all
replayed and an embedding still called out.
"""

from __future__ import annotations

import asyncio

import pytest

from llmtivo import MemoryStore, Mode, Recorder

pytest.importorskip("langchain_core")

from langchain_core.embeddings import Embeddings

from llmtivo.langchain import patched_langchain


class CountingEmbeddings(Embeddings):
    """A stand-in that records how many times it really ran."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(t)), 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # the real OpenAI implementation delegates like this, which is why nesting must not
        # record the same call twice
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def test_an_embedding_is_recorded_and_replayed():
    store = MemoryStore()
    live = CountingEmbeddings()
    with patched_langchain(Recorder(store, "emb::one", mode=Mode.RECORD)):
        vector = live.embed_query("kotlin multiplatform")
    assert live.calls == 1

    replayed_client = CountingEmbeddings()
    with patched_langchain(Recorder(store, "emb::one", mode=Mode.REPLAY)):
        replayed = replayed_client.embed_query("kotlin multiplatform")

    assert replayed == vector
    assert replayed_client.calls == 0, "replay re-ran the embedding for real"


def test_embed_query_records_ONE_interaction_not_two():
    """`embed_query` delegates to `embed_documents`, so intercepting both layers would record every
    embedding twice, double the ordinals, and replay the inner recording to the outer caller."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "emb::once", mode=Mode.RECORD)) as rec:
        CountingEmbeddings().embed_query("one call")
    assert rec.stats.recorded == 1, f"recorded {rec.stats.recorded}"
    assert [i.ordinal for i in rec.cassette.load()] == [1]


def test_embed_documents_is_recorded_directly_too():
    store = MemoryStore()
    live = CountingEmbeddings()
    with patched_langchain(Recorder(store, "emb::docs", mode=Mode.RECORD)) as rec:
        vectors = live.embed_documents(["a", "bb", "ccc"])
    assert rec.stats.recorded == 1
    with patched_langchain(Recorder(store, "emb::docs", mode=Mode.REPLAY)):
        assert CountingEmbeddings().embed_documents(["a", "bb", "ccc"]) == vectors


def test_an_async_embedding_replays():
    store = MemoryStore()
    live = CountingEmbeddings()
    with patched_langchain(Recorder(store, "emb::async", mode=Mode.RECORD)):
        vector = asyncio.run(live.aembed_query("async text"))

    cold = CountingEmbeddings()
    with patched_langchain(Recorder(store, "emb::async", mode=Mode.REPLAY)):
        assert asyncio.run(cold.aembed_query("async text")) == vector
    assert cold.calls == 0


def test_different_text_is_a_different_question():
    store = MemoryStore()
    with patched_langchain(Recorder(store, "emb::fp", mode=Mode.RECORD)):
        CountingEmbeddings().embed_query("the original text")

    from llmtivo.modes import FingerprintDrift

    with pytest.raises(FingerprintDrift):
        with patched_langchain(Recorder(store, "emb::fp", mode=Mode.REPLAY)):
            CountingEmbeddings().embed_query("an entirely different text")


def test_chat_and_embeddings_share_one_tape_in_call_order():
    """A RAG turn is embed -> retrieve -> chat. One ordinal sequence keeps it an account of the run."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    store = MemoryStore()
    with patched_langchain(Recorder(store, "emb::mixed", mode=Mode.RECORD)) as rec:
        CountingEmbeddings().embed_query("recall this")
        GenericFakeChatModel(messages=iter([AIMessage("answered")])).invoke("using the recall")

    tape = rec.cassette.load()
    assert [i.ordinal for i in tape] == [1, 2]
    assert tape[0].request.get("embedding") is True, tape[0].request


def test_an_embeddings_class_IMPORTED_LATER_is_still_intercepted():
    """Walking `__subclasses__()` at patch time only sees what is already imported.

    Providers are imported lazily on purpose — `from langchain_openai import OpenAIEmbeddings`
    inside the function that embeds, so importing the module needs no key. That class therefore does
    not exist when the patch goes in, the walk finds nothing, and every embedding reaches the
    network while the chat calls all replay. Classes defined DURING the block have to be caught too.
    """
    store = MemoryStore()
    with patched_langchain(Recorder(store, "emb::late", mode=Mode.RECORD)) as rec:

        class LateEmbeddings(Embeddings):  # defined after the patch, as a lazy import would be
            ran = 0

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                LateEmbeddings.ran += 1
                return [[1.0, 2.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

        vector = LateEmbeddings().embed_query("late import")

    assert rec.stats.recorded == 1, "a lazily imported embeddings class escaped the seam"
    assert LateEmbeddings.ran == 1

    with patched_langchain(Recorder(store, "emb::late", mode=Mode.REPLAY)):
        assert LateEmbeddings().embed_query("late import") == vector
    assert LateEmbeddings.ran == 1, "replay re-ran the lazily imported embedder"
