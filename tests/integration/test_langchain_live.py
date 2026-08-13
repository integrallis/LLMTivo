"""The LangChain seam against real provider packages.

`ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI` and `ChatDeepSeek` each talk to a different
SDK and none of them goes through LiteLLM. The seam's claim is that one patch on `BaseChatModel`
covers all of them because none overrides the public entry points — a claim only real clients can
test, since a fake model is exactly the thing that inherits everything.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from llmtivo.langchain import patched_langchain

from .conftest import requires

pytestmark = pytest.mark.integration


def openai_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=256)


def anthropic_model():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=256)


def gemini_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def deepseek_model():
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=256)


MODELS = [
    ("openai", openai_model, "OPENAI_API_KEY"),
    ("anthropic", anthropic_model, "ANTHROPIC_API_KEY"),
    ("gemini", gemini_model, "GEMINI_API_KEY"),
    ("deepseek", deepseek_model, "DEEPSEEK_API_KEY"),
]


@pytest.mark.parametrize("name,build,key", MODELS, ids=[m[0] for m in MODELS])
def test_a_real_invoke_replays_as_an_AIMessage(llmtivo, recording, name, build, key) -> None:
    """The response must come back as the TYPE the live call returned. A cassette is JSON, so
    without a codec this would replay as a string and `.content` would raise."""
    if recording:
        requires(key)

    with patched_langchain(llmtivo.recorder):
        response = build().invoke([HumanMessage("Reply with exactly: PONG")])

    assert isinstance(response, AIMessage), f"{name} replayed a {type(response).__name__}"
    assert "PONG" in str(response.content).upper(), response.content


@pytest.mark.parametrize("name,build,key", MODELS[:2], ids=[m[0] for m in MODELS[:2]])
def test_a_real_stream_replays_chunk_by_chunk(llmtivo, recording, name, build, key) -> None:
    """`stream()` returns a generator, and a generator is not a response. Recording its return
    value would put a generator object on the tape and replay something that yields nothing."""
    if recording:
        requires(key)

    with patched_langchain(llmtivo.recorder):
        chunks = list(build().stream([HumanMessage("Count: one two three four five")]))

    assert len(chunks) > 1, f"{name} did not actually stream ({len(chunks)} chunk)"
    assert all(isinstance(c, AIMessageChunk) for c in chunks), "chunks replay as chunks"
    assert "".join(str(c.content) for c in chunks).strip()


def test_a_real_async_invoke_records_the_awaited_value(llmtivo, recording) -> None:
    if recording:
        requires("ANTHROPIC_API_KEY")

    with patched_langchain(llmtivo.recorder):
        response = asyncio.run(anthropic_model().ainvoke([HumanMessage("Reply with: ASYNC")]))

    assert isinstance(response, AIMessage)
    assert "ASYNC" in str(response.content).upper(), response.content


def test_four_providers_share_one_tape_in_call_order(llmtivo, recording) -> None:
    """One ordinal sequence across four different SDKs — the account of the run that a per-client
    recorder cannot produce."""
    if recording:
        requires("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY")

    prompt = [HumanMessage("Reply with one word: hello")]
    with patched_langchain(llmtivo.recorder):
        for _, build, _ in MODELS:
            assert build().invoke(prompt).content

    tape = llmtivo.recorder.cassette.load()
    assert [i.ordinal for i in tape] == [1, 2, 3, 4]
    assert len({i.model for i in tape}) == 4, (
        f"expected four distinct models: {[i.model for i in tape]}"
    )
