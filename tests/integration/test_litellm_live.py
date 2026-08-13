"""The LiteLLM seam against real providers.

The claim the LiteLLM seam makes is that ONE patch covers every provider LiteLLM covers. A fake
`litellm` module cannot test that claim — it only proves the wrapper calls what it was told to.
These call OpenAI, Anthropic, Gemini and DeepSeek for real, through one seam, and the committed
tapes are the evidence.
"""

from __future__ import annotations

import pytest

from llmtivo.litellm import patched_litellm

from .conftest import requires

pytestmark = pytest.mark.integration

#: One cheap, fast model per provider. LiteLLM's own `provider/model` naming.
PROVIDERS = [
    ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
    ("anthropic/claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY"),
    ("gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
]


@pytest.mark.parametrize("model,key", PROVIDERS, ids=[p[0] for p in PROVIDERS])
def test_a_real_completion_records_and_replays(llmtivo, recording, model: str, key: str) -> None:
    """Four providers, one seam, one code path — which is the whole argument for the LiteLLM
    integration over an adapter per provider."""
    import litellm

    if recording:
        requires(key)

    with patched_litellm(llmtivo.recorder):
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=256,  # gemini-2.5 is a THINKING model: a small budget is
            temperature=0,  # spent on reasoning and leaves nothing for the answer
        )

    content = response["choices"][0]["message"]["content"]
    assert "PONG" in content.upper(), f"{model} returned {content!r}"


def test_one_tape_holds_two_providers_in_call_order(llmtivo, recording) -> None:
    """A run that mixes providers records them in the order they happened, so the tape is an
    account of the RUN rather than a set of per-provider fragments."""
    import litellm

    if recording:
        requires("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

    with patched_litellm(llmtivo.recorder):
        first = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Name one primary colour, one word."}],
            max_tokens=8,
            temperature=0,
        )
        second = litellm.completion(
            model="anthropic/claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Name one primary colour, one word."}],
            max_tokens=8,
            temperature=0,
        )

    tape = llmtivo.recorder.cassette.load()
    assert [i.ordinal for i in tape] == [1, 2]
    assert tape[0].model.startswith("openai/"), tape[0].model
    assert tape[1].model.startswith("anthropic/"), tape[1].model
    assert first["choices"][0]["message"]["content"]
    assert second["choices"][0]["message"]["content"]


def test_an_async_completion_records_the_awaited_value(llmtivo, recording) -> None:
    """`acompletion` returns a coroutine. Recording it without awaiting would put a coroutine
    object on the tape and never make the call — the failure the async branch exists to prevent."""
    import asyncio

    import litellm

    if recording:
        requires("OPENAI_API_KEY")

    async def ask() -> str:
        response = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly: ASYNC"}],
            max_tokens=16,
            temperature=0,
        )
        return str(response["choices"][0]["message"]["content"])

    with patched_litellm(llmtivo.recorder):
        content = asyncio.run(ask())

    assert "ASYNC" in content.upper(), content
    assert llmtivo.recorder.cassette.load()[0].response, "the awaited value reached the tape"
