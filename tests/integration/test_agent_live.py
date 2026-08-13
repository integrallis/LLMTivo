"""A real agent loop with a real NETWORK tool.

This is the case the tool seam was built for. Tavily is a paid search API: non-deterministic, billed
per call, and its result feeds the next prompt — so without recording it, a replayed suite calls a
paid API on every run and the model's tape drifts because the search results moved, not because the
model changed.

The agent is `langchain.agents.create_agent`, the same LangGraph loop DeepAgents is built on, so
what is exercised here is the framework's own tool execution rather than a hand-rolled imitation.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from llmtivo.langchain import patched_langchain

from .conftest import requires
from .test_langchain_live import openai_model

pytestmark = pytest.mark.integration


def tavily_tool():
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=2)


def test_an_agent_loop_with_a_real_search_tool(llmtivo, recording) -> None:
    """Model asks -> LangGraph runs Tavily for real -> model answers from the results. On replay
    the whole loop comes off the tape: no OpenAI call, no Tavily call, no network at all."""
    from langchain.agents import create_agent

    if recording:
        requires("OPENAI_API_KEY", "TAVILY_API_KEY")

    with patched_langchain(llmtivo.recorder):
        agent = create_agent(openai_model(), [tavily_tool()])
        result = agent.invoke(
            {
                "messages": [
                    ("user", "Search for what Kotlin Multiplatform is, then summarise in one line.")
                ]
            }
        )

    messages = result["messages"]
    assert any(getattr(m, "tool_calls", None) for m in messages), "the agent never called the tool"
    final = messages[-1]
    assert isinstance(final, AIMessage) and str(final.content).strip(), final

    tape = llmtivo.recorder.cassette.load()
    kinds = [i.request.get("tool") or f"model:{i.model}" for i in tape]
    assert any(k == "tavily_search" for k in kinds), (
        f"the tool execution is not on the tape: {kinds}"
    )
    assert len(tape) >= 3, f"expected model -> tool -> model, got {kinds}"


def test_replaying_the_agent_touches_no_network(llmtivo, recording) -> None:
    """The guarantee that makes a recorded agent test worth having.

    Replay runs inside the network guard the plugin installs, so if ANY part of this loop reached a
    socket — an un-recorded model call, a retry, a telemetry ping — it would raise rather than
    quietly billing the account. Passing in replay mode IS the proof.
    """
    from langchain.agents import create_agent

    if recording:
        requires("OPENAI_API_KEY", "TAVILY_API_KEY")

    with patched_langchain(llmtivo.recorder):
        agent = create_agent(openai_model(), [tavily_tool()])
        result = agent.invoke(
            {"messages": [("user", "Search for the capital of Portugal and answer in one word.")]}
        )

    assert "lisbon" in str(result["messages"][-1].content).lower(), result["messages"][-1].content
    if not recording:
        assert llmtivo.recorder.stats.replayed > 0, "nothing was replayed — did the tape load?"
        assert llmtivo.recorder.stats.recorded == 0, "replay recorded something: it called out"
