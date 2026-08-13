"""Real tool calling, both halves, against real providers.

A tool call is two events and LLMTivo has to survive both:

  1. the model ASKS — `tool_calls` on the message, which an agentic loop branches on. A tape that
     loses them replays a model that suddenly stopped calling tools.
  2. the tool RUNS — not a model call at all, and the reason the tool seam exists.

Recorded against OpenAI and Anthropic, whose tool-calling wire formats differ, so the round trip is
tested where it is actually hard rather than on a fake that echoes back what it was handed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from llmtivo.langchain import patched_langchain

from .conftest import requires
from .test_langchain_live import anthropic_model, openai_model

pytestmark = pytest.mark.integration


#: How many times the tool REALLY ran. `StructuredTool` is a pydantic model and refuses stray
#: attributes, so the counter lives beside it.
RAN: dict[str, int] = {"get_weather": 0}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    RAN["get_weather"] += 1
    return f"18C and overcast in {city}"


TOOL_MODELS = [
    ("openai", openai_model, "OPENAI_API_KEY"),
    ("anthropic", anthropic_model, "ANTHROPIC_API_KEY"),
]


@pytest.mark.parametrize("name,build,key", TOOL_MODELS, ids=[m[0] for m in TOOL_MODELS])
def test_a_real_model_asks_for_a_tool_and_the_request_survives(
    llmtivo, recording, name, build, key
) -> None:
    """The model's REQUEST — name and arguments — has to round-trip, or the loop takes a different
    branch on replay than the one that was recorded."""
    if recording:
        requires(key)

    with patched_langchain(llmtivo.recorder):
        response = (
            build()
            .bind_tools([get_weather])
            .invoke([HumanMessage("What is the weather in Lisbon? Use the tool.")])
        )

    assert isinstance(response, AIMessage)
    assert response.tool_calls, f"{name} did not ask for the tool"
    call = response.tool_calls[0]
    assert call["name"] == "get_weather"
    assert "lisbon" in str(call["args"]).lower(), call["args"]
    assert call.get("id"), "the call id is what the ToolMessage answers"


@pytest.mark.parametrize("name,build,key", TOOL_MODELS, ids=[m[0] for m in TOOL_MODELS])
def test_the_full_round_trip_model_tool_model(llmtivo, recording, name, build, key) -> None:
    """Ask -> run the tool -> feed the result back -> the model answers from it. Three interactions
    on one tape, in the order they happened, and the tool runs only while recording."""
    if recording:
        requires(key)

    before = RAN["get_weather"]
    question = HumanMessage("What is the weather in Lisbon? Use the tool, then answer in one line.")

    with patched_langchain(llmtivo.recorder):
        model = build().bind_tools([get_weather])
        asked = model.invoke([question])
        call = asked.tool_calls[0]
        observed = get_weather.invoke(call)
        answer = model.invoke([question, asked, observed])

    assert isinstance(observed, ToolMessage), type(observed).__name__
    assert "18C" in observed.content, observed.content
    assert "18" in str(answer.content) or "overcast" in str(answer.content).lower(), answer.content

    tape = llmtivo.recorder.cassette.load()
    assert [i.ordinal for i in tape] == [1, 2, 3]
    assert tape[1].request.get("tool") == "get_weather", (
        f"call 2 should be the tool: {tape[1].request}"
    )

    ran = RAN["get_weather"] - before
    if recording:
        assert ran == 1, "recording must actually run the tool"
    else:
        assert ran == 0, "replay ran the tool for real — the side effect happened again"
