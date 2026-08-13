"""MCP tools, executed for real over stdio, recorded and replayed.

An MCP tool is the strongest case for recording tool executions. It is not a function in the test
process — it is a JSON-RPC call to a separate server, which may be a subprocess, a container, or a
remote service, and which may do anything at all. `langchain-mcp-adapters` surfaces them as ordinary
`BaseTool` instances, so the same seam that catches a local `@tool` catches these, and the side
effect happens once, when recording.

The server here (`mcp_server.py`) is spawned over stdio and appends to a file, so "did the tool
really run?" is answered by the filesystem rather than by a mock's call count.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from llmtivo.langchain import patched_langchain

from .conftest import requires
from .test_langchain_live import openai_model

pytestmark = pytest.mark.integration

SERVER = Path(__file__).parent / "mcp_server.py"


def mcp_client(log: Path | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # The server runs as a SUBPROCESS, and MCP's stdio transport passes a filtered "safe"
    # environment rather than inheriting this process's — so the log path has to be handed over
    # explicitly or the side effect silently lands nowhere.
    env = dict(os.environ)
    if log is not None:
        env["LLMTIVO_MCP_LOG"] = str(log)

    return MultiServerMCPClient(
        {
            "testserver": {
                "command": sys.executable,
                "args": [str(SERVER)],
                "transport": "stdio",
                "env": env,
            }
        }
    )


async def load_tools(log: Path | None = None) -> list[BaseTool]:
    return await mcp_client(log).get_tools()


def test_mcp_tools_load_as_langchain_tools() -> None:
    """The premise of the whole file: an MCP tool IS a `BaseTool`, so it goes through the tool seam
    with no MCP-specific code in LLMTivo at all."""
    tools = asyncio.run(load_tools())
    names = sorted(t.name for t in tools)
    assert names == ["add", "note"], names
    assert all(isinstance(t, BaseTool) for t in tools)


def test_an_mcp_tool_execution_replays_without_reaching_the_server(llmtivo, recording) -> None:
    """The side effect is a file append, so the filesystem — not a mock — says whether the tool
    really ran. Replaying must leave the file untouched."""
    log = Path(tempfile.gettempdir()) / "llmtivo-mcp-side-effect.log"
    before = log.read_text().count("\n") if log.exists() else 0
    tools = {t.name: t for t in asyncio.run(load_tools(log))}

    with patched_langchain(llmtivo.recorder):
        result = asyncio.run(tools["note"].ainvoke({"text": "recorded once"}))

    assert "noted: recorded once" in str(result), result
    after = log.read_text().count("\n") if log.exists() else 0
    if recording:
        assert after == before + 1, "recording must actually reach the MCP server"
    else:
        assert after == before, "replay re-executed the MCP tool — the side effect happened again"


def test_a_real_model_drives_an_mcp_tool(llmtivo, recording) -> None:
    """The whole chain: a real model chooses an MCP tool, the MCP server really computes it, and
    the model answers from the result. All three interactions on one tape."""
    if recording:
        requires("OPENAI_API_KEY")

    tools = {t.name: t for t in asyncio.run(load_tools())}
    add = tools["add"]
    question = HumanMessage("What is 17 plus 25? Use the add tool, then reply with just the number.")

    with patched_langchain(llmtivo.recorder):
        model = openai_model().bind_tools([add])
        asked = model.invoke([question])
        assert asked.tool_calls, "the model did not call the MCP tool"
        observed = asyncio.run(add.ainvoke(asked.tool_calls[0]))
        answer = model.invoke([question, asked, observed])

    assert "42" in str(observed.content), observed.content
    assert "42" in str(answer.content), answer.content

    tape = llmtivo.recorder.cassette.load()
    kinds = [i.request.get("tool") or f"model:{i.model}" for i in tape]
    assert kinds[1] == "add", f"the MCP execution is not on the tape: {kinds}"
    assert len(tape) == 3, kinds
