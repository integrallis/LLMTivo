"""A tiny MCP server, run as a subprocess over stdio by the MCP integration test.

Deliberately trivial and side-effecting: `note` appends to a file so the test can prove whether the
tool REALLY ran, which is the difference between replaying a tool and re-executing it.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("llmtivo-test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def note(text: str) -> str:
    """Record a note. Appends to the file named by LLMTIVO_MCP_LOG — a real side effect."""
    path = os.environ.get("LLMTIVO_MCP_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return "noted: " + text


if __name__ == "__main__":
    mcp.run(transport="stdio")
