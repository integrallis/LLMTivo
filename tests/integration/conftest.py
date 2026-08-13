"""Integration tests against REAL providers, shipped as replayable tapes.

These are the tests that prove LLMTivo works on the thing it exists for. Everything in
`tests/` runs against fakes: fast, deterministic, and unable to notice that a provider changed a
response shape, that `ChatDeepSeek` puts reasoning content somewhere unexpected, or that a
tool-calling round trip loses `tool_calls` on the way back. These call the real APIs once, commit
the tape, and replay for free forever after — which is also the clearest possible demonstration of
what the library is for.

    pytest tests/integration                     # replay: free, offline, no keys needed
    pytest tests/integration --llmtivo=record    # re-record. costs money, needs keys.

The cassettes live beside these tests and ARE committed. They are the artifact under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmtivo.modes import Mode
from llmtivo.plugin import LLMTivo
from llmtivo.store import CassetteStore, FileStore

CASSETTES = Path(__file__).parent / "cassettes"

#: Every provider credential these tests can use, and the placeholder that stands in on replay.
#: A client constructor raises without SOMETHING here, so replaying a committed tape on a machine
#: with no keys — a contributor's laptop, CI — must not require inventing one per provider.
_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
)
_PLACEHOLDER = "replay-no-key-needed"


def pytest_configure(config: pytest.Config) -> None:
    """Load real credentials when they exist, and stand in placeholders when they do not."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    for name in _PROVIDER_KEYS:
        # `setdefault` is NOT enough: CI sets these to the EMPTY STRING to prove replay needs no
        # credential, and an empty value is present-but-useless — the client then raises "Missing
        # credentials" instead of being handed a placeholder. Blank counts as absent.
        if not os.environ.get(name, "").strip():
            os.environ[name] = _PLACEHOLDER
    # langchain_google_genai reads GOOGLE_API_KEY; the .env supplies GEMINI_API_KEY
    if os.environ.get("GOOGLE_API_KEY", "").strip() in ("", _PLACEHOLDER):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def requires(*env_names: str) -> None:
    """Fail — never skip — when a recording run is missing the credential it needs.

    A skip here would be the exact failure this library exists to prevent: a green summary line for
    a provider that was never contacted. On replay no credential is needed at all.
    """
    missing = [n for n in env_names if os.environ.get(n, _PLACEHOLDER) == _PLACEHOLDER]
    if missing:
        pytest.fail(f"recording needs {', '.join(missing)} in .env — set it, or run in replay mode")


@pytest.fixture(scope="session")
def llmtivo_store(pytestconfig: pytest.Config) -> CassetteStore:
    """Integration tapes live beside these tests, not in the unit-test cassette directory."""
    return FileStore(CASSETTES)


@pytest.fixture
def recording(llmtivo: LLMTivo) -> bool:
    """True when this run is talking to real providers."""
    return llmtivo.mode in (Mode.RECORD, Mode.RECORD_NEW, Mode.REPLAY_OR_RECORD)
