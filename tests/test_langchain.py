"""The LangChain seam — one adapter covering LangChain, LangGraph and DeepAgents.

LangGraph and DeepAgents have no model-call surface of their own: a graph node calls a chat model,
and DeepAgents is a graph. Every one of them bottoms out in `BaseChatModel`, and neither
`ChatAnthropic` nor `ChatOpenAI` overrides its public entry points, so patching the base class
covers every provider package at once. Three frameworks, one seam.
"""

from __future__ import annotations

import asyncio

import pytest

from llmtivo import MemoryStore, Mode, Recorder

langchain_core = pytest.importorskip("langchain_core")

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402

from llmtivo.langchain import langchain_request, patched_langchain  # noqa: E402


def model_of(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(r) for r in replies]))


def test_an_ai_message_replays_as_a_MESSAGE_not_a_string():
    """The codec requirement, and the reason this adapter is not just `patched(BaseChatModel)`.

    A cassette is JSON, and the default serialiser turns any unknown object into its `repr`. Replay
    would then hand the application a STRING, and the very next line — `response.content` — raises
    `AttributeError`. A replayed response has to be the same TYPE the live call returned.
    """
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::msg", mode=Mode.RECORD)):
        live = model_of("hello").invoke("hi")

    with patched_langchain(Recorder(store, "lc::msg", mode=Mode.REPLAY)):
        replayed = model_of("SHOULD NOT BE REACHED").invoke("hi")

    assert isinstance(replayed, AIMessage), f"replayed a {type(replayed).__name__}"
    assert replayed.content == live.content == "hello"


def test_tool_calls_survive_the_round_trip():
    """An agentic loop branches on `tool_calls`. A tape that loses them replays a model that
    suddenly stopped calling tools, and the graph takes a different path than the one recorded."""
    call = {"name": "search", "args": {"q": "kotlin"}, "id": "call_1", "type": "tool_call"}
    store = MemoryStore()

    with patched_langchain(Recorder(store, "lc::tools", mode=Mode.RECORD)):
        GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=[call])])).invoke(
            "find"
        )

    with patched_langchain(Recorder(store, "lc::tools", mode=Mode.REPLAY)):
        replayed = model_of("unused").invoke("find")

    assert replayed.tool_calls == [call]


def test_replay_never_reaches_the_model():
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::cold", mode=Mode.RECORD)):
        model_of("recorded").invoke("q")

    exhausted = GenericFakeChatModel(messages=iter([]))  # any live call raises StopIteration
    with patched_langchain(Recorder(store, "lc::cold", mode=Mode.REPLAY)):
        assert exhausted.invoke("q").content == "recorded"


def test_a_tool_bound_model_is_recorded_ONCE():
    """What a DeepAgents/LangGraph agent actually calls is a `bind_tools(...)` wrapper, which
    delegates down to `BaseChatModel.invoke`. Patching both layers would record every call twice,
    double the ordinals, and replay the inner recording to the outer caller."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::bound", mode=Mode.RECORD)) as rec:
        model_of("bound reply").bind(stop=["x"]).invoke("via a binding")

    assert rec.stats.recorded == 1, "a bound call is ONE interaction, not two"
    assert [i.ordinal for i in rec.cassette.load()] == [1]


def test_a_streamed_call_replays_chunk_by_chunk():
    """LangGraph and DeepAgents stream by default. `stream()` returns a GENERATOR — recording its
    return value puts a generator object on the tape, and replay hands back something that yields
    nothing at all."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::stream", mode=Mode.RECORD)):
        live = [c.content for c in model_of("one two three").stream("q")]
    assert len(live) > 1, "the fake model streams token by token"

    with patched_langchain(Recorder(store, "lc::stream", mode=Mode.REPLAY)):
        chunks = list(GenericFakeChatModel(messages=iter([])).stream("q"))

    assert [c.content for c in chunks] == live
    assert all(isinstance(c, AIMessageChunk) for c in chunks), "chunks replay as chunks"


def test_an_async_stream_replays():
    async def drain(model):
        return [c.content async for c in model.astream("q")]

    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::astream", mode=Mode.RECORD)):
        live = asyncio.run(drain(model_of("alpha beta")))

    with patched_langchain(Recorder(store, "lc::astream", mode=Mode.REPLAY)):
        replayed = asyncio.run(drain(GenericFakeChatModel(messages=iter([]))))

    assert replayed == live


def test_ainvoke_and_invoke_share_one_ordinal_sequence():
    """A graph mixes awaited and blocking nodes; the tape is an account of the RUN, in real order."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::mixed", mode=Mode.RECORD)) as rec:
        m = model_of("first", "second", "third")
        m.invoke("a")
        asyncio.run(m.ainvoke("b"))
        m.invoke("c")

    assert [i.ordinal for i in rec.cassette.load()] == [1, 2, 3]
    assert [i.response["content"] for i in rec.cassette.load()] == ["first", "second", "third"]


def test_the_request_records_the_model_name_and_the_messages():
    req = langchain_request(model_of("x"), ("who are you?",), {})
    assert "messages" in req
    assert isinstance(req["messages"], list | str)


def test_the_base_class_is_restored_even_when_the_test_raises():
    before = BaseChatModel.invoke
    with pytest.raises(ValueError, match="boom"):
        with patched_langchain(Recorder(MemoryStore(), "lc::raise", mode=Mode.RECORD)):
            raise ValueError("boom")
    assert BaseChatModel.invoke is before


def test_an_api_key_never_reaches_the_tape():
    """Cassettes are committed to git, so a leaked key is leaked permanently."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "lc::secret", mode=Mode.RECORD)) as rec:
        model_of("ok").invoke("q", api_key="sk-super-secret")

    assert "sk-super-secret" not in str(rec.cassette.load()[0].request)


# --- tool executions -------------------------------------------------------------------------
#
# The model saying "call search" is recorded as `tool_calls` on the message. The framework then
# ACTUALLY RUNS that tool, and that is not a model call — it never reaches the chat-model seam. A
# tool that hits an API is billed and non-deterministic on every replay, and its result feeds the
# next prompt, so a drifting tool invalidates the tape of a model that behaved identically.

from langchain_core.tools import BaseTool, tool  # noqa: E402


def counting_tool():
    """A tool that records how many times it really ran."""
    calls = {"n": 0}

    @tool
    def search(q: str) -> str:
        """Search for something."""
        calls["n"] += 1
        return "result for " + q

    return search, calls


def test_a_tool_execution_replays_without_running_the_tool():
    search, calls = counting_tool()
    store = MemoryStore()

    with patched_langchain(Recorder(store, "tool::once", mode=Mode.RECORD)):
        live = search.invoke({"q": "kotlin"})
    assert calls["n"] == 1

    with patched_langchain(Recorder(store, "tool::once", mode=Mode.REPLAY)):
        replayed = search.invoke({"q": "kotlin"})

    assert replayed == live == "result for kotlin"
    assert calls["n"] == 1, "replay ran the tool for real — the side effect happened again"


def test_a_tool_is_recorded_ONCE_not_twice():
    """`BaseTool.invoke` calls `self.run(...)`, so intercepting both layers would record every tool
    call twice, double the ordinals, and replay the inner recording to the outer caller."""
    search, _ = counting_tool()
    store = MemoryStore()
    with patched_langchain(Recorder(store, "tool::single", mode=Mode.RECORD)) as rec:
        search.invoke({"q": "x"})
    assert rec.stats.recorded == 1
    assert [i.ordinal for i in rec.cassette.load()] == [1]


def test_an_async_tool_is_recorded():
    """`StructuredTool` OVERRIDES `ainvoke` — verified to delegate down to `BaseTool.ainvoke`, so
    the base seam catches it. If it ever stopped delegating, async tools would record nothing."""
    ran = {"n": 0}

    @tool
    async def fetch(q: str) -> str:
        """Fetch."""
        ran["n"] += 1
        return "fetched " + q

    store = MemoryStore()
    with patched_langchain(Recorder(store, "tool::async", mode=Mode.RECORD)) as rec:
        live = asyncio.run(fetch.ainvoke({"q": "a"}))
    assert rec.stats.recorded == 1 and ran["n"] == 1

    with patched_langchain(Recorder(store, "tool::async", mode=Mode.REPLAY)):
        assert asyncio.run(fetch.ainvoke({"q": "a"})) == live
    assert ran["n"] == 1, "replay re-ran the async tool"


def test_model_and_tool_calls_share_one_ordinal_sequence():
    """The agentic loop is model -> tool -> model. One tape in real order is what makes it an
    account of the RUN rather than two unrelated fragments."""
    search, _ = counting_tool()
    store = MemoryStore()

    with patched_langchain(Recorder(store, "tool::loop", mode=Mode.RECORD)) as rec:
        m = model_of("thinking", "done")
        m.invoke("start")
        search.invoke({"q": "kmp"})
        m.invoke("finish")

    tape = rec.cassette.load()
    assert [i.ordinal for i in tape] == [1, 2, 3]
    assert tape[1].request.get("tool") == "search", f"call 2 should be the tool: {tape[1].request}"
    assert tape[1].request.get("args") == {"q": "kmp"}


def test_the_tool_name_and_args_are_fingerprinted():
    """Different arguments are a different question, so the recorded answer must not be served."""
    search, _ = counting_tool()
    store = MemoryStore()
    with patched_langchain(Recorder(store, "tool::fp", mode=Mode.RECORD)):
        search.invoke({"q": "first"})

    with pytest.raises(Exception, match="recorded for request"):  # FingerprintDrift
        with patched_langchain(Recorder(store, "tool::fp", mode=Mode.REPLAY)):
            search.invoke({"q": "SECOND — a different question"})


def test_a_tool_defined_inside_the_code_under_test_is_covered():
    """Patching the BASE class, not an instance, is what reaches a tool an agent builds itself."""
    store = MemoryStore()
    with patched_langchain(Recorder(store, "tool::inner", mode=Mode.RECORD)) as rec:

        @tool
        def built_later(x: str) -> str:
            """Made after the patch was installed."""
            return "late:" + x

        assert isinstance(built_later, BaseTool)
        built_later.invoke({"x": "y"})
    assert rec.stats.recorded == 1


def test_framework_injected_context_is_not_part_of_a_tools_identity():
    """A tool that declares a runtime/state parameter must still replay.

    MEASURED on a whole-build tape: deepagents' file tools take a `ToolRuntime`, and LangChain hands
    it in with the arguments. Its repr carries the entire message state, message IDs included — and
    those are fresh UUIDs on every run. So the fingerprint of `read_file('/skills/.../SKILL.md')`
    differed from itself between two identical runs, every tool call drifted, and a 14-minute
    recording could not be replayed once.

    The tool's identity is its NAME and the arguments the model chose. Runtime, config, callbacks and
    the injected store are how the framework delivers the call, not what was asked.
    """
    store = MemoryStore()

    class _Runtime:
        def __init__(self, tag):
            self.tag = tag

        def __repr__(self):  # what leaks into a fingerprint
            return f"ToolRuntime(state={{'id': '{self.tag}'}})"

    @tool
    def read_file(file_path: str) -> str:
        """Read a file."""
        return "contents of " + file_path

    with patched_langchain(Recorder(store, "tool::runtime", mode=Mode.RECORD)):
        live = read_file.invoke({"file_path": "/skills/kmp/SKILL.md", "runtime": _Runtime("run-1")})

    # a SECOND run: same question, a runtime whose repr differs in every byte that matters
    with patched_langchain(Recorder(store, "tool::runtime", mode=Mode.REPLAY)):
        replayed = read_file.invoke(
            {"file_path": "/skills/kmp/SKILL.md", "runtime": _Runtime("run-2")}
        )

    assert replayed == live, "the tool drifted on framework plumbing, not on the question"
