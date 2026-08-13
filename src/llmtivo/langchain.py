"""The LangChain seam — one adapter covering LangChain, LangGraph and DeepAgents.

LangGraph and DeepAgents look like three integrations and are one. Neither has a model-call surface
of its own: a LangGraph node calls a chat model, and DeepAgents is a graph over LangGraph. Every
path bottoms out in `langchain_core.language_models.BaseChatModel`, and no provider package
overrides its public entry points — `ChatAnthropic` and `ChatOpenAI` implement `_generate`/`_stream`
and inherit `invoke`/`ainvoke`/`stream`/`astream` unchanged. So ONE patch on the base class records
every provider, every framework, and a provider package that does not exist yet.

    from llmtivo import Recorder, FileStore, Mode
    from llmtivo.langchain import patched_langchain

    with patched_langchain(Recorder(FileStore("tests/cassettes"), test_id, mode=Mode.REPLAY)):
        run_the_graph()

## Why this and not only the LiteLLM seam

[llmtivo.litellm][] covers applications that route through LiteLLM. Most LangChain applications do
not: `ChatAnthropic` talks to the `anthropic` SDK directly and never enters LiteLLM at all, so the
LiteLLM seam would record NOTHING and the silence would look exactly like a test with no model
calls. Which seam an application needs is a question about its imports, not its dependencies.

## Why a codec

A cassette is JSON, and `json.dumps(default=str)` turns an unrecognised object into its `repr`.
Without translation a replayed `AIMessage` arrives as a string and the application's next line —
`response.content`, `response.tool_calls` — raises `AttributeError`. A replayed response has to be
the same TYPE the live call returned, so messages are dumped to plain dicts on the way in and
rebuilt on the way out.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from llmtivo.intercept import patched, patched_all
from llmtivo.recorder import Codec, Recorder

#: The public entry points, all four defined on `BaseChatModel` itself. `batch` is deliberately
#: absent: it is implemented in terms of `invoke`, so patching it too would record every batched
#: call twice.
_METHODS = ("invoke", "ainvoke", "stream", "astream")

#: The tool entry points, both defined on `BaseTool`. `run`/`arun` are deliberately absent:
#: `BaseTool.invoke` calls `self.run(...)`, so intercepting both layers would record every tool call
#: twice. `StructuredTool` overrides `ainvoke` but delegates down to the base, so the base seam
#: still catches async tools.
_TOOL_METHODS = ("invoke", "ainvoke")

#: The embedding entry points. All four are OVERRIDDEN by concrete classes — unlike `BaseChatModel`,
#: whose subclasses inherit `invoke` — so the base class is not the seam and each implementation has
#: to be patched. `embed_query` delegates to `embed_documents` in the common implementation, which is
#: why interception is re-entrancy-guarded rather than restricted to one layer.
_EMBEDDING_METHODS = ("embed_documents", "embed_query", "aembed_documents", "aembed_query")

#: Marks which message class a recorded payload came from, so replay rebuilds the same type. A
#: chunk replayed as a whole message breaks `+` concatenation in streaming consumers.
_TYPE_KEY = "__lc_class__"


def langchain_request(
    instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The request dict for a chat-model call.

    A LangChain chat model keeps its model name on the INSTANCE (`ChatAnthropic(model=...)`) and
    takes the input as the first positional argument, which may be a string, a message list, or a
    `PromptValue`. All three are reduced to a JSON-safe shape by the recorder's scrubber.
    """
    model = (
        getattr(instance, "model", None)
        or getattr(instance, "model_name", None)
        or getattr(instance, "_llm_type", None)
        or ""
    )
    request: dict[str, Any] = {"model": str(model)}
    if args:
        request["messages"] = args[0]
    for key in ("tools", "tool_choice", "response_format", "stop"):
        if key in kwargs:
            request[key] = kwargs[key]
    return request


def tool_request(instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request dict for a TOOL execution.

    A tool call is addressed by the tool's name and the arguments it was given, both of which go
    into the fingerprint: different arguments are a different question, and the recorded answer must
    not be served for them.
    """
    request: dict[str, Any] = {"tool": str(getattr(instance, "name", "") or "")}
    if args:
        payload = args[0]
        # LangGraph hands a tool either its argument mapping or the whole ToolCall; the arguments
        # are what identifies the call either way
        if isinstance(payload, dict) and "args" in payload and "name" in payload:
            payload = payload["args"]
        request["args"] = payload
    if kwargs:
        request["kwargs"] = kwargs
    return request


def embedding_request(
    instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The request dict for an EMBEDDING call.

    Embeddings are model calls that are not chat models, so a chat-model seam misses them entirely
    and a replayed RAG pipeline keeps calling out — billed, non-deterministic, and silent, because
    the chat calls all replay and only the retrieval reaches the network.
    """
    texts = kwargs.get("texts") or kwargs.get("text")
    if texts is None and args:
        texts = args[0]
    model = getattr(instance, "model", None) or getattr(instance, "model_name", None) or ""
    return {
        "embedding": True,
        "model": str(model),
        "texts": [texts] if isinstance(texts, str) else list(texts or []),
    }


def _embedding_classes(root: type) -> Iterator[type]:
    """Every loaded implementation under `root`, depth-first."""
    for sub in root.__subclasses__():
        yield sub
        yield from _embedding_classes(sub)


@contextlib.contextmanager
def _patched_embeddings(recorder: Recorder, stack: contextlib.ExitStack) -> Iterator[None]:
    """Intercept every `Embeddings` implementation, including ones imported LATER.

    The methods are overridden rather than inherited — unlike `BaseChatModel.invoke` — so the
    abstract base is not the seam and each implementation has to be patched.

    Walking `__subclasses__()` alone is not enough. Providers are imported lazily on purpose
    (`from langchain_openai import OpenAIEmbeddings` inside the function that embeds, so importing
    the module needs no key), which means the class does not exist when the patch goes in: the walk
    finds nothing and every embedding reaches the network while the chat calls all replay. So
    subclass CREATION is hooked for the duration too, and a class defined during the block is
    patched as it appears.
    """
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:  # pragma: no cover - embeddings are optional
        yield
        return

    def intercept(cls: type) -> None:
        for name in _EMBEDDING_METHODS:
            attr = cls.__dict__.get(name)
            if attr is not None and not getattr(attr, "__llmtivo_patched__", False):
                stack.enter_context(
                    patched(
                        cls,
                        name,
                        recorder,
                        build_request=embedding_request,
                        codec=MESSAGE_CODEC,
                        _finish=False,
                    )
                )

    for cls in [Embeddings, *_embedding_classes(Embeddings)]:
        intercept(cls)

    previous = Embeddings.__dict__.get("__init_subclass__")

    def on_new_subclass(cls: type, /, **kwargs: Any) -> None:
        if previous is not None:
            previous.__func__(cls, **kwargs)
        intercept(cls)

    # setattr, not assignment: `__init_subclass__` is an implicit classmethod slot and a direct
    # assignment is not expressible in the type system
    setattr(Embeddings, "__init_subclass__", classmethod(on_new_subclass))  # noqa: B010
    try:
        yield
    finally:
        if previous is not None:
            setattr(Embeddings, "__init_subclass__", previous)  # noqa: B010
        else:
            delattr(Embeddings, "__init_subclass__")


def _encode(response: Any) -> Any:
    """A message (or chunk) as a plain dict, tagged with the class to rebuild it as."""
    dump = getattr(response, "model_dump", None)
    if dump is None:
        return response
    payload = dump(mode="json")
    payload[_TYPE_KEY] = type(response).__name__
    return payload


def _decode(payload: Any) -> Any:
    """The inverse of [_encode][llmtivo.langchain._encode].

    An unrecognised class is returned as the raw dict rather than guessed at: handing back the wrong
    message type would fail later, somewhere unrelated to the cause.
    """
    if not isinstance(payload, dict) or _TYPE_KEY not in payload:
        return payload
    from langchain_core import messages as lc_messages

    fields = {k: v for k, v in payload.items() if k != _TYPE_KEY}
    cls = getattr(lc_messages, payload[_TYPE_KEY], None)
    return cls(**fields) if cls is not None else fields


#: Translates LangChain messages to and from the JSON on the tape.
MESSAGE_CODEC = Codec(_encode, _decode)


@contextlib.contextmanager
def patched_langchain(recorder: Recorder) -> Iterator[Recorder]:
    """Route every LangChain chat-model call through `recorder`, on ONE tape.

    Patches `BaseChatModel` itself, so a model constructed INSIDE the code under test is covered
    too — which is the normal case for a graph that builds its own nodes, and something a per-
    instance wrapper cannot reach.

    Raises if `langchain_core` is not installed: a silent no-op would look exactly like a test whose
    model calls were recorded, which is the failure this library exists to prevent.
    """
    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError as exc:  # pragma: no cover - exercised by the import-failure test
        raise RuntimeError(
            "langchain_core is not importable — install llmtivo[langchain], or use "
            "llmtivo.intercept.patched to name the client yourself"
        ) from exc

    targets: list[tuple[Any, ...]] = [
        (BaseChatModel, name, langchain_request)
        for name in _METHODS
        if hasattr(BaseChatModel, name)
    ]

    # Tool EXECUTIONS go on the same tape. The model saying "call search" is recorded as `tool_calls`
    # on the message, but the framework then actually RUNS that tool, and that is not a model call —
    # it never reaches the chat-model seam. A tool that hits an API is billed and non-deterministic
    # on every replay, and its result feeds the next prompt, so a drifting tool invalidates the tape
    # of a model that behaved identically. One ordinal sequence keeps model and tool in real order.
    with contextlib.suppress(ImportError):
        from langchain_core.tools import BaseTool

        targets += [
            (BaseTool, name, tool_request) for name in _TOOL_METHODS if hasattr(BaseTool, name)
        ]

    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched_embeddings(recorder, stack))
        rec = stack.enter_context(
            patched_all(targets, recorder, build_request=langchain_request, codec=MESSAGE_CODEC)
        )
        yield rec
