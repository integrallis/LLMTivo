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

from llmtivo.intercept import patched_all
from llmtivo.recorder import Codec, Recorder

#: The public entry points, all four defined on `BaseChatModel` itself. `batch` is deliberately
#: absent: it is implemented in terms of `invoke`, so patching it too would record every batched
#: call twice.
_METHODS = ("invoke", "ainvoke", "stream", "astream")

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

    targets = [(BaseChatModel, name) for name in _METHODS if hasattr(BaseChatModel, name)]
    with patched_all(
        targets, recorder, build_request=langchain_request, codec=MESSAGE_CODEC
    ) as rec:
        yield rec
