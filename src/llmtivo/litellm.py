"""The LiteLLM seam — one integration that covers every provider LiteLLM covers.

The alternative, and the reason this module exists: write an adapter per provider. A comparable
project ships five (OpenAI, Anthropic, Gemini, Ollama, Groq) and gains a maintenance obligation with
each one — a new provider, or a changed SDK surface, is a code change. LiteLLM already normalises
that surface across 100+ providers, so patching **its** entry points buys the same coverage from a
single place, and a provider LiteLLM adds arrives for free.

    from llmtivo import Recorder, FileStore, Mode
    from llmtivo.litellm import patched_litellm

    with patched_litellm(Recorder(FileStore("tests/cassettes"), test_id, mode=Mode.REPLAY)):
        run_the_thing_under_test()

Both the sync and async entry points are covered, along with LangChain's `ChatLiteLLM` when it is
installed — an application built on LangChain never calls `litellm.completion` directly, so patching
only the former would silently record nothing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from llmtivo.intercept import patched
from llmtivo.recorder import Recorder  # noqa: TC001 - used at runtime


def litellm_request(instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request dict for a LiteLLM call.

    LiteLLM's `completion(model=..., messages=[...])` is keyword-driven, so unlike a chat-client
    method the model is in the CALL rather than on the instance. Falls back to the instance for
    `ChatLiteLLM`, where it is an attribute.
    """
    model = (
        kwargs.get("model")
        or getattr(instance, "model", None)
        or getattr(instance, "model_name", None)
    )
    request: dict[str, Any] = {"model": str(model or "")}
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    if messages is not None:
        request["messages"] = messages
    for key in ("tools", "tool_choice", "response_format"):
        if key in kwargs:
            request[key] = kwargs[key]
    return request


def _targets() -> list[tuple[Any, str]]:
    """Every LiteLLM entry point present in this environment.

    Resolved at call time and tolerant of absence: LiteLLM is an OPTIONAL dependency, and a project
    using only `ChatLiteLLM` (or only the module functions) must not be forced to install the other.


    THE HIGH-LEVEL LAYER ONLY. `ChatLiteLLM.invoke` calls `litellm.completion` underneath, so
    patching both records every call TWICE — the ordinals double, and a replay serves the inner
    recording to the outer caller. When LangChain's wrapper is present it IS the layer the
    application calls, so it wins and the module functions are left alone.
    """
    with contextlib.suppress(ImportError):
        from langchain_litellm import ChatLiteLLM

        names = [n for n in ("invoke", "ainvoke") if hasattr(ChatLiteLLM, n)]
        if names:
            return [(ChatLiteLLM, n) for n in names]

    found: list[tuple[Any, str]] = []
    with contextlib.suppress(ImportError):
        import litellm

        for name in ("completion", "acompletion", "embedding", "aembedding"):
            if hasattr(litellm, name):
                found.append((litellm, name))
    return found


@contextlib.contextmanager
def patched_litellm(recorder: Recorder) -> Iterator[Recorder]:
    """Route every available LiteLLM entry point through `recorder`, on ONE tape.

    One tape, one ordinal sequence: a run that calls a chat model and an embedding model records
    them in the order they actually happened, which is what makes the tape an account of the run
    rather than a set of per-client fragments.

    Raises if LiteLLM is not installed at all — a silent no-op would look exactly like a test whose
    model calls were recorded, which is the failure this library exists to prevent.
    """
    targets = _targets()
    if not targets:
        raise RuntimeError(
            "neither litellm nor langchain_litellm is importable — install one, or use "
            "llmtivo.intercept.patched to name the client yourself"
        )

    with contextlib.ExitStack() as stack:
        for target, method in targets:
            stack.enter_context(patched(target, method, recorder, build_request=litellm_request))
        yield recorder
