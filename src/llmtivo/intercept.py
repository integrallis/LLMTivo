"""The interception seam — putting a [Recorder][llmtivo.recorder.Recorder] in front of a real client.

`Recorder.call` is the whole state machine and it takes a `perform` callable, so recording something
means routing that client's call through it. This module does that by PATCHING a method on a class
for the duration of a test, and restoring it afterwards.

## Why patching, and why it is narrow

The alternative — asking every call site to route through LLMTivo — does not work on code you did not
write, and it is exactly the "flag in the production path" hazard: a seam the application knows
about is a seam that can be switched on in production. Patching keeps LLMTivo entirely inside the
test process. The application under test contains no LLMTivo code and cannot accidentally ship it.

The patch is deliberately narrow:

  * **One method on one class at a time.** No import hooks, no HTTP-layer monkeying, no global
    transport swap — those catch calls you did not mean to record and break in surprising ways.
  * **Restored unconditionally**, even when the test raises, so a failure never leaks a patched
    class into the next test.
  * **Re-entrant-safe.** Patching the same target twice is refused rather than silently nesting,
    because an accidental double-patch would record every call twice.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from llmtivo.recorder import IDENTITY, Codec, Recorder

#: How a call's arguments become the request dict that gets fingerprinted and recorded.
#: The default handles the common `invoke(messages, **kwargs)` shape.
RequestBuilder = Callable[[Any, tuple[Any, ...], dict[str, Any]], dict[str, Any]]


def default_request(instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The request dict for a typical chat client call.

    Reads the model name off the client instance, since that is where every client of note keeps it,
    and takes the first positional argument as the messages.
    """
    model = (
        getattr(instance, "model", None)
        or getattr(instance, "model_name", None)
        or getattr(instance, "_model", None)
        or ""
    )
    request: dict[str, Any] = {"model": str(model)}
    if args:
        request["messages"] = args[0]
    request.update(kwargs)
    return request


@contextmanager
def patched(
    target: Any,
    method: str,
    recorder: Recorder,
    *,
    build_request: RequestBuilder = default_request,
    codec: Codec = IDENTITY,
    _finish: bool = True,
) -> Iterator[Recorder]:
    """Route `target.method` through `recorder` for the duration of the block.

        with patched(ChatAnthropic, "invoke", recorder):
            run_the_thing_under_test()

    The original is restored on the way out whatever happens, and `recorder.finish()` is called on a
    clean exit so a freshly recorded tape gets compacted.
    """
    original = getattr(target, method)
    if getattr(original, "__llmtivo_patched__", False):
        raise RuntimeError(
            f"{target.__name__}.{method} is already intercepted — a double patch would record "
            f"every call twice"
        )

    # A CLASS attribute is a method and receives `self`; a MODULE attribute is a plain function and
    # does not. litellm.completion is the latter, so assuming a bound method drops its first
    # argument and fails with "missing 1 required positional argument".
    is_method = isinstance(target, type)

    # An ASYNC target must be awaited before its value can be recorded. Wrapping one in the sync
    # path puts the coroutine OBJECT on the tape — it fails to serialise, and the call is never
    # awaited. `acompletion`/`ainvoke` are exactly this, so the distinction is not an edge case.
    is_async = inspect.iscoroutinefunction(original)

    # A STREAMING entry point returns a generator, not a response. Routing one through `call` puts
    # a generator object on the tape and replays something that yields nothing — the same class of
    # mistake as recording an un-awaited coroutine. `BaseChatModel.stream`/`astream` are exactly
    # this, and LangGraph and DeepAgents stream by default.
    is_gen = inspect.isgeneratorfunction(original)
    is_agen = inspect.isasyncgenfunction(original)

    # Each variant is named separately and then SELECTED, rather than six definitions sharing one
    # name: same-named variants differ in signature (bound vs plain, awaited vs yielded) and a
    # reader — or a type checker — cannot tell which one a branch installed.
    async def async_gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = build_request(self, args, kwargs)
        async for chunk in recorder.astream(
            request, lambda: original(self, *args, **kwargs), codec=codec
        ):
            yield chunk

    def gen_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = build_request(self, args, kwargs)
        return recorder.stream(request, lambda: original(self, *args, **kwargs), codec=codec)

    async def async_method_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = build_request(self, args, kwargs)
        return await recorder.acall(request, lambda: original(self, *args, **kwargs), codec=codec)

    async def async_function_wrapper(*args: Any, **kwargs: Any) -> Any:
        request = build_request(None, args, kwargs)
        return await recorder.acall(request, lambda: original(*args, **kwargs), codec=codec)

    def method_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = build_request(self, args, kwargs)
        return recorder.call(request, lambda: original(self, *args, **kwargs), codec=codec)

    def function_wrapper(*args: Any, **kwargs: Any) -> Any:
        request = build_request(None, args, kwargs)
        return recorder.call(request, lambda: original(*args, **kwargs), codec=codec)

    wrapper: Any
    if is_agen:
        wrapper = async_gen_wrapper
    elif is_gen:
        wrapper = gen_wrapper
    elif is_async:
        wrapper = async_method_wrapper if is_method else async_function_wrapper
    else:
        wrapper = method_wrapper if is_method else function_wrapper

    wrapper.__llmtivo_patched__ = True
    wrapper.__wrapped__ = original
    wrapper.__name__ = getattr(original, "__name__", method)
    wrapper.__doc__ = getattr(original, "__doc__", None)

    setattr(target, method, wrapper)
    try:
        yield recorder
        if _finish:  # patched_all finishes once, after ALL its targets are restored
            recorder.finish()  # clean exit only: never compact a half-recorded tape
    finally:
        setattr(target, method, original)


@contextmanager
def patched_all(
    targets: list[tuple[Any, str]],
    recorder: Recorder,
    *,
    build_request: RequestBuilder = default_request,
    codec: Codec = IDENTITY,
) -> Iterator[Recorder]:
    """Intercept several targets with ONE recorder, so their calls share a single ordinal sequence.

    A run that calls a chat model and an embedding model — or mixes blocking and awaited calls —
    records them in the order they actually happened, which is what makes the tape an account of the
    run rather than a set of per-client fragments.

    Delegates to [patched][llmtivo.intercept.patched] rather than repeating the wrapper. A second
    copy is how the async branch got missed here once already: `patched` learned to await and this
    did not, so a mixed run recorded a coroutine object.
    """
    with contextlib.ExitStack() as stack:
        for target, method in targets:
            stack.enter_context(
                patched(
                    target,
                    method,
                    recorder,
                    build_request=build_request,
                    codec=codec,
                    _finish=False,
                )
            )
        yield recorder
        recorder.finish()
