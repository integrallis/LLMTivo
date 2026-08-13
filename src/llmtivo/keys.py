"""Request fingerprinting — the drift detector behind ordinal keying.

A fingerprint is a short, stable digest of what was ASKED. It is never the cassette key (see
[llmtivo.cassette] for why order beats content-hashing); it is the evidence that the recording still
belongs to the call being replayed.

What goes into it is chosen so that the fingerprint changes when the *meaning* of the call changes
and not when something incidental does:

  * the model name — replaying Sonnet's answer for a Haiku call is a different experiment
  * the messages' roles and content
  * the tool/function names offered, if any — but not their full schemas, which churn
  * for a TOOL EXECUTION, the tool's name and the arguments it was given — without these every
    tool call digests identically, and the answer recorded for one set of arguments would be
    served for any other

Deliberately excluded: temperature, max_tokens, timeouts, api keys, request ids, and any other
transport or sampling detail. A test that retunes temperature has not changed the question it asked.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace — a reformatted template is not a different question.

    Prompts live in Jinja files that get re-wrapped and re-indented constantly. Without this, a
    cosmetic edit invalidates a cassette and bills a re-record for a prompt whose MEANING never
    changed. Semantic edits still change the digest, which is the whole point.
    """
    return _WS.sub(" ", text).strip()


#: Characters of the digest kept. 16 hex chars ≈ 64 bits: collision-free at any test-suite scale,
#: short enough to eyeball in a cassette.
WIDTH = 16


def _content_of(value: Any) -> str:
    """Canonical text for a message's content, whatever shape it arrives in.

    Content is not always a string: Anthropic returns a LIST OF BLOCKS for tool use, and multimodal
    messages carry lists everywhere. `str()` on those digests the Python REPR, whose dict key order
    is insertion order — so the same content, having been through JSON on its way to a cassette,
    orders its keys differently and hashes differently. Every replayed multi-turn agent run then
    drifts at the turn that quotes a previous structured response: the model behaved identically and
    the tape gets thrown away regardless. Sorted-key JSON is stable across that round trip.
    """
    if isinstance(value, str):
        return _normalize(value)
    if isinstance(value, (dict, list, tuple)):
        return _normalize(json.dumps(_stable(value), separators=(",", ":"), sort_keys=True))
    return _normalize(str(value))


def _messages_of(request: dict[str, Any]) -> list[dict[str, str]]:
    """The role/content pairs, however the caller spelled them."""
    raw = request.get("messages") or request.get("input") or []
    if isinstance(raw, str):
        return [{"role": "user", "content": _normalize(raw)}]
    out: list[dict[str, str]] = []
    for m in raw:
        if isinstance(m, dict):
            out.append(
                {"role": str(m.get("role", "")), "content": _content_of(m.get("content", ""))}
            )
        else:  # a LangChain message object, or anything else with .content
            role = getattr(m, "type", None) or getattr(m, "role", "") or m.__class__.__name__
            out.append({"role": str(role), "content": _content_of(getattr(m, "content", m))})
    return out


def _tool_names(request: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for t in request.get("tools") or []:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(str(name))
        elif (name := getattr(t, "name", None)) is not None:
            names.append(str(name))
    return sorted(names)


def fingerprint(request: dict[str, Any]) -> str:
    """A stable digest of the request's MEANING (see the module docstring for what counts)."""
    material: dict[str, Any] = {
        "model": str(request.get("model", "")),
        "messages": _messages_of(request),
        "tools": _tool_names(request),
    }
    # added only when present, so a chat call's digest is unchanged by tools existing as a concept
    # — a gratuitous change here would invalidate every cassette already committed
    if "texts" in request:  # an embedding: the texts ARE the question
        material["texts"] = _stable(request.get("texts"))
    if "tool" in request:
        material["tool"] = str(request.get("tool", ""))
        material["args"] = _stable(request.get("args"))
        material["kwargs"] = _stable(request.get("kwargs"))
    return _digest(material)


def _stable(value: Any) -> Any:
    """A JSON-safe, order-stable view of tool arguments."""
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _digest(material: dict[str, Any]) -> str:
    blob = json.dumps(material, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:WIDTH]
