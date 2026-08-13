# LLMTivo

[![MFCQI Score](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/llmtivo/main/.github/badges/llmtivo.json)](https://github.com/integrallis/llmtivo)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Record LLM calls once, then replay them from the filesystem.

A test suite that interacts with a language model faces two poor choices: call the real model (slow,
expensive, unpredictable), or write a fake by hand (free, quick, and ultimately meaningless). A
hand-crafted fake is the worse of these, since it passes tests but does not reflect what the model
would actually return.

LLMTivo offers a third way. It records the actual responses once, commits the tape, and then replays
it on every later run.

```python
from llmtivo import Recorder, FileStore, Mode

rec = Recorder(FileStore("tests/cassettes"), test_id, mode=Mode.REPLAY)
response = rec.call(request, perform=lambda: real_client.invoke(request))
```

## Modes

The mode answers one question: **when the tape does not have what this call needs, what happens?**
Every failure mode of a record/replay system comes down to that.

| Mode | On a miss | Use |
|---|---|---|
| `RECORD` | call the model, overwrite the tape | re-recording deliberately |
| `RECORD_NEW` | record if untaped, else replay | adding a test without re-billing the suite |
| `REPLAY` | **raise** | CI |
| `REPLAY_OR_RECORD` | call the model, append | local convenience |
| `OFF` | passthrough, record nothing | debugging against the real thing |

`REPLAY` never reaches the network. If a test quietly starts calling a paid API, that is a defect,
and a fallback that hides it is worse than a failure.

## Keying: call order, not prompt hash

An interaction is identified by its **call order within the test** — 1, 2, 3 — not by hashing the
request. This is the opposite of what an HTTP-level recorder does, and it is intentional.

Content hashing is precise but not useful in practice: change a single word in a prompt and every
cassette that includes it is invalid. Prompts are edited all the time. Order-based keying avoids
this entirely, which is the main reason to record model output rather than mock it.

The cost is that order alone cannot notice the code changed underneath the tape — reorder two calls
and each silently replays the other's answer. So every interaction also stores a **fingerprint** of
the request it was recorded for, covering the model, the messages and any tool names, but *not*
temperature, max tokens or transport details. Whitespace is normalised, so re-wrapping a Jinja
template is not a new question.

Order **addresses** an interaction; the fingerprint **validates** it. They have to match. A response
recorded under the old prompt is not an answer to the new one, so replaying it would assert
downstream behaviour against something the current code could never elicit:

- in `REPLAY` a mismatch **raises** — CI must not pass on a stale tape
- in a recordable mode the stale interaction **and everything after it** are dropped and re-recorded

That tail matters: in an agentic loop call N+1's prompt contains call N's response, so once N is
re-answered every later recording is a reply to a branch that no longer happens.

Order brings resilience. The fingerprint brings accuracy. Neither is enough by itself.

## Storage: filesystem by default

A cassette is an artifact of the repo — reviewable in a diff, carried by the same git history as the
test that produced it, needing nothing booted. A database is a legitimate **pluggable backend** for
teams that want one (implement the `CassetteStore` protocol), but it is never a prerequisite for
running the tests.

### Format: zstd-compressed JSON Lines, lossless

`<name>.jsonl.zst` — one interaction per line, one zstd frame per append.

Measured on a real corpus of model output (26 interactions, 65 KiB of generated Kotlin and prompts):

| codec | size | ratio | compress | decompress |
|---|---|---|---|---|
| gzip -6 | 12.3 KiB | 5.3x | 1.6 ms | 0.2 ms |
| gzip -9 | 12.2 KiB | 5.4x | 2.6 ms | 0.2 ms |
| zstd -3 | 13.2 KiB | 5.0x | 0.1 ms | <0.1 ms |
| **zstd -9** | **11.9 KiB** | **5.5x** | **0.6 ms** | **<0.1 ms** |
| zstd -19 | 11.4 KiB | 5.7x | 15.5 ms | <0.1 ms |
| msgpack + zstd -3 | 13.3 KiB | 4.9x | — | — |

zstd -9 beats gzip -9 on **both** axes — smaller, ~4x faster to write, ~5x faster to read — so there
is no compression/performance trade to make. Level 19 buys 4% more for 25x the write cost, which a
suite writing thousands of interactions would feel.

msgpack was rejected: the payload is text, so binary framing saves ~5% before compression and the
compressor erases even that, while costing the ability to read a tape with `zstdcat`.

JSON Lines rather than one document because recording **appends**. Each interaction is its own zstd
frame, so a run that dies halfway leaves every earlier interaction intact and replayable — zstd
decodes concatenated frames as a single stream, so appending stays a plain `open(..., "ab")`.

Nothing is lossy: a replayed response is byte-for-byte what the model returned.

## pytest

```bash
pip install llmtivo[pytest]
```

The plugin registers itself, so the `llmtivo` fixture is available with no conftest entry.

```python
def test_the_build(llmtivo):
    with llmtivo.patch(ChatAnthropic, "invoke"):
        assert build() == expected
```

```bash
pytest                        # replay — the default, CI-safe, costs nothing
pytest --llmtivo=record-new    # record only the tests with no tape yet
pytest --llmtivo=record        # re-record everything. costs money.
```

`@pytest.mark.llmtivo("off")` overrides one test. `--llmtivo-dir` moves the tapes. Interception
patches one method on one class for the duration of a test and restores it unconditionally — no
import hooks, no HTTP-layer monkeying — so the application under test contains no LLMTivo code and
cannot accidentally ship it. `patch_all` puts several clients on one tape in real call order.

## LiteLLM: one seam, every provider

```bash
pip install llmtivo[litellm]
```

```python
from llmtivo.litellm import patched_litellm

with patched_litellm(recorder):
    run_the_thing_under_test()
```

The alternative is an adapter per provider — OpenAI, Anthropic, Gemini, Ollama, Groq — and a
maintenance obligation with each. LiteLLM already normalises 100+ providers, so patching its entry
points covers them all from one place and a provider it adds arrives for free. Both sync and async
are covered, plus LangChain's `ChatLiteLLM`, because an application built on LangChain never calls
`litellm.completion` directly and patching only the latter would silently record nothing.

Sync and async are both handled — an awaited call records the **awaited value**, never the coroutine
object. Only the high-level layer is patched: `ChatLiteLLM.invoke` calls `litellm.completion`
underneath, so patching both would record every call twice and replay the inner recording to the
outer caller.

If neither package is importable it **raises** rather than no-opping: a silent no-op looks exactly
like a test whose calls were recorded, which is the failure this library exists to prevent.

## LangChain, LangGraph and DeepAgents: also one seam

```bash
pip install llmtivo[langchain]
```

```python
from llmtivo.langchain import patched_langchain

with patched_langchain(recorder):
    run_the_graph()
```

Three frameworks, one seam. Neither LangGraph nor DeepAgents has a model-call surface of its own — a
graph node calls a chat model, and DeepAgents is a graph over LangGraph — so every path bottoms out
in `BaseChatModel`. No provider package overrides its public entry points: `ChatAnthropic` and
`ChatOpenAI` implement `_generate`/`_stream` and inherit `invoke`/`ainvoke`/`stream`/`astream`
unchanged. Patching the base class therefore covers every provider at once, including one whose
package does not exist yet, and covers models constructed *inside* the code under test — the normal
case for a graph that builds its own nodes, and something a per-instance wrapper cannot reach.

**Pick this seam, not the LiteLLM one, if your app imports `ChatAnthropic` or `ChatOpenAI`.** Those
talk to the provider SDK directly and never enter LiteLLM, so the LiteLLM seam would record nothing —
and the silence would look exactly like a test with no model calls. Which seam an application needs
is a question about its imports, not its dependencies.

Streaming is recorded chunk by chunk. `stream()` returns a *generator*, and a generator is not a
response: recording its return value puts a generator object on the tape and replays something that
yields nothing. The chunks are what happened, so the chunks are what is recorded — yielded through as
they arrive, so a streamed test still observes streaming rather than one late burst. A consumer that
breaks out early records only what it consumed; the tape is an account of the run.

Responses replay as the **same type** they were. A cassette is JSON, and the default serialiser turns
an unknown object into its `repr` — without translation a replayed `AIMessage` arrives as a string
and the next line, `response.content` or `response.tool_calls`, raises `AttributeError`. Messages are
dumped to plain dicts on the way in and rebuilt on the way out, `tool_calls` included, so an agentic
loop takes the branch it recorded.

## Compaction

Appending per call buys crash-safety at a real cost: each interaction is its own zstd frame and
compresses independently. Measured on the same corpus, a recorded tape was 21.7 KiB appended against
12.2 KiB compacted — 44% wasted. `finish()` rewrites the tape as a single frame, atomically, and
**only after a clean finish**, so a recording that died halfway is never rewritten as if complete.
Durable while recording, compact at rest.

### Tool calls, both meanings

A model asking for a tool and a tool actually running are two different events, and both are on the
tape.

The **request** is part of the message: `tool_calls` survive the round trip intact, so an agentic
loop replays the branch it recorded rather than one where the model suddenly stopped calling tools.

The **execution** is not a model call at all — the framework runs the function, and nothing about
that reaches a chat-model seam. It is recorded anyway, because a tool that hits an API is billed and
non-deterministic on every replay, and its result feeds the next prompt: a drifting tool invalidates
the tape of a model that behaved identically. Replaying serves the recorded result **without running
the tool**, so side effects happen once, when recording.

A tool execution is addressed by the tool's name and the arguments it was given, both fingerprinted —
different arguments are a different question, and the recorded answer is not served for them.

`BaseTool.run`/`arun` are deliberately left alone: `invoke` calls `run` underneath, so intercepting
both layers would record every tool call twice.

Verified end to end through a real agent — one tape, in real order:

```
tape: [(1, 'model:scripted'), (2, 'lookup'), (3, 'model:scripted')]
replay -> 3 replayed, tool ran again: NO, run reproduced identically
```

## The network guard

`REPLAY` raises on a miss **at the seam**. That covers every client LLMTivo was pointed at, and
nothing else. A model call that goes *around* the seam — a provider SDK used directly, an embedding
client nobody patched, an HTTP call inside an agent's tool — reaches the real network from a suite
reporting itself as replaying. It is billed, it is non-deterministic, and it looks exactly like a
pass.

So replaying blocks outbound connections, and the block names the host it stopped:

```
network is blocked — api.openai.com was contacted while replaying. Some call is not going
through LLMTivo, so it would hit the real API and be billed.
```

This follows from the mode rather than a flag — an opt-in guard is one nobody sets. Only `connect`
and `connect_ex` are patched, never `socket.socket` itself, so local services keep working: a suite
legitimately talks to a database or a fixture server, and blocking those would turn a guard against
surprise *spending* into a guard against testing. Loopback is always allowed;
`--llmtivo-allowed-hosts` adds more.

## Secrets

Cassettes get committed, so anything secret that reaches one is leaked permanently by git history.
Filtering runs **on the way in** rather than being somebody's review responsibility, in two passes,
because one is not enough:

- **By key** — `api_key`, `authorization`, `token` and friends are dropped, along with transport
  objects that would not serialise anyway.
- **By value** — key-name filtering does nothing for a credential interpolated into a prompt, a
  system message or a URL, which is just *text* in `messages`. Every value in the environment under
  a `*_API_KEY` / `*_TOKEN` / `*_SECRET` / `*_PASSWORD` name is substituted with `<REDACTED>`
  wherever it appears, at any depth. No configuration needed to be right on the first run — the run
  that records the tape you commit.

## Layout

| Module | Responsibility |
|---|---|
| `store` | where tapes live — `FileStore` (default), `MemoryStore`, or your own |
| `cassette` | one tape per test; interaction records |
| `keys` | request fingerprinting |
| `modes` | the modes and what each promises on a miss |
| `recorder` | the state machine, testable without patching anything |
| `intercept` | the seam that puts a recorder in front of a real client |
| `guard` | blocks outbound connections while replaying |
| `litellm` | the LiteLLM integration — every provider from one patch |
| `langchain` | the LangChain / LangGraph / DeepAgents integration — one patch on `BaseChatModel` |
| `plugin` | the pytest integration |

## Integration tests, recorded with LLMTivo itself

`tests/integration/` calls **real** providers — OpenAI, Anthropic, Gemini, DeepSeek, Tavily and a
live MCP server over stdio — and the tapes are committed. They replay in CI with **no keys and no
network**, in about three seconds against forty-eight recording.

```bash
pytest tests/integration                    # replay: free, offline, no keys
pytest tests/integration --llmtivo=record   # re-record. costs money, needs keys in .env
```

They are not excluded by default. Excluding them is how a suite stops noticing that a provider
changed a response shape — and they found two defects that every fake had hidden:

- **The LiteLLM seam replayed a string.** `litellm.completion` returns a `ModelResponse` object; a
  cassette is JSON, so `default=str` wrote its repr and replay handed back text. The caller's next
  line, `response["choices"][0]["message"]["content"]`, raised `TypeError: string indices must be
  integers`. Every test using a fake `litellm` returned a plain dict and passed happily.
- **Structured content drifted on replay.** Anthropic returns content as a *list of blocks*, and
  the fingerprint canonicalised those with `str()` — the Python repr, whose dict key order is
  insertion order. After a JSON round trip the keys came back in a different order, so a
  multi-turn agent run drifted at the turn quoting a previous structured response: the model
  behaved identically and the tape was thrown away anyway.

Both are pinned by unit tests now, but neither was reachable without calling the real thing.

## Quality gate

This project holds itself to the bar it would demand of generated code: **MFCQI >= 0.75**, enforced
in `make quality-check` and in CI, with the badge regenerated on every push to main.

```bash
make mfcqi         # the gate — fails the build below MFCQI_MIN (default 0.75)
make mfcqi-badge   # regenerate .github/badges/llmtivo.json
make quality-check # format + lint + types + tests + the MFCQI gate
```
