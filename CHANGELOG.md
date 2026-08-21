# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-08-21

### Fixed

- **A tool's identity is what was asked, not how it was delivered.** LangChain v1 hands
  framework-injected parameters in alongside the model's own arguments, and a tool that declares one
  — deepagents' file tools take a `ToolRuntime` — had that object's repr folded into its fingerprint.
  A `ToolRuntime` reprs the whole message state, message IDs included, and those are fresh UUIDs on
  every run, so `read_file('/skills/kmp/SKILL.md')` fingerprinted DIFFERENTLY FROM ITSELF between two
  identical runs. Measured on a whole-build tape: every tool call drifted and a 14-minute recording
  could not be replayed once. `runtime`, `config`, `callbacks`, `run_manager`, `state`, `store` and
  `tool_call_id` are now excluded from a tool request; the tool's NAME and the arguments the MODEL
  chose still address it.

## [0.1.0] - 2026-08-12

Initial release.

### Added

- **Record/replay core.** `Recorder` serves a test's model calls from tape, from the network, or
  refuses — the whole state machine, testable without patching anything.
- **Five modes** (`RECORD`, `RECORD_NEW`, `REPLAY`, `REPLAY_OR_RECORD`, `OFF`), each answering what
  happens on a cassette miss. `REPLAY` is the default and never reaches the network.
- **Order keying with fingerprint validation.** Call order addresses an interaction so prompt edits
  do not invalidate a tape; a request fingerprint validates it so a stale tape cannot pass silently.
  A mismatch invalidates the interaction *and its tail*, since in an agentic loop a re-answered call
  changes every prompt after it.
- **Filesystem storage by default** — `<name>.jsonl.zst`, one tape per test. zstd level 9 chosen by
  measurement over gzip and msgpack; lossless, appendable, readable with `zstdcat`.
- **Compaction on clean finish**, recovering the ~44% that per-call framing costs, without giving up
  durability while recording.
- **Pluggable backends** via the `CassetteStore` protocol (`FileStore`, `MemoryStore`).
- **Narrow interception** (`patched`, `patched_all`) — one method on one class, restored
  unconditionally, so the application under test contains no LLMTivo code.
- **pytest plugin** — an `llmtivo` fixture, `--llmtivo` / `--llmtivo-dir`, and a per-test marker.
- **Secret scrubbing on the way in**, because cassettes are committed and git history is permanent.
