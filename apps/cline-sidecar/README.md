# Metis Cline sidecar

This package embeds the pinned `@cline/sdk` **0.0.86** runtime behind a small,
versioned NDJSON protocol. The process is local. Only provider inference leaves
the machine.

## Build and start

Node 22 or newer and pnpm 11.9.0 are required.

```sh
pnpm --dir apps/cline-sidecar build
node apps/cline-sidecar/dist/src/index.js \
  --stdio \
  --runtime cline \
  --data-dir /absolute/metis-owned/coding-sessions
```

`--data-dir` is mandatory, absolute, private (`0700`), and cannot be a symlink
or filesystem root. The sidecar sets all Cline data/session/database roots to
this directory before creating ClineCore. It never opens a TCP or Unix socket;
one supervised Metis process communicates over stdin/stdout. Stdout is reserved
for protocol frames.

SDK 0.0.86 enables provider-native web search by default for supported models.
The sidecar persists `web_search: false` in its isolated Cline settings and
verifies that value before creating ClineCore. Startup fails if the setting
cannot be written or read back. Native search runs at the provider and is not
covered by the local tool approval hooks; this startup check keeps project
coding inside Metis's web-disabled tool policy.

Use `--runtime fake` for deterministic contract tests with no model request.
`--max-replay-events N` changes the persisted per-session replay bound (default
256, maximum 4096).

## Wire protocol v1

Every line is one UTF-8 JSON object and cannot exceed 1 MiB.

```json
{"version":"1","id":"rpc_1","method":"restoreSlice","params":{"sessionId":"metis-1","provider":{"providerId":"cline-pass","modelId":"cline-pass/deepseek-v4-pro","apiKey":"..."}}}
```

Success and error responses are correlated by `id`:

```json
{"version":"1","id":"rpc_1","result":{"sessionId":"metis-1","state":"completed","finishReason":"completed","iterations":3,"toolCallCount":2,"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14,"requests":1},"model":{"providerId":"cline-pass","modelId":"cline-pass/deepseek-v4-pro"}}}
{"version":"1","id":"rpc_1","error":{"code":"POLICY_DENIED","message":"..."}}
```

Events are notifications containing bounded, sanitized fields only:

```json
{"version":"1","method":"event","params":{"sessionId":"metis-1","cursor":7,"type":"tool","tool":"editor","occurredAt":"2026-08-11T00:00:00.000Z"}}
```

Methods:

| Method | Parameters | Result |
| --- | --- | --- |
| `startSlice` | `sessionId?`, `prompt`, `systemPrompt?`, absolute `workspaceRoot`, `provider`, `limits?` | slice result |
| `continueSlice` | `sessionId`, host-chosen `recoverySessionId?`, `prompt`, `timeoutMs?`, same-route `provider?` | slice result |
| `abortSlice` | `sessionId`, `reason?` | slice result |
| `restoreSlice` | `sessionId`, same-route `provider?` | slice result |
| `restartWithModel` | `sessionId`, `newSessionId?`, `prompt`, new `provider`, `limits?` | new slice result with `parentSessionId` |
| `subscribe` | `sessionId`, `afterCursor?` | current/oldest cursor, replay count and overflow flag |
| `getUsage` | `sessionId` | accumulated usage |
| `deleteSession` | `sessionId` | requested session plus recursively deleted private ancestry |
| `getInfo` | `{}` | pinned protocol, engine/runtime, SDK, policy, and tool allowlist |
| `shutdown` | `{}` | `{state: "shutting_down"}` |

All objects reject unknown fields. Prompts, identifiers, headers, URLs, limits,
and frame sizes are bounded. `subscribe` first registers the live listener,
then replays retained events in cursor order. If the requested cursor was
compacted, the first notification is `replay_overflow`.

Raw agent-stream chunks and token-level content start/update/end fragments are
not persisted. They duplicate structured SDK events and can evict useful tool,
iteration, usage, state, and completion evidence from a bounded replay during a
long edit. Bounded stdout/stderr diagnostics remain observable, and final
summaries remain part of the settled slice result.

Settled slice results preserve Cline's exact `finishReason` (`completed`,
`aborted`, `error`, `mistake_limit`, or `max_iterations`) plus aggregate
iteration and tool-call counts. They never expose tool arguments or file
contents. The pinned SDK currently reports its own iteration-cap exception as
raw `finishReason: "error"`; the sidecar adds the orthogonal
`controlledStopReason: "max_iterations"` only when the raw reason, exact
configured iteration count, and exact pinned SDK message all agree. Metis uses
only that host-owned tag plus an independently imported application diff to
enter verification. All other unsuccessful reasons remain fail-closed.

### Provider mapping

- ClinePass: `providerId: "cline-pass"`; keep the qualified model ID such as
  `cline-pass/deepseek-v4-pro`; pass the Cline token as `apiKey`; omit `baseUrl`
  for the SDK default.
- Ollama local/cloud: `providerId: "ollama"`, provider-native model such as
  `deepseek-v4-pro:cloud`, configured Ollama `baseUrl`, and optional key.
- Other OpenAI-compatible endpoints: `providerId: "openai-compatible"` with an
  explicit HTTP(S) `baseUrl`, key, and provider-native model.

The sidecar does not rewrite provider or model aliases.

## Safety and lifecycle

Only `read_files`, workspace-root `search_codebase`, and `editor` are available.
They require approval through a fail-closed callback and pass a
second pre-tool hook. Unknown future tools stop the run. Shell, web, MCP config,
plugins, skills, questions, subagents, and teams are disabled. File operations
reject traversal, symlinks, `.git`, `.cline`, `.metis`, and secret `.env` files.
Around 60% of the configured iteration budget, Cline's native execution
reminder asks the model to finish remaining planned files, batch reads, keep
every editor text field at or below 5,500 characters, and yield to the host
verifier. The system policy gives the same editor contract before the first
turn. Successful editor results are reduced to a small acknowledgement instead
of retaining a rendered copy of the model's own diff. The hard ceiling and
controlled-stop verification remain the authority; these are only
efficiency/termination aids.

Credentials are never returned, logged, or intentionally persisted. After a
real SDK session starts, all Metis-owned artifacts are scanned for exact
credential canaries; a match deletes the session and hard-fails. Because keys
are not stored, `restoreSlice`/`continueSlice` must receive the same provider
connection after a child-process restart. ClineCore persists transcripts
but not a live runtime; the first post-restart continuation therefore forks the
persisted messages under a new session ID and reports the old ID as
`parentSessionId`. For a long model restart, the fork keeps the original user
contract plus a deterministic handoff instead of replaying stale tool payloads
and diffs. It preserves aggregate message metrics so usage lineage remains
cumulative and Metis can still compute the exact child delta.

After approval or rejection, Metis should call `deleteSession` and separately
delete the disposable workspace mirror. `deleteSession` refuses an active turn,
removes the active repair session and its private transcript ancestry through
ClineCore, and removes their sanitized event journals.

## Verification

```sh
pnpm --dir apps/cline-sidecar check
pnpm --dir apps/cline-sidecar test
```

Tests cover protocol strictness, frame bounds, real SDK configuration, a mocked
OpenAI-compatible ClineCore session, credential persistence scanning,
post-process restart recovery, tool/path denial, event replay/overflow,
fake-runtime limits, and cleanup. The mocked SDK test makes no external model
or network request.
