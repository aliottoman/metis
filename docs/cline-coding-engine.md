# ClineCore coding engine

Metis uses ClineCore for its inner project edit loop while keeping the rest of
the product boundary unchanged. New builds run as small verifier-gated vertical
slices; the former loop remains only for local checkpoints frozen before the
migration.

## What runs where

Metis still owns request/specification handling, the dependency-ordered file
plan, model routing, repository map, exact allowed paths, verification,
approval, and final writes to the user's project. For each planned vertical
slice it creates or rebases a private disposable mirror and supervises a local
Node child over stdin/stdout. The child embeds the exactly pinned `@cline/sdk`
`0.0.72` and owns that slice's causal read/edit/repair conversation. A clean
host verification closes the session; the next slice starts fresh against the
already verified overlay.

There is no Cline orchestration service in this path. The coordinator, session
database, SDK, tool loop, transcript data, event replay, and workspace all run
locally. Only inference requests to the explicitly selected ClinePass or Ollama
endpoint leave the machine. Generated code is independently diffed by Metis and
is still executed only by the existing networkless verifier before an approval
card can materialize it.

## Security boundary

- `.env`, credentials, key material, `.git`, `.cline`, `.metis`, appkit, binary
  files, links, and internal build state are excluded from the coding mirror.
- Provider credentials are supplied to one local request and are not written to
  Metis's durable coding-session record. Sidecar artifacts are scanned for the
  exact credential canary; a match deletes the SDK session and fails the run.
- Only workspace reads/search and patch/editor tools are enabled. Shell, web,
  MCP, skills, plugins, questions, subagents, and teams are denied. Unknown SDK
  tools fail closed rather than inheriting Cline's permissive defaults.
- Metis computes the mirror diff itself. Deletion, rename, links, disk drift,
  protected paths, and changes outside the host plan are rejected before bytes
  enter the ordinary staged overlay.
- The model cannot approve its own output or claim verification. Clean host and
  sandbox checks plus the user's existing batch approval remain mandatory.
- For staged Python changes, the sandbox discovers repository pytest files and
  runs a deterministic slice against the materialized overlay in a fresh child
  interpreter. The verifier—not Cline and not project configuration—owns the
  executable and flags: at most 32 files, 200 tests, four failures, and 45
  seconds, still under the container's network, process, CPU, memory, and wall
  limits. A red regression becomes structured repair evidence for the next
  Cline round; a dependency declared by the project but absent from the offline
  image remains advisory. The reviewed host-side verification recipe continues
  to be a separate, explicitly approved capability and is never run against the
  real project merely because Cline requested it.

## Model routing

The outer planner and the Cline coding model remain separate roles. The current
ClinePass ladder is Qwen3.7 Plus for planning, GLM 5.2 as planner fallback,
DeepSeek V4 Pro for the broad implementation, then Kimi K3 and Kimi K2.7 Code
for bounded repair/model-switch recovery. These are routing defaults, not a
claim that every project will pass.

For the temporary Ollama Cloud subscription, use a proven tool-capable planner
and coder from the local provider catalogue—for example GLM 5.2 plus DeepSeek
V4 Pro—and reserve Kimi for a verifier-guided repair. Do not add extra model
calls merely for review: deterministic host checks are the primary reviewer.

## Build and run

Node 22 or newer and the repository-pinned pnpm version are required.

```sh
make setup
make build
pnpm --dir apps/cline-sidecar test
```

`make build` compiles both the sidecar and web app; `make test` includes the
sidecar's offline mocked-SDK and policy suite. ClineCore is the only engine for
new project runs:

```dotenv
WAQIL_PROJECT_CODING_ENGINE=clinecore
```

For ClinePass also set `WAQIL_CLINE_API_KEY` and keep qualified
`cline-pass/...` model IDs. For Ollama, keep `WAQIL_OLLAMA_BASE_URL` pointed at
the local daemon; cloud model tags still travel through that explicit Ollama
endpoint. The default sidecar entrypoint is
`apps/cline-sidecar/dist/src/index.js`. At startup Metis requires a local
protocol handshake proving ClineCore, protocol v1, SDK `0.0.72`, policy v1, and
the exact `read_files`/`search_codebase`/`editor` allowlist. If the service is
missing or mismatched, the rest of
Metis stays available with degraded health while project coding fails closed;
it never silently falls back to the legacy coding loop.

## Recovery and cleanup

Each coding session durably records the sidecar identity, exact model route,
event cursor, mirror path, tree/overlay digests, and a full secret-free mirror
manifest. Provider secrets are deliberately absent, so a continuation after a
sidecar restart must re-supply the same route credentials. Cline restores the
persisted transcript and Metis re-imports bytes from the durable mirror using
the recorded baseline; it never trusts a sidecar-authored diff.

Metis also writes the complete staged overlay to a host-only recovery journal
beside—not inside—the project mirror, so Cline cannot read or edit it. The
journal is canonical JSON bounded by the normal staged file/byte limits, a
private single-link `0600` file, and is committed with file `fsync`, atomic
rename, and parent-directory `fsync`. Recovery accepts it only when rebuilding
the mirror reproduces both durable tree and overlay digests. A journal newer
than the database record is ignored and the prior baseline imports the mirror
diff again, avoiding a partial-overlay restore after a crash.

Cancellation aborts the active slice. Approval or rejection terminates the
coding session after the ordinary staged decision is durable. Cleanup itself
is a durable state machine: Metis first records the terminal target and every
parent/child sidecar identity, then idempotently deletes the SDK session and
mirror, and only then records `clean` plus the release time. A crash or deletion
failure leaves bounded backoff debt that startup and periodic host-only
maintenance retry; cleanup never calls a model and never changes the owning
run's verdict. A completed no-approval run that unexpectedly retains
model-staged bytes is held intact and emits `project.coding_cleanup_held`.

Maintenance also collects a mirror stranded by a crash before its coding row
was created, but only after 24 hours and only when it is an unreferenced,
non-symlink, canonical `metis-code-workspace-*` directory directly under the
configured private coding-workspace parent. Project roots, referenced mirrors,
recent activity, links, and lookalike directory names are never swept. If
recovery cannot prove session identity, baseline integrity, or cursor
continuity, the run stops for diagnosis instead of starting a fresh build and
calling it a continuation.

Inspect `project.coding_started`, `project.coding_round`, and
`project.coding_event` in the durable run timeline. The permanent evaluator
reports observed sessions, models, round states, and token/request usage.

## Legacy checkpoint transition

The former engine is no longer configurable for new runs. Its compatibility
branch remains temporarily in the local source so a checkpoint that already
froze `_coding_engine=legacy` can finish without changing protocols midway. Do
not delete `.data/coding-sessions` or `.data/coding-workspaces` while a run is
active. A rollback must never auto-approve or manually copy files from a
leftover mirror.

## Measured qualification gate

Previewing the route spends no tokens:

```sh
.venv/bin/python scripts/project_capability_eval.py \
  --coding-engine clinecore \
  --provider cline
```

Then run the smallest evidence funnel: sidecar/fake-runtime tests, one bounded
local smoke, and only then one fresh Meridian vertical slice with the intended
provider/model pair. A live qualification is explicit:

```sh
.venv/bin/python scripts/project_capability_eval.py \
  --live \
  --coding-engine clinecore \
  --provider cline \
  --orchestrator-model cline-pass/qwen3.7-plus \
  --coder-model cline-pass/deepseek-v4-pro \
  --repair-turns 1 \
  --fail-below 85 \
  --report /tmp/metis-meridian-clinecore.json
```

One run is a qualification pass only when its score is at least 85, its required
acceptance probe passes, the disposable changeset is approved, and there is no
false-clean result, data loss, security-policy bypass, or unverified repair
continuation. Do not enable the engine by default from one successful demo.

The initial binary gate is ten fresh runs: four OCI document applications,
three real project UI revamps, and three fault-injected repair/continuity cases.
Promotion requires at least 8/10 overall and at least two-thirds in every family
(therefore at least 3/4, 2/3, and 2/3 respectively), with zero false-clean or
data-loss incidents. A production candidate then needs a rolling twenty-run
window of at least 16/20 and at least 70% in each family under the same binary
pass definition.

Keep archived legacy reports as the historical baseline; count model
calls/tokens as well as pass/fail and score so a slower or more expensive win
remains visible. Spend live calls only when an earlier result can change the
release decision.

## Reading a run before you raise its budget

Three live Logivity runs ended inconclusive because two questions could not be
answered from the record: why an `editor` call failed, and whether the prompt
was growing every iteration. Nine `editor:failed` events carried empty message
fields, and one round reported 496,623 input tokens against 12,182 output with
nothing to say whether that was context re-sent or cache read.

Both are now recorded. Inspect these before changing any limit.

### Why a step failed — `project.coding_event`

| field | meaning |
| --- | --- |
| `tool` | the tool that ran, e.g. `editor`, `read_files` |
| `status` | the SDK event name, e.g. `tool_call_failed` |
| `message` | the SDK's own error text, redacted and bounded to 4 KiB. Never blank on a failure: when the SDK supplies nothing it reads `SDK returned a failed tool result without diagnostic text` |
| `failure_class` | coarse SDK label — an error `name`, `code` or `type` — when one exists |
| `path` | the tool's already-scoped target, when the SDK names one |
| `iteration` | 1-based agent iteration, counted from `iteration_start` events |
| `reason_code` | unchanged: host policy denials only |

A denial is `reason_code` with no `failure_class`; an SDK failure is the
reverse. If a run writes nothing, that distinction is the first thing to read —
it separates "the model was refused" from "the edit did not apply".

A session that ends in error now carries the provider's message too, so a
capacity refusal such as `model '…' is temporarily overloaded` is readable
without reconstructing it from `run.model_exhausted`.

### Whether context is compounding — `usage` and `usage_delta`

`project.coding_event` publishes `usage` with `usage_scope: "cumulative"`.
Cline reports session totals, so **never sum usage events** — that multiplies
the bill. The event carries `usage_delta` beside it: the non-negative increase
since the previous snapshot of the same session, with `first_snapshot: true`
on the first reading of a series.

Both objects use the wire spelling: `inputTokens`, `outputTokens`,
`totalTokens`, `requests`, and — only when the provider reports them —
`cacheReadTokens`, `cacheWriteTokens`, `costUsd`. **Absent means not reported,
not zero.**

To answer "is the prompt growing per iteration", read `usage_delta.inputTokens`
across consecutive `iteration` values in one session:

* roughly flat → the prompt is stable and more iterations cost roughly linearly;
* rising each iteration → context is compounding and raising the iteration
  ceiling multiplies cost superlinearly;
* large `inputTokens` with a comparable `cacheReadTokens` → the context is
  being read from cache, and the raw input figure overstates what is billed.

A `restartWithModel` recovery opens a new session id and therefore starts its
own delta series, so a child is never differenced against its parent's larger
totals. The ancestry-aware totals in `project_capability_eval` remain the only
source for what a run actually cost; `usage_delta` is diagnostic only.

### The one opted-in measurement run

No default limit changes. The values below are already inside the supported
ranges (`cline_sidecar_max_iterations` accepts 1–200, `max_output_tokens`
256–32768, `model_call_timeout_seconds` 30–1800), so a single evaluation opts
in explicitly and nothing else is affected:

```sh
WAQIL_CLINE_SIDECAR_MAX_ITERATIONS=60 \
WAQIL_CLINE_SIDECAR_MAX_ROUNDS=4 \
WAQIL_MAX_OUTPUT_TOKENS=32768 \
WAQIL_MODEL_CALL_TIMEOUT_SECONDS=1800 \
WAQIL_PROJECT_SANDBOX_TIMEOUT_SECONDS=300 \
.venv/bin/python scripts/project_capability_eval.py --live …
```

Protected-file enforcement, mirror import, the verification ladder, routed
repair, the no-change gate and the approval card are untouched by these values
and must stay untouched. Do not add a harness-side token ceiling that cancels a
live run: a budget that interrupts a round between a successful write and its
approval card destroys the evidence it was measuring. Bound inference by
admitting fewer rounds up front, never by cancelling one in flight.
