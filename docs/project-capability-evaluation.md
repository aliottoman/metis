# Realistic project capability evaluation

`scripts/project_capability_eval.py` runs one stable, production-shaped build
instead of a toy file-generation prompt. The scenario is **Meridian Evidence
Desk**: a responsive FastAPI application with document upload, local TXT invoice
extraction, OCI Responses support for images/PDFs, review and approval, cited
follow-up questions, SQLite persistence, and an audit trail.

Meridian remains the default and its schema-v1 JSON report is backward
compatible. Two smaller, token-conscious scenarios exercise the coding harness
against existing code rather than another greenfield build:

- `ui-revamp`: revamps a seeded Atlas FastAPI dashboard in exactly five UI,
  test, and documentation files. A host probe hashes four protected backend/API
  files, reruns their contracts, and checks the responsive, accessible UI and
  real PATCH interaction.
- `fastapi-repair`: repairs four deliberately broken files in a seeded incident
  service. The faults cover SQLite schema/row handling, transaction durability,
  HTTP 201/404 behavior, and persistence across a fresh application/client.

Preview either complete seed/write/probe contract without calling a model:

```sh
.venv/bin/python scripts/project_capability_eval.py --scenario ui-revamp
.venv/bin/python scripts/project_capability_eval.py --scenario fastapi-repair
```

New-scenario live reports retain the existing diagnostic 100-point score and
add a fail-closed `qualification` object. Its score is binary—100 only when the
run's durable scope contract resolves to the exact requested write scope, every
required path has attributed writes, post-write verification is clean, every
repair continuation is proven, the disposable changeset is approved, and its
required sandbox probe passes; otherwise it is 0 with the failed gates named.
`--fail-below` requires this binary qualification as well as the existing
release gates.

The production `cline_direct` path intentionally makes no planner-model call.
Its `project.direct_contract` freezes writable roots, task/project protections,
source hashes, check budgets, and approval requirements before inference. The
evaluator keeps `plan.present=false` honest, then uses the independently
observed mirror diff as the exact requested scope. Planner/slice compatibility
runs continue to use their file manifest. Both paths now flow through one
`scope_contract` report shape and the same overlay, verification, approval,
qualification, and release gates.

When the planner/slice compatibility path is explicitly selected, its default
model split and infrastructure backups stay inside ClinePass:

- orchestrator: `cline-pass/qwen3.7-plus` (not called by `cline_direct`)
- planner fallback: `cline-pass/glm-5.2`
- coder: `cline-pass/deepseek-v4-pro`
- deterministic host and sandbox checks own verification; the project build
  does not currently invoke a separate model reviewer

The preview and JSON report include the exact role ladders. A fallback is part
of this product-level evaluation and is recorded as `run.model_fallback`; an
exhausted ladder is recorded separately. That makes a successful backup run
visible instead of misattributing it to the selected primary pair. The CLI
rejects paid Cline gateway IDs so a ClinePass run cannot silently consume or
fail on pay-as-you-go credits.

Preview the exact request and route without calling a model:

```sh
.venv/bin/python scripts/project_capability_eval.py
```

For Ollama/local canaries, fallback ladders are opt-in and repeatable. No cloud
fallback is silently inserted. The preview and live report persist the exact
deduplicated role chains and per-turn output bound:

```sh
.venv/bin/python scripts/project_capability_eval.py \
  --scenario ui-revamp \
  --provider local \
  --orchestrator-model glm-5.2:cloud \
  --planner-fallback-model qwen3.7:cloud \
  --coder-model deepseek-v4-pro:cloud \
  --coder-fallback-model kimi-k3:cloud \
  --coder-fallback-model kimi-k2.7-code:cloud \
  --max-tokens-per-coding-turn 8192
```

`--max-tokens-per-coding-turn` is bounded to 4,096–32,768. Omitting it preserves
the configured `Settings.max_output_tokens`, so existing Meridian commands keep
their prior semantics.

Select the production coding harness independently from the provider/model route:

```sh
.venv/bin/python scripts/project_capability_eval.py \
  --coding-engine clinecore \
  --max-coding-iterations 24
```

`clinecore` is the production engine and uses the local supervised Cline SDK
sidecar. With the default `cline_direct` build path it owns the bounded coding
conversation under the frozen direct contract; it does not move verification,
approval, or project materialization to a cloud service. The report records
observed sessions, model switches, terminal round states, and token/request
usage. Existing `legacy` reports remain valid historical comparison artifacts.

## Cursor shadow benchmark

Cursor is available only as an evaluation adapter in `apps/cursor-shadow`; it
is not a `WAQIL_PROJECT_CODING_ENGINE` value and cannot write to or approve a
real Metis project. The pinned official `@cursor/sdk` runner requires Node
22.13+, an explicit `--live`, `CURSOR_API_KEY`, and a Metis-created disposable
workspace marker. It makes one SDK `send`, enables the Cursor local sandbox,
loads no ambient Cursor settings or MCP servers, removes web and subagent
tools, disables agent retries, and cancels after five minutes or 80,000
observed tokens. Its final diff must contain every requested file, no extra
file, and no protected file before it is even eligible for the same Metis
sandbox acceptance checks.

Inspect the complete free contract without a model request:

```sh
node apps/cursor-shadow/dist/src/index.js --describe
```

A live Cursor comparison remains deliberately separate from the ClinePass
qualification command. Run it only after its result can change the harness
decision, against the same seeded `ui-revamp` fixture and exact required and
protected path lists. Cursor reports its SDK tokens and billed usage where the
backend exposes them. No Cursor credential means no live comparison request.

`--max-coding-iterations` is the ClineCore SDK's inner tool-loop cap for one
coding session. It is bounded to 1–200 and is deliberately separate from
`--max-steps`, which caps Metis's outer project-planning graph. Omitting the
inner option preserves `WAQIL_CLINE_SIDECAR_MAX_ITERATIONS` (24 by default).
Both the preview and saved live report record the effective inner cap as
`max_coding_iterations` for each ClineCore slice.

The evaluator installs this value on a fresh `Settings` instance that is passed
only to its disposable `create_app()` runtime. It does not change the process
environment, the repository configuration, or another running Metis instance.

On a model restart, Cline seeds the child session with the parent's message
history, so the SDK's accumulated child token counters include inherited parent
tokens. Reports use the persisted sidecar ancestry to count only the child's
token delta while still counting each child-local request. `usage_lineage`
retains both raw and counted counters for auditability; older reports without an
explicit parent ID infer only the ordered restart within the same durable coding
session.

Run the live evaluation and retain a JSON report:

```sh
.venv/bin/python scripts/project_capability_eval.py \
  --live \
  --coding-engine clinecore \
  --provider cline \
  --orchestrator-model cline-pass/qwen3.7-plus \
  --coder-model cline-pass/deepseek-v4-pro \
  --repair-turns 1 \
  --report /tmp/metis-meridian-eval.json
```

Qwen3.7 Plus is the measured default, not a paper preference: in two fresh
production-shaped attempts GLM 5.2 returned neither the required manifest call
nor recoverable text after its bounded call, while Qwen produced the complete
ordered 16-file plan in both attempts. GLM remains the next planner rung. An
explicit `--orchestrator-model`, saved role chain, or
`WAQIL_CLINE_ORCHESTRATOR_MODEL` still owns the first rung; repeated model
identities are removed rather than retried as a nominal fallback.

Every run uses a new temporary project and data directory. A turn is bounded by
the outer `--max-steps`, the ClineCore-only inner `--max-coding-iterations`, and
`--timeout` (30 minutes by default), while any one model call is capped at five
minutes and then advances the visible role ladder.
On the planner/slice compatibility path, Metis treats the planner's file order
as dependency order and checks bounded contiguous prefixes. The default direct
path instead lets the supervised coding session inspect and change the project
under its frozen contract, then Metis imports the independent diff and runs the
same final verification and acceptance scenarios.

Repair follow-ups are bounded by `--repair-turns` (0–3). A clean changeset is
approved only in that disposable project. “Clean” includes the scenario's
host-owned acceptance probe: before auto-approval, the evaluator reads the
exact pending bytes from that run's checkpoint and gives them directly to the
existing sandbox materializer. The materializer copies the source project to a
temporary networkless workspace, layers those pending bytes over the copy, and
adds the host probe there. It never imports generated code on the host and does
not write the source project during this pre-approval check. The evaluator also
requires the exact requested scope—planner manifest for the compatibility path
or observed final diff for the direct path—every required staged path, no
unapproved scope, no protected path in the overlay, and unchanged protected
source hashes.

If product verification passes but this hidden qualification probe fails, one
configured repair turn may consume the sandbox's bounded exact findings. This
is a fair coding-agent test rather than an answer-key leak: the original prompt
already states the behavior, real coding agents iterate from test failures, and
the follow-up spends the same declared repair budget. It remains in the same
conversation and project while the prior approval is undecided, using Metis's
supported staged-overlay continuation path. Before spending that turn, the
evaluator requires the durable `project.staged_resumed` event to name the
immediately preceding run and a non-empty carried file set. Before approval it
also requires an observed in-plan content change and verification after that
change. Missing continuity, an out-of-plan/no-op repair, or an unavailable
sandbox fails closed and receives no convergence or clean-rebuild credit.

After approval, the host probe runs once more against the materialized
disposable project. Reports retain every pre-approval probe under
`acceptance_history`, the exact overlay digest and path-only guard evidence on
the corresponding attempt, and the final post-approval result under
`acceptance`; generated source text is never copied into the report.

Generated code is never imported on the host: the final chained
UI/API/SQLite workflow executes inside Metis's existing rootless, networkless,
read-only Podman boundary.

The report retains raw signals and a transparent 100-point score:

| Category | Points | Signal |
| --- | ---: | --- |
| Planning/scope | 20 | Durable admission contract, required-file coverage, approval/check bounds |
| Execution | 25 | Unique authorized paths successfully written, refused-write rate, step bound |
| Verification | 20 | Verification ran and the final changeset has no blocker |
| Repair convergence | 15 | Reduction in blocking findings across follow-ups |
| Acceptance | 20 | Required chained page, upload, review, approval, Q&A, and SQLite audit probe |

Every write event records its normalized target path. Repeatedly patching one
file therefore counts as one completed plan path, not several completed files.
Reports produced from older path-less events remain readable, but label their
completion evidence as unavailable and award no completion credit for unknown
targets. A partially attributed run receives credit only for its known unique
planned paths. A `project.plan_revised` event supplies the current merged plan
while retaining the original intent and acceptance scenarios.

Acceptance is strict: a failed or unavailable required probe receives zero
acceptance points. `checks_total` and `checks_passed` describe that one required
product probe; the generic import and route-smoke counts remain available under
`sandbox_checks_total` and `sandbox_checks_passed` for diagnosis and are not
presented as completed Meridian workflows.

The required sandbox probe also checks the product artifacts around the live
workflow: the HTML must genuinely load `app.js`, the starter README must be
replaced with OCI and no-OCI setup guidance, workflow tests must cover the
critical routes, and directly used runtime dependencies must be explicit.

`--include-events` adds the durable event trace for diagnosis. `--keep-workspace`
retains the otherwise deleted disposable project. `--fail-below N` is a strict
CI/release gate: in addition to the numeric threshold, the disposable changeset
must have been approved, the required acceptance probe must have passed, and
every successful write in every attempt must have complete path evidence.
Without it, the command reports evidence without pretending every experimental
model comparison is a release gate.
