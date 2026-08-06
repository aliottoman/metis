# Tool-calling decode for hosted models

A plan for letting the Ollama provider drive a hosted model, which today it
cannot. Written 2026-08-05 from measured results, not from design taste.

## Why

Local models have hit a ceiling on whole-application builds. Benchmarked on
the same specification: every planned file delivered, and still two to seven
blocking defects; zero of eight attempts to land a repair patch; forty to
sixty-eight minutes per repair turn at roughly a minute per structured step.
The models are not confused about the task — they are imprecise across a
thirty-decision horizon. Larger hosted models are the available answer, and
Ollama Cloud is the cheapest route to them.

The blocker is not capability or cost. It is that **Ollama Cloud does not
enforce structured output.**

## What was measured

Against the real `ProjectBuildStepWireV1` contract, through the Ollama daemon:

| Model (cloud) | `format` schema | Tool calling |
|---|---|---|
| gpt-oss:120b | ignored — invented `{action,path,content}` | enforced |
| gpt-oss:20b | — | enforced |
| gemma4:31b | ignored — non-JSON | enforced |
| minimax-m3 | ignored — non-JSON | not honoured |

Also tested, and the reason this plan does not go further:

- Local `qwen3-coder:30b` tool calling: **1 of 3**, leaking raw `<fun…` tags.
  Under grammar constraint the same class of model produced **zero** malformed
  replies in ~75 steps.
- A live Ledger build on the hosted coder died after five steps on three
  unreadable replies, having written three of twelve planned files.

Two conclusions follow. The non-enforcement is a property of the platform
rather than of any model, so **a paid subscription will not fix it** — kimi,
glm and the rest will land in the same place. And converging everything on
tool calling would be a regression: the grammar path is load-bearing for the
weak models it was built for.

## The shape

Not a third protocol. Metis already speaks both dialects — the OCI provider
uses real function schemas, the Ollama provider uses grammar-constrained
JSON — and the decode protocol becomes a property of the *transport*, chosen
by one rule:

> Constrain generation where the runtime can enforce a grammar. Where it
> cannot, hand the model function schemas and validate what comes back.

Concretely, in `OllamaModelProvider`: a hosted model name (`is_cloud_model`,
already in `model_preference`) selects a tool-calling decode; everything else
keeps the existing path untouched. The tool definitions already exist as
`_unrestricted_project_tools()`, and the OCI provider already contains the
logic for turning a returned call into a `ProjectToolCallV1`.

## Milestones

Each one lands on its own and is independently verifiable. Nothing after M1
should require revisiting M1.

### M1 — Tool definitions become provider-neutral

Lift `_unrestricted_project_tools()` and the narrowing in `_project_tools()`
out of the OCI provider into a shared module, still derived from the single
canonical roster (`PROJECT_TOOL_REQUIRED_ARGUMENTS`).

*Done when:* the OCI path is byte-identical in behaviour, the existing parity
tests still pass unchanged, and a new test asserts both providers advertise
the same tool set.

### M2 — A tool-calling decode in the Ollama provider

Add the branch: send `tools`, read `message.tool_calls`, convert to
`ProjectAgentStepV1`. Reuse the OCI conversion. Handle the three failure
modes seen in testing — no tool call at all (prose), arguments as a JSON
string rather than an object, and an unknown tool name — as `ModelProviderError`
so the loop's existing malformed-reply handling covers them.

*Done when:* unit tests cover each failure mode, and a scripted provider test
drives a full build turn through the branch without a network.

### M3 — Protocol selection by transport

`_decode_structured` picks the protocol from the model name. Hosted → tools;
local → grammar, unchanged. No new setting: the model name already says which
transport it is.

*Done when:* a local model still compiles its grammar (`make verify-schemas`
passes untouched), a hosted model takes the tool path, and switching between
them mid-conversation is covered.

### M4 — Completion and the build-turn gate

The grammar path makes an empty completion *unexpressible* on a build turn
(`ProjectBuildStepWireV1`). Tool calling cannot do that structurally, so the
equivalent is tool availability: withhold nothing, but keep the host-side
premature-finish guard that already exists and is provider-independent.

*Note:* the OCI provider deliberately keeps `finish_project_task` available —
withholding it was tried and measured worse, because a model with no legal
move burns the whole budget. Match that, do not re-litigate it.

*Done when:* a hosted build turn that fabricates a completion is declined by
the same guard that declines it on the OCI path, with a test.

### M5 — Live validation on the real specification

Run the Ledger build end to end on a hosted model, then compare against the
recorded baselines: Grok at 100s / 21 steps / 16-of-17 tool calls, local at
40+ minutes with two to seven defects.

*Done when:* the run reaches a terminal card without malformed-reply
failures, and the numbers are written down next to the others.

### M6 — Per-model capability record

Tool calling is a training property, not a platform one — `minimax-m3` fails
it on the same endpoint where `gemma4` succeeds. Record which hosted models
are usable, and refuse to route a build to one that is not.

*Done when:* an unusable hosted model produces a clear configuration error at
selection time rather than three malformed replies at step five.

## Risks

- **Tool calling is a soft guarantee.** The grammar path makes malformed
  output impossible; this path makes it merely unlikely, and the existing
  malformed-streak breaker becomes load-bearing rather than a backstop. This
  is acceptable for large hosted models and is exactly why the local path is
  not being migrated.
- **Silent divergence between the two paths.** Two protocols reaching the same
  contract can drift. The canonical roster and the parity tests are the
  defence; extend them rather than adding a second source of truth.
- **Per-model variance.** M6 exists because of it. Do not assume a subscription
  model behaves like the free one that was tested.

## Out of scope

Migrating the local path to tool calling; retiring `grammar_schema` and the
flat wire contracts; any new provider or vendor. The measured 1-of-3 result
settles the first two for now, and revisiting them needs new evidence, not a
new opinion.

## Results — M5 live run, 2026-08-05

The Ledger build (the exact 5,582-char spec), end to end through the real
HTTP and approval path on `gpt-oss:120b-cloud` via the local daemon, with the
tool-calling decode from M1–M4. Throwaway data directory, one model for every
role, artifact harness forked from `scripts/project_build_smoke.py`.

| | hosted, pre-M2 (`format`) | **hosted, tool decode (this run)** | Grok / OCI | local qwen3-coder:30b |
|---|---|---|---|---|
| outcome | died step 5 | **terminal card, blocked with 19 named findings** | blocked card | blocked card |
| wall time to card | 17.8 min | **5.3 min (315 s)** | 100 s | 42.8 min |
| agent steps | 5 | **48 (full budget)** | 21 | 46 |
| malformed replies | **3 — the failure** | **0** | 0 | 0 |
| files written / planned | 3 / 12 | **7 / 12** (+6 scaffold, 13 staged) | 17 staged | 18 / 18 |
| refused writes | — | **0** | 1 of 11 | 0 |

The protocol failure is gone: every one of 48 steps decoded to a valid tool
call, the build-plan manifest decoded through the single-function path on the
first try, and the blocked card is the verification gate doing its job on
real defects (`F821 BaseModel`, a renamed `client`, three planned files never
written and named honestly).

The residual is model behaviour, not transport: gpt-oss spent 16 steps on
host-refused repeat reads/lists and left `README.md`, `app.js`, `style.css`
unwritten rather than finishing the manifest — the same per-step semantic
slip already on record for the local models, now at ~6.6 s/step instead of
~60. Worth trying next: the same run on a subscription-tier coder model
(kimi, glm), which M6's capability record now gates on measurement rather
than hope.

### Follow-up lanes, same day

**Cohere Command A+** (`command-a-plus-05-2026`, trial key, the new fourth
provider riding the same shared roster and conversion): terminal card at
**836 s / 42 steps / 0 malformed replies**, 9 of 12 planned files written,
blocked with **7** named findings (an `await` outside async, one type
default, three planned files unwritten). Better code than gpt-oss:120b-cloud
(7 findings vs 19, 9/12 files vs 7/12), slower wall clock — the trial key is
rate-limited to 20 calls/min and the model spends thinking tokens. The
transport generalised: two hosted platforms, one seam, zero protocol
failures on either.

**devstral-small-2:24b** (local, 15 GB, the strongest untried coder that fits
48 GB — devstral-2 is 123B/75 GB and qwen3-coder-next is 52 GB, both out):
the best *write discipline* any local lane has shown — 10 of 12 planned
files staged in the first 10 steps, zero refused calls, zero malformed
replies — and then a step whose generation outran the 600 s model-call
timeout, putting the turn into the timeout/repair cycle; the run was stopped
there. Per-step wall clock (~2.5 min rising with context) is the blocker on
this hardware, not correctness. A fair full run needs
`model_call_timeout_seconds` raised toward 1200 for dense-24B locals; until
then qwen3-coder:30b (MoE, ~6× faster per step) keeps the local seat despite
its messier tool use.
