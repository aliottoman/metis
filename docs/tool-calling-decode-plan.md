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
