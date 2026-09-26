# Chat evidence routing evaluation — 2026-09-26

## Failure under review

The reported question asked what Cline had released recently for its harness.
Metis treated it as a private-knowledge question. The original run retrieved 23
local snippets, no web pages, and ended by saying the provided context contained
no Cline information. A second answer attempt made the run slower without
repairing the missing source. The earlier [capability review](chat-capability-review-2026-09-26.md)
records the event trace and timings.

The immediate bug was a missed word form in a freshness regex. That exposed a
larger design fault: Auto chose **one** source lane from keywords before it knew
what evidence the question actually needed. A question could need public facts,
private records, both, or neither. A selected project also has its own action
gate. Adding more release keywords could not make that decision reliable across
paraphrases or mixed questions.

## Research and design choice

[Cline SDK 0.0.83](https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md)
enabled provider-native web search by default on supported model/provider
combinations in non-yolo sessions. That is a ClineCore agent-session feature.
Metis's ordinary chat sends ClinePass through its Python model provider, so it
does not acquire the ClineCore tool loop or native search automatically. Its
project coding session does use ClineCore, but has native search disabled by
Metis's coding policy. The SDK's [agent package](https://github.com/cline/cline/blob/main/sdk/packages/agents/README.md)
and [tool documentation](https://github.com/cline/cline/blob/main/docs/sdk/tools.mdx)
show a richer agent path for future integration; that path needs explicit host
permissions and evaluation before becoming ordinary chat.

The hosted [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search)
and [Claude web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
tools illustrate the distinction between allowing search and requiring it for
a current claim. Metis supports multiple model providers, so the new source
decision is host-owned and provider-independent. A structured model call plans
the evidence; the host validates public queries, owns explicit scope, performs
retrieval, keeps the project action policy separate, and attaches URLs to
citable passages. For changeable claims, a second bounded check can identify a
coverage gap and request one follow-up search before answer generation.

## Implemented path

1. **Source plan.** Auto asks a structured planner for `web`, `private`, both,
   or neither; up to two public-only search queries; short focus terms; and a
   read-only/action advisory. The planner gets bounded recent conversation to
   resolve follow-ups. User-selected Web and Notion scopes take precedence.
2. **Privacy boundary.** Only validated public queries reach the search engine.
   Auto never falls back to searching the raw message, including when the
   planner fails. User-supplied public URLs can be opened directly. Private
   markers, credentials, local paths, and suspicious tokens are rejected by
   the query filter. This is a conservative filter, not a proof that every
   novel private name can be recognized.
3. **Retrieval.** Web and private knowledge can be combined; public fetching
   begins while private context loads. The user can compare personal notes with
   current release records without losing either source. A needed web source
   that cannot be retrieved stops a project edit before the coding session.
4. **Coverage and answer.** Recent public claims may get one source-coverage
   review and one follow-up search. A narrow host check recognizes a
   substantive first-party GitHub changelog for the requested release track
   and skips that extra model call. For harness requests, a public SDK
   changelog query is added when the semantic plan omits that release track.
   The answer prompt labels each passage's
   provider, favors first-party changelogs for release claims, cites the
   numbered passage, and states any remaining unsupported point. Public
   references are passed into the Cline project prompt as bounded, untrusted
   task evidence for research-and-edit requests.
5. **Document extraction.** GitHub Markdown changelog pages are fetched from
   the public raw host and cited with their readable GitHub URL. The excerpt
   samples substantial bullets from recent version sections. A live read of
   the Cline SDK changelog reduced 75,454 source characters to a 3,170-character
   excerpt that retained 0.0.86 through 0.0.83 and the provider-native search
   bullet. Redirects and resolved addresses remain checked before fetching.
6. **Failure behavior and activity.** A planner/provider error emits
   `evidence.plan_failed` and preserves an apparent public-source requirement
   without releasing the whole prompt as a query. The activity panel remains
   folded for ordinary chat; full source and model events remain available in
   the task details. The `model.response` event now reports ClinePass as Cline
   when provider metadata is absent.

The source planner does not authorize writes or registered tool calls. Those
still pass through Metis's existing action planner and policy gate. In
particular, the model's `action=answer` advisory is not used as permission to
bypass that gate.

## Evaluation

The evaluation corpus contains 29 labelled source questions with public,
private, mixed, stable, explicit-scope, and pasted-instruction cases. It also
contains five project/action cases and four source-gap cases. These examples
measure source decisions and public-query handling; they do not prove answer
quality. A second holdout set uses unrelated product names and varied private
wording.

| Measure | Previous keyword route | Structured source plan |
| --- | ---: | ---: |
| Labelled source choices | 14 / 29 | 29 / 29 |
| Labelled project/action advisories | — | 5 / 5 |
| Median source-plan latency, configured ClinePass | — | 5.68 s |
| p95 source-plan latency, configured ClinePass | — | 11.06 s |

The new result is one pass over a small, deliberately varied synthetic set.
It is evidence of improvement over this route and these cases, not a general
accuracy guarantee. On a separately written 18-question holdout using other
software products, private phrasing, mixed requests, and quoted text, the
keyword route scored 3/18 and the structured plan scored 17/18. Its one miss
unnecessarily selected private retrieval for a memo fully included in the
message; it did not search the web or leak the quoted instruction. The holdout
median source-plan latency was 7.75 seconds, with a 13.38-second p95.

The structured planner is an extra model call, so simple
questions can start their answers later. For live end-to-end quality, the exact
reported Cline question and a stable Python question are run against the real
configured ClinePass model in a disposable Metis data directory. No existing
conversation or project is modified by that smoke test.

The final exact-question smoke completed in **50.85 seconds**, with its first
answer text at **42.22 seconds**. Its final answer cited the official SDK and
main Cline changelogs, named SDK 0.0.83–0.0.86, and included provider-native
web search with the supported-provider condition. The first ~8.1 seconds were
source planning; public fetch and the official-changelog check finished by
~10.0 seconds; most of the remaining wait was ClinePass answer prefill before
the first streamed text. A stable Python list-comprehension question completed
in 19.11 seconds with no web or corpus search and first text at 16.19 seconds.
These are single runs, not a statistically meaningful latency comparison.

The intermediate live passes were useful negative controls: one retrieved the
SDK changelog but placed it behind generic summaries, then omitted native web
search; another missed the SDK track and spent 35 seconds on a model coverage
check before a provider failure. The source-track query validation and narrow
official-changelog check address those observed faults. They do not remove the
remaining answer-model delay.

## Limits and next qualification

- The source planner can still misclassify unseen language or produce a weak
  public query. Private proper nouns with no obvious private marker need more
  adversarial testing before claiming a complete query privacy guarantee.
- The current public search backend is DuckDuckGo HTML. It is keyless and
  bounded, but result markup and coverage can change. A production search API
  or an evaluated provider-native search path would be more resilient.
- The current follow-up is limited to one extra search. It does not yet offer a
  full model-driven search/open/read loop for ordinary chat.
- The answer is still subject to model latency and grounding errors. Compare
  full answers, citations, first-token time, and false refusals on real public,
  private, and mixed tasks before expanding the default tool surface.
- A selected project still uses the existing project route and coding policy.
  The source planner changes what evidence it gets, not what writes are
  authorized. Cline SDK features such as native search should be qualified in
  that policy before enabling them there.

## Follow-up: model speed and Cline native search

These are disposable probes, not changes to the running app or a basis for a
global model switch. All research turns used the same PR branch and temporary
Metis data. A short synthetic answer looked much faster on MiMo, but full turns
varied widely:

| Full turn | Qwen 3.7 Plus | MiMo 2.5 |
| --- | ---: | ---: |
| Stable question | 18.94 s | 23.23 s |
| SDK 0.0.83 question | 37.02 s | 69.87 s |
| Current Dubai weather | 33.85 s | 46.33 s |

MiMo 2.5 did answer the exact reported Cline question in 26.64 seconds with an
SDK changelog citation, compared with the earlier 50.85-second Qwen run. On the
three matched questions Qwen was faster overall, and both models had weak
weather sourcing. MiMo 2.6 Flash routed six synthetic cases and started a
short answer quickly, but missed the SDK changelog in the exact Cline question
at that stage of the source-ranking work.
`cline-pass/deepseek-v4-flash` and `cline-pass/kimi-k2.6` returned model-not-found
from the current gateway, so they were removed from Metis's advertised list;
the live-verified `cline-pass/mimo-v2.6-flash` was added. The default remained
Qwen at this point in the evaluation; the final chat-only rollout is below.

The live saved preference also combined the Cline provider with a pinned Ollama
model ID, `gpt-oss:120b-cloud`. Cline's direct chat provider ignored that pin;
the UI could therefore claim a model selection that never reached Cline. The
branch now passes valid Cline pins to the gateway and presents an incompatible
old pin as split mode. It rejects new mismatched pins at save time.

An isolated ClineCore 0.0.86 research pilot confirmed ClinePass can invoke its
provider-native `web_search`. With Qwen 3.7 Plus, a search-only run spent 34.8
seconds on eight searches and returned no answer. A second run allowed only
read-only fetching of official Cline GitHub pages, spent 50.5 seconds on six
searches and five fetch attempts, and reached its five-iteration limit without
an answer. The fetched SDK changelog did include the requested 0.0.83 feature.
These results do not make native search a reliable or faster replacement for
the current host research path. A separate read-only Cline research lane needs
more model and prompt evaluation before it can serve user chat.

## Optional production search backend

The branch now supports Brave's LLM Context API when
`WAQIL_BRAVE_SEARCH_API_KEY` is configured. Its linked, extracted passages are
used directly as source evidence, so ordinary results do not need an extra
page fetch. User-supplied URLs keep the same validated direct-read path. With
no key, Metis continues to use DuckDuckGo HTML. A configured Brave error is
surfaced as a web retrieval failure rather than silently falling back to HTML.
Brave's [API documentation](https://api-dashboard.search.brave.com/documentation/services/llm-context)
defines the endpoint and response shape. The account currently has no Brave
search key, so only mocked contract tests have run; source quality and latency
need a live, key-backed comparison before treating it as qualified. This
adapter also does not yet give the answer model an iterative search/open loop.

## Final chat-only rollout

After repairing the subject-bearing public query, filtering similarly named
third-party repositories when the official component changelog is present,
and surfacing distinct release capabilities before synthesis, the exact
reported Cline question produced an answer citing Cline's SDK and app
changelogs. It included SDK 0.0.83 provider-native web search, its supported
provider/model condition, hooks, concurrent subagents, compaction, and grouped
retry fixes. With MiMo 2.6 Flash for answer synthesis, the disposable full turn
finished in 35.80 seconds and began answer text at 25.27 seconds. The earlier
Qwen turn on the same question took 82.85 seconds and omitted web search.
These are individual live runs with different answer prompts and network
conditions, not a controlled speed benchmark or a guarantee for every turn.

A stable Python question took 13.79 seconds, began answer text at 8.12 seconds,
and made no public search call. In split ClinePass mode, Flash now serves chat
answers only; configured planner chains continue to handle planning, and a
pinned single model still applies to all roles. The final guard can add one
cited information-access finding from a validated official changelog if a
broad app-adoption answer omits it. It makes no extra model call and never
copies arbitrary page text into the answer. The active app still uses the
keyless DuckDuckGo HTML backend until a Brave key is configured; native Cline
web search remains disabled in the guarded coding sidecar.
