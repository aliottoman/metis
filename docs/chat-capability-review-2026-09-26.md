# Chat capability review — 2026-09-26

This review follows the reported Cline release question through Metis's actual
chat path. It separates changes shipped in this branch from capabilities that
still require implementation or live qualification.

## What happened in the reported run

Run `run_a1f45f0589144d5da6289ab17dd49f63` took about 66 seconds. The
`context.retrieved` event recorded `knowledge_scope=auto`,
`web_requested=false`, `web_source_count=0`, and 23 local knowledge snippets.
Those 23 snippets were not Cline release sources. The initial Auto rule matched
`recent` but missed `recently`; “new features” also had no freshness cue. The
question therefore entered private corpus embedding and reranking instead of
web research. A planning call then found no relevant registered action tool.
The first answer used none of the local passages, and the score-only grounding
gate ordered a second answer attempt. The final answer said that the supplied
context contained no Cline release information, which was true of the wrong
context but failed the user's question.

The durable event times put about 9 seconds in corpus retrieval, 13 seconds in
planning, 20 seconds in the first answer, and 24 seconds in its revision. The
interface displayed 18 internal events and counted all 23 retrieved snippets as
“grounded” sources even though the answer cited none. These are run-specific
measurements, not general latency benchmarks.

## Current routing and tool boundaries

| Request | Current path | Capability and limit |
| --- | --- | --- |
| Ordinary chat | Python control plane: ingest, retrieve, plan, synthesize, grounding review, publish | The answer call receives context as text. It has no iterative search/open/tool loop. Some clear questions skip the planner. |
| Sources: Auto | Heuristic classification, then either public web retrieval or private corpus retrieval | This branch recognizes the reported release wording. Other mixed public/private questions can still be misclassified. |
| Sources: Web | Metis host web retriever searches or opens a supplied URL, reads up to four public pages, and adds linked snippets | This is a bounded one-shot search. DuckDuckGo HTML is a fragile search backend; the model cannot refine queries or open a second result after seeing the first answer. |
| Sources: Notion | Synced Notion corpus only | The answer fails closed when it finds no Notion evidence. |
| Project building | Metis supervises a pinned ClineCore SDK sidecar in a disposable coding mirror | Cline handles the edit loop with read, search, editor, and host check tools. Web, shell, MCP, skills, subagents, and other tools are disabled by the coding policy. Metis independently verifies and gates final writes. |
| ClinePass direct chat | Python model provider calls the ClinePass chat completion gateway | This is a model transport, not a ClineCore agent session. It does not inherit ClineCore's search tools or tool loop. |

The sidecar pins `@cline/sdk` 0.0.86. [Cline's SDK changelog](https://github.com/cline/cline/blob/main/sdk/CHANGELOG.md)
describes provider-native search becoming the default for supported non-yolo
sessions in 0.0.83. Metis explicitly disables native search before its coding
sidecar starts ClineCore and checks the stored setting. That change cannot add
search to Metis's direct chat, which does not run through ClineCore. The
[provider capability manifest](https://github.com/cline/cline/blob/main/sdk/packages/llms/src/providers/builtins.ts)
also limits native search to supported provider and model routes. Metis must
retain a provider-independent web path for OCI, Cohere, Ollama, and other
unsupported routes. Disabling `fetch_web_content` alone does not control
provider-native search, so the sidecar also asserts the separate native-search
setting at startup.

## Changes in this branch

1. Auto now recognizes the reported “released recently” question and similar
   release/changelog wording as a public freshness request. A regression test
   sends the exact prompt through the API and proves it uses web evidence,
   avoids corpus embedding/reranking and a separate planner call, and returns
   a linked source from one answer call. Recent release questions also search
   for focused release notes in parallel with the original wording and rank
   changelogs ahead of reviews, social posts, and index pages.
2. Safe factual questions can take the direct answer path; explicit action
   requests still go through planning and policy.
3. The conversation's activity row stays folded, is about one line tall, and
   expands to a short list of meaningful milestones. The task panel retains
   the complete event record. Internal verifier and model events no longer
   dominate a chat answer.
4. The answer footer counts sources actually cited in the delivered answer.
   An uncited answer receives no “grounded” badge, and the review timeline no
   longer claims that a score check proves grounding. Grouped markers such as
   `[1, 2]` now link both sources; invalid numbers and code indices cannot
   silently inflate the source count.
5. Earlier commits in this branch added Metis's bounded web source with linked
   citations, chat composer and scroll refinements, run lifecycle fixes, and
   incremental ClinePass gateway streaming.
6. Direct OCI Responses and Cohere chat generation now emit answer text as
   their providers stream it. Planning and structured calls retain their
   existing response handling; streamed errors and cancellation are bounded.
7. The coding sidecar now pins Cline SDK 0.0.86 with an explicit, verified
   native-search opt-out in its isolated settings. Protocol handshakes and
   tests use the same version; user chat still uses Metis's research path.
8. ClinePass ordinary chat can answer stable general-knowledge questions when
   no snippets are available. Its structured and project prompts remain
   bounded to supplied context. A disposable live question about Python list
   comprehensions completed in 13 seconds with an appropriate answer.
9. The project capability evaluator can explicitly launch a selected local
   Ollama model before a disposable live run, then restore its role routing;
   it never launches a model by default.
10. The live project run exposed a misleading direct-coding prompt: with no
    host file manifest, it said there were no existing files and no files to
    create. The prompt now asks Cline to inspect the actual tree, follow the
    task's requested paths, and use host checks instead of scratch test files.
    This prompt change has focused tests but has not passed a new live build.

## Remaining work, in priority order

1. **Research agent:** give user-facing chat a bounded search/open/refine loop
   with clear public-query extraction, provider-independent result objects,
   URL validation, source provenance, and citations tied to actual claims.
   Preserve private scopes. Compare results with Cline's native search on
   supported ClinePass models, using the same release, news, URL, and mixed
   local/public evaluation prompts.
2. **Answer quality and speed:** make corpus relevance a semantic decision,
   not only a rerank-score threshold. Do not order a full second generation
   because irrelevant snippets were uncited. Record and graph retrieval,
   planning, first-token, generation, and revision latency separately. Keep
   single-call factual answers while retaining planning for mutations and
   registered tools.
3. **Streaming measurement:** all four direct chat provider paths now expose
   incremental text, but OCI and Cohere were validated with protocol mocks,
   not live credentials. Measure first-token latency and cancellation for
   every selected provider in an actual app session, and catch any provider
   wire-format differences there.
4. **Cline SDK live qualification:** the 0.0.86 sidecar compiles and passes its
   offline policy and integration checks. Evaluate the newer SDK's
   transient-error retries, more accurate compaction, output-limit recovery,
   and faster event forwarding in real project runs against Metis's own retry
   limits. Keep native web research in user chat separate from code-edit
   permissions.
5. **Project capability gate:** mocked sidecar and deterministic end-to-end
   tests pass, but one live local repair failed its qualification gate. Run the
   existing live qualification funnel in
   `docs/cline-coding-engine.md` and `docs/project-capability-evaluation.md`
   with actual model routes. Record task success, exact deliverables,
   first-token time, tool failures, iterations, retries, cost, and false-clean
   outcomes before changing default project capabilities.
6. **Conversation polish:** visually test streaming scroll lock, long-draft
   composer resize, keyboard focus, reduced motion, mobile layout, and source
   links against live API streams. The UI code now handles these cases, but a
   live browser smoke remains useful before calling the experience finished.

## Verification for this branch

The focused web/routing suite passed 43 tests and Ruff. The provider suite
passed 63 tests, including an installed OpenAI SDK SSE decoder test, and Ruff.
Frontend TypeScript and five source-link tests passed. The upgraded sidecar's
direct Node suite passed 70 tests. A full API suite before the SDK upgrade in
the restricted environment passed 1,943 tests;
10 failures were caused by blocked local bind or an unavailable sidecar build.
After compiling the sidecar and running those four affected test files with
local bind available, all 23 tests in those files passed. After the SDK upgrade,
51 focused API project tests passed in the restricted environment; the remaining
Podman-dependent case passed when rerun with Podman available. A combined
follow-up suite covering the revised search, citations, ClinePass prompt, and
evaluator and direct coding prompt passed 147 tests and Ruff. A live ClinePass
chat run for the exact
reported question completed in 36 seconds, requested four web pages, and
answered rather than refusing; it exposed grouped citations that were not
linked. The next live run completed in 28 seconds with linked citations and
official changelogs among the retrieved pages, but its answer still favored
general SDK articles over the latest version details. After source ranking and
release-answer guidance changed, a third run completed in 32 seconds. It put
the official SDK changelog first, linked it, and described specific 0.0.86
recovery and persistence changes. It still omitted provider-native web search
from 0.0.83 and did not name the version in its prose. A final small source
ranking fix demoted generic blog index pages; its ordering test passed, but it
was not separately tested against a live model. These are single-run
measurements, not a latency or accuracy benchmark. The one-shot search and
answer path remains less complete than a bounded research agent that can open
and refine sources after the first search.

The `fastapi-repair` live evaluator ran ClineCore 0.0.86 with the installed
`qwen3-coder:30b` local model on a disposable project. It ran for 514 seconds,
reported about 605,000 cumulative input/output tokens, and staged eight writes
across six unique paths. The model did not modify the required
`tests/test_incidents.py`, added temporary test files outside the requested
four-file scope, and left a `app/repository.py` type error. Metis correctly
blocked approval; all five non-continuity qualification gates failed. This
shows real sidecar file/check activity and a real task failure, not a proven
project-building success. A separate attempt using Ollama Cloud's
`deepseek-v4-pro:cloud` never began coding because that account's current
entitlement excluded the model. Neither result measures ClinePass's project
coder route. The current direct contract permits project-wide writes so new
files can be created; it does not enforce a user-requested exact file list at
tool time. That is a separate contract and policy change to design and test,
not something the prompt fix alone guarantees.
