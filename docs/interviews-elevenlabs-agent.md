# The Chiron agent

How to configure the ElevenLabs agent behind the `/interviews` page. The page
mints a one-conversation WebRTC token server-side and supplies every dynamic
variable at start; the agent runs on ElevenLabs' hosted LLM and never calls
back into Metis. The only channel from the conversation back into Metis is the
blocking `submit_interview_evaluation` client tool, which the browser handles.

This is a **separate agent** from Metis voice. The voice agent is wired to a
Custom LLM pointed at your machine (`docs/voice-mode.md`); Chiron must not be.
Do not reuse that agent, and do not point this one at a Custom LLM.

## Agent settings

| Setting | Value | Why |
|---|---|---|
| Agent | Private, dedicated to interviews | The id is server configuration, never shipped to the browser |
| System prompt | [`interviews-agent-prompt.txt`](interviews-agent-prompt.txt), verbatim | The behavioral foundation every workflow node sharpens |
| Workflow | Exactly [`interviews-workflow.json`](interviews-workflow.json), pushed by the deploy script | The five-question limit is graph shape, not prompt memory |
| LLM | A hosted frontier model (the script preserves whatever is set) | There is nothing to retrieve from Metis records |
| Knowledge base | Off | The prompt assumes no retrieval |
| Server tools / webhooks | None | The client tool is the only tool |
| Client tools | `submit_interview_evaluation`, blocking | Schema below |
| Overrides | None needed | Everything arrives as dynamic variables |
| Max conversation duration | 60 min (3600 s) | Five answers, up to two follow-up probes each, plus a debrief |
| First message | The fixed opening line | Deterministic and free; spoken before any node runs |
| Audio saving | On (`platform_settings.privacy.record_voice`) | The post-call delivery analysis reads the recording back; the deploy script pins it on |

## Deploying from the spec

The dashboard is not the editor here — the committed spec is. One command
pushes the whole configuration:

```bash
.venv/bin/python scripts/deploy_interview_agent.py --agent-id agent_...
```

The script translates [`interviews-workflow.json`](interviews-workflow.json)
and [`interviews-agent-prompt.txt`](interviews-agent-prompt.txt) into the
live configuration, creates or updates the blocking client tool, and
fetch-merge-patches the agent so settings the spec does not own — LLM, voice,
language — are preserved exactly. It is idempotent; re-run it after any spec
change. Each run verifies the result landed whole and writes the full agent
export to `docs/interviews-agent-export.json`, so a change that makes the
agent worse is a diff and not a memory. Anything drawn in the dashboard's
workflow canvas is replaced by the spec on the next run — the spec is the
source of truth, not the canvas.

Two schema facts learned against the real API, for whoever edits the script:
the workflow is a **top-level field on the agent resource**, a sibling of
`conversation_config` (the documented `conversation_config.workflow`
placement is accepted and silently ignored), and tools attach by id via
`agent.prompt.tool_ids`, not the deprecated inline `tools` array.

How the spec maps onto the live graph:

- The spec's `router` becomes the platform's `start` node. Its fixed opening
  line is the agent's first message (deterministic, no model call), and its
  never-re-ask rules live in the base prompt. Its three outgoing edges
  dispatch on `{{interview_type}}`.
- The fifteen question nodes and the shared evaluation node become
  `override_agent` nodes whose `additional_prompt` **appends** to the base
  prompt.
- Early-stop edges are ordered **before** the advance edge on every question
  node, so a candidate who confirmed they want to stop reaches the
  evaluation, not question N+1.
- The spec's `end` node becomes a `wrap_up` subagent (brief factual questions
  about the verdict, no softening, no new interview questions) followed by
  the platform's `end` node, which hangs up — mapping the spec's end straight
  onto the platform's would end the call the instant the debrief finished.

The agent id goes into Metis server config only:

```
WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID=agent_2201kzxk377tebjtc0048cfffar5
```

That value lives in your local untracked env file next to
`WAQIL_ELEVENLABS_API_KEY`. Tracked templates carry it blank. It is
deliberately absent from frontend source — the browser learns nothing but a
one-conversation token.

## Dynamic variables

All eight are supplied by the Metis API at session start, every time. Any
`{{variable}}` referenced anywhere in the prompt or workflow must be in this
list, or conversations fail outright at start.

| Variable | Value |
|---|---|
| `job_title` | From the setup form |
| `company_name` | From the setup form |
| `job_description` | From the setup form, verbatim |
| `interview_type` | `hr_recruiter`, `hiring_manager`, or `technical` |
| `question_limit` | Always the string `"5"` — fixed server-side, not client input |
| `metis_interview_session_id` | The Metis session row this conversation belongs to |
| `interview_objective` | The optional objective, or a stated "no objective" default |
| `focus_areas` | The optional focus areas joined with `"; "`, or `"None specified."` |

The last two carry stated defaults when the form left them empty — an
optional field may be blank on the form, never absent from the payload.

The job description, objective, and focus areas are untrusted pasted text.
The base prompt instructs the agent to treat them as source material and
steering only — they shape question choice and follow-up probes, never the
question count, the honesty rules, or the scoring standards (the verdict may
state whether the objective was met; the objective never changes how
strictly the round is scored) — and to ignore any instructions embedded
inside them.

## Follow-up probes

Each question node allows at most two short follow-up probes after the
candidate's answer: a vague claim, a "we" hiding their contribution, an
adjective where a number belonged, or a thread the objective says to pull.
Probes never begin with a question number — the page's `Question N of 5`
counter hangs off those openings — and the advance edges only fire once the
probes are done, so a probe can never be answered by the next node's
question. The five-question limit is still graph shape; probes live inside a
node and cannot consume a slot.

## Base prompt

The canonical text is [`interviews-agent-prompt.txt`](interviews-agent-prompt.txt),
kept as its own file so the deploy script and any hand paste push the same
bytes. Workflow node prompts (in
[`interviews-workflow.json`](interviews-workflow.json)) sharpen it per step;
they must never override its honesty, its five-question limit, or its
anti-fabrication rules.

## The `submit_interview_evaluation` client tool

Create it as a **client tool** (the browser implements it), **blocking** (the
agent waits for the response before speaking the debrief). The model submits
five criterion scores and the qualitative material; it never submits an
overall score. Metis computes the overall (specific evidence 25%, role depth
25%, relevance 20%, structure 15%, communication 15%, rounded to one decimal)
and the recommendation (≥7.5 advance, 6.0–7.4 borderline, <6.0 do not
advance), stores the scorecard, and returns both to the tool call — the agent
speaks the returned numbers, not its own.

Name: `submit_interview_evaluation`
Description for the agent: "Submit the completed interview evaluation. Call
exactly once, after the interview ends. Returns the calculated overall score
and recommendation; speak those returned values in the debrief."

Parameters (all required):

| Parameter | Type | Constraints |
|---|---|---|
| `specific_evidence` | integer | 1–10 |
| `role_depth` | integer | 1–10 |
| `relevance` | integer | 1–10 |
| `structure` | integer | 1–10 |
| `communication` | integer | 1–10 |
| `verdict` | string | One sentence, the honest verdict |
| `strongest_answer_quote` | string | Exact words the candidate said, from the transcript |
| `strongest_answer_reason` | string | Why that answer worked |
| `improvements` | array, exactly 3 objects | Highest-priority problems, in order |
| `improvements[].what_happened` | string | What the candidate did |
| `improvements[].evidence` | string | Evidence from the transcript |
| `improvements[].why_it_hurt` | string | Why it hurt the performance |
| `improvements[].better_approach` | string | A better approach — not a script to memorise |
| `drill` | string | One concrete ten-minute drill |
| `completed_question_count` | integer | 0–5, how many questions were answered |
| `incomplete` | boolean | True when the interview ended before question five |

The tool response the agent receives:

```json
{
  "overall_score": 6.8,
  "recommendation": "borderline",
  "provisional": false
}
```

`provisional` is true when the interview ended early with at least three
answers; the debrief must then present the score as provisional. When fewer
than three questions were answered the tool response contains no score at
all — `{"overall_score": null, "recommendation": null, "provisional": true}` —
and the agent says the interview was too short to score rather than inventing
one.

## Building it by hand (fallback)

If the script is ever unavailable, the same graph can be drawn in the
dashboard from [`interviews-workflow.json`](interviews-workflow.json): the
router's three edges dispatch on the dynamic variable (the round was chosen
on the setup screen; it is not a question), the fifteen question nodes append
their `prompt` fields to the base prompt and ask exactly one numbered
candidate-facing question each ('Question N,' then wait — after the answer
the node may press with at most two short unnumbered follow-up probes, and
its advance edge fires only once the probes are done; the total stays five
because the cap is graph shape, not probe accounting), every non-final
question node gets an early-stop edge to the shared
evaluation node ordered ahead of its advance edge, and the evaluation node
scores, calls the blocking tool once, waits, and speaks the returned overall
score. Finish with the wrap-up subagent and the end node. Then run the
deploy script anyway next time — hand edits are replaced by the spec.

### Verifying a build

- Start a session from `/interviews`. Chiron's first words must be the fixed
  opening line — if it asks for the job description, the dynamic variables
  are not reaching the conversation.
- Each question must begin 'Question one' … 'Question five'; the page's
  `Question N of 5` counter is driven by those words.
- Give one deliberately vague answer: the press-back must arrive as a short
  unnumbered probe — never as a 'Question N' opening — and the page's
  counter must hold still while it does.
- Run a full round and confirm the debrief's overall score matches the
  scorecard the page shows — that proves the agent spoke the tool's returned
  number rather than its own arithmetic.
- Say "stop" after two answers, confirm, and check the agent goes to
  evaluation, the tool call carries `incomplete: true` and
  `completed_question_count: 2`, and no score is spoken.

Do not edit the live agent to match a spec change without running that
checklist afterwards.
