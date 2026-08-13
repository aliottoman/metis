# Interview assistant

A voice interview coach that runs on its own ElevenLabs agent. It takes a job
description, tells you what the round will probe, plays the interviewer, and
then scores you honestly.

This is deliberately **not** part of Metis. No shared code, no shared config, no
shared agent. It reaches nothing in `apps/`, and Metis reaches nothing here.

- [`agent-prompt.md`](agent-prompt.md) — the base system prompt. Paste it whole.
- [`workflow.md`](workflow.md) — the five nodes, their prompts, and every edge
  condition.
- [`evaluation-node.md`](evaluation-node.md) — the debrief node, written out.
- [`app/`](app/) — a standalone page that runs the whole thing: context form,
  talk button, transcript.

## Running it

```bash
pip install -r interview-assistant/requirements.txt
```

```bash
INTERVIEW_ELEVENLABS_API_KEY=sk_... INTERVIEW_ELEVENLABS_AGENT_ID=agent_... uvicorn app.main:app --reload --port 8931 --app-dir interview-assistant
```

Then open `http://127.0.0.1:8931`. `GET /api/health` tells you which of the two
variables is missing, and `POST /api/session` names it in the error rather than
failing vaguely.

Deliberately its own variable names, its own port, its own runtime. Nothing
here reads Metis's config and nothing in Metis reads this.

## Why this page is so much simpler than Metis voice

Metis's voice mode is complicated because **Metis is the brain**: ElevenLabs
calls back into your machine over a Custom LLM endpoint, which is why it needs
cloudflared, the isolated ingress on :8788, Cloudflare Access, and a shared
secret (`docs/voice-mode.md`).

The interview agent runs on ElevenLabs' own hosted LLM. There is nothing to call
back into, so none of that machinery exists here:

| Metis voice | Interviews |
|---|---|
| Custom LLM → your tunnel | Hosted LLM, no callback |
| cloudflared + named tunnel | None |
| Voice ingress on :8788 | None |
| Cloudflare Access + shared secret | None |
| Loopback SSE for written answers | None — the transcript is enough |
| Lease renewal, sweep, teardown | Just `endSession()` |

What is shared is the *shape*: mint a conversation token server-side, hand it to
`@elevenlabs/react`'s `useConversation` with `connectionType: "webrtc"`, and let
the SDK carry the audio — the same pattern as
`apps/web/hooks/use-voice-session.ts`. When this becomes a Metis page, that hook
is the thing to copy, minus the second channel and the lease logic.

The API key stays server-side in both. The browser only ever sees a token good
for one conversation.

If this later becomes a page in Metis, that is a merge decision made on purpose,
not something that happens by accident because the two shared a directory.

## Setting up the agent

Create a **new** agent in the ElevenLabs console. Do not reuse the Metis voice
agent — that one is wired to a Custom LLM pointed at your own machine
(`docs/voice-mode.md`), and this one should not be.

| Setting | Value | Why |
|---|---|---|
| System prompt | The whole of `agent-prompt.md` | — |
| LLM | A hosted frontier model | Metis's own models are the wrong lane here; there is nothing to retrieve from your records |
| Knowledge base | Off | The prompt assumes no retrieval and says so out loud |
| Server tools | None | Same |
| Max conversation duration | 45–60 min | A mock round plus a debrief runs long; the default cuts you off mid-answer |
| Post-call webhook | Off for now | Nothing is listening |

**First message** — set it to something that gets straight to intake:

> Hey, I'm Alex. Tell me who you're interviewing with and what the role is, and paste the job description in when you get a chance.

Settings labels move around in the ElevenLabs console; the mechanisms are right
even where a name has since changed.

## Getting the job description in

You asked whether you can just paste it as text instead of reading it aloud. You
can, two ways.

**Today, no build:** turn on text input for the agent's widget (the text/voice
multimodality setting). The agent asks for the JD, you paste it into the box
mid-call, and it arrives as an ordinary user turn. The prompt already handles
this — it acknowledges a long paste in one line instead of reading it back at
you, which is the failure mode you would otherwise get.

**Later, small build:** a launcher page with a textarea and the ElevenLabs SDK,
passing the JD at conversation start so the agent has it before it says hello.
Two ways to wire it — a dynamic variable, or a prompt override. If you go the
dynamic-variable route, add this line under `# Environment` in the prompt:

```
This session is about {{company_name}}. The job description is: {{job_description}}
```

The footgun: **every** `{{variable}}` referenced anywhere in the prompt must be
supplied at conversation start, or the conversation fails outright. That is why
the variables are not in `agent-prompt.md` already — the moment they go in, a
session started from the plain widget stops working. Add them when the launcher
exists, not before.

The evaluation node uses both variables, so it has the same requirement.

## Workflow shape

```
start → intake → prep → round (interviewer subagent) → evaluation → end
                              ↑                            │
                              └──── re-run / harder ───────┘
```

`agent-prompt.md` is the base every node inherits — the two registers, the
character discipline, the anti-fabrication rules. Node prompts add the specifics
for their step. The debrief node is written out in
[`evaluation-node.md`](evaluation-node.md), including the one setting that will
quietly break it: if the evaluation node does not inherit the round's
conversation history, it is grading an interview it cannot see.

## What changed from the original prompt, and why

The bones were fine. Five things were actively working against you:

**It told the agent to research the company.** "Conduct internal research on the
company, its values, recent news, and industry position" — with no web access
and no knowledge base, that instruction cannot be followed, so the model
satisfies it by inventing plausible facts about a real employer you are about to
sit in front of. This is the worst failure mode the system has. The prompt now
states the limit in `# Environment`, and `# Guardrails` requires the agent to
mark the difference between "I know this about this company" and "I'm reasoning
from the role and the industry."

**Feedback after every answer meant it was never a mock interview.** Real
interviewers do not stop to tell you how you are doing. Feedback now banks
silently during the round and lands in the debrief, with per-answer coaching
available as an explicit opt-in when you want to drill rather than simulate.

**One tone for two incompatible jobs.** "Warm, thoughtful, encouraging, 2-3
sentences" is right for a coach and disastrous for a hiring manager — a mock
interviewer who is nice to you teaches you nothing. There are now two registers
and a hard rule against mixing them, plus a `pause` safe word that breaks
character on demand so a round never becomes a trap.

**The no-answers guardrail collided with the whole point.** "Never provide
actual interview answers" versus your ask for tips and prep. The real line is
not answers-versus-no-answers, it is *their* material versus invented material:
the agent can structure, sharpen, and build worked examples from what you have
actually done, and cannot hand you an achievement you never had or a script to
memorise.

**It promised delivery feedback it cannot produce.** You want to be told about
the "um"s. The agent sees a transcript, and transcripts are typically cleaned of
disfluency before the model ever sees them — so the honest instruction is to
report fillers only when they actually appear, never to estimate a count. What
it *can* always see is the higher-value stuff anyway: hedging, rambling,
self-undermining, adjectives where a number belonged.

Two things were added that were not there at all: five distinct interviewer
personas with their own registers, and a fixed five-dimension rubric with
anchored scores, so a 3 next month means what a 3 meant today.

## Not built

- Anything written down. The debrief is spoken; the call transcript is the only
  record. A post-call webhook writing a structured scorecard is the obvious next
  step and is not started.
- Live company research. Real web search means an isolated webhook or MCP
  server, and that is its own task.
- Any UI. Widget only.
