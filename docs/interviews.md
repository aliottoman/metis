# Interviews

A five-question spoken mock interview at `/interviews`. You give it a job
title, a company, the pasted job description, and which round to run; Chiron —
a private ElevenLabs agent — asks exactly five questions, one at a time, then
scores the round honestly. No coaching mid-round, no rehearsed praise, and no
score the backend didn't calculate.

## Why this is not voice mode

Metis voice mode is complicated because Metis is the brain: ElevenLabs calls
back into this machine over a Custom LLM endpoint, which is why voice needs
cloudflared, the isolated ingress, leases, and a shared secret
(`docs/voice-mode.md`). The interview agent runs on ElevenLabs' own hosted
model. There is nothing to call back into, so none of that machinery exists
here:

| Voice mode | Interviews |
|---|---|
| Custom LLM → tunnel into this machine | Hosted LLM, no callback |
| cloudflared + ingress + shared secret | None |
| Lease renewal and sweeps | `endSession()` and an idempotent end call |
| Loopback SSE for written answers | None — the transcript panel is enough |

What is shared is the shape that matters: the API key stays server-side, and
the browser receives only a WebRTC conversation token good for one
conversation. The one channel back into Metis is the blocking
`submit_interview_evaluation` client tool, which the browser handles by
calling the Metis API.

## Configuration

Two settings, both server-side:

```
WAQIL_ELEVENLABS_API_KEY=            # already used by dictation/meetings/voice
WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID= # the Chiron agent, created by you
```

Create the agent once in the ElevenLabs console, then deploy its entire
configuration — base prompt, workflow graph, blocking client tool, duration
cap — with `scripts/deploy_interview_agent.py`, which translates the
version-controlled spec ([`interviews-workflow.json`](interviews-workflow.json)
and [`interviews-agent-prompt.txt`](interviews-agent-prompt.txt)) onto the
live agent and writes the export back to `docs/interviews-agent-export.json`.
Details in [`interviews-elevenlabs-agent.md`](interviews-elevenlabs-agent.md).
The agent id is configuration, not code: it appears in your local `.env`,
never in frontend source. `GET /api/v1/interviews/availability` names
anything missing.

## How a session runs

1. The setup form collects role, company, job description, and round. The
   draft persists in localStorage, so accidental navigation does not eat a
   pasted JD. Start stays disabled until all four are valid.
2. `POST /api/v1/interviews/sessions` persists the session, mints the token,
   and returns the dynamic variables — `job_title`, `company_name`,
   `job_description`, `interview_type`, `question_limit` (always `"5"`,
   stamped server-side), and `metis_interview_session_id`. The job
   description is passed as untrusted source material; the agent prompt tells
   Chiron to ignore any instructions inside it.
3. The browser starts the WebRTC conversation with those variables. The page
   tracks progress by the spoken "Question one" … "Question five" openings,
   shows a `Question N of 5` counter, an elapsed timer, mute, a typed-reply
   fallback, and a subdued transcript panel. Every transcript line is
   appended to the session as it is heard.
4. After the fifth answer Chiron says "That's the interview. I'm evaluating
   it now." and calls the blocking `submit_interview_evaluation` tool with
   five criterion scores and the qualitative material — never an overall
   score. The backend computes the overall (specific evidence 25%, role
   depth 25%, relevance 20%, structure 15%, communication 15%, rounded half-up
   to one decimal) and the recommendation (≥7.5 would advance, 6.0–7.4
   borderline, <6.0 would not advance), stores the scorecard, and returns
   both to the tool. Chiron speaks the returned number; the page shows the
   same stored scorecard. If the tool is never called, the page shows a
   recoverable error and keeps the transcript — it never invents a score.

### The product state machine

The page's states are explicit, not inferred from the SDK connection:
`setup → ready → connecting → listening ⇄ agent_speaking → evaluating →
complete`, with `failed` and `ended_early` as exits. Teardown fires on
explicit stop, unmount, navigation, `pagehide`, and connection failure; the
server treats a repeated end as a no-op, so overlapping teardowns cannot
conflict.

### Ending early

The End button confirms once, then ends the round unscored. Telling Chiron
you want to stop is different: it confirms once, then evaluates what exists.
With at least three answered questions the scorecard comes back labeled
**Provisional**; with fewer, the evaluation stores but no number is computed
— two answers are not an interview, and the honest response is to say so.

## API surface

| Route | What it does |
|---|---|
| `GET /api/v1/interviews/availability` | Whether a session can start; names every missing setting |
| `POST /api/v1/interviews/sessions` | Persist context, mint token, fix `question_limit` to 5 |
| `GET /api/v1/interviews/sessions/{id}` | Session, transcript, scorecard |
| `PATCH /api/v1/interviews/sessions/{id}` | Attach ElevenLabs' conversation id |
| `POST /api/v1/interviews/sessions/{id}/turns` | Append transcript turns; re-sent ordinals absorbed |
| `POST /api/v1/interviews/sessions/{id}/evaluation` | Validate, compute score, store once, return stored |
| `POST /api/v1/interviews/sessions/{id}/end` | Idempotent close (`completed` / `ended_early` / `failed`) |
| `DELETE /api/v1/interviews/sessions/{id}` | Remove session + transcript; already-gone is success |

Persistence is `interview_sessions` + `interview_turns` (schema v29). The
scorecard is one JSON column written exactly once — a retried tool call gets
the stored verdict back, never a rewrite of a score the candidate already
heard.

## Where things live

- API: `apps/api/src/waqil_api/interviews.py` (service + scoring),
  contracts in `contracts.py`, routes in `api.py`, tables in `database.py`.
- Web: `apps/web/app/interviews/page.tsx`,
  `components/interviews-workbench.tsx`, `hooks/use-interview-session.ts`,
  pure logic in `lib/interviews.ts`, API calls in `lib/api.ts`.
- Tests: `apps/api/tests/test_interviews.py` (includes the workflow-spec
  checks), `apps/web/tests/interviews-api.test.ts`.
- Agent deployment: `scripts/deploy_interview_agent.py`, reading
  `docs/interviews-workflow.json` + `docs/interviews-agent-prompt.txt` and
  writing `docs/interviews-agent-export.json`.

## Manual test script

1. With `WAQIL_INTERVIEW_ELEVENLABS_AGENT_ID` unset, open `/interviews`: the
   setup screen names the missing variable and Start is disabled. Set it and
   restart.
2. Fill everything but the job description — Start stays disabled and the
   problem shows at the field. Paste a JD, pick a round, reload the page:
   the draft survived.
3. Begin the interview with the microphone blocked: a recoverable error
   explains how to allow it; allowing and retrying connects.
4. Run a full round. Check: Chiron opens with the fixed line and never asks
   for the JD; each question opens "Question N" and the counter follows; the
   transcript panel fills; mute works; the timer runs.
5. Mid-round, type a reply instead of speaking — Chiron treats it as your
   answer and it appears once in the transcript.
6. After the fifth answer, Chiron announces the evaluation, then speaks a
   verdict whose overall score matches the scorecard on screen exactly.
   The scorecard shows the recommendation, five rubric scores, a quoted
   strongest answer, three problems with evidence, and one drill. No
   confetti.
7. Click "Retry this round" — a fresh session starts with the same context.
8. Start another round and click End interview after one answer: it asks
   once, then lands on "Ended early — not scored" with the transcript kept.
9. Start again and *tell Chiron* you want to stop after three answers:
   the scorecard arrives labeled Provisional.
10. Kill the network mid-round: the page lands on a recoverable failure with
    the transcript preserved and no invented score.
11. Narrow the window to phone width: the form stacks, the stage and
    scorecard stay usable. With reduced motion enabled, the live stage shows
    a still orb instead of the shader.
