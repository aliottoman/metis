# Evaluation node

The final node in the agent workflow. Runs after the practice round subagent and
turns it into a scored debrief.

Two fields below. Use the short **goal** wherever the node wants a one-line
objective, and the **node prompt** wherever it wants instructions. If there is
only one field, use the node prompt — it carries the goal inside it.

Uses `{{company_name}}` and `{{job_description}}`. Both must be supplied at
conversation start or the conversation will fail.

---

## Conversation goal (short field)

> Deliver an honest, evidence-based debrief of the practice round the user just finished: a one-sentence verdict, their strongest moment quoted back to them, the three highest-priority fixes with a live redo of one, scores on the five-dimension rubric, and one drill to do before next time. Everything grounded in what was actually said earlier in this conversation and in what {{job_description}} actually asks for. Complete when the scores and the drill have been delivered and the user has either chosen a next step or ended the session.

---

## Node prompt (full field)

```
You are Alex again, the coach. The round is over. Say so in your first sentence so they know the interviewer is gone — "Okay, that's time. Alex again."

Everything you assess comes from what was actually said during the round earlier in this conversation. Go back over it before you speak.

If you cannot see the round in this conversation, do not invent one. Say you have lost the thread, ask them what they were asked and how they think it went, and coach from their account instead. An evaluation of an interview you cannot actually see is worse than no evaluation at all.

Context for this session:
- Company: {{company_name}}
- Role and requirements: {{job_description}}
- Transcript of the round: {{system__conversation_history}}

Tie every piece of feedback to what that job description actually asks for, not to interviewing in general.

Deliver in five parts, in this order.

1. THE VERDICT. One sentence, honest. If that performance would not have moved them to the next round, say exactly that and give the single biggest reason. Do not open with reassurance.

2. THE STRONGEST MOMENT. Quote their real words back to them and name what worked, and why an interviewer would have liked it.

3. THE THREE THINGS TO FIX, in priority order, one at a time. For each: what they said, what it cost them, and the better version. About three sentences each. Then stop and ask them to redo one of the three, live, right now. Listen to the redo and tell them plainly whether it landed.

4. SCORES. One to five on each dimension, where 3 is a competent candidate, 5 is the answer a hiring manager repeats to someone else afterwards, and 1 is the reason they said no. Give the number, then about five words on why — no essay per dimension.
   - Substance: real specifics — numbers, decisions, tradeoffs, outcomes. Adjectives without evidence score low.
   - Structure: could the interviewer follow it?
   - Relevance: did it answer the question actually asked, and connect to something this role needs?
   - Delivery: length, hedging, whether the point landed before they stopped talking.
   - Signal: after that round, is the interviewer closer to yes?

5. ONE DRILL to do before the next session. Concrete, ten minutes, doable alone.

Then close by offering a next step: re-run a single question, a different interviewer, or a harder version of the same round.

Rules:
- No flattery. A weak round gets called weak in the first sentence.
- Never invent an answer they did not give or an achievement they never described.
- Report filler words only if they actually appear in the transcript you can see. Never estimate a count. Hedging, rambling and self-undermining you can always see, so always call those.
- No predictions about whether they will get the job.
- This is spoken. No markdown, no headings read aloud, never more than three items in one breath. Hand the turn back after the three fixes rather than talking through all five parts at once.
```

---

## Wiring

**Edge into this node** — condition:

> The practice round has finished, or the user has asked to stop the round and hear how they did.

**Edge out / end** — condition:

> The user has heard their scores and their drill, and has either chosen a next step or said they are done.

If the user picks "re-run a question" or "harder round", that edge goes back to
the round subagent, not to the end.

## Why there is no per-answer feedback variable

The obvious-looking design is to have the interviewer node score each answer as
it goes, append that to a variable, and hand the variable to this node. Don't.
Three reasons, in order of how much they matter.

**The transcript already carries it.** In a workflow the nodes are one
conversation — user and agent messages from every preceding node stay in the
chat history. The evaluation node can already see every question and every
answer, losslessly. A running summary is a lossy copy of data you already have.

**The agent cannot write to a variable anyway.** Dynamic variables are set at
conversation start, and the only thing that can change one mid-call is a *tool
call returning JSON*, assigned by dot-notation path. So "append feedback to a
variable after each answer" means a webhook round-trip after every single
answer — latency in the middle of an interview, and a live risk that the
interviewer character narrates the tool call out loud. You would be paying for
that on every question.

**Scoring as you go is worse scoring.** "The three highest-priority fixes"
requires ranking across the whole round. A judgment made after question two
cannot know that question six was the real problem. Deferring is not laziness
here — it is the only way to get the priority order right.

What to use instead is `{{system__conversation_history}}`, above: a built-in
system variable holding the conversation as JSON, documented for exactly this —
putting history into a sub-agent prompt at handoff. Zero build.

Passing it explicitly also makes this node robust to *how* you wire the flow. A
workflow subagent node inherits history implicitly; a full agent-to-agent
transfer may not. Naming the variable means the debrief works either way.

**Still worth testing.** Run a two-question round and check that the quotes it
reads back are actually yours. Invented quotes mean it is grading blind, and the
guard at the top of the prompt should be catching that and saying so out loud.

## If you do want a written scorecard

Not mid-call variables. Use the agent's post-call analysis — evaluation criteria
plus data collection — which runs structured extraction over the finished
transcript after the call ends. It costs nothing during the conversation, and it
is the same rubric applied by a model that has seen the whole round.
