import assert from "node:assert/strict";
import test from "node:test";

import { elicitationFrom, latestPendingElicitation } from "../lib/elicitations.ts";
import type { RunEventV1 } from "../lib/types.ts";

function event(partial: Partial<RunEventV1> & { type: string; sequence: number }): RunEventV1 {
  return {
    id: `e${partial.sequence}`,
    run_id: "run-1",
    thread_id: "conv-1",
    payload: {},
    ...partial,
  };
}

const requested = event({
  type: "elicitation.requested",
  sequence: 1,
  payload: {
    id: "elic_1",
    run_id: "run-1",
    question: "Which environment?",
    options: ["staging", "prod"],
    allow_text: true,
  },
});

test("elicitationFrom parses the request event and ignores status events", () => {
  const parsed = elicitationFrom(requested);
  assert.ok(parsed);
  assert.equal(parsed?.id, "elic_1");
  assert.equal(parsed?.question, "Which environment?");
  assert.deepEqual(parsed?.options, ["staging", "prod"]);
  assert.equal(parsed?.allow_text, true);

  assert.equal(elicitationFrom(event({ type: "run.awaiting_input", sequence: 2 })), null);
  assert.equal(
    elicitationFrom(event({ type: "elicitation.answered", sequence: 3, payload: { elicitation_id: "elic_1" } })),
    null,
  );
});

test("a request with no id or no question is not actionable", () => {
  assert.equal(
    elicitationFrom(event({ type: "elicitation.requested", sequence: 1, payload: { question: "hi" } })),
    null,
  );
  assert.equal(
    elicitationFrom(event({ type: "elicitation.requested", sequence: 1, payload: { id: "elic_1" } })),
    null,
  );
});

test("latestPendingElicitation surfaces the open question", () => {
  const pending = latestPendingElicitation([requested], new Set());
  assert.equal(pending?.id, "elic_1");
});

test("an answered question is no longer pending", () => {
  const events = [
    requested,
    event({ type: "elicitation.answered", sequence: 2, payload: { elicitation_id: "elic_1" } }),
  ];
  assert.equal(latestPendingElicitation(events, new Set()), null);
});

test("a locally answered id is hidden before its events arrive", () => {
  assert.equal(latestPendingElicitation([requested], new Set(["elic_1"])), null);
});

test("a finished run has nothing waiting, even with a request in history", () => {
  const events = [requested, event({ type: "run.completed", sequence: 4 })];
  assert.equal(latestPendingElicitation(events, new Set()), null);
});

test("the newest unanswered question wins", () => {
  const second = event({
    type: "elicitation.requested",
    sequence: 5,
    payload: { id: "elic_2", question: "Second?", options: [], allow_text: true },
  });
  const pending = latestPendingElicitation([requested, second], new Set());
  assert.equal(pending?.id, "elic_2");
});
