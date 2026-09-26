import assert from "node:assert/strict";
import test from "node:test";

import { latestPendingApproval } from "../lib/approvals.ts";
import { latestPendingElicitation } from "../lib/elicitations.ts";
import { mergeExecutionEvents, retainExecutionMilestone } from "../lib/execution-milestones.ts";
import { deriveTaskExecution } from "../lib/task-execution.ts";
import type { RunEventV1 } from "../lib/types.ts";

function event(type: string, sequence: number, payload: Record<string, unknown> = {}, runId = "run-1"): RunEventV1 {
  return { id: `${runId}-${sequence}`, run_id: runId, sequence, type, payload };
}

test("a replay longer than 300 events keeps its plan, decisions, and outputs", () => {
  const replay = [
    event("run.started", 1),
    event("plan.created", 2, { summary: "Build the search page", route: "existing_tool" }),
    event("project.build_planned", 3, { files: ["search.ts"], slices: ["Working search"], scenarios: ["Search shows matching records"] }),
    event("approval.applied", 4, { approval_id: "approval-1", status: "approved" }),
    event("artifact.created", 5, { id: "file-1", filename: "search-report.pdf" }),
    ...Array.from({ length: 600 }, (_, index) => event("message.delta", index + 6, { delta: "text" })),
    event("project.plan_revised", 606, { files: ["search.ts", "query.ts"], reason: "Add the query adapter" }),
  ];
  const milestones = replay.reduce(retainExecutionMilestone, [] as RunEventV1[]);
  const merged = mergeExecutionEvents(milestones, replay.slice(-300), "run-1");
  const execution = deriveTaskExecution(merged, { runId: "run-1", active: true, connection: "live" });
  assert.equal(execution.plan.summary, "Build the search page");
  assert.deepEqual(execution.plan.files.map((file) => file.path), ["search.ts", "query.ts"]);
  assert.deepEqual(execution.plan.scenarios, ["Search shows matching records"]);
  assert.equal(execution.decisions.length, 1);
  assert.ok(merged.some((item) => item.type === "artifact.created"));
  assert.equal(new Set(merged.map((item) => item.id)).size, merged.length);
});

test("thousands of checks remain bounded without evicting the first event, latest plan, approval, or output", () => {
  let milestones: RunEventV1[] = [];
  for (const item of [event("run.started", 1), event("project.build_planned", 2, { files: ["app.ts"] }), event("approval.required", 3, { id: "pending-1" }), event("artifact.created", 4, { id: "artifact-1" })]) milestones = retainExecutionMilestone(milestones, item);
  for (let index = 0; index < 2_000; index += 1) milestones = retainExecutionMilestone(milestones, event("project.check_result", index + 5, { name: `Check ${index}`, ok: index % 2 === 0 }));
  milestones = retainExecutionMilestone(milestones, event("project.build_planned", 2005, { files: ["app.ts", "tests.ts"] }));
  assert.ok(milestones.length <= 256);
  assert.ok(milestones.some((item) => item.sequence === 1));
  assert.ok(milestones.some((item) => item.type === "approval.required"));
  assert.ok(milestones.some((item) => item.type === "artifact.created"));
  assert.deepEqual(milestones.find((item) => item.type === "project.build_planned")?.payload.files, ["app.ts", "tests.ts"]);
  assert.ok(milestones.some((item) => item.payload.name === "Check 1999"));
});

test("routine deltas reuse the retained collection and a new run excludes old milestones", () => {
  const milestones = retainExecutionMilestone([], event("plan.created", 1, { summary: "Old task" }));
  assert.equal(retainExecutionMilestone(milestones, event("message.delta", 2)), milestones);
  const current = retainExecutionMilestone(milestones, event("run.started", 1, {}, "run-2"));
  assert.deepEqual(current.map((item) => item.run_id), ["run-2"]);
  assert.deepEqual(mergeExecutionEvents(milestones, current, "run-2"), current);
});

function decisionReplay(later: RunEventV1[] = []) {
  const replay = [
    event("approval.required", 1, { id: "approval-1", title: "Apply the reviewed change?" }),
    event("elicitation.requested", 2, { id: "question-1", question: "Which environment?" }),
    ...later,
    ...Array.from({ length: 600 }, (_, index) => event("message.delta", index + 10, { delta: "text" })),
  ];
  return {
    milestones: replay.reduce(retainExecutionMilestone, [] as RunEventV1[]),
    recent: replay.slice(-300),
  };
}

test("long tasks keep actionable approvals and questions after their requests leave recent activity", () => {
  const { milestones, recent } = decisionReplay();
  assert.equal(latestPendingApproval(recent, new Set()), null);
  assert.equal(latestPendingElicitation(recent, new Set()), null);
  const events = mergeExecutionEvents(milestones, recent, "run-1");
  assert.equal(latestPendingApproval(events, new Set())?.id, "approval-1");
  assert.equal(latestPendingElicitation(events, new Set())?.id, "question-1");
  assert.equal(latestPendingApproval(events, new Set(["approval-1"])), null);
  assert.equal(latestPendingElicitation(events, new Set(["question-1"])), null);
});

test("retained decisions do not resurface after another window resolves them", () => {
  const { milestones, recent } = decisionReplay([
    event("approval.applied", 3, { approval_id: "approval-1" }),
    event("elicitation.answered", 4, { elicitation_id: "question-1" }),
  ]);
  const events = mergeExecutionEvents(milestones, recent, "run-1");
  assert.equal(latestPendingApproval(events, new Set()), null);
  assert.equal(latestPendingElicitation(events, new Set()), null);
});

test("completed tasks and switching runs cannot expose retained decisions from the old run", () => {
  const { milestones, recent } = decisionReplay([event("run.completed", 3)]);
  for (const runId of ["run-1", "run-2", null]) {
    const events = mergeExecutionEvents(milestones, recent, runId);
    assert.equal(latestPendingApproval(events, new Set()), null);
    assert.equal(latestPendingElicitation(events, new Set()), null);
  }
});
