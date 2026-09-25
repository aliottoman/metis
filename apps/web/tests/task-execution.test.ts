import assert from "node:assert/strict";
import test from "node:test";

import { deriveTaskExecution } from "../lib/task-execution.ts";
import type { RunEventV1 } from "../lib/types.ts";

const options = { runId: "run-1", active: true, connection: "live" };
function event(type: string, payload: Record<string, unknown> = {}, sequence = 1, runId = "run-1"): RunEventV1 {
  return { id: `${runId}-${sequence}`, run_id: runId, sequence, type, payload };
}

test("an empty or closed stream never invents a plan, checks, or completion", () => {
  const execution = deriveTaskExecution([], { runId: null, active: false, connection: "closed" });
  assert.equal(execution.state, "idle");
  assert.equal(execution.substantial, false);
  assert.equal(execution.checksReported, false);
  assert.deepEqual(execution.plan.steps, []);
  assert.deepEqual(execution.plan.files, []);
});

test("a published plan preserves its real steps and assumptions without claiming each step passed", () => {
  const execution = deriveTaskExecution([
    event("plan.created", { summary: "Prepare the presentation", route: "document", assumptions: ["Use the supplied notes"], steps: [{ id: "outline", title: "Write an outline", description: "Cover the main findings", kind: "respond" }, { id: "render", title: "Generate the deck", description: "Create the slides", kind: "tool" }] }),
    event("run.completed", {}, 2),
  ], { ...options, active: false, connection: "closed" });
  assert.equal(execution.substantial, true);
  assert.equal(execution.state, "completed");
  assert.equal(execution.plan.summary, "Prepare the presentation");
  assert.equal(execution.plan.steps[0]?.title, "Write an outline");
  assert.equal(execution.plan.steps.some((step) => step.checked), false);
  assert.deepEqual(execution.plan.assumptions, ["Use the supplied notes"]);
});

test("a file-plan revision preserves its acceptance criteria and named slices", () => {
  const execution = deriveTaskExecution([
    event("project.build_planned", { scope: "whole_app", files: ["app.ts"], slices: ["Working search"], scenarios: ["Search finds saved items"] }),
    event("project.plan_revised", { files: ["app.ts", "search.ts"], reason: "The search helper needs a repair" }, 2),
    event("project.coding_round", { changed_paths: ["search.ts"] }, 3),
  ], options);
  assert.deepEqual(execution.plan.scenarios, ["Search finds saved items"]);
  assert.equal(execution.plan.steps[0]?.title, "Working search");
  assert.equal(execution.plan.steps[0]?.checked, false);
  assert.deepEqual(execution.plan.files, [{ path: "app.ts", state: "planned" }, { path: "search.ts", state: "changed" }]);
  assert.equal(execution.plan.revision, "The search helper needs a repair");
  assert.equal(execution.plan.summary, "A plan for the whole application.");
});

test("only a named passing verification marks the matching slice and its files checked", () => {
  const execution = deriveTaskExecution([
    event("project.build_planned", { files: ["search.ts", "export.ts"], slices: ["Search", "Export"] }),
    event("project.vertical_slice_checked", { name: "Search", files: ["search.ts"], errors: 0, ran: 2 }, 2),
  ], options);
  assert.deepEqual(execution.plan.steps.map((step) => step.checked), [true, false]);
  assert.deepEqual(execution.plan.files.map((file) => file.state), ["checked", "planned"]);
});

test("changing a checked file requires fresh verification", () => {
  const execution = deriveTaskExecution([
    event("project.build_planned", { files: ["app.ts"] }),
    event("project.build_checked", { files: ["app.ts"], errors: 0, ran: 2 }, 2),
    event("project.coding_round", { changed_paths: ["app.ts"] }, 3),
  ], options);
  assert.equal(execution.plan.files[0]?.state, "changed");
});

test("a failed recheck removes stale passing evidence and a successful retry clears blockers", () => {
  const events = [
    event("project.build_planned", { files: ["app.ts"] }),
    event("project.staged_verified", { errors: 0, ran: 3 }, 2),
    event("project.staged_verified", { errors: 1, ran: 3, findings: [{ path: "app.ts", error: "Search returns no results", severity: "error" }] }, 3),
  ];
  const failed = deriveTaskExecution(events, options);
  assert.equal(failed.checksReported, false);
  assert.equal(failed.issues.length, 1);
  assert.match(failed.issues[0]!.detail, /app.ts: Search returns no results/);
  const recovered = deriveTaskExecution([...events, event("project.staged_verified", { errors: 0, ran: 3 }, 4)], options);
  assert.equal(recovered.issues.length, 0);
  assert.equal(recovered.checksPassed, 3);
});

test("repeated named check results are reconciled instead of inflating passed counts", () => {
  const execution = deriveTaskExecution([
    event("project.check_result", { name: "Typecheck", ok: false, error: "Invalid return value" }),
    event("project.check_result", { name: "Typecheck", ok: true }, 2),
    event("project.check_result", { name: "Typecheck", ok: true }, 3),
  ], options);
  assert.equal(execution.checksPassed, 1);
  assert.deepEqual(execution.issues, []);
});

test("events from a previously viewed run cannot contaminate this task's plan or status", () => {
  const execution = deriveTaskExecution([
    event("run.failed", { error: "Old failure" }, 1, "old-run"),
    event("project.build_planned", { files: ["old.ts"] }, 2, "old-run"),
    event("stage.entered", { stage: "planning", label: "Reading the request" }, 3),
  ], options);
  assert.equal(execution.state, "working");
  assert.deepEqual(execution.issues, []);
  assert.deepEqual(execution.plan.files, []);
});

test("pending decisions show a real pause and terminal events take precedence over stale flags", () => {
  assert.equal(deriveTaskExecution([], { ...options, pendingDecision: true }).state, "waiting");
  assert.equal(deriveTaskExecution([event("run.cancelled")], { ...options, pendingDecision: true }).state, "cancelled");
  assert.equal(deriveTaskExecution([], { ...options, connection: "reconnecting" }).state, "reconnecting");
});

test("stopping retains recorded outputs and decision history instead of pretending work was undone", () => {
  const execution = deriveTaskExecution([
    event("project.coding_round", { changed_paths: ["draft.md"] }),
    event("approval.applied", { status: "approved" }, 2),
    event("elicitation.answered", { text: "Use the concise version" }, 3),
    event("run.cancelled", {}, 4),
  ], { ...options, active: false, connection: "closed" });
  assert.equal(execution.state, "cancelled");
  assert.deepEqual(execution.changedFiles, ["draft.md"]);
  assert.equal(execution.decisions.length, 2);
});
