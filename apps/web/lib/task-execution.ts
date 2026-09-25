import type { RunEventV1 } from "./types.ts";

export type ExecutionState = "idle" | "working" | "waiting" | "reconnecting" | "completed" | "failed" | "cancelled";
export type ExecutionPlanStep = { id: string; title: string; description?: string; checked: boolean };
export type ExecutionIssue = { id: string; title: string; detail: string };
export type ExecutionPlan = {
  summary: string | null;
  steps: ExecutionPlanStep[];
  files: Array<{ path: string; state: "planned" | "changed" | "checked" }>;
  scenarios: string[];
  assumptions: string[];
  revision: string | null;
};

const TERMINAL = new Map<string, ExecutionState>([
  ["run.completed", "completed"], ["completed", "completed"],
  ["run.failed", "failed"], ["failed", "failed"],
  ["run.cancelled", "cancelled"], ["cancelled", "cancelled"],
]);
const HIDDEN = new Set(["message.delta", "message.reasoning", "assistant.message", "message.completed", "message.created", "run.created", "input.ingested"]);
const DECISIONS = new Set(["approval.applied", "approval.decided", "elicitation.answered", "project.verification_decided"]);
const VERIFICATION = new Set(["project.staged_verified", "project.build_checked", "project.vertical_slice_checked"]);

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.flatMap((item) => text(item) ? [text(item)!] : []) : [];
}
function count(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return null;
  return value;
}

/** A task summary built from recorded evidence. Plan items are never marked
 * complete merely because time passed or the stream closed. */
export function deriveTaskExecution(events: readonly RunEventV1[], options: {
  runId: string | null;
  active: boolean;
  connection: string;
  pendingDecision?: boolean;
  project?: boolean;
}) {
  const ordered = events.filter((event) => !options.runId || event.run_id === options.runId).slice().sort((a, b) => a.sequence - b.sequence);
  const plan: ExecutionPlan = { summary: null, steps: [], files: [], scenarios: [], assumptions: [], revision: null };
  const changed = new Set<string>();
  const checked = new Set<string>();
  const checkedSlices = new Set<string>();
  const issues = new Map<string, ExecutionIssue>();
  const toolChecks = new Map<string, boolean>();
  let aggregateChecks: number | null = null;
  let slices: string[] = [];
  let plannedFiles: string[] = [];
  let state: ExecutionState = options.active ? "working" : "idle";
  let terminal: RunEventV1 | null = null;
  let nontrivialPlan = false;

  for (const event of ordered) {
    const payload = event.payload;
    const terminalState = TERMINAL.get(event.type);
    if (terminalState) { state = terminalState; terminal = event; }
    if (["run.resumed", "run.recovered", "run.started"].includes(event.type)) {
      terminal = null;
      state = "working";
      issues.delete("run");
    }
    if (event.type === "plan.created") {
      plan.summary = text(payload.summary);
      plan.assumptions = strings(payload.assumptions);
      plan.steps = Array.isArray(payload.steps) ? payload.steps.flatMap((entry, index) => {
        const step = record(entry);
        const title = text(step.title);
        return title ? [{ id: text(step.id) ?? `step-${index}`, title, description: text(step.description) ?? undefined, checked: false }] : [];
      }) : [];
      nontrivialPlan = plan.steps.length > 1 || (typeof payload.route === "string" && !["direct", "ask_user"].includes(payload.route));
    }
    if (event.type === "project.build_planned") {
      plannedFiles = strings(payload.files);
      slices = strings(payload.slices);
      plan.scenarios = strings(payload.scenarios);
      plan.summary = plan.summary ?? (payload.scope === "whole_app" ? "A plan for the whole application." : payload.scope === "narrow" ? "A focused change within the project." : null);
      plan.revision = null;
    }
    if (event.type === "project.plan_revised") {
      // A revision changes only the fields it carries; acceptance scenarios
      // and named slices still belong to the original plan.
      if (Array.isArray(payload.files)) plannedFiles = strings(payload.files);
      if (Array.isArray(payload.slices)) slices = strings(payload.slices);
      if (Array.isArray(payload.scenarios)) plan.scenarios = strings(payload.scenarios);
      plan.revision = text(payload.reason);
    }
    if (event.type === "project.coding_round") strings(payload.changed_paths).forEach((path) => { changed.add(path); checked.delete(path); });
    if (VERIFICATION.has(event.type)) {
      const errors = count(payload.errors);
      if (errors === 0) {
        strings(payload.files).forEach((path) => checked.add(path));
        if (event.type === "project.vertical_slice_checked" && text(payload.name)) checkedSlices.add(text(payload.name)!);
        // A successful whole-changeset check supersedes earlier failures.
        if (event.type !== "project.vertical_slice_checked") {
          for (const key of issues.keys()) if (key.startsWith("check:")) issues.delete(key);
        }
      }
      const key = event.type === "project.vertical_slice_checked" ? `check:slice:${text(payload.name) ?? "current"}` : "check:build";
      if (errors !== null) {
        if (errors > 0) {
          if (event.type !== "project.vertical_slice_checked") {
            aggregateChecks = null;
            checked.clear();
            checkedSlices.clear();
          }
          const findings = Array.isArray(payload.findings) ? payload.findings.map(record).filter((item) => item.severity !== "warning") : [];
          const first = findings.find((item) => text(item.error));
          issues.set(key, {
            id: key,
            title: `${errors} ${errors === 1 ? "check needs" : "checks need"} attention`,
            detail: first ? [text(first.path), text(first.error)].filter(Boolean).join(": ") : text(payload.outcome) ?? text(payload.name) ?? "The latest verification reported blocking findings.",
          });
          strings(payload.files).forEach((path) => checked.delete(path));
          if (text(payload.name)) checkedSlices.delete(text(payload.name)!);
        } else issues.delete(key);
      }
      if (event.type !== "project.vertical_slice_checked" && errors === 0 && count(payload.ran) !== null) aggregateChecks = count(payload.ran);
    }
    if (event.type === "project.check_result") {
      const name = text(payload.name) ?? "Verification check";
      if (typeof payload.ok === "boolean") {
        toolChecks.set(name, payload.ok);
        if (payload.ok) issues.delete(`check:${name}`);
        else issues.set(`check:${name}`, { id: `check:${name}`, title: name, detail: text(payload.error) ?? (payload.timed_out ? "This check timed out." : "This check did not pass.") });
      }
    }
    if (event.type === "run.model_exhausted") {
      issues.set("model", { id: "model", title: "No model could continue", detail: "The available model routes were exhausted. Review the activity below, then adjust the model or task and try again." });
    }
    if (event.type === "run.model_fallback" || event.type === "model.started") issues.delete("model");
    if (event.type === "run.failed" || event.type === "failed") {
      issues.set("run", { id: "run", title: "Task stopped before completion", detail: text(payload.error) ?? text(payload.message) ?? text(payload.summary) ?? "Review the latest activity and send a revised request to continue." });
    }
  }

  if (slices.length) plan.steps = slices.map((title, index) => ({ id: `slice-${index}`, title, checked: checkedSlices.has(title) }));
  plan.files = Array.from(new Set(plannedFiles)).map((path) => ({ path, state: checked.has(path) ? "checked" : changed.has(path) ? "changed" : "planned" }));
  if (!terminal) {
    state = options.pendingDecision ? "waiting"
      : options.active && ["reconnecting", "error"].includes(options.connection) ? "reconnecting"
        : options.active ? "working" : "idle";
  }
  const progress = ordered.filter((event) => !HIDDEN.has(event.type) && !event.type.includes("delta") && !DECISIONS.has(event.type));
  const decisions = ordered.filter((event) => DECISIONS.has(event.type));
  const substantial = Boolean(options.runId) && (Boolean(options.project) || nontrivialPlan || ordered.some((event) => event.type.startsWith("project.") || event.type === "artifact.created") || progress.filter((event) => event.type.startsWith("tool.")).length > 0);
  return {
    state,
    substantial,
    plan,
    progress,
    decisions,
    issues: Array.from(issues.values()),
    changedFiles: Array.from(changed),
    checksPassed: aggregateChecks ?? Array.from(toolChecks.values()).filter(Boolean).length,
    checksReported: aggregateChecks !== null || toolChecks.size > 0,
    latest: progress[progress.length - 1] ?? null,
    terminal,
  };
}

export type TaskExecution = ReturnType<typeof deriveTaskExecution>;

export const EXECUTION_STATE_LABELS: Record<ExecutionState, string> = {
  idle: "Ready", working: "In progress", waiting: "Waiting for you", reconnecting: "Reconnecting",
  completed: "Completed", failed: "Needs attention", cancelled: "Stopped",
};
